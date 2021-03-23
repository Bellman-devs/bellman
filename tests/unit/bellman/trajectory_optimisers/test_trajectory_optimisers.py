# Copyright 2021 The Bellman Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import chain, repeat
from typing import Optional

import numpy as np
import pytest
import tensorflow as tf
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories.time_step import StepType, restart, time_step_spec
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.typing import types
from tf_agents.utils.common import replicate

from bellman.distributions.utils import create_uniform_distribution_from_spec
from bellman.environments.environment_model import EnvironmentModel
from bellman.environments.initial_state_distribution_model import (
    DeterministicInitialStateModel,
    create_uniform_initial_state_distribution,
)
from bellman.environments.termination_model import ConstantFalseTermination
from bellman.trajectory_optimisers.trajectory_optimisers import (
    PolicyStateUpdater,
    PolicyTrajectoryOptimiser,
)
from tests.tools.bellman.environments.reward_model import ConstantReward
from tests.tools.bellman.environments.transition_model.transition_model import (
    StubTrainableTransitionModel,
)
from tests.tools.bellman.samplers.samplers import get_optimiser_and_environment_model
from tests.tools.bellman.trajectory_optimisers.environment_model_components import (
    OBSERVATION_SPACE_SPEC,
    TrajectoryOptimiserTerminationModel,
    TrajectoryOptimiserTransitionModel,
)


def test_trajectory_optimiser_with_particles_actions_shape(
    action_space, horizon, population_size, number_of_particles
):
    observation = create_uniform_distribution_from_spec(OBSERVATION_SPACE_SPEC).sample(
        sample_shape=(population_size * number_of_particles,)
    )
    transition_model = TrajectoryOptimiserTransitionModel(action_space, repeat(observation))
    reward = ConstantReward(OBSERVATION_SPACE_SPEC, action_space, -1.0)
    termination_model = ConstantFalseTermination(OBSERVATION_SPACE_SPEC)
    environment_model = EnvironmentModel(
        transition_model=transition_model,
        reward_model=reward,
        termination_model=termination_model,
        initial_state_distribution_model=DeterministicInitialStateModel(StepType.FIRST),
        batch_size=population_size * number_of_particles,
    )

    time_step_space = time_step_spec(OBSERVATION_SPACE_SPEC)

    policy = RandomTFPolicy(time_step_space, action_space, automatic_state_reset=False)
    trajectory_optimiser = PolicyTrajectoryOptimiser(
        policy,
        horizon=horizon,
        population_size=population_size,
        number_of_particles=number_of_particles,
        max_iterations=2,
    )

    initial_time_step = restart(tf.expand_dims(observation[0], axis=0))
    optimal_actions = trajectory_optimiser.optimise(initial_time_step, environment_model)

    assert optimal_actions.shape == (horizon + 1,) + action_space.shape


def test_mismatch_between_optimizer_and_environment_model_batch_size(
    observation_space, action_space, optimiser_policy_trajectory_optimiser_factory
):
    time_step_space = time_step_spec(observation_space)
    environment_model = EnvironmentModel(
        StubTrainableTransitionModel(
            observation_space, action_space, predict_state_difference=True
        ),
        ConstantReward(observation_space, action_space),
        ConstantFalseTermination(observation_space),
        create_uniform_initial_state_distribution(observation_space),
    )
    population_size = environment_model.batch_size + 1
    trajectory_optimiser = optimiser_policy_trajectory_optimiser_factory(
        time_step_space, action_space, 1, population_size, 1
    )
    # remember the time step comes from the real environment with batch size 1
    observation = create_uniform_distribution_from_spec(observation_space).sample(
        sample_shape=(1,)
    )
    time_step = restart(observation, batch_size=1)
    with pytest.raises(AssertionError) as excinfo:
        _ = trajectory_optimiser.optimise(time_step, environment_model)

    assert "batch_size parameter is not equal to environment_model.batch_size" in str(excinfo)


class StubPolicyStateUpdater(PolicyStateUpdater):
    def __init__(self):
        self.step_types = []

    def update(
        self,
        policy_state: types.NestedTensor,
        trajectories: Trajectory,
        number_of_particles: int,
    ) -> types.NestedTensor:
        self.step_types.append(trajectories.step_type)


def test_trajectory_optimiser_pathological_trajectories(action_space, horizon, batch_size):
    """
    The replay buffer is a FIFO buffer of fixed capacity. Ensure that the capacity is sufficient
    such that the initial observation is still present in the buffer even in the pathological case
    where all trajectories are of length 2.
    """

    # construct the environment model
    observations = list(
        chain.from_iterable(
            repeat(
                [
                    replicate(tf.constant(StepType.FIRST), [batch_size]),
                    replicate(tf.constant(StepType.LAST), [batch_size]),
                ],
                horizon,
            )
        )
    )

    transition_model = TrajectoryOptimiserTransitionModel(action_space, iter(observations))
    reward = ConstantReward(OBSERVATION_SPACE_SPEC, action_space, -1.0)
    termination_model = TrajectoryOptimiserTerminationModel(OBSERVATION_SPACE_SPEC)
    environment_model = EnvironmentModel(
        transition_model=transition_model,
        reward_model=reward,
        termination_model=termination_model,
        initial_state_distribution_model=DeterministicInitialStateModel(StepType.FIRST),
        batch_size=batch_size,
    )

    time_step_space = time_step_spec(OBSERVATION_SPACE_SPEC)
    policy = RandomTFPolicy(time_step_space, action_space)
    stub_policy_state_updater = StubPolicyStateUpdater()
    trajectory_optimiser = PolicyTrajectoryOptimiser(
        policy,
        horizon,
        population_size=batch_size,
        max_iterations=1,
        policy_state_updater=stub_policy_state_updater,
    )

    time_step = restart(tf.expand_dims(tf.constant(StepType.FIRST), axis=0), batch_size=1)

    trajectory_optimiser.optimise(time_step, environment_model)

    stored_trajectory = stub_policy_state_updater.step_types[0]
    assert stored_trajectory[0][0] == StepType.FIRST


@pytest.mark.parametrize("max_iterations", [1, 5, 10])
def test_trajectory_optimiser_each_iteration_starts_with_the_initial_observation(
    action_space, horizon, batch_size, max_iterations
):
    class WrappedRandomTFPolicy(TFPolicy):
        def __init__(
            self,
            ts_spec: ts.TimeStep,
            action_spec: types.NestedTensorSpec,
            env_model: EnvironmentModel,
        ):
            super().__init__(ts_spec, action_spec)

            self._internal_policy = RandomTFPolicy(ts_spec, action_space)

            self._environment_model = env_model

        def _action(
            self,
            time_step: ts.TimeStep,
            policy_state: types.NestedTensor,
            seed: Optional[types.Seed],
        ) -> policy_step.PolicyStep:
            np.testing.assert_array_equal(
                time_step.observation, self._environment_model.current_time_step().observation
            )
            return self._internal_policy._action(time_step, policy_state, seed)

        def _distribution(
            self, time_step: ts.TimeStep, policy_state: types.NestedTensorSpec
        ) -> policy_step.PolicyStep:
            raise NotImplementedError()

    observations = list(
        repeat(
            replicate(tf.constant(StepType.MID), [batch_size]), max_iterations * (horizon + 1)
        )
    )

    transition_model = TrajectoryOptimiserTransitionModel(action_space, iter(observations))
    reward = ConstantReward(OBSERVATION_SPACE_SPEC, action_space, -1.0)
    termination_model = TrajectoryOptimiserTerminationModel(OBSERVATION_SPACE_SPEC)
    environment_model = EnvironmentModel(
        transition_model=transition_model,
        reward_model=reward,
        termination_model=termination_model,
        initial_state_distribution_model=DeterministicInitialStateModel(StepType.FIRST),
        batch_size=batch_size,
    )

    time_step_space = time_step_spec(OBSERVATION_SPACE_SPEC)

    policy = WrappedRandomTFPolicy(time_step_space, action_space, environment_model)
    trajectory_optimiser = PolicyTrajectoryOptimiser(
        policy,
        horizon=horizon,
        population_size=batch_size,
        max_iterations=max_iterations,
    )

    initial_time_step = restart(
        tf.expand_dims(tf.constant(StepType.FIRST), axis=0), batch_size=1
    )

    trajectory_optimiser.optimise(initial_time_step, environment_model)


def test_tf_env_wrapper_is_reset_at_the_start_of_each_iteration(action_space):

    observations_array = [
        # First iteration
        [StepType.FIRST, StepType.FIRST],
        [StepType.LAST, StepType.MID],
        [StepType.FIRST, StepType.MID],
        [StepType.LAST, StepType.LAST],
        # Second iteration
        [StepType.FIRST, StepType.FIRST],
        [StepType.MID, StepType.MID],
        [StepType.MID, StepType.MID],
        [StepType.MID, StepType.LAST],
        [StepType.MID, StepType.FIRST],
    ]
    observations = [tf.concat(ob_array, axis=0) for ob_array in observations_array]

    transition_model = TrajectoryOptimiserTransitionModel(action_space, iter(observations))
    reward = ConstantReward(OBSERVATION_SPACE_SPEC, action_space, -1.0)
    termination_model = TrajectoryOptimiserTerminationModel(OBSERVATION_SPACE_SPEC)
    environment_model = EnvironmentModel(
        transition_model=transition_model,
        reward_model=reward,
        termination_model=termination_model,
        initial_state_distribution_model=DeterministicInitialStateModel(StepType.FIRST),
        batch_size=2,
    )

    time_step_space = time_step_spec(OBSERVATION_SPACE_SPEC)

    policy = RandomTFPolicy(
        time_step_space, action_space, automatic_state_reset=False, validate_args=False
    )
    policy_state_updater = StubPolicyStateUpdater()
    trajectory_optimiser = PolicyTrajectoryOptimiser(
        policy,
        horizon=3,
        population_size=2,
        max_iterations=2,
        policy_state_updater=policy_state_updater,
    )

    initial_time_step = restart(
        tf.expand_dims(tf.constant(StepType.FIRST), axis=0), batch_size=1
    )

    trajectory_optimiser.optimise(initial_time_step, environment_model)

    for stored_trajectories in policy_state_updater.step_types:
        np.testing.assert_equal(stored_trajectories[:, 0], np.array([0, 0]))
