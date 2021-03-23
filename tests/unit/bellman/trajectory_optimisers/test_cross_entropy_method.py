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

import pytest
import tensorflow as tf
from tf_agents.trajectories.time_step import StepType, restart, time_step_spec

from bellman.distributions.utils import create_uniform_distribution_from_spec
from bellman.environments.environment_model import EnvironmentModel
from bellman.environments.initial_state_distribution_model import (
    create_uniform_initial_state_distribution,
)
from bellman.environments.termination_model import ConstantFalseTermination
from bellman.environments.transition_model.keras_model.keras import KerasTransitionModel
from bellman.environments.transition_model.keras_model.linear import LinearTransitionNetwork
from bellman.trajectory_optimisers.cross_entropy_method import (
    CrossEntropyMethodPolicyStateUpdater,
    cross_entropy_method_trajectory_optimisation,
)
from tests.tools.bellman.environments.reward_model import ConstantReward
from tests.tools.bellman.trajectories.trajectory import generate_dummy_trajectories


@pytest.mark.parametrize("invalid_learning_rate", [-0.1, 1.1])
def test_invalid_learning_rate(
    observation_space, action_space, horizon, invalid_learning_rate
):
    time_step_space = time_step_spec(observation_space)
    with pytest.raises(AssertionError) as excinfo:
        cross_entropy_method_trajectory_optimisation(
            time_step_space,
            action_space,
            horizon=horizon,
            population_size=5,
            number_of_particles=1,
            num_elites=2,
            learning_rate=invalid_learning_rate,
            max_iterations=1,
        )

    assert "learning rate" in str(excinfo)


def test_invalid_num_elites(observation_space, action_space, horizon):

    # some fixed parameters
    population_size = 10
    number_of_particles = 1

    # set up the environment model
    network = LinearTransitionNetwork(observation_space)
    model = KerasTransitionModel([network], observation_space, action_space)
    environment_model = EnvironmentModel(
        model,
        ConstantReward(observation_space, action_space),
        ConstantFalseTermination(observation_space),
        create_uniform_initial_state_distribution(observation_space),
        population_size,
    )

    # set up the trajectory optimizer
    time_step_space = time_step_spec(observation_space)
    optimiser = cross_entropy_method_trajectory_optimisation(
        time_step_space,
        action_space,
        horizon=horizon,
        population_size=population_size,
        number_of_particles=number_of_particles,
        num_elites=population_size + 1,
        learning_rate=0.1,
        max_iterations=1,
    )

    # remember the time step comes from the real environment with batch size 1
    observation = create_uniform_distribution_from_spec(observation_space).sample(
        sample_shape=(1,)
    )
    initial_time_step = restart(observation, batch_size=1)

    # run
    with pytest.raises(AssertionError) as excinfo:
        optimiser.optimise(initial_time_step, environment_model)

    assert "num_elites" in str(excinfo)


def test_update_policy_state_with_trajectories_that_reset_mid_way(
    observation_space, action_space, horizon
):
    policy_state_updater = CrossEntropyMethodPolicyStateUpdater(
        num_elites=1, learning_rate=0.1
    )
    policy_state = None
    trajectory = generate_dummy_trajectories(
        observation_space, action_space, batch_size=1, trajectory_length=2
    )
    trajectory = trajectory.replace(step_type=tf.constant([[StepType.LAST, StepType.FIRST]]))
    with pytest.raises(AssertionError) as excinfo:
        policy_state_updater.update(policy_state, trajectory, 1)

    assert "contain a terminal state" in str(excinfo)


def test_update_policy_state_with_trajectories_that_do_not_terminate(
    observation_space, action_space, horizon
):
    policy_state_updater = CrossEntropyMethodPolicyStateUpdater(
        num_elites=1, learning_rate=0.1
    )
    policy_state = None
    trajectory = generate_dummy_trajectories(
        observation_space, action_space, batch_size=1, trajectory_length=2
    )
    with pytest.raises(AssertionError) as excinfo:
        policy_state_updater.update(policy_state, trajectory, 1)

    assert "must end in a terminal state" in str(excinfo)
