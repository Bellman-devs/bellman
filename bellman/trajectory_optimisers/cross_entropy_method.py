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

"""
This module defines a method which can be used to optimise trajectories according to the cross
entropy method as described in:

Botev, Z.; Kroese, D.; Rubinstein, R.; L’Ecuyer, P. (2013)
The Cross-Entropy Method for Optimization. In Handbook of Statistics, Vol: 31, Page: 35-59

The helper function `cross_entropy_method_trajectory_optimisation` constructs a
`TrajectoryOptimiser` object which has been configured to use the cross entropy method.
"""

import tensorflow as tf
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.typing import types
from tf_agents.typing.types import NestedBoundedTensorSpec

from bellman.policies.cross_entropy_method_policy import (
    CrossEntropyMethodPolicy,
    sample_action_batch,
)
from bellman.trajectory_optimisers.particles import averaged_particle_returns
from bellman.trajectory_optimisers.trajectory_optimisers import (
    PolicyStateUpdater,
    PolicyTrajectoryOptimiser,
    TrajectoryOptimiser,
    TrajectorySelector,
)


def cross_entropy_method_trajectory_optimisation(
    time_step_spec: TimeStep,
    action_spec: NestedBoundedTensorSpec,
    horizon: int,
    population_size: int,
    number_of_particles: int,
    num_elites: int,
    learning_rate: float,
    max_iterations: int,
) -> TrajectoryOptimiser:
    """
    Helper function to construct a trajectory optimiser which uses the cross entropy method.

    :param time_step_spec: A `TimeStep` spec of the expected time_steps.
    :param action_spec: A nest of BoundedTensorSpec representing the actions.
    :param horizon: Number of steps taken in the environment in each virtual rollout.
    :param population_size: The number of candidate sequences of actions at each iteration.
    :param number_of_particles: Number of monte-carlo rollouts of each action trajectory.
    :param num_elites: The number of elite trajectories used for updating the parameters
        of distribution for each action. This should be a proportion of `population_size`
        rollouts.
    :param learning_rate: The learning rate for updating the distribution parameters.
    :param max_iterations: The maximum number of iterations to use for optimisation.

    :return: A `TrajectoryOptimiser` object which uses the cross entropy method.
    """
    policy = CrossEntropyMethodPolicy(time_step_spec, action_spec, horizon, population_size)
    policy_state_updater = CrossEntropyMethodPolicyStateUpdater(
        num_elites=num_elites, learning_rate=learning_rate
    )
    trajectory_selector = CrossEntropyMethodTrajectorySelector(action_spec)
    return PolicyTrajectoryOptimiser(
        policy,
        horizon,
        population_size,
        number_of_particles=number_of_particles,
        max_iterations=max_iterations,
        trajectory_selector=trajectory_selector,
        policy_state_updater=policy_state_updater,
    )


class CrossEntropyMethodTrajectorySelector(TrajectorySelector):
    """
    Use the mean of the distribution as the optimal trajectory.
    """

    def add_candidate_trajectories(
        self,
        trajectories: Trajectory,
        policy_state: types.NestedTensor,
        number_of_particles: int,
    ):
        self._optimal_trajectory.assign(policy_state[0])


class CrossEntropyMethodPolicyStateUpdater(PolicyStateUpdater):
    """
    This `PolicyStateUpdater` updates the policy state for the cross entropy method policy.
    """

    def __init__(self, num_elites: int, learning_rate: float):
        """
        :param num_elites: number of samples to use to update sampling distribution
        :param learning_rate: in [0,1] determines how quickly to update sampling distribution
        """
        assert 0.0 <= learning_rate <= 1, "Invalid learning rate"

        self._num_elites = num_elites
        self._lr = learning_rate

    def update(
        self,
        policy_state: types.NestedTensor,
        trajectories: Trajectory,
        number_of_particles: int,
    ) -> types.NestedTensor:
        """
        Update the policy state at the end of each iteration.

        Note that the each of the trajectories in the batch should be of the same length.
        Trajectories cannot terminate and restart.

        :param policy_state: A nest of tensors with details about policy.
        :param trajectories: A time-stacked trajectory object.
        :param number_of_particles: Number of monte-carlo rollouts of each action trajectory.
        """
        assert (
            self._num_elites <= trajectories.discount.shape[0]
        ), "num_elites needs to be smaller than population size"
        assert tf.equal(
            tf.reduce_all(trajectories.is_boundary()[:, :-1]), False
        ), "No trajectories in the batch should contain a terminal state before the final step."
        assert tf.equal(
            tf.reduce_all(trajectories.is_boundary()[:, -1]), True
        ), "All trajectories in the batch must end in a terminal state."

        returns = averaged_particle_returns(
            trajectories.reward, trajectories.discount, number_of_particles
        )

        sorted_idx = tf.argsort(returns, direction="DESCENDING")
        elite_idx = sorted_idx[: self._num_elites]
        elites = tf.gather(
            trajectories.action, elite_idx
        )  # shape = (number of elites, horizon) + action_spec.shape

        elites_mean = tf.reduce_mean(elites, axis=0)  # shape = (horizon,) + action_spec.shape
        elites_var = tf.reduce_mean(
            tf.math.square(elites - elites_mean), axis=0
        )  # shape = (horizon,) + action_spec.shape

        old_mean, old_var, low, high, _, step_index = policy_state

        new_mean = (
            1.0 - self._lr
        ) * old_mean + self._lr * elites_mean  # shape = (horizon,) + action_spec.shape
        new_var = (
            1.0 - self._lr
        ) * old_var + self._lr * elites_var  # shape = (horizon,) + action_spec.shape

        new_actions = sample_action_batch(new_mean, new_var, low, high, returns.shape[0])

        return tf.nest.pack_sequence_as(
            policy_state,
            [new_mean, new_var, low, high, new_actions, tf.zeros_like(step_index)],
        )
