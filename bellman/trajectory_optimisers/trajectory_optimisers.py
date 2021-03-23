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
This module provides functions and classes for trajectory optimisation.
"""

from abc import ABC, abstractmethod
from math import inf
from typing import Optional
from warnings import warn

import tensorflow as tf
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.specs import BoundedTensorSpec
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils.nest_utils import get_outer_shape

from bellman.environments.environment_model import EnvironmentModel
from bellman.environments.utils import virtual_rollouts_buffer_and_driver
from bellman.trajectory_optimisers.particles import (
    averaged_particle_returns,
    decorate_policy_with_particles,
)


class TrajectoryOptimiser(ABC):
    """
    A trajectory optimiser optimises a sequence of actions given a model of the
    the environment that is used for virtual rollouts.
    """

    def __init__(self, batch_size: int):
        """
        Initializes the class.

        :param batch_size: The number of virtual rollouts that are simulated in each
            iteration during optimization.
        """
        self._batch_size = batch_size

    @property
    def batch_size(self):
        """
        Return the batch size parameter that determines the number of virtual rollouts
        that are simulated in each iteration during optimization.
        """
        return self._batch_size

    @abstractmethod
    def optimise(self, time_step: TimeStep, environment_model: EnvironmentModel):
        """
        Optimise trajectories starting from an initial time_step.

        :param time_step: Initial `TimeStep` from which rollouts are starting.
        :param environment_model: An `EnvironmentModel` is a model of the MDP that represents
            the environment, consisting of a transition, reward, termination and initial state
            distribution model, of which some are trainable and some are fixed.

        :return: Action trajectory (Tensor with shape (horizon,) + action_shape).
        """
        pass

    def _time_step_to_initial_observation(
        self,
        time_step: TimeStep,
        environment_model: EnvironmentModel,
    ):
        """
        Construct initial observation from time step.

        :param time_step: Initial time step from the real environment with nominal batch size of 1
            (because the real environment is assumed to be not "batchable").
        :param environment_model: An `EnvironmentModel` is a model of the MDP that represents
            the environment, consisting of a transition, reward, termination and initial state
            distribution model, of which some are trainable and some are fixed.

        :return: Initial observation that has the appropriate batch size as first dimension.
        """

        observation = time_step.observation
        batch_size = get_outer_shape(observation, environment_model.observation_spec())
        # the time step comes from the real environment
        assert batch_size == (
            1,
        ), f"batch_size of time_step.observation = {batch_size} and it should be 1"
        initial_observation = tf.repeat(observation, repeats=self._batch_size, axis=0)
        return initial_observation


class TrajectorySelector(ABC):
    """
    Base class for optimal trajectory selection.

    The process of optimising trajectories generates many candidates for the optimal actions. Each
    batch of candidates should be passed to this class, and subclasses should define the approach
    used to choose the optimal actions.
    """

    def __init__(self, action_spec: BoundedTensorSpec):
        """
        Initializes the class.

        :param action_spec: A nest of `BoundedTensorSpec` representing the actions.
        """
        self._action_spec = action_spec
        self._initial_value = tf.cast(0.0, dtype=action_spec.dtype)
        self._optimal_trajectory = tf.Variable(
            self._initial_value, name="optimal_trajectory", shape=tf.TensorShape(None)
        )

    @abstractmethod
    def add_candidate_trajectories(
        self,
        trajectories: Trajectory,
        policy_state: types.NestedTensor,
        number_of_particles: int,
    ):
        """
        Add candidate trajectories from which to compute the optimal sequence of actions.

        :param trajectories: A time-stacked trajectory object.
        :param policy_state: A nest of tensors with details about policy.
        :param number_of_particles: Number of monte-carlo rollouts of each action trajectory.
        """
        pass

    def reset(self):
        """
        Reset the optimal trajectory. This method should be called between iterations.
        """
        self._optimal_trajectory.assign(self._initial_value)

    def get_optimal_actions(self) -> tf.Tensor:
        """
        Return the optimal actions.
        """
        return self._optimal_trajectory.read_value()


class HighestReturnTrajectorySelector(TrajectorySelector):
    """
    A trajectory selector which uses the cumulative reward to choose trajectories. The trajectory
    with the largest cumulative reward is considered optimal, and those actions are returned.
    """

    def __init__(self, action_spec: BoundedTensorSpec):
        super().__init__(action_spec)

        self._highest_return = common.create_variable("highest_reward", -inf, dtype=tf.float32)

    def add_candidate_trajectories(
        self,
        trajectories: Trajectory,
        policy_state: types.NestedTensor,
        number_of_particles: int,
    ):
        returns = averaged_particle_returns(
            trajectories.reward, trajectories.discount, number_of_particles
        )

        best_trajectory_index = tf.argmax(returns)
        best_return = returns[best_trajectory_index]

        if best_return > self._highest_return:
            self._highest_return.assign(returns[best_trajectory_index])
            self._optimal_trajectory.assign(trajectories.action[best_trajectory_index])

    def reset(self):
        super().reset()
        self._highest_return.assign(-inf)


class PolicyStateUpdater(ABC):
    """
    Policies which need the policy state to be updated between iterations should subclass this
    class and use that implementation with the `PolicyTrajectoryOptimiser`.
    """

    @abstractmethod
    def update(
        self,
        policy_state: types.NestedTensor,
        trajectories: Trajectory,
        number_of_particles: int,
    ) -> types.NestedTensor:
        """
        Update the policy state at the end of each iteration.

        :param policy_state: A nest of tensors with details about policy.
        :param trajectories: A time-stacked trajectory object.
        :param number_of_particles: Number of monte-carlo rollouts of each action trajectory.
        """
        pass


class PolicyTrajectoryOptimiser(TrajectoryOptimiser):
    """
    In a decision-time planning setting, the policy trajectory optimiser samples trajectories from
    the environment model using a given policy, and returns the best action sequence found.
    """

    def __init__(
        self,
        policy: TFPolicy,
        horizon: int,
        population_size: int,
        number_of_particles: int = 1,
        max_iterations: int = 1,
        trajectory_selector: Optional[TrajectorySelector] = None,
        policy_state_updater: Optional[PolicyStateUpdater] = None,
    ):
        """
        Initializes the class.

        :param policy: A `TFPolicy` instance.
        :param horizon: Number of steps taken in the environment in each virtual rollout.
        :param population_size: The number of candidate sequences of actions at each iteration.
        :param number_of_particles: Number of monte-carlo rollouts of each action trajectory.
        :param max_iterations: Number of iterations in the optimization.
        :param trajectory_selector: An optional `TrajectorySelector` used for selecting the best
            trajectory out of simulated trajectories.
        :param policy_state_updater: An optional `PolicyStateUpdater` used for updating the
            policy state, for stateful policies.
        """

        assert horizon > 0, "horizon has to be > 0"
        assert population_size > 0, "population_size has to be > 0"
        assert number_of_particles > 0, "number_of_particles has to be > 0"
        assert max_iterations > 0, "max_iterations has to be > 0"

        super().__init__(population_size * number_of_particles)

        if number_of_particles > 1:
            warn(
                f"A number of particles greater than one has been specified "
                f"({number_of_particles}). This requires the policy and the environment model"
                f" to be stochastic."
            )
            policy = decorate_policy_with_particles(policy, number_of_particles)

        self._policy = policy
        self._horizon = horizon
        self._population_size = population_size
        self._number_of_particles = number_of_particles
        self._max_iterations = max_iterations

        self._trajectory_selector = (
            trajectory_selector
            if trajectory_selector
            else HighestReturnTrajectorySelector(policy.action_spec)
        )

        self._policy_state_updater = policy_state_updater

    def optimise(self, time_step: TimeStep, environment_model: EnvironmentModel):

        assert (
            self._batch_size == environment_model.batch_size
        ), "batch_size parameter is not equal to environment_model.batch_size"
        # TODO: monitor the "virtual_rollouts_buffer_and_driver", it could potentially
        #       cause performance issues (memory leaks)
        virtual_buffer, virtual_driver, wrapped_env_model = virtual_rollouts_buffer_and_driver(
            environment_model, self._policy, self._horizon
        )
        initial_observation = self._time_step_to_initial_observation(
            time_step, wrapped_env_model
        )

        self._trajectory_selector.reset()
        policy_state = self._policy.get_initial_state(self._population_size)
        for _ in range(self._max_iterations):
            time_step = wrapped_env_model.set_initial_observation(initial_observation)
            _, policy_state = virtual_driver.run(time_step, policy_state)

            trajectories = virtual_buffer.gather_all()
            virtual_buffer.clear()

            if self._policy_state_updater:
                policy_state = self._policy_state_updater.update(
                    policy_state, trajectories, self._number_of_particles
                )

            self._trajectory_selector.add_candidate_trajectories(
                trajectories, policy_state, self._number_of_particles
            )

        return self._trajectory_selector.get_optimal_actions()
