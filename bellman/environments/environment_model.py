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
This module contains the `EnvironmentModel` class.
"""

import numpy as np
import tensorflow as tf
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.trajectories.time_step import StepType, TimeStep, restart, time_step_spec
from tf_agents.utils.nest_utils import get_outer_shape

from bellman.environments.initial_state_distribution_model import InitialStateDistributionModel
from bellman.environments.mixins import BatchSizeUpdaterMixin
from bellman.environments.reward_model import RewardModel
from bellman.environments.termination_model import TerminationModel
from bellman.environments.transition_model.transition_model import TransitionModel


class EnvironmentModel(TFEnvironment):
    """
    An approximate MDP 𝓜̂.

    The approximate MDP has the following form::
        𝓜̂ = (S, A, r̂, P̂, ρ̂₀, γ)

    where S is the state space, A is the action space, r̂ is the approximate reward function, P̂
    is the approximate state transition distribution, ρ̂₀ is the approximate initial state
    distribution and γ is the discount factor of the cumulative reward. Note that the terms "state"
    and "observation" are used interchangeably.

    This class also requires a `TerminationModel`. This function maps 𝒔 ∊ S to a boolean. If
    the `TerminationModel` returns `True` then that state is an absorbing state of the MDP and
    the episode is terminated. The model should include all termination criteria which
    are intrinsic to the MDP.

    Extrinsic termination criteria should be handled in a wrapper around this class.
    """

    def __init__(
        self,
        transition_model: TransitionModel,
        reward_model: RewardModel,
        termination_model: TerminationModel,
        initial_state_distribution_model: InitialStateDistributionModel,
        batch_size: int = 1,
    ):
        """
        :param transition_model: The state transition distribution that maps a state-action pair
            (𝒔 ∊ S, 𝒂 ∊ A) to the next state 𝒔' ∊ S in a (possibly) probabilistic fashion
        :param reward_model: The reward model that maps a state-action-next-state tuple
            (𝒔 ∊ S, 𝒂 ∊ A, 𝒔' ∊ S) to a scalar real value
        :param termination_model: Termination model. For each state 𝒔 ∊ S, this should return `True`
            if state 𝒔 terminates an episode, and `False` otherwise.
        :param initial_state_distribution_model: Distribution from which the starting state 𝒔 ∊ S of
            a new episode will be sampled. The starting state must not be terminal.
        :param batch_size: The batch size expected for the actions and observations, it should
            be greater than 0.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size is " + str(batch_size) + " and it should be > 0")
        super().__init__(
            time_step_spec(transition_model.observation_space_spec),
            transition_model.action_space_spec,
            batch_size,
        )

        self._transition_model = transition_model
        self._reward_model = reward_model
        self._termination_model = termination_model
        self._initial_state_distribution_model = initial_state_distribution_model

        self._time_step: TimeStep

        self._initialise_trajectory()

    @property
    def termination_model(self) -> TerminationModel:
        """
        Return the `TerminationModel`.
        """
        return self._termination_model

    def _current_time_step(self):
        """
        Return the current TimeStep object.
        """
        return self._time_step

    def _ensure_no_terminal_observations(self, observation):
        """
        Raise error when any observation in the observation batch is terminal.
        :param observation: A batch of observations, one for each batch element (the batch size is
            the first dimension)
        """
        has_terminated = self._termination_model.terminates(observation)
        has_terminated_numpy = has_terminated.numpy()
        if any(has_terminated_numpy):
            termination_indices = np.where(has_terminated_numpy)[0]
            raise ValueError(
                "Terminal observations occurred at indices "
                + np.array_str(termination_indices)
            )

    def _set_initial_observation(self, observation):
        """
        Set initial observation of the environment model.

        :param observation: A batch of observations, one for each batch element (the batch size is
            the first dimension)
        """

        # Make sure that the observation shape is as expected
        batch_size = get_outer_shape(
            observation, self._transition_model.observation_space_spec
        )
        assert batch_size == self._batch_size, batch_size

        # Raise error when any initial observation is terminal
        self._ensure_no_terminal_observations(observation)

        # Create `TimeStep` object from observation tensor. Note that this will mark the observation
        # as FIRST.
        self._time_step = restart(observation, batch_size=self._batch_size)

    def _initialise_trajectory(self):
        """
        Sample initial state to start the trajectory.
        """
        observation = self._initial_state_distribution_model.sample(
            sample_shape=(self._batch_size,)
        )

        self._set_initial_observation(observation)

    def _reset(self):
        """
        Sample a TimeStep from the initial distribution, set as FIRST
        and return the sampled TimeStep.
        """
        self._initialise_trajectory()
        return self._time_step

    def _step(self, action):
        """
        Return predictions of next states for each member of the batch.

        :param action: A batch of actions (the batch size is the first dimension)
        :return: A batch of next state predictions in the form of a `TimeStep` object
        """
        # Make sure that action shape is as expected
        batch_size = get_outer_shape(action, self._transition_model.action_space_spec)
        assert batch_size == self._batch_size

        # Get observation from current time step
        observation = self._time_step.observation

        # Identify observation batch elements in the previous time step that have terminated. Note
        # the conversion to numpy is for performance reasons
        is_last = self._time_step.is_last()
        is_any_last = any(is_last.numpy())

        # Elements of the observation batch that terminated on the previous time step require reset.
        if is_any_last:
            # Identify number of elements to be reset and their corresponding indexes
            number_resets = tf.math.count_nonzero(is_last)
            reset_indexes = tf.where(is_last)

            # Sample reset observations from initial state distribution
            reset_observation = self._initial_state_distribution_model.sample((number_resets,))

            # Raise error when any terminal observations are left after re-initialization
            self._ensure_no_terminal_observations(reset_observation)

        # Get batches of next observations, update observations that were reset
        next_observation = self._transition_model.step(observation, action)
        if is_any_last:
            next_observation = tf.tensor_scatter_nd_update(
                next_observation, reset_indexes, reset_observation
            )

        # Get batches of rewards, set rewards from reset batch elements to 0
        reward = self._reward_model.step_reward(observation, action, next_observation)
        if is_any_last:
            reward = tf.where(condition=is_last, x=tf.constant(0.0), y=reward)

        # Get batches of termination flags
        has_terminated = self._termination_model.terminates(next_observation)

        # Get batches of step types, set step types from reset batch elements to FIRST
        step_type = tf.where(condition=has_terminated, x=StepType.LAST, y=StepType.MID)
        if is_any_last:
            step_type = tf.where(condition=is_last, x=StepType.FIRST, y=step_type)

        # Get batches of discounts, set discounts from reset batch elements to 1
        discount = tf.where(condition=has_terminated, x=tf.constant(0.0), y=tf.constant(1.0))
        if is_any_last:
            discount = tf.where(condition=is_last, x=tf.constant(1.0), y=discount)

        # Create TimeStep object and return
        self._time_step = TimeStep(step_type, reward, discount, next_observation)
        return self._time_step

    def set_initial_observation(self, observation):
        """
        Set initial observation of the environment model.

        :param observation: A batch of observations, one for each batch element (the batch size is
            the first dimension)
        :return: A batch of initials states in the form of a `TimeStep` object
        """
        self._set_initial_observation(observation)
        return self._time_step

    @property
    def batch_size(self):
        """
        Re-implementing the batch_size property of TFEnvironment in order to define a
        setter method.
        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        if batch_size > 0:
            self._batch_size = batch_size
            if isinstance(self._transition_model, BatchSizeUpdaterMixin):
                self._transition_model.update_batch_size(batch_size)
        else:
            raise ValueError(f"batch_size is " + str(batch_size) + " and it should be > 0")

    def render(self):
        raise NotImplementedError("No rendering support.")
