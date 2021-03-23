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
This module implements additional classes for reward, termination or initial state distribution
models for some OpenAI Gym classic control environments. These models are sometimes assumed
to be known and only transition dynamics is learned. In such cases these models are required
as inputs to the model-based reinforcement learning algorithms.
"""

import gin
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gym import spaces
from gym.envs.classic_control import cartpole
from gym.envs.classic_control.pendulum import angle_normalize
from tf_agents.environments import suite_gym

from bellman.environments.initial_state_distribution_model import (
    ProbabilisticInitialStateDistributionModel,
)
from bellman.environments.reward_model import RewardModel
from bellman.environments.termination_model import TerminationModel


class PendulumReward(RewardModel):
    def _step_reward(
        self, observation: tf.Tensor, action: tf.Tensor, next_observation: tf.Tensor
    ) -> tf.Tensor:
        angles = tf.math.atan2(observation[:, 1], observation[:, 0])
        angles = angle_normalize(angles)
        return -(angles ** 2 + 0.1 * observation[:, 2] ** 2 + 0.001 * action ** 2)[0]


def create_pendulum_environment(max_episode_steps):
    return suite_gym.load("Pendulum-v0", max_episode_steps=max_episode_steps)


class MountainCarTermination(TerminationModel):
    """
    The OpenAI gym termination criteria for Mountain Car are different depending on
    whether the environment is discrete or continuous. The goal position in the
    discrete environment is 0.5, while in the continuous environment it is 0.45.
    """

    def __init__(self, observation_spec: tf.TensorSpec, goal_position=0.5):
        super().__init__(observation_spec)
        self._goal_position = tf.constant(goal_position, dtype=observation_spec.dtype)

    def _terminates(self, observation: tf.Tensor) -> bool:
        return tf.greater_equal(observation[:, 0], self._goal_position)


def _calculate_mechanical_energy(state: tf.Tensor) -> tf.Tensor:
    potential_energy = 0.0025 * tf.sin(3 * state[:, 0])
    kinetic_energy = 0.5 * tf.square(state[:, 1])

    return potential_energy + kinetic_energy


@gin.configurable
class MountainCarReward(RewardModel):
    """
    Reward function for the mountain car environment, based on the change
    in mechanical energy.

    Mechanical energy is defined as the sum of the potential and kinetic energy of the car.
    """

    def _step_reward(
        self, observation: tf.Tensor, action: tf.Tensor, next_observation: tf.Tensor
    ) -> tf.Tensor:
        reward = 100 * (
            _calculate_mechanical_energy(next_observation)
            - _calculate_mechanical_energy(observation)
        )
        return tf.cast(reward, self._reward_spec.dtype)


@gin.configurable
class MountainCarInitialState(ProbabilisticInitialStateDistributionModel):
    """
    Initial state distribution for the mountain car environment, ranges are based on
    OpenAI Gym implementation.
    """

    def __init__(self, observation_spec: tf.TensorSpec):
        distribution = tfp.distributions.Uniform(
            low=tf.convert_to_tensor([-0.6, 0.0], dtype=observation_spec.dtype),
            high=tf.convert_to_tensor([-0.4, 0], dtype=observation_spec.dtype),
        )
        super().__init__(distribution)


class CartPoleSwingUp(cartpole.CartPoleEnv):
    """
    Customise the OpenAI gym cartpole environment so that it is a swing up task rather than a
    balancing task.
    """

    def __init__(self) -> None:
        super().__init__()

        self.x_threshold = 4.8

        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
            ]
        )

        self.observation_space = spaces.Box(-high, high)
        self._steps_beyond_done = None

    def step(self, action):
        """
        Override the reward value and done flag from the ``CartPoleEnv`` to be suitable for the
        swing up problem.
        :param action: the chosen action
        :return: A tuple of next state, reward, episode terminated and additional dictionary.
        """
        state, _, _, info = super().step(action)
        reward, done = self._reward_function(state)
        return state, reward, done, info

    def _reward_function(self, state):
        """
        Reward function as defined by https://arxiv.org/pdf/1604.06778.pdf
        """
        x, _, theta, _ = state

        done = np.abs(x) > self.x_threshold

        reward = np.cos(theta)
        # The episode is ended if the cart leaves the observation space.
        if done:
            reward -= 100

            if self._steps_beyond_done is None:
                self._steps_beyond_done = 0
            else:
                self._steps_beyond_done += 1
        else:
            self._steps_beyond_done = None

        self.steps_beyond_done = self._steps_beyond_done

        return reward, done

    def reset(self):
        super().reset()
        self.state[2] += np.pi
        return np.array(self.state)


class CartPoleSwingUpTermination(TerminationModel):
    def __init__(self, observation_spec: tf.TensorSpec):
        super().__init__(observation_spec)

        self._x_threshold = 4.8

    @property
    def observation_spec(self) -> tf.TensorSpec:
        return self._observation_spec

    def _terminates(self, observation: tf.Tensor) -> tf.Tensor:
        return tf.math.abs(observation[:, 0]) > self._x_threshold


class CartPoleSwingUpReward(RewardModel):
    def __init__(self, termination: CartPoleSwingUpTermination, action_spec: tf.TensorSpec):
        super().__init__(termination.observation_spec, action_spec)

        self._termination = termination

    def _step_reward(
        self, observation: tf.Tensor, action: tf.Tensor, next_observation: tf.Tensor
    ) -> tf.Tensor:
        done = self._termination.terminates(next_observation)
        theta = next_observation[:, 2]

        reward = tf.cos(theta)
        # The episode is ended if the cart leaves the observation space.
        if any(done.numpy()):
            termination_indexes = tf.where(done)
            number_terminations = termination_indexes.shape[0]
            reward = tf.tensor_scatter_nd_sub(
                reward,
                termination_indexes,
                tf.constant(100, shape=(number_terminations,), dtype=tf.float32),
            )

        return reward
