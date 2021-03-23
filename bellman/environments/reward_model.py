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
This module provides interfaces and implementations for reward models.
"""

from abc import ABC, abstractmethod

import tensorflow as tf
from tf_agents.trajectories.time_step import TimeStep, time_step_spec
from tf_agents.utils.nest_utils import is_batched_nested_tensors

RewardSpec = tf.TensorSpec(tuple(), dtype=tf.float32)


class RewardModel(ABC):
    """
    Abstract base class for reward models.
    """

    def __init__(self, observation_spec: tf.TensorSpec, action_spec: tf.TensorSpec):
        """
        :param observation_spec: The `TensorSpec` object which specifies the observation tensors
        :param action_spec: The `TensorSpec` object which specifies the action tensors
        """
        self._observation_spec = observation_spec
        self._action_spec = action_spec

        self._reward_spec = time_step_spec(observation_spec).reward

    @abstractmethod
    def _step_reward(
        self, observation: tf.Tensor, action: tf.Tensor, next_observation: tf.Tensor
    ) -> tf.Tensor:
        pass

    def step_reward(
        self, observation: tf.Tensor, action: tf.Tensor, next_observation: tf.Tensor
    ) -> tf.Tensor:
        """
        Return the step reward for the transition from `observation` to `next_observation` via
        `action`.
        """
        is_batched_nested_tensors(
            [observation, action, next_observation],
            [self._observation_spec, self._action_spec, self._observation_spec],
        )

        rewards = self._step_reward(observation, action, next_observation)

        is_batched_nested_tensors(rewards, self._reward_spec)
        return rewards
