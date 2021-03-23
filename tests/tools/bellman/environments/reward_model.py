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

import tensorflow as tf

from bellman.environments.reward_model import RewardModel


class ConstantReward(RewardModel):
    """
    Constant reward function.
    """

    def __init__(
        self,
        observation_spec: tf.TensorSpec,
        action_spec: tf.TensorSpec,
        constant_reward: float = 0.0,
    ):
        super().__init__(observation_spec, action_spec)
        self._constant_reward = constant_reward

    def _step_reward(
        self, observation: tf.Tensor, action: tf.Tensor, next_observation: tf.Tensor
    ) -> tf.Tensor:
        return tf.constant(
            self._constant_reward,
            shape=(observation.shape[0],),
            dtype=self._reward_spec.dtype,
        )
