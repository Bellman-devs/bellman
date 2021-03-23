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

import numpy as np
import pytest
import tensorflow as tf
from tf_agents.specs import BoundedTensorSpec

from bellman.distributions.utils import create_uniform_distribution_from_spec
from bellman.environments.reward_model import RewardModel
from tests.tools.bellman.environments.reward_model import ConstantReward

REWARD_TARGET = 2.3


@pytest.fixture(name="test_data")
def _fixture(observation_space, action_space, batch_size):
    observation_distr = create_uniform_distribution_from_spec(observation_space)
    batch_observations = observation_distr.sample(batch_size)

    reward = ConstantReward(observation_space, action_space, REWARD_TARGET)
    action_distr = create_uniform_distribution_from_spec(action_space)
    batch_actions = action_distr.sample(batch_size)

    return reward, batch_observations, batch_actions, batch_size


def test_batched_reward_shape(test_data):
    reward, batch_observations, batch_actions, batch_size = test_data
    reward_batch = reward.step_reward(batch_observations, batch_actions, batch_observations)

    assert reward_batch.shape == [batch_size]


def test_batched_constant_reward_value(test_data):
    reward, batch_observations, batch_actions, batch_size = test_data
    reward_batch = reward.step_reward(batch_observations, batch_actions, batch_observations)

    np.testing.assert_almost_equal(np.ones((batch_size,)) * REWARD_TARGET, reward_batch)


def test_batched_reward_throws_with_single_observation(test_data):
    reward, batch_observations, batch_actions, _ = test_data
    single_observation = batch_observations[0]

    with pytest.raises(ValueError):
        reward.step_reward(single_observation, batch_actions, batch_observations)


def test_batched_reward_throws_with_single_action(test_data):
    reward, batch_observations, batch_actions, _ = test_data
    single_action = batch_actions[0]

    with pytest.raises(ValueError):
        reward.step_reward(batch_observations, single_action, batch_observations)


def test_batched_reward_throws_with_single_next_observation(test_data):
    reward, batch_observations, batch_actions, _ = test_data
    single_observation = batch_observations[0]

    with pytest.raises(ValueError):
        reward.step_reward(batch_observations, batch_actions, single_observation)


def test_reward_model_wrong_reward_dtype():
    class _WrongDtypeRewardModel(RewardModel):
        def _step_reward(
            self, observation: tf.Tensor, action: tf.Tensor, next_observation: tf.Tensor
        ) -> tf.Tensor:
            return tf.constant(0, shape=(1,), dtype=tf.int8)

    observation = tf.zeros(shape=(1,), dtype=tf.float64)
    action = tf.zeros(shape=(1,), dtype=tf.float64)
    observation_spec = BoundedTensorSpec(shape=(1,), dtype=tf.float64, minimum=0, maximum=1)
    action_spec = BoundedTensorSpec(shape=(1,), dtype=tf.float64, minimum=0, maximum=1)
    reward_model = _WrongDtypeRewardModel(observation_spec, action_spec)

    with pytest.raises(TypeError) as excinfo:
        reward_model.step_reward(observation, action, observation)

    assert "Tensor dtypes do not match spec dtypes" in str(excinfo)
