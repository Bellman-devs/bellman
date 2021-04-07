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
from tf_agents.trajectories.time_step import time_step_spec

from bellman.environments.transition_model.keras_model.keras import KerasTransitionModel
from bellman.environments.transition_model.keras_model.linear import LinearTransitionNetwork
from examples.utils.trajectories import sample_uniformly_distributed_transitions
from tests.tools.bellman.environments.reward_model import ConstantReward


@pytest.fixture(name="transitions")
def _transition_fixture(mountain_car_environment, batch_size):
    network = LinearTransitionNetwork(mountain_car_environment.observation_spec())
    transition_model = KerasTransitionModel(
        [network],
        mountain_car_environment.observation_spec(),
        mountain_car_environment.action_spec(),
    )

    reward_model = ConstantReward(
        mountain_car_environment.observation_spec(), mountain_car_environment.action_spec()
    )

    transition = sample_uniformly_distributed_transitions(
        transition_model, 2 * batch_size, reward_model
    )

    return mountain_car_environment, transition


def test_batch_of_samples_observation(transitions, batch_size):
    tf_env, transition = transitions
    observation = transition.observation

    assert observation.shape[0] == 2 * batch_size
    assert tf_env.observation_spec().is_compatible_with(observation[0, ...])


def test_batch_of_samples_action(transitions, batch_size):
    tf_env, transition = transitions
    action = transition.action

    assert action.shape[0] == 2 * batch_size
    assert tf_env.action_spec().is_compatible_with(action[0, ...])


def test_batch_of_samples_reward(transitions, batch_size):
    tf_env, transition = transitions
    reward = transition.reward

    reward_spec = time_step_spec(tf_env.observation_spec()).reward

    assert reward.shape[0] == 2 * batch_size
    assert reward_spec.is_compatible_with(reward[0, ...])


def test_batch_of_samples_next_observation(transitions, batch_size):
    tf_env, transition = transitions
    next_observation = transition.next_observation

    assert next_observation.shape[0] == 2 * batch_size
    assert tf_env.observation_spec().is_compatible_with(next_observation[0, ...])
