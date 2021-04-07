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
from tf_agents.trajectories.time_step import StepType, _as_float32_array

from bellman.environments.environment_model import EnvironmentModel
from bellman.environments.initial_state_distribution_model import (
    create_uniform_initial_state_distribution,
)
from bellman.environments.termination_model import ConstantFalseTermination
from bellman.environments.transition_model.keras_model.keras import KerasTransitionModel
from bellman.environments.transition_model.keras_model.linear import LinearTransitionNetwork
from tests.tools.bellman.environments.reward_model import ConstantReward


def _create_wrapped_environment(observation_space, action_space, reward):
    network = LinearTransitionNetwork(observation_space)
    model = KerasTransitionModel([network], observation_space, action_space)
    return EnvironmentModel(
        model,
        ConstantReward(observation_space, action_space, reward),
        ConstantFalseTermination(observation_space),
        create_uniform_initial_state_distribution(observation_space),
    )


@pytest.fixture(name="first_step")
def _first_step_fixture(observation_space, action_space):
    environment_wrapper = _create_wrapped_environment(observation_space, action_space, 0.0)
    return environment_wrapper.reset()


def test_reset_model_time_step_step_type(first_step):
    assert first_step.step_type == StepType.FIRST


def test_reset_model_time_step_reward(first_step):
    assert first_step.reward == _as_float32_array(0.0)


def test_reset_model_time_step_discount(first_step):
    assert first_step.discount == _as_float32_array(1.0)


def test_reset_model_time_step_observation_from_observation_space(
    observation_space, action_space
):
    environment_wrapper = _create_wrapped_environment(observation_space, action_space, 0.0)
    first_step = environment_wrapper.reset()

    assert observation_space.is_compatible_with(first_step.observation[0])
