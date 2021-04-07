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
import tensorflow as tf
from tf_agents.trajectories.time_step import StepType, TimeStep, _as_float32_array

from bellman.environments.environment_model import EnvironmentModel
from bellman.environments.initial_state_distribution_model import (
    create_uniform_initial_state_distribution,
)
from bellman.environments.termination_model import ConstantFalseTermination
from bellman.environments.transition_model.keras_model.keras import KerasTransitionModel
from bellman.environments.transition_model.keras_model.linear import LinearTransitionNetwork
from tests.tools.bellman.environments.reward_model import ConstantReward


def _create_wrapped_linear_model(observation_space, action_space, constant_reward=0.0):
    network = LinearTransitionNetwork(observation_space)
    model = KerasTransitionModel([network], observation_space, action_space)
    wrapped_model = EnvironmentModel(
        model,
        ConstantReward(observation_space, action_space, constant_reward),
        ConstantFalseTermination(action_space),
        create_uniform_initial_state_distribution(observation_space),
    )

    return wrapped_model


def _get_time_step_from_wrapped_linear_model(
    observation_space, action_space, constant_reward=0.0
) -> TimeStep:
    wrapped_model = _create_wrapped_linear_model(
        observation_space, action_space, constant_reward
    )
    selected_action = tf.zeros((1,) + action_space.shape, action_space.dtype)

    return wrapped_model.step(selected_action)


def test_step_model_step_type(observation_space, action_space):
    time_step = _get_time_step_from_wrapped_linear_model(observation_space, action_space)
    assert time_step.step_type == StepType.MID


def test_step_model_step_reward(observation_space, action_space):
    time_step = _get_time_step_from_wrapped_linear_model(observation_space, action_space, 0.5)
    assert time_step.reward == _as_float32_array(0.5)


def test_step_model_step_discount(observation_space, action_space):
    time_step = _get_time_step_from_wrapped_linear_model(observation_space, action_space)
    assert time_step.discount == _as_float32_array(1.0)


def test_step_model_step_shape(observation_space, action_space):
    time_step = _get_time_step_from_wrapped_linear_model(observation_space, action_space)
    assert time_step.step_type.shape == (1,)


def test_step_model_step_observation_properties(observation_space, action_space):
    time_step = _get_time_step_from_wrapped_linear_model(observation_space, action_space)
    assert observation_space.is_compatible_with(time_step.observation[0])


def test_step_model_next_observation_is_different_from_observation(
    observation_space, action_space
):
    wrapped_model = _create_wrapped_linear_model(observation_space, action_space, 0.0)

    first_time_step = wrapped_model.current_time_step()
    selected_action = tf.zeros((1,) + action_space.shape, action_space.dtype)
    next_time_step = wrapped_model.step(selected_action)

    assert not np.array_equal(
        first_time_step.observation.numpy(), next_time_step.observation.numpy()
    )
