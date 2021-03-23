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
import tensorflow as tf

from tests.tools.bellman.environments.transition_model.transition_model import (
    StubTrainableTransitionModel,
)


def test_call_step_with_an_action_with_the_wrong_shape(observation_space, action_space):
    model = StubTrainableTransitionModel(
        observation_space, action_space, predict_state_difference=True
    )
    starting_observation = tf.zeros((1,) + observation_space.shape, observation_space.dtype)
    selected_action = tf.zeros((2,) + action_space.shape + (100, 100), action_space.dtype)

    with pytest.raises(ValueError) as excinfo:
        model.step(starting_observation, selected_action)

    assert "Received a mix of batched and unbatched Tensors" in str(excinfo)


def test_call_step_with_an_action_with_the_wrong_dtype(observation_space, action_space):
    model = StubTrainableTransitionModel(
        observation_space, action_space, predict_state_difference=True
    )
    starting_observation = tf.zeros((1,) + observation_space.shape, observation_space.dtype)
    selected_action = tf.zeros((1,) + action_space.shape, dtype=tf.int8)

    with pytest.raises(TypeError) as excinfo:
        model.step(starting_observation, selected_action)

    assert "Tensor dtypes do not match spec dtypes:" in str(excinfo.value)


def test_call_step_with_an_observation_with_the_wrong_shape(observation_space, action_space):
    model = StubTrainableTransitionModel(
        observation_space, action_space, predict_state_difference=True
    )
    starting_observation = tf.zeros(
        (1,) + observation_space.shape + (100, 100), observation_space.dtype
    )
    selected_action = tf.zeros((1,) + action_space.shape, action_space.dtype)

    with pytest.raises(ValueError) as excinfo:
        model.step(starting_observation, selected_action)

    assert "Received a mix of batched and unbatched Tensors" in str(excinfo)


def test_call_step_with_an_observation_with_the_wrong_dtype(observation_space, action_space):
    model = StubTrainableTransitionModel(
        observation_space, action_space, predict_state_difference=True
    )
    starting_observation = tf.zeros((1,) + observation_space.shape, dtype=tf.int8)
    selected_action = tf.zeros((1,) + action_space.shape, action_space.dtype)

    with pytest.raises(TypeError) as excinfo:
        model.step(starting_observation, selected_action)

    assert "Tensor dtypes do not match spec dtypes:" in str(excinfo.value)


def test_call_step_with_different_observation_and_action_batch_sizes(
    observation_space, action_space
):
    model = StubTrainableTransitionModel(
        observation_space, action_space, predict_state_difference=True
    )
    starting_observation = tf.ones((3,) + observation_space.shape, observation_space.dtype)
    selected_action = tf.zeros((2,) + action_space.shape, action_space.dtype)

    with pytest.raises(AssertionError) as excinfo:
        model.step(starting_observation, selected_action)

    assert "should have equal batch sizes." in str(excinfo)


def test_call_step_with_no_batch_dim(observation_space, action_space):
    model = StubTrainableTransitionModel(
        observation_space, action_space, predict_state_difference=True
    )
    starting_observation = tf.ones(observation_space.shape, observation_space.dtype)
    selected_action = tf.zeros(action_space.shape, action_space.dtype)

    with pytest.raises(AssertionError):
        model.step(starting_observation, selected_action)
