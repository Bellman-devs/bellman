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

from bellman.distributions.utils import create_uniform_distribution_from_spec
from bellman.environments.environment_model import EnvironmentModel
from bellman.environments.initial_state_distribution_model import (
    DeterministicInitialStateModel,
)
from bellman.environments.tf_wrappers import TFTimeLimit
from bellman.environments.transition_model.keras_model.keras import KerasTransitionModel
from tests.tools.bellman.environments.reward_model import ConstantReward
from tests.tools.bellman.environments.termination_model import MutableBatchConstantTermination
from tests.tools.bellman.environments.transition_model.keras_models.dummy_ensemble import (
    DummyEnsembleTransitionNetwork,
)


def _create_env_model(observation_space, action_space):
    batch_size = 3
    time_limit = 5

    terminations = MutableBatchConstantTermination(observation_space, batch_size)
    observation = create_uniform_distribution_from_spec(observation_space).sample()
    network = DummyEnsembleTransitionNetwork(observation_space)
    model = KerasTransitionModel([network], observation_space, action_space)
    env_model = TFTimeLimit(
        EnvironmentModel(
            transition_model=model,
            reward_model=ConstantReward(observation_space, action_space, -1.0),
            termination_model=terminations,
            initial_state_distribution_model=DeterministicInitialStateModel(observation),
            batch_size=batch_size,
        ),
        duration=time_limit,
    )

    actions = create_uniform_distribution_from_spec(action_space).sample((batch_size,))

    # Initial time step
    env_model.reset()

    observations = np.squeeze(
        np.repeat(np.expand_dims(observation, axis=0), batch_size, axis=0)
    )
    return terminations, observations, actions, env_model


def test_time_step_constructor(observation_space, action_space):
    terminations, observation, actions, env_model = _create_env_model(
        observation_space, action_space
    )

    ###########################
    # Constructing First step #
    ###########################
    first_terminates = tf.convert_to_tensor([False, False, False])
    terminations.should_terminate = first_terminates
    first_time_step = env_model.step(actions)

    np.testing.assert_equal(first_time_step.step_type.numpy(), [1, 1, 1])
    np.testing.assert_equal(first_time_step.reward.numpy(), [-1, -1, -1])
    np.testing.assert_equal(first_time_step.discount.numpy(), [1, 1, 1])
    # no observations have been reset
    assert not np.any(np.isclose(first_time_step.observation, observation))

    ############################
    # Constructing Second step #
    ############################
    second_terminates = tf.convert_to_tensor([True, False, False])
    terminations.should_terminate = second_terminates
    second_time_step = env_model.step(actions)

    # step type on termination is 2
    np.testing.assert_equal(second_time_step.step_type.numpy(), [2, 1, 1])
    np.testing.assert_equal(second_time_step.reward.numpy(), [-1, -1, -1])
    # discount on termination is 0
    np.testing.assert_equal(second_time_step.discount.numpy(), [0, 1, 1])
    # no observations have been reset
    assert not np.any(np.isclose(second_time_step.observation, observation))

    ############################
    # Constructing Third step #
    ############################
    third_terminates = tf.convert_to_tensor([False, False, False])
    terminations.should_terminate = third_terminates
    third_time_step = env_model.step(actions)

    # step type on reset is 0
    np.testing.assert_equal(third_time_step.step_type.numpy(), [0, 1, 1])
    # reward on reset is 0
    np.testing.assert_equal(third_time_step.reward.numpy(), [0, -1, -1])
    np.testing.assert_equal(third_time_step.discount.numpy(), [1, 1, 1])
    # the first observation in the batch has been reset
    np.testing.assert_allclose(third_time_step.observation[0], observation[0])
    assert not np.any(np.isclose(third_time_step.observation[1:], observation[1:]))

    ############################
    # Constructing Fourth step #
    ############################
    fourth_terminates = tf.convert_to_tensor([False, True, False])
    terminations.should_terminate = fourth_terminates
    fourth_time_step = env_model.step(actions)

    # step type on termination is 2
    np.testing.assert_equal(fourth_time_step.step_type.numpy(), [1, 2, 1])
    np.testing.assert_equal(fourth_time_step.reward.numpy(), [-1, -1, -1])
    # discount on termination is 0
    np.testing.assert_equal(fourth_time_step.discount.numpy(), [1, 0, 1])
    assert not np.any(np.isclose(fourth_time_step.observation, observation))

    ############################
    # Constructing Last step #
    ############################
    last_terminates = tf.convert_to_tensor([False, False, False])
    terminations.should_terminate = last_terminates
    last_time_step = env_model.step(actions)

    # step type on reset is 0, and on termination is 2
    np.testing.assert_equal(last_time_step.step_type.numpy(), [1, 0, 2])
    # reward on reset is 0
    np.testing.assert_equal(last_time_step.reward.numpy(), [-1, 0, -1])
    # discount on termination is 0
    np.testing.assert_equal(last_time_step.discount.numpy(), [1, 1, 0])
    assert not np.any(np.isclose(last_time_step.observation[0], observation[0]))
    # the second observation in the batch has been reset
    np.testing.assert_allclose(last_time_step.observation[1], observation[1])
    assert not np.any(np.isclose(last_time_step.observation[2], observation[2]))
