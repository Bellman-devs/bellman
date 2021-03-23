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
from tf_agents.agents import TFAgent
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.specs import BoundedTensorSpec

from bellman.training.utils import TRAIN_ARGSPEC_COMPONENT_ID
from tests.tools.bellman.training.agent_trainer import (
    MultiComponentAgentTrainer,
    SingleComponentAgentTrainer,
)
from tests.tools.bellman.training.schedule import MultiComponentAgent, SingleComponentAgent


def test_agent_trainer_creates_valid_single_component_training_schedule(mocker):
    """
    Ensure that the agent trainer can create a training scheduler that trains a single component
    agent at the specified schedule.
    """
    single_component_agent_trainer = SingleComponentAgentTrainer()
    mock_agent = mocker.MagicMock(spec=TFAgent)
    mock_replay_buffer = mocker.MagicMock(spec=ReplayBuffer)

    training_scheduler = single_component_agent_trainer.create_training_scheduler(
        mock_agent, mock_replay_buffer
    )
    loss_dictionary = training_scheduler.maybe_train(tf.ones(tuple(), dtype=tf.int64))

    assert (
        loss_dictionary[SingleComponentAgent.COMPONENT].extra
        == SingleComponentAgent.COMPONENT.name
    )


def test_agent_trainer_creates_valid_multi_component_training_schedule(mocker):
    """
    Ensure that the agent trainer can create a training scheduler that trains each component of a
    multi-component agent at the specified schedule.
    """
    multi_component_agent_trainer = MultiComponentAgentTrainer()
    mock_agent = mocker.MagicMock(spec=TFAgent)
    mock_train_argspec = mocker.PropertyMock(
        return_value={TRAIN_ARGSPEC_COMPONENT_ID: BoundedTensorSpec((), tf.int64, 0, 2)}
    )
    type(mock_agent).train_argspec = mock_train_argspec
    mock_replay_buffer = mocker.MagicMock(spec=ReplayBuffer)

    training_scheduler = multi_component_agent_trainer.create_training_scheduler(
        mock_agent, mock_replay_buffer
    )

    loss_dictionary_1 = training_scheduler.maybe_train(1 * tf.ones(tuple(), dtype=tf.int64))
    loss_dictionary_2 = training_scheduler.maybe_train(2 * tf.ones(tuple(), dtype=tf.int64))
    loss_dictionary_3 = training_scheduler.maybe_train(3 * tf.ones(tuple(), dtype=tf.int64))

    assert len(loss_dictionary_1) == 1
    assert (
        loss_dictionary_1[MultiComponentAgent.COMPONENT_1].extra
        == MultiComponentAgent.COMPONENT_1.name
    )

    assert len(loss_dictionary_2) == 2
    assert (
        loss_dictionary_2[MultiComponentAgent.COMPONENT_1].extra
        == MultiComponentAgent.COMPONENT_1.name
    )
    assert (
        loss_dictionary_2[MultiComponentAgent.COMPONENT_2].extra
        == MultiComponentAgent.COMPONENT_2.name
    )

    assert len(loss_dictionary_3) == 2
    assert (
        loss_dictionary_3[MultiComponentAgent.COMPONENT_1].extra
        == MultiComponentAgent.COMPONENT_1.name
    )
    assert (
        loss_dictionary_3[MultiComponentAgent.COMPONENT_3].extra
        == MultiComponentAgent.COMPONENT_3.name
    )
