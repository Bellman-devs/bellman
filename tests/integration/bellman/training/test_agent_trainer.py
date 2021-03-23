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
from tf_agents.agents import TFAgent
from tf_agents.metrics.tf_metrics import EnvironmentSteps
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.specs import BoundedTensorSpec
from tf_agents.trajectories.trajectory import single_step

from bellman.training.utils import TRAIN_ARGSPEC_COMPONENT_ID
from tests.tools.bellman.training.agent_trainer import MultiComponentAgentTrainer
from tests.tools.bellman.training.schedule import MultiComponentAgent


def test_agent_trainer_with_environment_steps_metric(mocker):
    """
    Use a mock agent and the environment steps metric to trigger the training, as in the
    experiment harness.
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

    # After zero environment steps, do not train any models
    environment_steps_metric = EnvironmentSteps()
    loss_dictionary = training_scheduler.maybe_train(environment_steps_metric.result())
    assert not loss_dictionary

    # After one environment step, train the first model
    single_step_trajectory = single_step(
        tf.zeros(()), tf.zeros(()), (), tf.zeros(()), tf.zeros(())
    )
    environment_steps_metric.call(single_step_trajectory)
    loss_dictionary_1 = training_scheduler.maybe_train(environment_steps_metric.result())
    assert len(loss_dictionary_1) == 1
    assert (
        loss_dictionary_1[MultiComponentAgent.COMPONENT_1].extra
        == MultiComponentAgent.COMPONENT_1.name
    )

    # After two environment steps, train the first and second models
    environment_steps_metric.call(single_step_trajectory)
    loss_dictionary_2 = training_scheduler.maybe_train(environment_steps_metric.result())
    assert len(loss_dictionary_2) == 2
    assert (
        loss_dictionary_2[MultiComponentAgent.COMPONENT_1].extra
        == MultiComponentAgent.COMPONENT_1.name
    )
    assert (
        loss_dictionary_2[MultiComponentAgent.COMPONENT_2].extra
        == MultiComponentAgent.COMPONENT_2.name
    )

    # After three environment steps, train the first and third models
    environment_steps_metric.call(single_step_trajectory)
    loss_dictionary_3 = training_scheduler.maybe_train(environment_steps_metric.result())
    assert len(loss_dictionary_3) == 2
    assert (
        loss_dictionary_3[MultiComponentAgent.COMPONENT_1].extra
        == MultiComponentAgent.COMPONENT_1.name
    )
    assert (
        loss_dictionary_3[MultiComponentAgent.COMPONENT_3].extra
        == MultiComponentAgent.COMPONENT_3.name
    )
