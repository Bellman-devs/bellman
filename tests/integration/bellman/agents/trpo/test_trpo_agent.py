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
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer

from bellman.agents.components import ModelFreeAgentComponent
from bellman.agents.trpo.trpo_agent import TRPOLossInfo
from bellman.training.model_free_agent_trainer import OnPolicyModelFreeAgentTrainer
from tests.tools.bellman.agents.trpo.trpo_agent import (
    create_trpo_agent_factory,
    dummy_trajectory_batch,
)


@pytest.fixture(name="trajectory_batch")
def _trajectory(batch_size=2, n_steps=5, obs_dim=2):
    return dummy_trajectory_batch(batch_size, n_steps, obs_dim)


@pytest.fixture(name="create_trpo_agent")
def _create_trpo_agent_fixture():
    return create_trpo_agent_factory()


def test_trpo_agent_agent_trainer(mocker, create_trpo_agent, trajectory_batch):
    trpo_agent = create_trpo_agent()
    replay_buffer = mocker.MagicMock(spec=ReplayBuffer)
    replay_buffer.gather_all.return_value = trajectory_batch

    trpo_agent_trainer = OnPolicyModelFreeAgentTrainer(10)
    tf_training_scheduler = trpo_agent_trainer.create_training_scheduler(
        trpo_agent, replay_buffer
    )
    training_losses = tf_training_scheduler.maybe_train(tf.constant(10, dtype=tf.int64))

    assert isinstance(
        training_losses[ModelFreeAgentComponent.MODEL_FREE_AGENT].extra, TRPOLossInfo
    )
