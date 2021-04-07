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

"""
Agent trainers for TF-Agents' model free agents.
"""

from enum import Enum
from typing import Dict, cast

import tensorflow as tf
from tf_agents.agents import DdpgAgent, PPOAgent, SacAgent, Td3Agent, TFAgent
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer

from bellman.agents.components import ModelFreeAgentComponent
from bellman.agents.trpo.trpo_agent import TRPOAgent
from bellman.training.agent_trainer import AgentTrainer
from bellman.training.schedule import TFTrainingScheduler, TrainingDefinition


class OnPolicyModelFreeAgentTrainer(AgentTrainer):
    """
    An `AgentTrainer` specialised for on-policy model free agents.
    """

    def __init__(self, steps_per_policy_update: int):
        """
        :param steps_per_policy_update: steps between policy updates
        """
        self._steps_per_policy_update = steps_per_policy_update

    def create_training_scheduler(
        self, agent: TFAgent, replay_buffer: ReplayBuffer
    ) -> TFTrainingScheduler:
        assert isinstance(
            agent, (TRPOAgent, PPOAgent)
        ), "Expect an on-policy model free agent."

        def train_step() -> LossInfo:
            trajectory = replay_buffer.gather_all()
            replay_buffer.clear()
            return agent.train(trajectory)

        schedule = {
            ModelFreeAgentComponent.MODEL_FREE_AGENT: TrainingDefinition(
                self._steps_per_policy_update, train_step
            )
        }
        return TFTrainingScheduler(cast(Dict[Enum, TrainingDefinition], schedule))


class OffPolicyModelFreeAgentTrainer(AgentTrainer):
    """
    An `AgentTrainer` specialised for off-policy model free agents.
    """

    def __init__(self, steps_per_policy_update: int, training_data_batch_size: int = 1):
        """
        :param steps_per_policy_update: steps between policy updates.
        :param training_data_batch_size: size of mini-batch for each training step.
        """
        self._steps_per_policy_update = steps_per_policy_update
        self._training_data_batch_size = training_data_batch_size

    def create_training_scheduler(
        self, agent: TFAgent, replay_buffer: ReplayBuffer
    ) -> TFTrainingScheduler:
        assert isinstance(
            agent, (DdpgAgent, Td3Agent, SacAgent)
        ), "Expect an off-policy model free agent."

        def _not_boundary(trajectories, _):
            return ~trajectories.is_boundary()[0]

        dataset = (
            replay_buffer.as_dataset(num_steps=2)
            .filter(_not_boundary)
            .batch(self._training_data_batch_size)
        )
        iterator = iter(dataset)

        def train_step():
            if (
                tf.data.experimental.cardinality(dataset).numpy()
                >= self._training_data_batch_size
            ):
                experience, _ = next(iterator)
                return agent.train(experience)
            return LossInfo(None, None)

        schedule = {
            ModelFreeAgentComponent.MODEL_FREE_AGENT: TrainingDefinition(
                self._steps_per_policy_update, train_step
            )
        }
        return TFTrainingScheduler(cast(Dict[Enum, TrainingDefinition], schedule))
