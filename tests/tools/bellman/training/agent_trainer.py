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

from enum import Enum
from typing import Dict, cast

from tf_agents.agents import TFAgent
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer

from bellman.training.agent_trainer import AgentTrainer
from bellman.training.schedule import TFTrainingScheduler, TrainingDefinition
from tests.tools.bellman.training.schedule import (
    IdentifiableComponentTrainer,
    MultiComponentAgent,
    SingleComponentAgent,
)


class SingleComponentAgentTrainer(AgentTrainer):
    def __init__(self, interval: int = 1):
        self._interval = interval

    def create_training_scheduler(
        self, agent: TFAgent, replay_buffer: ReplayBuffer
    ) -> TFTrainingScheduler:
        schedule = {
            SingleComponentAgent.COMPONENT: TrainingDefinition(
                self._interval,
                IdentifiableComponentTrainer(SingleComponentAgent.COMPONENT.name),
            )
        }
        return TFTrainingScheduler(cast(Dict[Enum, TrainingDefinition], schedule))


class MultiComponentAgentTrainer(AgentTrainer):
    def create_training_scheduler(
        self, agent: TFAgent, replay_buffer: ReplayBuffer
    ) -> TFTrainingScheduler:
        schedule = {
            MultiComponentAgent.COMPONENT_1: TrainingDefinition(
                1, IdentifiableComponentTrainer(MultiComponentAgent.COMPONENT_1.name)
            ),
            MultiComponentAgent.COMPONENT_2: TrainingDefinition(
                2, IdentifiableComponentTrainer(MultiComponentAgent.COMPONENT_2.name)
            ),
            MultiComponentAgent.COMPONENT_3: TrainingDefinition(
                3, IdentifiableComponentTrainer(MultiComponentAgent.COMPONENT_3.name)
            ),
        }
        return TFTrainingScheduler(cast(Dict[Enum, TrainingDefinition], schedule))
