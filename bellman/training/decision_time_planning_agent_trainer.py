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
This module defines an agent trainer for training a decision-time planning agent.
"""

from enum import Enum
from typing import Dict, cast

from tf_agents.agents import TFAgent
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer

from bellman.agents.components import EnvironmentModelComponents
from bellman.agents.decision_time_planning.decision_time_planning_agent import (
    DecisionTimePlanningAgent,
)
from bellman.training.agent_trainer import AgentTrainer
from bellman.training.schedule import TFTrainingScheduler, TrainingDefinition
from bellman.training.utils import TRAIN_ARGSPEC_COMPONENT_ID


class DecisionTimePlanningAgentTrainer(AgentTrainer):
    """
    An `AgentTrainer` for a decision-time planning agent which only trains a transition model.
    """

    def __init__(self, steps_per_transition_model_update: int):
        self._steps_per_transition_model_update = steps_per_transition_model_update

    def create_training_scheduler(
        self, agent: TFAgent, replay_buffer: ReplayBuffer
    ) -> TFTrainingScheduler:
        assert isinstance(
            agent, DecisionTimePlanningAgent
        ), "Expect a `DecisionTimePlanningAgent`."

        train_kwargs_dict = {
            TRAIN_ARGSPEC_COMPONENT_ID: EnvironmentModelComponents.TRANSITION.value
        }

        def train_step() -> LossInfo:
            trajectory = replay_buffer.gather_all()
            return agent.train(trajectory, **train_kwargs_dict)

        schedule = {
            EnvironmentModelComponents.TRANSITION: TrainingDefinition(
                self._steps_per_transition_model_update, train_step
            )
        }
        return TFTrainingScheduler(cast(Dict[Enum, TrainingDefinition], schedule))
