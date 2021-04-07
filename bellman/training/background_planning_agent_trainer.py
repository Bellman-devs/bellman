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
This module defines an agent trainer for training a background planning agent.
"""

from enum import Enum
from typing import Dict, cast

from tf_agents.agents import TFAgent
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer

from bellman.agents.background_planning.background_planning_agent import (
    BackgroundPlanningAgent,
)
from bellman.agents.components import EnvironmentModelComponents, ModelFreeAgentComponent
from bellman.training.agent_trainer import AgentTrainer
from bellman.training.schedule import TFTrainingScheduler, TrainingDefinition
from bellman.training.utils import TRAIN_ARGSPEC_COMPONENT_ID


class BackgroundPlanningAgentTrainer(AgentTrainer):
    """
    An `AgentTrainer` for a background planning agent which trains a transition model and a
    model-free agent.
    """

    def __init__(
        self, steps_per_transition_model_update: int, steps_per_model_free_agent_update: int
    ):
        """
        :param steps_per_transition_model_update: steps between transition model updates.
        :param steps_per_model_free_agent_update: steps between model-free agent updates.
        """
        assert steps_per_transition_model_update > 0
        assert steps_per_model_free_agent_update > 0
        self._steps_per_transition_model_update = steps_per_transition_model_update
        self._steps_per_model_free_agent_update = steps_per_model_free_agent_update

        # to indicate whether transition model training has occurred already (this is important
        # for model-free agents, who should not be trained at all if the transition model has never
        # been trained before).
        self._has_transition_model_been_trained = False

    def create_training_scheduler(
        self, agent: TFAgent, replay_buffer: ReplayBuffer
    ) -> TFTrainingScheduler:
        assert isinstance(
            agent, BackgroundPlanningAgent
        ), "Expect a `BackgroundPlanningAgent`."

        # specify the train step for training the transition model
        train_transition_model_kwargs_dict = {
            TRAIN_ARGSPEC_COMPONENT_ID: EnvironmentModelComponents.TRANSITION.value
        }

        def train_transition_model_step() -> LossInfo:
            self._has_transition_model_been_trained = True
            trajectory = replay_buffer.gather_all()
            return agent.train(trajectory, **train_transition_model_kwargs_dict)

        # specify the train step for training the model-free agent
        train_model_free_agent_kwargs_dict = {
            TRAIN_ARGSPEC_COMPONENT_ID: ModelFreeAgentComponent.MODEL_FREE_AGENT.value
        }

        def train_model_free_agent_step() -> LossInfo:
            if not self._has_transition_model_been_trained:
                return LossInfo(None, None)
            trajectory = replay_buffer.gather_all()
            return agent.train(trajectory, **train_model_free_agent_kwargs_dict)

        # create scheduler
        schedule = {
            EnvironmentModelComponents.TRANSITION: TrainingDefinition(
                self._steps_per_transition_model_update, train_transition_model_step
            ),
            ModelFreeAgentComponent.MODEL_FREE_AGENT: TrainingDefinition(
                self._steps_per_model_free_agent_update, train_model_free_agent_step
            ),
        }
        return TFTrainingScheduler(cast(Dict[Enum, TrainingDefinition], schedule))
