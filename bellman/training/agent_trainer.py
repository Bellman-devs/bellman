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
This module provides a base class for specifying how an agent should be trained. For each trainable
component this includes the number of real environment steps which should elapse between training
steps together with the data which should be used in the training step.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict

from tf_agents.agents import TFAgent
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer

from bellman.training.schedule import TFTrainingScheduler, TrainingDefinition
from bellman.training.utils import TRAIN_ARGSPEC_COMPONENT_ID


class AgentTrainer(ABC):
    """
    Abstract base class which is responsible for creating an instance of a `TFTrainingScheduler`
    that has been configured to train the various components of an agent. This combines the number
    of real environment steps which should have elapsed before training a component as well as the
    method for sampling data from the replay buffer.
    """

    @abstractmethod
    def create_training_scheduler(
        self, agent: TFAgent, replay_buffer: ReplayBuffer
    ) -> TFTrainingScheduler:
        """
        Subclasses should implement this method which creates a training schedule for an agent.

        :param agent: The agent object that is trained.
        :param replay_buffer: The replay buffer containing data for training the agent.

        :return: A `TFTrainingScheduler` for scheduling training.
        """
