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
This module provides enumerations of commonly used groups of trainable components, which can be
used to define the training schedules of single- or multi-component agents.
"""

from enum import Enum

import tensorflow as tf


class EnvironmentModelComponents(Enum):
    """
    This class defines names of individual components of the MDP that can potentially be modelled
    in model-based reinforcement learning algorithms. These names should be used when calling the
    training method of model-based reinforcement learning agents, to specify which model should be
    trained.
    """

    TRANSITION = tf.constant("transition_model")
    REWARD = tf.constant("reward_model")
    TERMINATION = tf.constant("termination_model")
    INITIALSTATE = tf.constant("initial_state_model")


class ModelFreeAgentComponent(Enum):
    """
    For consistency of training between model-based and model-free agents, this class defines a
    name which can be used for training a model-free agent.
    """

    MODEL_FREE_AGENT = tf.constant("model_free_agent")
