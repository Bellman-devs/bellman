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

import tensorflow as tf
from tf_agents.agents.tf_agent import LossInfo


class IdentifiableComponentTrainer:
    def __init__(self, identifier: str):
        self._identifier = identifier

    def __call__(self) -> LossInfo:
        return LossInfo(0.0, extra=self._identifier)


class SingleComponentAgent(Enum):
    COMPONENT = tf.constant("component")


class MultiComponentAgent(Enum):
    COMPONENT_1 = tf.constant("component_1")
    COMPONENT_2 = tf.constant("component_2")
    COMPONENT_3 = tf.constant("component_3")
