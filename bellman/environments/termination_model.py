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
This module provides interfaces and implementations for termination models.
"""

from abc import ABC, abstractmethod

import tensorflow as tf
from tf_agents.utils.nest_utils import get_outer_shape, is_batched_nested_tensors

TerminationSpec = tf.TensorSpec(tuple(), dtype=tf.bool)


class TerminationModel(ABC):
    """
    Abstract base class for termination models.
    """

    def __init__(self, observation_spec: tf.TensorSpec):
        """
        :param observation_spec: The `TensorSpec` object which defines the observation tensors
        """
        self._observation_spec = observation_spec

    @abstractmethod
    def _terminates(self, observation: tf.Tensor) -> tf.Tensor:
        pass

    def terminates(self, observation: tf.Tensor) -> tf.Tensor:
        """
        Return a boolean tensor to describe whether the `observation` is a terminal state.
        """
        is_batched_nested_tensors([observation], [self._observation_spec])

        return self._terminates(observation)


class ConstantFalseTermination(TerminationModel):
    """
    Termination model that always returns false.
    """

    def _terminates(self, observation: tf.Tensor) -> tf.Tensor:
        """
        Return a boolean tensor to describe whether the `observation` is a terminal state.
        """
        batch_size = get_outer_shape([observation], [self._observation_spec])
        return tf.constant(False, shape=batch_size, dtype=tf.bool)
