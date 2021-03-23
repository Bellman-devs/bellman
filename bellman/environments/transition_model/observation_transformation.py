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
This module is for transforming observations for transition models.
"""

from abc import ABC, abstractmethod

import tensorflow as tf
from tf_agents.specs import BoundedTensorSpec


class ObservationTransformation(ABC):
    """
    Abstract base class to transform observations and invert latent observations.
    """

    def __init__(self, latent_observation_space_spec: BoundedTensorSpec) -> None:
        """
        :param latent_observation_space_spec: The latent observation spec from the model.
        """
        self.latent_observation_space_spec = latent_observation_space_spec

    @abstractmethod
    def forward_observation(self, observation: tf.Tensor) -> tf.Tensor:
        """
        :param observation: observation tensor.
        :return: latent observation tensor.
        """
        pass

    @abstractmethod
    def invert_observation(
        self,
        latent_observation: tf.Tensor,
        previous_observation: tf.Tensor,
    ) -> tf.Tensor:
        """
        :param latent_observation: latent observation tensor.
        :param previous_observation: previous non-latent observation tensor.
        :return: observation tensor.
        """
        pass


class IdentityObservationTransformation(ObservationTransformation):
    """
    Class for identity transforms.
    """

    def forward_observation(self, observation: tf.Tensor) -> tf.Tensor:
        return observation

    def invert_observation(
        self,
        latent_observation: tf.Tensor,
        previous_observation: tf.Tensor,
    ) -> tf.Tensor:
        return latent_observation
