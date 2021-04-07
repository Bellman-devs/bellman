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
This module provides interfaces and implementations for initial state distributions.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.specs import BoundedTensorSpec

from bellman.distributions.utils import create_uniform_distribution_from_spec


class InitialStateDistributionModel(ABC):
    """
    Interface for sampling an initial state/observation. This is consumed by the
    `EnvironmentModel`.
    """

    @abstractmethod
    def sample(self, sample_shape: Tuple[int]) -> tf.Tensor:
        """
        Sample from the initial state distribution.

        :param sample_shape: The leading dimensions of the sample.
        :return: A tensor of shape (sample_shape,) + distribution sample shape
        """
        pass


class ProbabilisticInitialStateDistributionModel(InitialStateDistributionModel):
    """
    An InitialStateDistributionModel which wraps a probability distribution. The state/observation
    is sampled from the distribution and returned.
    """

    def __init__(self, distribution: tfp.distributions.Distribution):
        """
        :param distribution: The distribution from which to sample states/observations
        """
        self._distribution = distribution

    def sample(self, sample_shape: Tuple[int]) -> tf.Tensor:
        """
        Sample from the initial state distribution.

        :param sample_shape: The leading dimensions of the sample.
        :return: A tensor of shape (sample_shape,) + distribution sample shape
        """
        return self._distribution.sample(sample_shape=sample_shape)


class DeterministicInitialStateModel(ProbabilisticInitialStateDistributionModel):
    """
    A deterministic InitialStateDistributionModel with mass concentrated
    on a particular starting state 𝒔
    """

    def __init__(self, state: tf.Tensor):
        """
        :param state: The tensor corresponding to the starting state 𝒔.
        """
        super().__init__(tfp.distributions.Deterministic(state))


def create_uniform_initial_state_distribution(
    state_spec: BoundedTensorSpec,
) -> InitialStateDistributionModel:
    """
    Helper function to create uniform initial state distributions.
    """
    state_sampler = ProbabilisticInitialStateDistributionModel(
        create_uniform_distribution_from_spec(state_spec)
    )
    return state_sampler
