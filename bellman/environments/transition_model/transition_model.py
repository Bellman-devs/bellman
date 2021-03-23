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
This module defines a base class for transition models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

import tensorflow as tf
from tf_agents.specs import BoundedTensorSpec
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.utils.nest_utils import is_batched_nested_tensors

from bellman.environments.transition_model.observation_transformation import (
    IdentityObservationTransformation,
    ObservationTransformation,
)


@dataclass
class TransitionModelTrainingSpec:
    """
    Specification data class for model training. Models that require additional parameters for
    training should create a subclass of this class and add additional properties.
    """

    epochs: int
    training_batch_size: int


class TransitionModel(ABC):
    """
    Interface for transition models.

    A transition model forms part of a model of a Markov decision process (MDP).

    Denote the state of the MDP by 𝒔 ∊ ℝⁿ, and the action by 𝒂 ∊ ℝᵐ. Then this transition model
    should define the (probabilistic) function 𝒇 : ℝⁿ⁺ᵐ ↦ ℝⁿ, such that 𝒔ₜ₊₁ = 𝒇 (𝒔ₜ, 𝒂ₜ).
    """

    @property
    @abstractmethod
    def observation_space_spec(self) -> BoundedTensorSpec:
        """
        :return: The state space specification from the environment.
        """
        pass

    @property
    @abstractmethod
    def action_space_spec(self) -> BoundedTensorSpec:
        """
        :return: The action space specification from the environment.
        """
        pass

    @abstractmethod
    def step(self, observation: tf.Tensor, action: tf.Tensor) -> tf.Tensor:
        """
        Returns the sampled next state of the MDP, given a state and an action.

        :param observation: The current state.
        :param action: The current chosen action.
        :return: The sampled next state.
        """
        pass


T = TypeVar("T")
""" :var T: Return type from the `train` method of the `TrainableTransitionModel`. """
TS = TypeVar("TS", bound=TransitionModelTrainingSpec)
""" :var TS: Specification of the model training for the `TrainableTransitionModel`. """


class TrainableTransitionModel(TransitionModel, Generic[T, TS]):
    """
    Abstract base class for transition models. This implementation of the `TransitionModel`
    interface adds the abstract `train` method and validates the shapes and dtype of the input
    tensors.

    Subclasses of this class should define the `_step` method to sample the next state (or next
    batch of states) from the current state and action (or current batch of states and actions),
    and should also define the `train` method. Note that both the `_step` and the `_train` methods
    need to internally take care of transforming observations back and forth between observed and
    latent spaces (if transition models operate on a latent observation space different from the
    environment observation space).
    """

    def __init__(
        self,
        observation_space_spec: BoundedTensorSpec,
        action_space_spec: BoundedTensorSpec,
        predict_state_difference: bool = False,
        observation_transformation: Optional[ObservationTransformation] = None,
    ) -> None:
        """
        :param observation_space_spec: The observation spec from the environment.
        :param action_space_spec: The action spec from the environment.
        :param predict_state_difference: Boolean to specify whether the transition model should
            return the next (latent) state or the difference between the current (latent) state and
            the next (latent) state
        :param observation_transformation: To transform observations to latent observations that
            are used by the transition model, and back. None will internally create an identity
            transform.
        """
        self._observation_space_spec = observation_space_spec
        self._action_space_spec = action_space_spec
        self._predict_state_difference = predict_state_difference

        if observation_transformation is None:
            self.observation_transformation = IdentityObservationTransformation(
                observation_space_spec
            )
        else:
            self.observation_transformation = observation_transformation  # type: ignore

    @property
    def observation_space_spec(self) -> BoundedTensorSpec:
        return self._observation_space_spec

    @property
    def latent_observation_space_spec(self) -> BoundedTensorSpec:
        """
        :return: The latent state space specification from the transition model.
        """
        return self.observation_transformation.latent_observation_space_spec

    @property
    def action_space_spec(self) -> BoundedTensorSpec:
        return self._action_space_spec

    @property
    def predict_state_difference(self) -> bool:
        """
        :return: The predict state difference flag from the transition model.
        """
        return self._predict_state_difference

    @abstractmethod
    def _step(self, latent_observation: tf.Tensor, action: tf.Tensor) -> tf.Tensor:
        pass

    def step(self, observation: tf.Tensor, action: tf.Tensor) -> tf.Tensor:
        """
        From an observation and an action, return the sampled next observation.
        Both input tensors should have the same (non-zero) batch dimension.

        :return: next observation
        """
        tensors = [observation, action]
        specs = [self._observation_space_spec, self._action_space_spec]

        assert is_batched_nested_tensors(tensors, specs, num_outer_dims=1)
        assert (
            observation.shape[0] == action.shape[0]
        ), f"{observation} and {action} should have equal batch sizes."

        latent_observation = self.observation_transformation.forward_observation(observation)
        next_latent_observation = self._step(latent_observation, action)

        if self._predict_state_difference:
            next_latent_observation += latent_observation

        next_observation = self.observation_transformation.invert_observation(
            next_latent_observation, observation
        )

        assert is_batched_nested_tensors(
            [next_observation], [self._observation_space_spec], num_outer_dims=1
        )

        return next_observation

    @abstractmethod
    def _train(self, latent_trajectories: Trajectory, training_spec: TS) -> T:
        pass

    def train(self, trajectories: Trajectory, training_spec: TS) -> T:
        """
        Train the transition model using data from the `Trajectory`, according to the specification
        `training_spec`.

        :param trajectories: The training data.
        :param training_spec: The training specification.
        :return: A summary of the training process, for example the model loss over the training
            data.
        """
        latent_observation = self.observation_transformation.forward_observation(
            trajectories.observation
        )
        latent_trajectories = trajectories.replace(observation=latent_observation)
        return self._train(latent_trajectories, training_spec)
