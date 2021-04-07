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
This module contains factory methods.
"""

from typing import Callable, List, Optional, Tuple

import tensorflow as tf
from tf_agents.typing import types

from bellman.environments.transition_model.keras_model.keras import (
    KerasTrainingSpec,
    KerasTransitionModel,
)
from bellman.environments.transition_model.keras_model.multilayer import (
    MultilayerFcTransitionNetwork,
)
from bellman.environments.transition_model.keras_model.network import KerasTransitionNetwork
from bellman.environments.transition_model.keras_model.probabilistic import (
    DiagonalGaussianTransitionNetwork,
)
from bellman.environments.transition_model.keras_model.trajectory_sampler_types import (
    TrajectorySamplerType,
)
from bellman.environments.transition_model.keras_model.trajectory_sampling import (
    InfiniteHorizonTrajectorySampling,
    MeanTrajectorySamplingStrategy,
    OneStepTrajectorySampling,
    TrajectorySamplingStrategy,
)
from bellman.environments.transition_model.keras_model.transition_model_types import (
    TransitionModelType,
)
from bellman.environments.transition_model.observation_transformation import (
    ObservationTransformation,
)


def build_transition_model_and_training_spec_from_type(
    observation_spec: types.NestedTensorSpec,
    action_spec: types.NestedTensorSpec,
    transition_model_type: TransitionModelType,
    num_hidden_layers: int,
    num_hidden_nodes: int,
    activation_function: Callable,
    ensemble_size: int,
    predict_state_difference: bool,
    epochs: int,
    training_batch_size: int,
    callbacks: List[tf.keras.callbacks.Callback],
    trajectory_sampler: TrajectorySamplingStrategy,
    observation_transformation: Optional[ObservationTransformation] = None,
    verbose: int = 0,
) -> Tuple[KerasTransitionModel, KerasTrainingSpec]:
    """
    Custom function to build a keras transition model plus training spec from arguments.

    :param observation_spec: A nest of BoundedTensorSpec representing the observations.
    :param action_spec: A nest of BoundedTensorSpec representing the actions.
    :param transition_model_type: An indicator which of the available transition models
        should be used - list can be found in `TransitionModelType`. A component of the
        environment model that describes the transition dynamics.
    :param num_hidden_layers: A transition model parameter, used for constructing a neural
        network. A number of hidden layers in the neural network.
    :param num_hidden_nodes: A transition model parameter, used for constructing a neural
        network. A number of nodes in each hidden layer. Parameter is shared across all layers.
    :param activation_function: A transition model parameter, used for constructing a neural
        network. An activation function of the hidden nodes.
    :param ensemble_size: A transition model parameter, used for constructing a neural
        network. The number of networks in the ensemble.
    :param predict_state_difference: A transition model parameter, used for constructing a
        neural network. A boolean indicating whether transition model will be predicting a
        difference between current and a next state or the next state directly.
    :param epochs: A transition model parameter, used by Keras fit method. A number of epochs
        used for training the neural network.
    :param training_batch_size: A transition model parameter, used by Keras fit method. A
        batch size used for training the neural network.
    :param callbacks: A transition model parameter, used by Keras fit method. A list of Keras
        callbacks used for training the neural network.
    :param trajectory_sampler: Trajectory sampler determines how predictions from an ensemble of
        neural networks that model the transition dynamics are sampled. Works only with ensemble
        type of transition models.
    :param observation_transformation: To transform observations to latent observations that
        are used by the transition model, and back. None will internally create an identity
        transform.
    :param verbose: A transition model parameter, used by Keras fit method. A level of how
        detailed the output to the console/logger is during the training.

    :return: The keras transition model object and the corresponding training spec.
    """

    networks: List[KerasTransitionNetwork] = None
    if transition_model_type == TransitionModelType.Deterministic:
        networks = [
            MultilayerFcTransitionNetwork(
                observation_spec,
                num_hidden_layers,
                [num_hidden_nodes] * num_hidden_layers,
                [activation_function] * num_hidden_layers,
            )
        ]
        transition_model = KerasTransitionModel(
            networks,
            observation_spec,
            action_spec,
            predict_state_difference=predict_state_difference,
            observation_transformation=observation_transformation,
        )
    elif transition_model_type == TransitionModelType.DeterministicEnsemble:
        networks = [
            MultilayerFcTransitionNetwork(
                observation_spec,
                num_hidden_layers,
                [num_hidden_nodes] * num_hidden_layers,
                [activation_function] * num_hidden_layers,
                bootstrap_data=True,
            )
            for _ in range(ensemble_size)
        ]
        transition_model = KerasTransitionModel(
            networks,
            observation_spec,
            action_spec,
            predict_state_difference=predict_state_difference,
            observation_transformation=observation_transformation,
            trajectory_sampling_strategy=trajectory_sampler,
        )
    elif transition_model_type == TransitionModelType.Probabilistic:
        networks = [
            DiagonalGaussianTransitionNetwork(
                observation_spec,
                num_hidden_layers,
                [num_hidden_nodes] * num_hidden_layers,
                [activation_function] * num_hidden_layers,
            )
        ]
        transition_model = KerasTransitionModel(
            networks,
            observation_spec,
            action_spec,
            predict_state_difference=predict_state_difference,
            observation_transformation=observation_transformation,
        )
    elif transition_model_type == TransitionModelType.ProbabilisticEnsemble:
        networks = [
            DiagonalGaussianTransitionNetwork(
                observation_spec,
                num_hidden_layers,
                [num_hidden_nodes] * num_hidden_layers,
                [activation_function] * num_hidden_layers,
                bootstrap_data=True,
            )
            for _ in range(ensemble_size)
        ]
        transition_model = KerasTransitionModel(
            networks,
            observation_spec,
            action_spec,
            predict_state_difference=predict_state_difference,
            observation_transformation=observation_transformation,
            trajectory_sampling_strategy=trajectory_sampler,
        )
    else:
        raise RuntimeError("Unknown transition model")

    training_spec = KerasTrainingSpec(
        epochs=epochs,
        training_batch_size=training_batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )

    return transition_model, training_spec


def build_trajectory_sampler_from_type(
    ensemble_size: int,
    trajectory_sampler_type: TrajectorySamplerType,
    batch_size: int,
) -> TrajectorySamplingStrategy:
    """
    Custom function to build a trajectory sampler from arguments.

    :param ensemble_size: A transition model parameter, used for constructing a neural
        network. The number of networks in the ensemble.
    :param trajectory_sampler_type: An indicator which of the available trajectory samplers
        should be used - list can be found in `TrajectorySamplerType`. Trajectory sampler
        determines how predictions from an ensemble of neural networks that model the
        transition dynamics are sampled. Works only with ensemble type of transition models.
    :param batch_size: A trajectory optimiser parameter. The number of virtual rollouts
        that are simulated in each iteration during trajectory optimization.
    :return: The trajectory sample object.
    """
    assert ensemble_size > 1, "For ensemble transition models ensemble_size should be > 1"

    trajectory_sampler: Optional[TrajectorySamplingStrategy] = None
    if trajectory_sampler_type == TrajectorySamplerType.TS1:
        trajectory_sampler = OneStepTrajectorySampling(
            batch_size,
            ensemble_size,
        )
    elif trajectory_sampler_type == TrajectorySamplerType.TSinf:
        trajectory_sampler = InfiniteHorizonTrajectorySampling(
            batch_size,
            ensemble_size,
        )
    elif trajectory_sampler_type == TrajectorySamplerType.Mean:
        trajectory_sampler = MeanTrajectorySamplingStrategy(ensemble_size)
    else:
        raise RuntimeError("Unknown trajectory sampler")

    return trajectory_sampler
