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
This module defines a transition model implemented by a Keras model of one or more sequential
feed-forward neural networks.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tf_agents.specs import BoundedTensorSpec
from tf_agents.trajectories.trajectory import Trajectory

from bellman.environments.mixins import BatchSizeUpdaterMixin
from bellman.environments.transition_model.keras_model.network import KerasTransitionNetwork
from bellman.environments.transition_model.keras_model.trajectory_sampling import (
    SingleFunction,
    TrajectorySamplingStrategy,
)
from bellman.environments.transition_model.keras_model.utils import (
    create_concatenated_inputs,
    pack_transition_into_ensemble_training_data_set,
)
from bellman.environments.transition_model.observation_transformation import (
    ObservationTransformation,
)
from bellman.environments.transition_model.transition_model import (
    TrainableTransitionModel,
    TransitionModelTrainingSpec,
)
from bellman.environments.transition_model.utils import extract_transitions_from_trajectories


@dataclass
class KerasTrainingSpec(TransitionModelTrainingSpec):
    """
    Specification data class for Keras models. These are the same arguments as in the keras fit
    method. Arguments that are not implemented are x, y, and validation_data (including
    validation_steps and validation_freq that are only relevant if validation_data is specified),
    since data is preprocessed by the `KerasTransitionModel` class.
    """

    callbacks: List[tf.keras.callbacks.Callback] = None
    verbose: int = 1
    validation_split: float = 0.0
    shuffle: bool = True
    class_weight: Optional[Dict[int, float]] = None
    sample_weight: Optional[np.ndarray] = None
    initial_epoch: int = 0
    steps_per_epoch: int = None
    validation_batch_size: int = None
    max_queue_size: int = 10
    workers: int = 1
    use_multiprocessing: bool = False


class KerasTransitionModel(TrainableTransitionModel, BatchSizeUpdaterMixin):
    """
    This class defines transition models which are implemented using Keras. This class acts as an
    adapter for the TF-Agents configuration objects and data, ensuring that the model definition is
    separate from the data representation in TF-Agents.

    A Keras model consists of one or more sequential feed forward neural networks. The structure of
    each of the neural networks are defined by each of the `KerasTransitionNetwork` objects which
    are passed to the constructor of this class. This class defines the inputs to each of the
    networks in the ensemble, which have shape:

    [batch_dim, features_vector]

    where the batch_dim dimension is undefined (set to None), and the features vector by default is
    the flattened and concatenated latent observations and actions. More precisely, the latent
    observations and actions tensors are flattened and then concatenated to form the features
    vector.

    The `KerasTransitionNetwork` objects provide the loss functions, metrics, and other Keras
    compilation details for the neural networks.

    If a custom transition model needs to be trained on a subset of latent observations and/or
    actions, or certain transformations need to be performed on them, subclasses of this class
    should implement `_get_inputs` and `_build_model` method. Most likely the `_step` and `_train`
    methods will need to be overridden as well.
    """

    def __init__(
        self,
        keras_transition_networks: List[KerasTransitionNetwork],
        observation_space_spec: BoundedTensorSpec,
        action_space_spec: BoundedTensorSpec,
        predict_state_difference: bool = False,
        observation_transformation: Optional[ObservationTransformation] = None,
        trajectory_sampling_strategy: TrajectorySamplingStrategy = SingleFunction(),
        optimizer: Union[tf.keras.optimizers.Optimizer, str] = tf.keras.optimizers.Adam(),
    ):
        """
        :param keras_transition_networks: A list of `KerasTransitionNetwork` objects. The ensemble
            will consist of this collection of networks.
        :param observation_space_spec: The observation spec from the environment.
        :param action_space_spec: The action spec from the environment.
        :param predict_state_difference: Boolean to specify whether the transition model should
            return the next (latent) state or the difference between the current (latent) state and
            the next (latent) state
        :param observation_transformation: To transform observations to latent observations that
            are used by the transition model, and back. None will internally create an identity
            transform.
        :param trajectory_sampling_strategy: Strategy for propagating the elements of the batch
            through the ensemble.
        :param optimizer: An optimizer for the networks in the ensemble.
        """
        assert isinstance(observation_space_spec, BoundedTensorSpec)
        assert isinstance(action_space_spec, BoundedTensorSpec)
        assert trajectory_sampling_strategy.ensemble_size == len(keras_transition_networks)

        super().__init__(
            observation_space_spec,
            action_space_spec,
            predict_state_difference,
            observation_transformation,
        )

        self._keras_transition_networks = keras_transition_networks
        self._trajectory_sampling_strategy = trajectory_sampling_strategy
        self._optimizer = optimizer

        self._model = self._build_model()
        self._compile_model()

    def _create_inputs(
        self, input_tensor_name_suffix: str = ""
    ) -> Tuple[tf.keras.layers.Layer, List[tf.keras.layers.Layer]]:
        """
        Defines input layer and input tensors that will be used for building each network in the
        model. This method should be overwritten if one wants to train a custom model, using a
        subset of observations/actions.

        :return: The layer and input tensors of the model.
        """
        concatenated_network_inputs, raw_inputs = create_concatenated_inputs(
            [self.latent_observation_space_spec, self.action_space_spec],
            input_tensor_name_suffix,
        )
        return concatenated_network_inputs, raw_inputs

    def _build_model(self) -> tf.keras.Model:
        """
        Defines and returns model.

        This method uses each of the `KerasTransitionModel` objects to build an element of the
        ensemble. All of these networks are collected together into the model.

        :return: The model.
        """
        inputs = []
        outputs = []
        for index, network in enumerate(self._keras_transition_networks):
            input_tensor_name_suffix = "_" + str(index)
            concatenated_network_inputs, raw_inputs = self._create_inputs(
                input_tensor_name_suffix
            )
            inputs.extend(raw_inputs)

            network_output_layer = network.build_model(concatenated_network_inputs)
            if not self.latent_observation_space_spec.is_compatible_with(
                network_output_layer[0].type_spec
            ):
                raise ValueError(
                    f"The output layer of network {index}: {network_output_layer} "
                    f"is not compatible with the latent observation tensor spec: "
                    f"{self.latent_observation_space_spec}"
                )

            outputs.append(network_output_layer)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def _compile_model(self) -> None:
        """
        Compiles the model, with the loss function, optimizer and metrics from each of the
        individual networks. Optimizer is shared among the networks.
        """
        losses = [network.loss() for network in self._keras_transition_networks]
        metrics = [network.metrics() for network in self._keras_transition_networks]
        self._model.compile(optimizer=self._optimizer, loss=losses, metrics=metrics)

    def _step(self, latent_observation: tf.Tensor, action: tf.Tensor) -> tf.Tensor:
        """
        Predict the next latent state of the environment with a forward pass through the model. The
        output tensor is cast to the latent observation dtype. This cannot be part of the network
        definition in the case of integer dtypes because they are incompatible with the loss
        function.

        This method would require overwriting if there is a custom model on a subset of
        observations or actions.

        :param latent_observation: Latent observations for which you are predicting the successor
            latent observation with the model.
        :param action: Actions for which you are predicting the successor latent observation with
            the model.

        :return: Next latent observation predicted by the model.
        """
        inputs = self._trajectory_sampling_strategy.transform_step_inputs(
            [latent_observation, action]
        )

        tensors_or_distributions = self._model.call(inputs)
        outputs = tf.nest.map_structure(tf.convert_to_tensor, tensors_or_distributions)

        new_latent_observation = self._trajectory_sampling_strategy.transform_step_outputs(
            outputs
        )

        return tf.cast(
            new_latent_observation, dtype=self.latent_observation_space_spec.dtype
        )  # pylint: disable=all

    def _train(
        self, latent_trajectories: Trajectory, training_spec: KerasTrainingSpec
    ) -> tf.keras.callbacks.History:
        """
        Train the ensemble model, using the training Keras model, according to the specifications
        in `training_spec`. For each of the networks in the ensemble it prepares the training data
        from trajectories, creating a batched TensorFlow dataset which is used to fit the training
        Keras model.

        This method would require overwriting if there is a custom model on a subset of
        observations or actions, since all state information and actions are by default packed into
        the dataset.

        :param latent_trajectories: Trajectories of latent states, actions, rewards and next latent
            states.
        :param training_spec: Keras training specifications. Currently only `training_batch_size`,
        `epochs` and `callbacks` are used.

        :return: Keras History object with model training information.
        """

        # TODO: trajectory sampler should not care about training
        self._trajectory_sampling_strategy.train_model()

        transition = extract_transitions_from_trajectories(
            latent_trajectories,
            self.latent_observation_space_spec,
            self.action_space_spec,
            self.predict_state_difference,
        )
        training_data_set = pack_transition_into_ensemble_training_data_set(
            transition,
            self._model.input_names,
            self._model.output_names,
            self._keras_transition_networks,
        )
        batched_training_data_set = training_data_set.batch(training_spec.training_batch_size)
        # TODO: validation_split, workers and use_multiprocessing do not work with BatchDataset
        training_spec_dict = dict(training_spec.__dict__)
        training_spec_dict.pop("training_batch_size", None)
        history = self._model.fit(batched_training_data_set, **training_spec_dict)

        return history

    def update_batch_size(self, batch_size: int) -> None:
        if isinstance(self._trajectory_sampling_strategy, BatchSizeUpdaterMixin):
            self._trajectory_sampling_strategy.update_batch_size(batch_size)
