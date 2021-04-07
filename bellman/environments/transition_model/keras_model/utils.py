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
This module contains auxiliary functions useful for Keras transition models.
"""

from typing import List, Tuple

import tensorflow as tf
from tf_agents.specs import TensorSpec

from bellman.environments.transition_model.keras_model.network import KerasTransitionNetwork
from bellman.environments.transition_model.utils import Transition
from bellman.networks.cast_layer import Cast


def _create_input_layer(
    bounded_tensor_spec: TensorSpec, tensor_name_suffix: str
) -> tf.keras.layers.Layer:
    input_shape = bounded_tensor_spec.shape
    dtype = bounded_tensor_spec.dtype
    name = bounded_tensor_spec.name + tensor_name_suffix
    return tf.keras.Input(shape=input_shape, dtype=dtype, name=name)


def create_concatenated_inputs(
    tensor_specs: List[TensorSpec], input_tensor_name_suffix: str
) -> Tuple[tf.keras.layers.Layer, List[tf.keras.layers.Layer]]:
    """
    Concatenates inputs (states and actions) in the transition model into a single layer. This
    conversion is necessary as in TF agents they are separate objects while for modeling with
    Keras, they need to be treated as a single input.
    :param tensor_specs: A list of tensor specifications (e.g. one for states/observations and one
        for actions).
    :param input_tensor_name_suffix: String suffix to add to the names of the input tensors, to
        ensure that the input tensors have globally unique names.
    :return: A tuple with concatenated input layer and input tensor.
    """
    input_tensors = []
    inputs_layers = []
    for tensor_spec in tensor_specs:
        input_tensor = _create_input_layer(tensor_spec, input_tensor_name_suffix)

        # ensure that the `input_tensor` is cast to the Keras `floatx` dtype.
        cast_input_tensor = Cast()(input_tensor)

        input_tensors.append(input_tensor)
        inputs_layers.append(
            tf.keras.layers.Flatten(dtype=cast_input_tensor.dtype)(cast_input_tensor)
        )

    return tf.keras.layers.Concatenate()(inputs_layers), input_tensors


def pack_transition_into_ensemble_training_data_set(
    transition: Transition,
    model_input_names: List[str],
    model_output_names: List[str],
    keras_transition_networks: List[KerasTransitionNetwork],
) -> tf.data.Dataset:
    """
    The `transition` contains all of the data which will be used to train the keras transition
    model. These data are converted into a TensorFlow `Dataset` which can be used to train the
    ensemble model.
    Each member of the ensemble will be trained on a separate data set, as defined by the
    `KerasTransitionNetwork:transform_training_data` method.
    """
    inputs = {}
    targets = {}

    # Iterate over pairs of input names corresponding to the observation and action tensors
    # for each network.
    input_names_iterator = iter(model_input_names)
    training_input_names_iterator = zip(input_names_iterator, input_names_iterator)

    for network, training_input_names, training_target_name in zip(
        keras_transition_networks,
        training_input_names_iterator,
        model_output_names,
    ):
        transformed_transition = network.transform_training_data(transition)

        training_inputs = {
            training_input_names[0]: transformed_transition.observation,
            training_input_names[1]: transformed_transition.action,
        }
        training_targets = {training_target_name: transformed_transition.next_observation}

        inputs.update(training_inputs)
        targets.update(training_targets)

    return tf.data.Dataset.from_tensor_slices((inputs, targets))
