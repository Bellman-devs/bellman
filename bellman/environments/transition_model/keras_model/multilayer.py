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
This module defines fully connected feedforward multilayer transition network using Keras.
"""

from typing import Callable, List, Optional

import numpy as np
import tensorflow as tf

from bellman.environments.transition_model.keras_model.network import KerasTransitionNetwork
from bellman.environments.transition_model.utils import size


class MultilayerFcTransitionNetwork(KerasTransitionNetwork):
    """
    This class defines a multilayer transition model using Keras, fully connected type. If defined
    with zero layers (default) we obtain a network equivalent to linear regression.
    If number of hidden layers is one or more then all arguments to the dense Keras layer
    can be set individually for each layer.
    """

    def __init__(
        self,
        observation_space_spec: tf.TensorSpec,
        num_hidden_layers: int = 0,
        units: Optional[List[int]] = None,
        activation: Optional[List[Callable]] = None,
        use_bias: Optional[List[bool]] = None,
        kernel_initializer: Optional[List[Callable]] = None,
        bias_initializer: Optional[List[Callable]] = None,
        kernel_regularizer: Optional[List[Callable]] = None,
        bias_regularizer: Optional[List[Callable]] = None,
        activity_regularizer: Optional[List[Callable]] = None,
        kernel_constraint: Optional[List[Callable]] = None,
        bias_constraint: Optional[List[Callable]] = None,
        bootstrap_data: bool = False,
    ):
        """
        :param observation_space_spec: Environment observation specifications.
        :param num_hidden_layers: A number of hidden layers in the network. If larger than zero
            (default), then all other arguments to the `Dense` Keras layer have to have
            same length, if specified.
        :param units: Number of nodes in each hidden layer.
        :param activation: Activation function of the hidden nodes.
        :param use_bias: Boolean, whether the layer uses a bias vector.
        :param kernel_initializer: Initializer for the kernel weights matrix.
        :param bias_initializer: Initializer for the bias vector.
        :param kernel_regularizer: Regularizer function applied to the kernel weights matrix.
        :param bias_regularizer: Regularizer function applied to the bias vector.
        :param activity_regularizer: Regularizer function applied to the output of the layer.
        :param kernel_constraint: Constraint function applied to the kernel weights matrix.
        :param bias_constraint: Constraint function applied to the bias vector.
        :param bootstrap_data: Re-sample data with replacement.
        """
        assert num_hidden_layers >= 0, "num_hidden_layers must be an integer >= 0"
        if num_hidden_layers > 0:
            assert units is not None, "if num_hidden_layers > 0, units cannot be None"
            assert num_hidden_layers == len(units), (
                "if num_hidden_layers > 0, units has to be a list with a "
                + "number of elements equal to num_hidden_layers"
            )
            for i in [
                activation,
                use_bias,
                kernel_initializer,
                bias_initializer,
                kernel_regularizer,
                bias_regularizer,
                activity_regularizer,
                kernel_constraint,
                bias_constraint,
            ]:
                if i is not None:
                    assert num_hidden_layers == len(i)

        super().__init__(bootstrap_data)

        self._observation_space_spec = observation_space_spec
        self._num_hidden_layers = num_hidden_layers
        self._units = units
        self._activation = activation
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._activity_regularizer = activity_regularizer
        self._kernel_constraint = kernel_constraint
        self._bias_constraint = bias_constraint

    def gen_hidden_dense_layers(self, hidden_layer):
        """Generate a sequence of dense Keras layers"""
        if self._num_hidden_layers > 0:
            for id_layer in range(self._num_hidden_layers):
                hidden_layer = tf.keras.layers.Dense(units=self._units[id_layer])(hidden_layer)
                if self._activation is not None:
                    hidden_layer.activation = self._activation[id_layer]
                if self._use_bias is not None:
                    hidden_layer.use_bias = self._use_bias[id_layer]
                if self._kernel_initializer is not None:
                    hidden_layer.kernel_initializer = self._kernel_initializer[id_layer]
                if self._bias_initializer is not None:
                    hidden_layer.bias_initializer = self._bias_initializer[id_layer]
                if self._kernel_regularizer is not None:
                    hidden_layer.kernel_regularizer = self._kernel_regularizer[id_layer]
                if self._bias_regularizer is not None:
                    hidden_layer.bias_regularizer = self._bias_regularizer[id_layer]
                if self._activity_regularizer is not None:
                    hidden_layer.activity_regularizer = self._activity_regularizer[id_layer]
                if self._kernel_constraint is not None:
                    hidden_layer.kernel_constraint = self._kernel_constraint[id_layer]
                if self._bias_constraint is not None:
                    hidden_layer.bias_constraint = self._bias_constraint[id_layer]
        return hidden_layer

    def build_model(self, inputs: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        observation_space_nodes = size(self._observation_space_spec)
        hidden_layer = inputs
        hidden_layer = self.gen_hidden_dense_layers(hidden_layer)
        output_layer = tf.keras.layers.Dense(observation_space_nodes, activation="linear")(
            hidden_layer
        )
        return tf.keras.layers.Reshape(self._observation_space_spec.shape)(output_layer)

    def loss(self) -> tf.keras.losses.Loss:
        return tf.keras.losses.MeanSquaredError()
