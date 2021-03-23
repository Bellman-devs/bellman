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

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from bellman.environments.transition_model.keras_model.network import KerasTransitionNetwork
from bellman.environments.transition_model.utils import size


class DummyEnsembleTransitionNetwork(KerasTransitionNetwork):
    """
    This dummy class implements a dummy network ensemble that accepts batches of observations and
    actions and outputs batches of next_observations.

    In order to mimic ensemble behaviour we create a neural network with inputs shape [obs_dim,
    action_dim], with a hidden and output shape [obs_dim].
    """

    def __init__(self, observation_space_spec: tf.TensorSpec, bootstrap_data: bool = True):
        super().__init__(bootstrap_data)
        self._observation_space_spec = observation_space_spec

    def build_model(self, inputs: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        outputs_size = size(self._observation_space_spec)

        x = layers.Dense(64, activation="relu", dtype=tf.float32)(inputs)
        outputs = layers.Dense(outputs_size, activation=None, dtype=tf.float32)(x)
        return tf.keras.layers.Reshape(self._observation_space_spec.shape)(outputs)

    def loss(self) -> tf.keras.losses.Loss:
        return tf.keras.losses.MeanSquaredError()
