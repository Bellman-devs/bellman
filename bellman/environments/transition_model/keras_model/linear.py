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
This module defines linear transition models using Keras.
"""

import tensorflow as tf

from bellman.environments.transition_model.keras_model.multilayer import (
    MultilayerFcTransitionNetwork,
)


class LinearTransitionNetwork(MultilayerFcTransitionNetwork):
    """
    This class defines a linear transition model using Keras, i.e. a neural network with no
    hidden layers, equivalent to a linear regression.
    """

    def __init__(self, observation_space_spec: tf.TensorSpec, bootstrap_data: bool = False):
        """
        :param observation_space_spec: Environment observation specifications.
        :param bootstrap_data: Create an ensemble version of the network where data is resampled
        with replacement.
        """
        super().__init__(
            observation_space_spec, num_hidden_layers=0, bootstrap_data=bootstrap_data
        )
