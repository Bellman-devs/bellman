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
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.specs import tensor_spec

from bellman.environments.transition_model.keras_model.probabilistic import (
    DiagonalGaussianTransitionNetwork,
    GaussianTransitionNetwork,
)
from bellman.environments.transition_model.keras_model.utils import create_concatenated_inputs

tfpl = tfp.layers


@pytest.mark.parametrize("num_hidden_layers", [0, 1, 3])
@pytest.mark.parametrize("num_hidden_nodes", [1, 10])
@pytest.mark.parametrize(
    "prob_network", [GaussianTransitionNetwork, DiagonalGaussianTransitionNetwork]
)
def test_probabilistic_network_nparams(
    observation_space, action_space, num_hidden_layers, num_hidden_nodes, prob_network
):
    """
    Ensure we have a correct number of nodes/parameters in the network.
    """
    concatenated_network_inputs, raw_inputs = create_concatenated_inputs(
        [observation_space, action_space], ""
    )
    input_nodes = int(np.prod(concatenated_network_inputs.shape[1:]))
    output_nodes = int(np.prod(observation_space.shape))
    if prob_network == GaussianTransitionNetwork:
        output_nodes = tfpl.MultivariateNormalTriL.params_size(output_nodes)
    elif prob_network == DiagonalGaussianTransitionNetwork:
        output_nodes = tfpl.IndependentNormal.params_size(output_nodes)
    network = prob_network(
        observation_space,
        num_hidden_layers,
        [num_hidden_nodes] * num_hidden_layers,
    )
    network_output = network.build_model(concatenated_network_inputs)
    model = tf.keras.Model(inputs=raw_inputs, outputs=network_output)

    # number of parameters
    nparams = model.count_params()

    # expected number of parameters
    if num_hidden_layers == 0:
        nparams_exp = input_nodes * output_nodes + output_nodes
    elif num_hidden_layers > 0:
        nparams_exp = (
            (input_nodes + 1) * num_hidden_nodes
            + (num_hidden_layers - 1) * num_hidden_nodes * (num_hidden_nodes + 1)
            + output_nodes * (num_hidden_nodes + 1)
        )

    assert nparams == nparams_exp
