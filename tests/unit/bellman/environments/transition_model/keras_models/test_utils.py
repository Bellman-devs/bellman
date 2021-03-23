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

import pytest
import tensorflow as tf
from tf_agents.specs import TensorSpec

from bellman.environments.transition_model.keras_model.utils import create_concatenated_inputs


@pytest.fixture(name="number_input_tensor_specs", params=[2, 5, 11])
def _number_input_tensor_specs_fixture(request):
    return request.param


@pytest.fixture(name="dummy_keras_model")
def _dummy_keras_model_fixture(number_input_tensor_specs):
    input_specs = [
        TensorSpec(shape=(4,), dtype=tf.float64, name=f"input_spec_{i}")
        for i in range(number_input_tensor_specs)
    ]
    output_layer, input_tensors = create_concatenated_inputs(input_specs, "")
    return tf.keras.Model(inputs=input_tensors, outputs=output_layer)


def test_create_concatenated_inputs_layer_is_flattened(dummy_keras_model):
    """
    Ensure that the output fragment of the model produced by the `create_concatenated_inputs`
    function has rank 2.

    The output shape should be [batch_size, feature vector].
    """
    assert len(dummy_keras_model.output_shape) == 2


def test_create_concatenated_inputs_single_concatenate_layer(dummy_keras_model):
    """
    The ensemble model assumes that the fragment of the model produced by the
    `create_concatenated_inputs` function has exactly one `Concatenate` layer, and that is the
    output tensor of the fragment.
    """
    is_concatenate_layer = [
        isinstance(layer, tf.keras.layers.Concatenate) for layer in dummy_keras_model.layers
    ]

    # `index` will return the index of the first instance of `True` in the list, so we assert that
    # this should be equal to the final index in the list, which confirms that there is exactly one
    # `Concatenate` layer, and that it is the final layer in the model.
    assert is_concatenate_layer.index(True) == len(dummy_keras_model.layers) - 1


def test_create_concatenated_inputs_no_trainable_parameters(dummy_keras_model):
    """
    The `create_concatenated_inputs` function creates layers and tensors as inputs to a keras
    model. This test asserts that these tensors contain no trainable parameters.
    """
    assert not dummy_keras_model.trainable_variables
