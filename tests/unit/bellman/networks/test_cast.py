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
from tensorflow.python.keras.testing_utils import (  # pylint: disable=no-name-in-module
    layer_test,
)

from bellman.networks.cast_layer import Cast


@pytest.mark.parametrize("target_dtype", [tf.float32, tf.float64, tf.int32, tf.int64])
def test_cast(dtype, target_dtype):
    input_layer = tf.keras.Input(shape=(2, 3), dtype=dtype, name="Test_Input")
    cast_layer = Cast(target_dtype)(input_layer)

    assert cast_layer.dtype == target_dtype


def test_flatten_casts_floats():
    """
    The casting behaviour of tf.keras Layers does not include casting from integer tensors to
    floating point tensors, despite the fact that the underlying op does support this conversion.

    We have worked around this behaviour. This is a confirmatory test to spot when they have changed
    this behaviour. If this test fails then the TensorFlow keras behaviour has probably been changed
    and our workaround may need to be updated or removed.
    """
    assert tf.keras.backend.floatx() == tf.float32

    input_layer = tf.keras.Input(shape=(2, 3), dtype=tf.float64, name="Test_Input")
    flat_input = tf.keras.layers.Flatten()(input_layer)

    assert flat_input.dtype == tf.float32


def test_flatten_does_not_cast_ints():
    """
    The casting behaviour of tf.keras Layers does not include casting from integer tensors to
    floating point tensors, despite the fact that the underlying op does support this conversion.

    We have worked around this behaviour. This is a negative test to spot when they have changed
    this behaviour. If this test fails then the TensorFlow keras behaviour has probably been changed
    and our workaround may need to be updated or removed.
    """
    assert tf.keras.backend.floatx() == tf.float32

    input_layer = tf.keras.Input(shape=(2, 3), dtype=tf.int32, name="Test_Input")
    flat_input = tf.keras.layers.Flatten()(input_layer)

    assert flat_input.dtype == tf.int32


def test_layer_test_cast_layer():
    """
    The keras `layer_test` function checks several properties of the layer, i.e. shapes and dtypes.
    """
    output = layer_test(Cast, input_shape=(3, 4, 5))

    assert output.dtype == tf.keras.backend.floatx()
