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
This module contains the keras layer `Cast`, which casts all input tensors to the appropriate data
type.
"""

import tensorflow as tf


class Cast(tf.keras.layers.Layer):
    """
    This class is a workaround for the TensorFlow keras autocasting behaviour. The keras
    documentation states that tensors in the first argument to the `call` method are automatically
    cast to the layer's dtype, where the dtype of the layer is either specified in the constructor
    or defaults to the keras `floatx` value, which in turn defaults to float32.

    The keras base layer only actually casts an input tensor if it is a floating point tensor. The
    casting is performed by the TensorFlow dtype `cast` function, which can cast between integer and
    floating point types.

    This class is a "catch all" cast, which casts all tensors which are not already of the correct
    type.
    """

    def __init__(self, dtype: tf.DType = None, **kwargs) -> None:
        self._target_dtype = dtype if dtype is not None else tf.keras.backend.floatx()
        super().__init__(dtype=self._target_dtype, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({"dtype": self._target_dtype})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs) -> tf.Tensor:
        def f(x):
            if x.dtype.base_dtype.name != self._target_dtype:
                return tf.dtypes.cast(x, self._target_dtype)
            else:
                return x

        return tf.nest.map_structure(f, inputs)


tf.keras.utils.get_custom_objects()["Cast"] = Cast
