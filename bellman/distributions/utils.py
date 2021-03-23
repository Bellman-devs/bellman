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
Utilities related to distributions.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.specs import BoundedTensorSpec


def create_uniform_distribution_from_spec(
    spec: BoundedTensorSpec,
) -> tfp.distributions.Distribution:
    """
    Helper function which returns either a categorical distribution or a uniform distribution over
    the values of the tensors specified by the tensor spec.

    :param spec: A bounded tensor spec defining a space.
    :return: A distribution over the input space which is uniform (resp. categorical) if the input
             space is continuous (resp. discrete).
    """
    if spec.dtype.is_integer:
        zero_logits = tf.zeros(
            spec.shape + (spec.maximum - spec.minimum + 1,), dtype=tf.float32
        )
        return tfp.distributions.Categorical(zero_logits, dtype=spec.dtype)
    else:
        return tfp.distributions.Uniform(
            low=tf.convert_to_tensor(
                np.broadcast_to(spec.minimum, spec.shape), dtype=spec.dtype
            ),
            high=tf.convert_to_tensor(
                np.broadcast_to(spec.maximum, spec.shape), dtype=spec.dtype
            ),
        )
