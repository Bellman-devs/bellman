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

from bellman.agents.trpo.utils import (
    conjugate_gradient,
    flatten_tensors,
    hessian_vector_product,
    unflatten_tensor,
)


@tf.function
def _example_fun(x: tf.Tensor, y: tf.Tensor):
    """ function used to test hessian computations"""
    return x ** 3.0 - 2 * x * y - y ** 6.0


@pytest.mark.parametrize(
    "shape_list, vector",
    [
        ([(2,)], tf.ones((2, 1))),
        ([(2, 3)], tf.ones((6, 1))),
        ([(2, 3), (2, 3)], tf.ones((12, 1))),
        ([(2,), (2, 3), (2, 3, 4)], tf.ones((32, 1))),
    ],
)
def test_unflatten_size(shape_list, vector):
    """ Test unflatten returns list of variables with correct shapes"""
    var_tensors = [tf.Variable(np.zeros(shape), dtype=np.float32) for shape in shape_list]
    unflattened = unflatten_tensor(vector, var_tensors)
    for t, s in zip(unflattened, shape_list):
        assert tuple(tf.shape(t)) == tuple(s)


@pytest.mark.parametrize(
    "tensor_list, correct_size",
    [
        ([tf.ones(2)], 2),
        ([tf.ones((2, 3))], 6),
        ([tf.ones((2, 3)), tf.ones((2, 3))], 12),
        ([tf.ones((2,)), tf.ones((2, 3)), tf.ones((2, 3, 4))], 32),
    ],
)
def test_flatten_size(tensor_list, correct_size):
    """Test that flattening list of tensors returns vector of correct size"""
    flat = flatten_tensors(tensor_list)
    assert len(flat.numpy().shape) == 2
    assert flat.numpy().shape[-1] == 1
    assert flat.numpy().size == correct_size


@pytest.mark.parametrize(
    "tensor_list",
    [
        [tf.range(10)],
        [tf.reshape(tf.range(6), (2, 3))],
        [tf.reshape(tf.range(6), (2, 3)), tf.reshape(tf.range(6), (2, 3))],
        [
            tf.ones((2,)),
            tf.reshape(tf.range(6.0), (2, 3)),
            tf.reshape(tf.range(24.0), (2, 3, 4)),
        ],
    ],
)
def test_flatten_unflatten(tensor_list):
    """test unflattened tensorlist matches tensorlist before flattening it"""
    variables = [tf.Variable(np.zeros_like(t.numpy())) for t in tensor_list]
    flat = flatten_tensors(tensor_list)
    unflattened = unflatten_tensor(flat, variables)
    for before, after in zip(tensor_list, unflattened):
        np.testing.assert_array_almost_equal(before.numpy(), after.numpy())


@pytest.mark.parametrize(
    "a_matrix, b_vector",
    [
        (tf.eye(2), tf.ones((2, 1))),
        (
            tf.constant([[4.0, 1.0], [1.0, 3.0]], dtype=tf.float64),
            tf.constant([[1.0], [2.0]], dtype=tf.float64),
        ),
    ],
)
def test_cg(a_matrix, b_vector):
    """
    Test that cg solution matches exact solution computed in numpy.
    """

    def _matrix_times_x(x):
        return a_matrix @ x

    solution = conjugate_gradient(_matrix_times_x, b_vector, max_iter=100)

    # Note: accuracy seems very dependent on specific matrices
    np.testing.assert_array_almost_equal(
        solution.numpy(), (tf.linalg.inv(a_matrix) @ b_vector).numpy()
    )


@pytest.mark.parametrize(
    "vector, params",
    [
        (tf.ones((2, 1)), [tf.constant([1.0]), tf.constant([2.0])]),
        (tf.constant([[1.0], [2.0]]), [tf.constant([1.0]), tf.constant([2.0])]),
        (tf.ones((2, 1)), [tf.constant([0.0]), tf.constant([0.0])]),
    ],
)
def test_hvp_list(vector, params):
    """ Check Hessian vector product matches explicit computation"""
    fun = lambda p: _example_fun(p[0], p[1])

    @tf.function
    def _compute_hessians(f, ps):
        return tf.hessians(f(ps), ps)

    hessian = _compute_hessians(fun, tf.concat(params, axis=0))
    hvp_ref = tf.matmul(hessian, vector)

    # Compute hvp by double backprop
    hvp = hessian_vector_product(fun, params, vector)

    np.testing.assert_array_almost_equal(hvp_ref[0], hvp)
