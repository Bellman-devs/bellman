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
Utilities for implementing TRPO.
"""

from typing import Callable, List

import numpy as np
import tensorflow as tf

# constant used for numerical stability
EPS = 1e-8


def flatten_tensors(list_of_tensors: List[tf.Tensor]) -> tf.Tensor:
    """flatten and concatenate list of tensors into single (dx1) vector"""
    assert len(list_of_tensors) >= 1, "at least 1 tensor needs to be given"
    return tf.concat(
        [tf.reshape(x, [-1, 1]) for x in list_of_tensors], axis=0
    )  # pylint: disable=all


def unflatten_tensor(flat_tensor: tf.Tensor, list_of_vars: List[tf.Tensor]) -> List[tf.Tensor]:
    """
    Copy vector into list of variables. Each variable gets assigned values from the given vector in
    order. Combined size of variables must match size of vector
    """

    total_size = sum(np.prod(v.shape) for v in list_of_vars)
    assert tf.size(flat_tensor).numpy() == total_size, "Incompatible shapes"

    idx = 0
    for var in list_of_vars:
        siz = np.prod(var.shape)
        var.assign(tf.reshape(flat_tensor[idx : idx + siz, 0], var.shape))
        idx += siz
    return list_of_vars


def conjugate_gradient(
    matrix_vector_fun: Callable,
    vector: tf.Tensor,
    x0=None,
    max_iter: int = 30,
    tolerance: float = 1e-10,
) -> tf.Tensor:
    """
    Approximately solve system Ax=b using conjugate gradients.


    :param matrix_vector_fun: function x -> Ax that computes product Ax for any input vector x
    :param vector: vector b for the system Ax = b
    :param x0: initial guess for solution
    :param max_iter: maximum iterations to run
    :param tolerance: absolute residual for accepting solution
    :return: solution vector x ~= A^-1 * b
    """

    vector = tf.reshape(vector, (-1, 1))

    if x0 is None:
        solution = tf.zeros_like(vector)
    else:
        solution = tf.identity(x0)

    residual = vector - matrix_vector_fun(solution)
    squared_residual = tf.transpose(residual) @ residual
    direction = tf.identity(residual)

    for _ in range(max_iter):
        matrix_times_direction = matrix_vector_fun(direction)
        step_size = squared_residual / (tf.transpose(direction) @ matrix_times_direction + EPS)
        solution += step_size * direction
        residual -= step_size * matrix_times_direction
        new_squared_residual = tf.transpose(residual) @ residual

        if tf.sqrt(new_squared_residual) < tolerance:
            break
        direction = residual + (new_squared_residual / squared_residual) * direction
        squared_residual = new_squared_residual
    return solution


def hessian_vector_product(
    f: Callable, params: List[tf.Tensor], vector: tf.Tensor
) -> tf.Tensor:
    """
    Compute product of hessian H with vector v by using back-prop twice.

    :param f: scalar function for which to compute hessian vector product
    :param params: points at which to evaluate the function f
    :param vector: vector to multiply with hessian
    :return: flattened product of Hessian and vector
    """
    with tf.GradientTape() as g_tape:
        g_tape.watch(params)
        with tf.GradientTape() as f_tape:
            f_tape.watch(params)
            f_output = f(flatten_tensors(params))
        grads = flatten_tensors(f_tape.gradient(f_output, params))
        grad_vec = tf.reduce_sum(tf.multiply(grads, tf.stop_gradient(vector)))

    return flatten_tensors(g_tape.gradient(grad_vec, params))
