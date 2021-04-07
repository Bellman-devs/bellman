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

from bellman.environments.mixins import BatchSizeUpdaterMixin
from bellman.environments.transition_model.keras_model.trajectory_sampling import (
    InfiniteHorizonTrajectorySampling,
    MeanTrajectorySamplingStrategy,
    OneStepTrajectorySampling,
)


def test_batch_partition_is_reversible(
    trajectory_sampling_strategy_factory, batch_size, ensemble_size
):
    strategy = trajectory_sampling_strategy_factory(batch_size, ensemble_size)
    starting_tensor = tf.range(batch_size)
    input_tensors = strategy.transform_step_inputs([starting_tensor])
    output_tensor = strategy.transform_step_outputs(input_tensors)

    np.testing.assert_array_equal(output_tensor, starting_tensor)


def test_update_batch_batch_partition_is_reversible(
    trajectory_sampling_strategy_factory, batch_size, ensemble_size
):
    strategy = trajectory_sampling_strategy_factory(batch_size, ensemble_size)
    strategy.update_batch_size(2 * batch_size)
    starting_tensor = tf.range(2 * batch_size)
    input_tensors = strategy.transform_step_inputs([starting_tensor])
    output_tensor = strategy.transform_step_outputs(input_tensors)

    np.testing.assert_array_equal(output_tensor, starting_tensor)


def test_batch_partition_is_reversible_within_tf_function(
    trajectory_sampling_strategy_factory, batch_size, ensemble_size
):
    strategy = trajectory_sampling_strategy_factory(batch_size, ensemble_size)
    starting_tensor = tf.range(batch_size)

    @tf.function
    def inner_function():
        input_tensors = strategy.transform_step_inputs([starting_tensor])
        return strategy.transform_step_outputs(input_tensors)

    output_tensor = inner_function()
    np.testing.assert_array_equal(output_tensor, starting_tensor)


def test_batch_partition_is_consistent_on_input_tensors(
    trajectory_sampling_strategy_factory, batch_size, ensemble_size
):
    strategy = trajectory_sampling_strategy_factory(batch_size, ensemble_size)

    # Set up the values of the tensors such that they are different by scaling the second tensor.
    input_tensors = [tf.range(batch_size), batch_size * tf.range(batch_size)]

    transformed_input_tensors = strategy.transform_step_inputs(input_tensors)
    transformed_input_tensors_iter = iter(transformed_input_tensors)

    for tensor_pair in zip(transformed_input_tensors_iter, transformed_input_tensors_iter):
        # Scale the values of the first tensor to ensure that the same indices were used to
        # partition both input tensors.
        np.testing.assert_array_equal(batch_size * tensor_pair[0], tensor_pair[1])


@pytest.fixture(name="fix_random_seed")
def _fix_random_seed_fixture():
    global_seed, _ = tf.compat.v1.random.get_seed(None)
    tf.random.set_seed(1)

    yield

    tf.random.set_seed(global_seed)


def test_one_step_trajectory_sampling_resample_indices(fix_random_seed):
    batch_size = 5
    ensemble_size = 10
    strategy = OneStepTrajectorySampling(batch_size, ensemble_size)

    input_tensor = tf.range(batch_size)
    first_transformed_tensors = strategy.transform_step_inputs([input_tensor])
    first_indices = [ind for ind, el in enumerate(first_transformed_tensors) if len(el)]

    second_transformed_tensors = strategy.transform_step_inputs([input_tensor])
    second_indices = [ind for ind, el in enumerate(second_transformed_tensors) if len(el)]

    assert not np.array_equal(first_indices, second_indices)


def test_infinite_horizon_trajectory_sampling_do_not_resample_indices_each_time(batch_size):
    ensemble_size = 100
    strategy = InfiniteHorizonTrajectorySampling(batch_size, ensemble_size)

    input_tensor = tf.range(batch_size)
    first_transformed_tensors = strategy.transform_step_inputs([input_tensor])
    first_indices = [ind for ind, el in enumerate(first_transformed_tensors) if len(el)]

    second_transformed_tensors = strategy.transform_step_inputs([input_tensor])
    second_indices = [ind for ind, el in enumerate(second_transformed_tensors) if len(el)]

    np.testing.assert_array_equal(first_indices, second_indices)


def test_infinite_horizon_trajectory_sampling_resample_indices(fix_random_seed):
    batch_size = 5
    ensemble_size = 10
    strategy = InfiniteHorizonTrajectorySampling(batch_size, ensemble_size)

    input_tensor = tf.range(batch_size)
    first_transformed_tensors = strategy.transform_step_inputs([input_tensor])
    first_indices = [ind for ind, el in enumerate(first_transformed_tensors) if len(el)]

    strategy.train_model()

    second_transformed_tensors = strategy.transform_step_inputs([input_tensor])
    second_indices = [ind for ind, el in enumerate(second_transformed_tensors) if len(el)]

    assert not np.array_equal(first_indices, second_indices)


def test_mean_trajectory_sampling_duplicate_input_tensors(batch_size, ensemble_size):
    strategy = MeanTrajectorySamplingStrategy(ensemble_size)

    # Set up the values of the tensors such that they are different by scaling the second tensor.
    input_tensors = [tf.range(batch_size), batch_size * tf.range(batch_size)]

    transformed_input_tensors = strategy.transform_step_inputs(input_tensors)
    assert len(transformed_input_tensors) == 2 * ensemble_size

    transformed_input_tensors_iter = iter(transformed_input_tensors)
    for tensor_pair in zip(transformed_input_tensors_iter, transformed_input_tensors_iter):
        np.testing.assert_array_equal(tensor_pair, input_tensors)


def test_mean_trajectory_sampling_transform_outputs(batch_size, ensemble_size):
    strategy = MeanTrajectorySamplingStrategy(ensemble_size)

    # Specify tensors which will have an integer mean to avoid numerical issues.
    output_tensors = [2 * (i + 1) * tf.range(batch_size) for i in range(ensemble_size)]

    transformed_output_tensors = strategy.transform_step_outputs(output_tensors)

    np.testing.assert_array_equal(
        transformed_output_tensors, (ensemble_size + 1) * tf.range(batch_size)
    )
