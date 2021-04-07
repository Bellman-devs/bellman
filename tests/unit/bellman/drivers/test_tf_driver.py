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
Unit tests for the `TFBatchDriver`. These are based on the unit tests for the TF-Agents `TFDriver`.
"""

from math import ceil

import pytest
import tensorflow as tf
from tf_agents.drivers.test_utils import (
    NumEpisodesObserver,
    NumStepsTransitionObserver,
    PyEnvironmentMock,
    TFPolicyMock,
)
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from bellman.drivers.tf_driver import TFBatchDriver

TRAJECTORY_LENGTH = 3


def _create_environment_and_policy(batch_size):
    tf_batched_environment = TFPyEnvironment(
        BatchedPyEnvironment(
            [PyEnvironmentMock(final_state=TRAJECTORY_LENGTH) for _ in range(batch_size)]
        )
    )
    policy = TFPolicyMock(
        tf_batched_environment.time_step_spec(),
        tf_batched_environment.action_spec(),
        batch_size=batch_size,
    )

    return tf_batched_environment, policy


def test_ensemble_driver_run_once_number_of_episodes(batch_size):
    env, policy = _create_environment_and_policy(batch_size)

    observer = NumEpisodesObserver()

    driver = TFBatchDriver(
        env, policy, observers=[observer], min_episodes=1, disable_tf_function=True
    )

    initial_time_step = env.reset()
    initial_policy_state = policy.get_initial_state(batch_size=batch_size)
    driver.run(initial_time_step, initial_policy_state)

    assert observer.num_episodes.read_value() == batch_size


def _expected_num_steps(trajectory_length):
    """
    The `TFPolicyMock` alternates between actions `1` and `2`. The `PyEnvironmentMock` terminates
    when the alternating sum exceeds `trajectory_length`.
    """
    return ceil(trajectory_length * (2 / 3))


def test_ensemble_driver_run_once_number_of_steps(batch_size):
    env, policy = _create_environment_and_policy(batch_size)

    observer = NumStepsTransitionObserver()

    driver = TFBatchDriver(
        env,
        policy,
        observers=[],
        transition_observers=[observer],
        min_episodes=1,
        disable_tf_function=True,
    )

    initial_time_step = env.reset()
    initial_policy_state = policy.get_initial_state(batch_size=batch_size)
    driver.run(initial_time_step, initial_policy_state)

    assert observer.num_steps.read_value() == batch_size * _expected_num_steps(
        TRAJECTORY_LENGTH
    )


@pytest.mark.parametrize("min_trajectories", [1, 2, 5])
def test_ensemble_driver_run_once_minimum_trajectories_in_each_batch_member(
    batch_size, min_trajectories
):
    env, policy = _create_environment_and_policy(batch_size)

    buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        policy.trajectory_spec, batch_size=env.batch_size, max_length=10 * TRAJECTORY_LENGTH
    )

    driver = TFBatchDriver(
        env,
        policy,
        observers=[buffer.add_batch],
        min_episodes=min_trajectories,
        disable_tf_function=True,
    )

    initial_time_step = env.reset()
    initial_policy_state = policy.get_initial_state(batch_size=batch_size)
    driver.run(initial_time_step, initial_policy_state)

    trajectories = buffer.gather_all()
    episode_termination_mask = tf.cast(trajectories.is_last(), dtype=tf.float32)
    episodes_per_batch = tf.reduce_sum(episode_termination_mask, axis=1)
    minimum_number_of_episodes_in_batch = tf.reduce_min(episodes_per_batch)

    assert minimum_number_of_episodes_in_batch >= min_trajectories
