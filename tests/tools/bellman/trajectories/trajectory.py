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

import tensorflow as tf
from tf_agents.specs import BoundedTensorSpec
from tf_agents.trajectories.trajectory import from_episode
from tf_agents.utils import common


def generate_dummy_trajectories(
    observation_space: BoundedTensorSpec,
    action_space: BoundedTensorSpec,
    batch_size: int,
    trajectory_length: int,
):
    observation_space_shape = observation_space.shape.as_list()
    observation = tf.zeros(
        [batch_size, trajectory_length] + observation_space_shape,
        dtype=observation_space.dtype,
    )

    action_space_shape = action_space.shape.as_list()
    action = tf.zeros(
        [batch_size, trajectory_length] + action_space_shape, dtype=action_space.dtype
    )

    reward = tf.zeros((batch_size, trajectory_length), dtype=tf.float32)
    discount = tf.ones((batch_size, trajectory_length), dtype=tf.float32)

    trajectory = from_episode(observation, action, (), reward, discount)

    batched_step_type = tf.transpose(
        tf.repeat(trajectory.step_type[None, ...], repeats=[trajectory_length], axis=0)
    )
    batched_next_step_type = tf.transpose(
        tf.repeat(trajectory.next_step_type[None, ...], repeats=[trajectory_length], axis=0)
    )

    trajectory = trajectory.replace(step_type=batched_step_type)
    trajectory = trajectory.replace(next_step_type=batched_next_step_type)

    return trajectory
