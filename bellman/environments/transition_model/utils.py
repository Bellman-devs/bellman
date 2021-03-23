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
This module contains utility functions for adapting TF-Agents data structures for use with Keras
models.
"""

from collections import namedtuple

import numpy as np
import tensorflow as tf
from tf_agents.specs import TensorSpec
from tf_agents.trajectories.trajectory import Trajectory

Transition = namedtuple("Transition", ["observation", "action", "reward", "next_observation"])


def extract_transitions_from_trajectories(
    trajectory: Trajectory,
    observation_spec: TensorSpec,
    action_spec: TensorSpec,
    predict_state_difference: bool,
) -> Transition:
    """
    TF-Agents returns a batch of trajectories from a buffer as a `Trajectory` object. This function
    transforms the data in the batch into a `Transition` tuple which can be used used for training
    the model.

    :param trajectory: The TF-Agents trajectory object
    :param observation_spec: The `TensorSpec` object which defines the observation tensors
    :param action_spec: The `TensorSpec` object which defines the action tensors
    :param predict_state_difference: Boolean to specify whether the transition model should
        return the next (latent) state or the difference between the current (latent) state and
        the next (latent) state

    :return: A `Transition` tuple which contains the observations and actions which can be used to
            train the model.
    """
    mask = ~trajectory.is_boundary()[:, :-1]  # to filter out boundary elements

    trajectory_observation = trajectory.observation
    # [batch_size, time_dim, features...]
    tf.ensure_shape(trajectory_observation, [None, None] + observation_spec.shape)
    next_observation = tf.boolean_mask(trajectory_observation[:, 1:, ...], mask)
    observation = tf.boolean_mask(trajectory_observation[:, :-1, ...], mask)

    trajectory_action = trajectory.action
    # [batch_size, time_dim, features...]
    tf.ensure_shape(trajectory_action, [None, None] + action_spec.shape)
    action = tf.boolean_mask(trajectory_action[:, :-1, ...], mask)

    trajectory_reward = trajectory.reward
    # [batch_size, time_dim]
    tf.ensure_shape(trajectory_reward, [None, None])
    reward = tf.boolean_mask(trajectory_reward[:, :-1], mask)

    if predict_state_difference:
        next_observation -= observation

    return Transition(
        observation=observation,
        action=action,
        reward=reward,
        next_observation=next_observation,
    )


def size(tensor_spec: TensorSpec) -> int:
    """
    Equivalent to `np.size` for `TensorSpec` objects.
    """
    return int(np.prod(tensor_spec.shape))
