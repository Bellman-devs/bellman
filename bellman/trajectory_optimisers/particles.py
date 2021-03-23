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
When the policy used for trajectory optimisation is unconditional (samples actions without
conditioning on the state) it is possible to get a better monte-carlo estimate of the value of an
action trajectory by using the trajectory to propagate a cloud of particles and averaging the
returns.

This module adds support for particles to the trajectory optimisers. By default the value of an
action trajectory is estimated by a single particle.
"""

from typing import Callable, Optional

import tensorflow as tf
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.typing.types import NestedTensor, PolicyStep, Seed, Tensor, TimeStep
from tf_agents.utils.common import get_contiguous_sub_episodes
from tf_agents.utils.nest_utils import split_nested_tensors, tile_batch

from bellman.policies.cross_entropy_method_policy import CrossEntropyMethodPolicy


def decorate_policy_with_particles(policy: TFPolicy, number_of_particles: int) -> TFPolicy:
    """
    Decorate a policy's `action` method to duplicate the actions of an element of the batch over a
    set of particles.

    :param policy: An instance of `tf_policy.TFPolicy` representing the agent's current policy.
    :param number_of_particles: Number of monte-carlo rollouts of each action trajectory.

    :return: A decorated policy.
    """
    assert isinstance(
        policy, (CrossEntropyMethodPolicy, RandomTFPolicy)
    ), "Particles can only be used with state-unconditioned policies."

    def _wrapper(
        action_method: Callable[[TimeStep, NestedTensor, Optional[Seed]], PolicyStep]
    ):
        def action_method_method_wrapper(
            time_step: TimeStep, policy_state: NestedTensor = (), seed: Optional[Seed] = None
        ) -> PolicyStep:
            """
            The incoming `time_step` has a batch size of `population_size * number_of_particles`.
            This function reduces the batch size of `time_step` to be equal to `population_size`
            only. It does not matter which observations are retained because the policy must be
            state-unconditioned.

            The reduced time step is passed to the policy, and then each action is duplicated
            `number_of_particles` times to create a batch of
            `population_size * number_of_particles` actions.
            """
            reduced_time_step = split_nested_tensors(
                time_step, policy.time_step_spec, number_of_particles
            )[0]

            policy_step = action_method(reduced_time_step, policy_state, seed)
            actions = policy_step.action

            tiled_actions = tile_batch(actions, number_of_particles)

            return policy_step.replace(action=tiled_actions)

        return action_method_method_wrapper

    policy.action = _wrapper(policy.action)
    return policy


def averaged_particle_returns(
    reward: Tensor, discount: Tensor, number_of_particles: int
) -> Tensor:
    """
    Compute the returns from a set of trajectories, averaging over a number of particles per
    element of the batch.

    :param reward: A batch of trajectories of step rewards. The batch size is the number of action
                   trajectories multiplied by the number of monte-carlo rollouts of each action
                   trajectory.
    :param discount: A batch of trajectories of step discounts. The batch size is the number of
                     action trajectories multiplied by the number of monte-carlo rollouts of each
                     action trajectory.
    :param number_of_particles: Number of monte-carlo rollouts of each action trajectory.

    :return: Monte-carlo estimate of the returns from each action trajectory.
    """
    # Looks weird but is correct! At first sight, digging into it, it evokes the impression
    # that the last reward signal is missed. However, tf's cumprod method has an `exclusive`
    # flag which is set to True such that the last reward signal is included.
    mask = get_contiguous_sub_episodes(discount)

    particles_returns = tf.reduce_sum(reward * mask, axis=1)  # shape = (batch_size,)
    batch_particles_returns = reshape_create_particle_axis(
        particles_returns, number_of_particles
    )
    return tf.reduce_mean(batch_particles_returns, axis=1)


def reshape_create_particle_axis(
    batched_particles_tensor: Tensor, number_of_particles: int
) -> Tensor:
    """
    Given a tensor with a single batch dimension and shape
    [population_size * number_of_particles, ...], return a reshaped tensor with two batch
    dimensions and shape [population_size, number_of_particles, ...].

    :param batched_particles_tensor: A `Tensor` with shaped `[batch_size, ...]`.
    :param number_of_particles: An integer corresponding to the number of duplicates of the initial
                                batched tensor.

    :return: A `Tensor` with shape `[reduced_batch_size, multiplier, ...]`.
    """
    inner_shape = batched_particles_tensor.shape.dims[1:]
    return tf.reshape(batched_particles_tensor, [-1, number_of_particles] + inner_shape)
