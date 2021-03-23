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
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories.time_step import restart, time_step_spec
from tf_agents.typing.types import NestedTensorSpec, PolicyStep, TimeStep
from tf_agents.utils.nest_utils import tile_batch

from bellman.distributions.utils import create_uniform_distribution_from_spec
from bellman.trajectory_optimisers.particles import (
    averaged_particle_returns,
    decorate_policy_with_particles,
    reshape_create_particle_axis,
)


class _DummyPolicy(TFPolicy):
    def _distribution(self, time_step: TimeStep, policy_state: NestedTensorSpec) -> PolicyStep:
        raise NotImplementedError()


def test_decorate_policy_with_particles_wrong_policy(observation_space, action_space):
    time_step_space = time_step_spec(observation_space)
    policy = _DummyPolicy(time_step_space, action_space)

    with pytest.raises(AssertionError) as excinfo:
        decorate_policy_with_particles(policy, 1)

    assert "state-unconditioned policies" in str(excinfo)


def test_decorate_policy_with_particles_action_shapes(
    observation_space, action_space, population_size, number_of_particles
):
    time_step_space = time_step_spec(observation_space)
    policy = RandomTFPolicy(time_step_space, action_space)
    decorated_policy = decorate_policy_with_particles(policy, number_of_particles)

    observation = create_uniform_distribution_from_spec(observation_space).sample(
        sample_shape=(population_size * number_of_particles,)
    )
    initial_time_step = restart(observation, batch_size=population_size * number_of_particles)
    policy_step = decorated_policy.action(initial_time_step)
    actions = policy_step.action
    assert actions.shape == [population_size * number_of_particles] + action_space.shape.dims


def test_averaged_particle_returns(population_size, number_of_particles, horizon):
    batched_rewards = tf.random.normal(shape=(population_size * number_of_particles, horizon))
    batched_discount = tf.ones_like(batched_rewards)

    population_rewards = tf.split(batched_rewards, population_size)
    expected_rewards = [
        tf.reduce_mean(tf.reduce_sum(tensor, axis=1), axis=0) for tensor in population_rewards
    ]
    expected_reward = tf.concat(expected_rewards, axis=0)

    averaged_rewards = averaged_particle_returns(
        batched_rewards, batched_discount, number_of_particles
    )

    np.testing.assert_array_equal(expected_reward, averaged_rewards)


@pytest.mark.parametrize("multiplier", [1, 2, 10])
def test_reshape_create_minibatch_axis_from_tile_batch(multiplier):
    tensor = tf.random.uniform(shape=(2, 3, 4), maxval=10, dtype=tf.int32)
    batched_tensor = tile_batch(tensor, multiplier)

    minibatched_tensor = reshape_create_particle_axis(batched_tensor, multiplier)

    np.testing.assert_array_equal(minibatched_tensor.shape, (2, multiplier, 3, 4))
