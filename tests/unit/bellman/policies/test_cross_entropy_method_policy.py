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
from tf_agents.trajectories.time_step import StepType, TimeStep, time_step_spec

from bellman.environments.environment_model import EnvironmentModel
from bellman.environments.initial_state_distribution_model import (
    create_uniform_initial_state_distribution,
)
from bellman.environments.termination_model import ConstantFalseTermination
from bellman.environments.transition_model.keras_model.keras import KerasTransitionModel
from bellman.environments.transition_model.keras_model.linear import LinearTransitionNetwork
from bellman.policies.cross_entropy_method_policy import (
    CrossEntropyMethodPolicy,
    sample_action_batch,
)
from tests.tools.bellman.environments.reward_model import ConstantReward


def test_cross_entropy_method_policy_action_shape(observation_space, action_space, horizon):
    env_model, policy = get_cross_entropy_policy(observation_space, action_space, horizon, 1)
    initial_time_step = env_model.reset()
    initial_policy_state = policy.get_initial_state(env_model.batch_size)
    policy_step = policy.action(initial_time_step, initial_policy_state)

    assert policy_step.action.shape[0] == env_model.batch_size
    assert action_space.is_compatible_with(policy_step.action[0])


def test_initial_policy_state(observation_space, action_space, horizon):
    env_model, policy = get_cross_entropy_policy(observation_space, action_space, horizon, 1)
    (
        mean_spec,
        var_spec,
        low_spec,
        high_spec,
        actions_spec,
        step_index_spec,
    ) = policy.policy_state_spec
    mean, var, low, high, actions, step_index = policy.get_initial_state(env_model.batch_size)

    assert mean_spec.is_compatible_with(mean)
    assert var_spec.is_compatible_with(var)
    assert low_spec.is_compatible_with(low)
    assert high_spec.is_compatible_with(high)
    assert actions_spec.is_compatible_with(actions)
    assert step_index_spec.is_compatible_with(step_index)


def test_step_index_increments(observation_space, action_space):
    env_model, policy = get_cross_entropy_policy(observation_space, action_space, 1, 1)
    initial_time_step = env_model.reset()
    initial_policy_state = policy.get_initial_state(env_model.batch_size)

    assert initial_policy_state[5] == 0

    policy_step = policy.action(initial_time_step, initial_policy_state)

    assert policy_step.state[5] == 1


def test_cross_entropy_method_assert_step_index(observation_space, action_space, horizon):
    env_model, policy = get_cross_entropy_policy(observation_space, action_space, horizon, 2)
    initial_time_step = env_model.reset()
    initial_policy_state = policy.get_initial_state(env_model.batch_size)

    policy_step = policy.action(initial_time_step, initial_policy_state)
    mid_time_step = TimeStep(
        np.array([StepType.MID, StepType.LAST]),
        initial_time_step.reward,
        initial_time_step.discount,
        initial_time_step.observation,
    )
    for _ in range(horizon):
        policy_step = policy.action(mid_time_step, policy_step.state)

    final_time_step = TimeStep(
        np.array([StepType.MID, StepType.LAST]),
        initial_time_step.reward,
        initial_time_step.discount,
        initial_time_step.observation,
    )

    with pytest.raises(AssertionError) as excinfo:
        policy.action(final_time_step, policy_step.state)

    assert f"Max step index {horizon + 1} is out of range (> {horizon})" in str(excinfo)


def get_cross_entropy_policy(observation_space, action_space, horizon, batch_size):
    time_step_space = time_step_spec(observation_space)
    network = LinearTransitionNetwork(observation_space)
    model = KerasTransitionModel([network], observation_space, action_space)
    env_model = EnvironmentModel(
        model,
        ConstantReward(observation_space, action_space),
        ConstantFalseTermination(observation_space),
        create_uniform_initial_state_distribution(observation_space),
        batch_size,
    )
    policy = CrossEntropyMethodPolicy(time_step_space, action_space, horizon, batch_size)
    return env_model, policy


def test_sample_actions_batch_shape(action_space, horizon, batch_size):
    sample_shape = (horizon + 1,) + action_space.shape
    mean = tf.zeros(sample_shape)
    var = tf.ones(sample_shape)
    low = tf.zeros(sample_shape)
    high = tf.ones(sample_shape)

    actions_batch = sample_action_batch(mean, var, low, high, batch_size)

    assert actions_batch.shape == (batch_size, horizon + 1) + action_space.shape
