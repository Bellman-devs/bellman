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
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.utils.tensor_normalizer import StreamingTensorNormalizer

from bellman.agents.trpo.trpo_agent import compute_return_and_advantage
from tests.tools.bellman.agents.trpo.trpo_agent import (
    create_trpo_agent_factory,
    dummy_trajectory_batch,
)


@pytest.fixture(name="trajectory_batch")
def _trajectory(batch_size=2, n_steps=5, obs_dim=2):
    return dummy_trajectory_batch(batch_size, n_steps, obs_dim)


@pytest.fixture(name="time_step_batch")
def _time_step_batch(trajectory_batch):
    return trajectory.to_transition(trajectory_batch)[-1]


@pytest.fixture(name="create_trpo_agent")
def _create_trpo_agent_fixture():
    return create_trpo_agent_factory()


def _compute_gae(rewards, values, gamma, lambda_):
    """ generalised advantage computation"""
    deltas = rewards + gamma * values[:, 1:] - values[:, :-1]
    coeff = lambda_ * gamma
    result = np.zeros_like(rewards)
    accumulator = 0
    for delta_idx in reversed(range(deltas.shape[-1])):
        accumulator = deltas[:, delta_idx] + coeff * accumulator
        result[:, delta_idx] = accumulator

    return result


def _normalise(array, eps=1e-8):
    """mean / std normalisation"""
    return (array - array.mean(keepdims=True)) / (array.std(keepdims=True) + eps)


def test_policy(create_trpo_agent):
    """ Test policy returns correct action shapes"""
    trpo_agent = create_trpo_agent()
    observations = tf.constant([[1, 2]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=1)
    action_step = trpo_agent.policy.action(time_steps)
    actions = action_step.action
    assert tuple(actions.shape.as_list()) == (1, 1)


def test_value_estimation_loss(create_trpo_agent):
    """ Test computation of value estimation loss"""
    trpo_agent = create_trpo_agent()

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    returns = tf.constant([1.9, 1.0], dtype=tf.float32)
    weights = tf.ones_like(returns)

    expected_loss = 123.205
    loss = trpo_agent.value_estimation_loss(time_steps, returns, weights).numpy()

    np.testing.assert_allclose(loss, expected_loss)


def test_compute_return_and_advantage(create_trpo_agent, time_step_batch):
    """ Test computation of normalised returns and advantages """
    trpo_agent = create_trpo_agent()

    values = np.ones(
        (time_step_batch.reward.shape[0], time_step_batch.reward.shape[1] + 1),
        dtype=np.float32,
    )
    # manually computed values
    expected_gaes = np.array([[1.72262475, 1.48005, 0.99, 0.0]] * 2)
    ref_return = np.array([[2.72262475, 2.48005, 1.99, 1.0]] * 2)
    ref_gaes = _normalise(expected_gaes)

    discount = trpo_agent._discount_factor
    lambda_ = trpo_agent._lambda

    ret, gae = compute_return_and_advantage(
        discount, lambda_, time_step_batch.reward, time_step_batch, values
    )

    np.testing.assert_array_almost_equal(gae.numpy(), ref_gaes)
    np.testing.assert_array_almost_equal(ret.numpy(), ref_return)


def test_policy_gradient_loss(create_trpo_agent):
    """ Test computation of value policy gradient loss"""
    trpo_agent = create_trpo_agent()

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    actions = tf.constant([[0], [1]], dtype=tf.float32)
    sample_action_log_probs = tf.constant([0.9, 0.3], dtype=tf.float32)
    advantages = tf.constant([1.9, 1.0], dtype=tf.float32)
    weights = tf.ones_like(advantages)

    current_policy_distribution, unused_network_state = trpo_agent._actor_net(
        time_steps.observation, time_steps.step_type, ()
    )

    expected_loss = -0.01646461
    loss = trpo_agent.policy_gradient_loss(
        time_steps,
        actions,
        sample_action_log_probs,
        advantages,
        current_policy_distribution,
        weights,
    ).numpy()
    np.testing.assert_allclose(loss, expected_loss)


def test_policy_gradient(create_trpo_agent):
    """ Test computation of policy gradient"""
    trpo_agent = create_trpo_agent()

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    actions = tf.constant([[0], [1]], dtype=tf.float32)
    advantages = tf.constant([1.9, 1.0], dtype=tf.float32)
    weights = tf.ones_like(advantages)

    policy_info = {
        "dist_params": {
            "loc": tf.constant([[0.0], [0.0]], dtype=tf.float32),
            "scale": tf.constant([[1.0], [1.0]], dtype=tf.float32),
        }
    }
    policy_steps = policy_step.PolicyStep(action=actions, state=(), info=policy_info)

    # manually computed values for dummy policy
    expected_loss = -0.09785123
    expected_grads = [
        np.array([[0.01901411, -0.00523423], [0.03126473, -0.008375]], dtype=np.float32),
        np.array([0.01225063, -0.00314077], dtype=np.float32),
    ]
    loss, grads = trpo_agent.policy_gradient(time_steps, policy_steps, advantages, weights)

    np.testing.assert_allclose(loss, expected_loss)
    for g, g_ref in zip(grads, expected_grads):
        np.testing.assert_array_almost_equal(g, g_ref)


def test_natural_gradient(create_trpo_agent):
    """ Test computation of natural policy gradient"""
    trpo_agent = create_trpo_agent()

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    actions = tf.constant([[0], [1]], dtype=tf.float32)
    weights = tf.ones(2)

    # Note: policy needs to be sufficiently close to DummyActorNet policy or values blow up
    policy_info = {
        "dist_params": {
            "loc": tf.constant([[8.8], [15.1]], dtype=tf.float32),
            "scale": tf.constant([[7.5], [12.3]], dtype=tf.float32),
        },
    }
    policy_steps = policy_step.PolicyStep(action=actions, state=(), info=policy_info)

    policy_gradients = [
        tf.constant([[0.27840388, -0.07645298], [0.45947012, -0.12277764]], dtype=tf.float32),
        tf.constant([0.18106624, -0.04632466], dtype=tf.float32),
    ]

    # values computed with exact hessian and OpenAI TRPO conj grad implementation
    ref_nat_grad = np.array(
        [-6.99209238, 0.57774081, 5.61016746, -0.65180424, 12.60223692, -1.22954106]
    ).reshape((-1, 1))

    ref_coeff = 0.0815904435

    nat_grad, coeff = trpo_agent.natural_policy_gradient(
        time_steps, policy_steps, policy_gradients, weights
    )

    np.testing.assert_allclose(coeff, ref_coeff, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_almost_equal(nat_grad, ref_nat_grad, decimal=4)


def test_train_does_not_update_policy_with_constraint(create_trpo_agent, trajectory_batch):
    """
    Ensure calling train does not updates policy  variables when KL constraint is violated
    """
    # minimum possible KL is ~ 0.825
    trpo_agent = create_trpo_agent(max_kl=0.5)

    values_before = {
        v.name: v.numpy().copy() for v in trpo_agent._actor_net.trainable_variables
    }
    trpo_agent.train(trajectory_batch)
    values_after = {
        v.name: v.numpy().copy() for v in trpo_agent._actor_net.trainable_variables
    }
    for v in trpo_agent._actor_net.trainable_variables:
        assert np.array_equal(values_before[v.name], values_after[v.name]), v.name


def test_train_updates_policy(create_trpo_agent, trajectory_batch):
    """
    Ensure calling train updates policy variables when KL constraint is not violated
    """
    # minimum possible KL is ~ 0.825
    trpo_agent = create_trpo_agent(max_kl=0.9)

    values_before = {
        v.name: v.numpy().copy() for v in trpo_agent._actor_net.trainable_variables
    }
    trpo_agent.train(trajectory_batch)
    values_after = {
        v.name: v.numpy().copy() for v in trpo_agent._actor_net.trainable_variables
    }
    for v in trpo_agent._actor_net.trainable_variables:
        assert not np.array_equal(values_before[v.name], values_after[v.name]), v.name


def test_train_updates_value_function(create_trpo_agent, trajectory_batch):
    """
    Ensure calling train updates value function variables
    """
    trpo_agent = create_trpo_agent()

    values_before = {
        v.name: v.numpy().copy() for v in trpo_agent._value_net.trainable_variables
    }
    trpo_agent.train(trajectory_batch)
    values_after = {
        v.name: v.numpy().copy() for v in trpo_agent._value_net.trainable_variables
    }

    # should always update, independent of policy constraints
    for v in trpo_agent._value_net.trainable_variables:
        assert not np.array_equal(values_before[v.name], values_after[v.name]), v.name
