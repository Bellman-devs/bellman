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

from bellman.distributions.utils import create_uniform_distribution_from_spec
from examples.utils.classic_control import CartPoleSwingUp, MountainCarReward
from examples.utils.pendulum import generate_pendulum_trajectories


def _get_cartpole_swingup_environment_with_state(x_position, theta):
    """
    Set the state vector of the cartpole swing up environment and return the environment.

    Sets the x_dot and theta_dot to values sampled from a standard normal (which matches their
    initialisation in the environment).

    :param x_position: x position of the cart
    :param theta: angle of the pole
    :return: Cartpole Swing up environment
    """
    environment = CartPoleSwingUp()
    random_state = np.random.randn(4)

    random_state[0] = x_position
    random_state[2] = theta

    environment.state = random_state

    return environment


@pytest.mark.parametrize("x_position,expected_reward", [(200, -101), (-200, -101), (0, -1)])
@pytest.mark.parametrize("action", [0, 1])
def test_discrete_action_space_cartpole_swingup_x_position_expected_reward(
    x_position, expected_reward, action
):
    environment = _get_cartpole_swingup_environment_with_state(x_position, np.pi)
    _, reward, _, _ = environment.step(action)

    np.testing.assert_almost_equal(reward, expected_reward, decimal=1)


@pytest.mark.parametrize("x_position,expected_done", [(200, True), (-200, True), (0, False)])
@pytest.mark.parametrize("action", [0, 1])
def test_discrete_action_space_cartpole_swingup_x_position_expected_done(
    x_position, expected_done, action
):
    environment = _get_cartpole_swingup_environment_with_state(x_position, np.pi)
    _, _, done, _ = environment.step(action)

    assert done == expected_done


@pytest.mark.parametrize(
    "x_position,expected_steps_beyond_done", [(200, 0), (-200, 0), (0, None)]
)
@pytest.mark.parametrize("theta", np.linspace(-np.pi, np.pi, num=5))
@pytest.mark.parametrize("action", [0, 1])
def test_discrete_action_space_cartpole_swingup_x_position_expected_steps_beyond_done(
    x_position, expected_steps_beyond_done, theta, action
):
    environment = _get_cartpole_swingup_environment_with_state(x_position, theta)
    environment.step(action)

    assert expected_steps_beyond_done == environment.steps_beyond_done


@pytest.mark.parametrize(
    "theta,expected_reward", zip(np.linspace(-np.pi, np.pi, num=5), [-1, 0, 1, 0, -1])
)
@pytest.mark.parametrize("action", [0, 1])
def test_discrete_action_space_cartpole_swingup_theta_expected_reward(
    theta, expected_reward, action
):
    environment = _get_cartpole_swingup_environment_with_state(0, theta)
    _, reward, _, _ = environment.step(action)

    np.testing.assert_almost_equal(reward, expected_reward, decimal=1)


@pytest.mark.parametrize("theta", np.linspace(-np.pi, np.pi, num=5))
@pytest.mark.parametrize("action", [0, 1])
def test_discrete_action_space_cartpole_swingup_theta_expected_done(theta, action):
    environment = _get_cartpole_swingup_environment_with_state(0, theta)
    _, _, done, _ = environment.step(action)

    assert not done


@pytest.fixture(name="test_data")
def _fixture(mountain_car_environment, batch_size):
    observation_space = mountain_car_environment.observation_spec()
    action_space = mountain_car_environment.action_spec()

    observation_distr = create_uniform_distribution_from_spec(observation_space)
    batch_observations = observation_distr.sample(batch_size)

    reward = MountainCarReward(observation_space, action_space)
    action_distr = create_uniform_distribution_from_spec(action_space)
    batch_actions = action_distr.sample(batch_size)

    return reward, batch_observations, batch_actions, batch_size


def test_mountain_car_reward_shape(test_data):
    reward, batch_observations, batch_actions, batch_size = test_data
    reward_batch = reward.step_reward(batch_observations, batch_actions, batch_observations)

    assert reward_batch.shape == [batch_size]


def test_pendulum_trajectory_shape(batch_size, trajectory_length):
    """
    Ensure that the `Trajectory` object contains data with the correct shape.
    """
    trajectory, _, _ = generate_pendulum_trajectories(batch_size, trajectory_length)
    assert trajectory.reward.shape == (batch_size, trajectory_length + 1)
