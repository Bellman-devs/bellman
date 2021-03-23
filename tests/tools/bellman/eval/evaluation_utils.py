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
Utilities for evaluating policies.
"""

from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.utils import common


def policy_evaluation(
    environment: TFEnvironment,
    policy: TFPolicy,
    num_episodes: int = 1,
    max_buffer_capacity: int = 200,
    use_function: bool = True,
) -> Trajectory:
    """
    Evaluate `policy` on the `environment`.

    :param environment: tf_environment instance.
    :param policy: tf_policy instance used to step the environment.
    :param num_episodes: Number of episodes to compute the metrics over.
    :param max_buffer_capacity:  Maximum capacity of replay buffer
    :param use_function: Option to enable use of `tf.function` when collecting the trajectory.
    :return: The recorded `Trajectory`.
    """

    time_step = environment.reset()
    policy_state = policy.get_initial_state(environment.batch_size)

    buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        policy.trajectory_spec,
        batch_size=environment.batch_size,
        max_length=max_buffer_capacity,
    )

    driver = dynamic_episode_driver.DynamicEpisodeDriver(
        environment, policy, observers=[buffer.add_batch], num_episodes=num_episodes
    )
    if use_function:
        common.function(driver.run)(time_step, policy_state)
    else:
        driver.run(time_step, policy_state)

    return buffer.gather_all()
