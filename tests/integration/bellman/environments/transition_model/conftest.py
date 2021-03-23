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

import pytest
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from examples.utils.classic_control import create_pendulum_environment
from tests.tools.bellman.eval.evaluation_utils import policy_evaluation


@pytest.fixture(name="pendulum_training_data", scope="session")
def _pendulum_training_data_fixture():
    max_steps = 50
    num_episodes = 80

    tf_env = tf_py_environment.TFPyEnvironment(create_pendulum_environment(max_steps))

    collect_policy = RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())

    trajectories = policy_evaluation(
        tf_env,
        collect_policy,
        num_episodes=num_episodes,
        max_buffer_capacity=1000,
        use_function=True,
    )

    tf_env.close()

    return trajectories, tf_env


@pytest.fixture(name="mountain_car_data", scope="session")
def _mountain_car_data_fixture():
    tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load("MountainCar-v0"))

    collect_policy = RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())
    replay_buffer_capacity = 5000
    model_training_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        collect_policy.trajectory_spec,
        batch_size=1,
        max_length=replay_buffer_capacity,
    )

    collect_episodes_per_iteration = 10
    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        collect_policy,
        observers=[model_training_buffer.add_batch],
        num_episodes=collect_episodes_per_iteration,
    )

    collect_driver.run()
    tf_env.close()

    return tf_env, model_training_buffer.gather_all()
