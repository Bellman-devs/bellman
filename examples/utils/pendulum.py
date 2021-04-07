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

from typing import Tuple

from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import BoundedTensorSpec
from tf_agents.trajectories.trajectory import Trajectory

from examples.utils.classic_control import create_pendulum_environment


def generate_pendulum_trajectories(
    batch_size: int, max_steps: int
) -> Tuple[Trajectory, BoundedTensorSpec, BoundedTensorSpec]:
    """
    Utility function for generating batches of trajectories from the Pendulum-v0 gym environment.

    :param batch_size: Number of trajectories to generate
    :param max_steps: Length of trajectories
    :return: A tuple consisting of
        * A `Trajectory` object containing the batch of trajectories
        * The observation spec from the Pendulum-v0 environment
        * The action spec from the Pendulum-v0 environment
    """
    tf_env = tf_py_environment.TFPyEnvironment(
        BatchedPyEnvironment(
            [create_pendulum_environment(max_steps) for _ in range(batch_size)]
        )
    )

    collect_policy = RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())
    replay_buffer_capacity = 1000
    model_training_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        collect_policy.trajectory_spec,
        batch_size=batch_size,
        max_length=replay_buffer_capacity,
    )

    collect_episodes_per_iteration = 1
    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        collect_policy,
        observers=[model_training_buffer.add_batch],
        num_episodes=collect_episodes_per_iteration,
    )

    collect_driver.run()
    tf_env.close()

    training_data = model_training_buffer.gather_all()

    return training_data, tf_env.observation_spec(), tf_env.action_spec()
