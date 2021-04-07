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

from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories.time_step import StepType

from bellman.environments.environment_model import EnvironmentModel
from bellman.environments.tf_wrappers import TFTimeLimit
from bellman.environments.transition_model.keras_model.keras import KerasTransitionModel
from bellman.environments.transition_model.keras_model.linear import LinearTransitionNetwork
from examples.utils.classic_control import MountainCarInitialState, MountainCarTermination
from tests.tools.bellman.environments.reward_model import ConstantReward


def test_sample_trajectory_for_mountain_car():
    tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load("MountainCar-v0"))

    network = LinearTransitionNetwork(tf_env.observation_spec())
    model = KerasTransitionModel(
        [network],
        tf_env.observation_spec(),
        tf_env.action_spec(),
    )
    reward = ConstantReward(tf_env.observation_spec(), tf_env.action_spec(), -1.0)
    terminates = MountainCarTermination(tf_env.observation_spec())
    initial_state_sampler = MountainCarInitialState(tf_env.observation_spec())
    environment = TFTimeLimit(
        EnvironmentModel(model, reward, terminates, initial_state_sampler), duration=200
    )

    collect_policy = RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())
    replay_buffer_capacity = 1001
    policy_training_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        collect_policy.trajectory_spec, batch_size=1, max_length=replay_buffer_capacity
    )

    collect_episodes_per_iteration = 2
    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        environment,
        collect_policy,
        observers=[policy_training_buffer.add_batch],
        num_episodes=collect_episodes_per_iteration,
    )

    collect_driver.run()

    trajectory = policy_training_buffer.gather_all()

    first_batch_step_type = trajectory.step_type[0, :]
    assert (
        first_batch_step_type[0] == StepType.FIRST
        and first_batch_step_type[-1] == StepType.LAST
    )
