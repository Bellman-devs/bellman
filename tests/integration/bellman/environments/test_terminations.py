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
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories.time_step import time_step_spec

from bellman.environments.environment_model import EnvironmentModel
from bellman.environments.initial_state_distribution_model import (
    create_uniform_initial_state_distribution,
)
from bellman.environments.termination_model import ConstantFalseTermination
from bellman.environments.tf_wrappers import TFTimeLimit
from bellman.environments.transition_model.keras_model.keras import KerasTransitionModel
from bellman.environments.transition_model.keras_model.linear import LinearTransitionNetwork
from tests.tools.bellman.environments.reward_model import ConstantReward


def test_tf_time_limit_wrapper_with_environment_model(
    observation_space, action_space, trajectory_length
):
    """
    This test checks that the environment wrapper can in turn be wrapped by the `TimeLimit`
    environment wrapper from TF-Agents.
    """
    ts_spec = time_step_spec(observation_space)

    network = LinearTransitionNetwork(observation_space)
    environment = KerasTransitionModel([network], observation_space, action_space)
    wrapped_environment = TFTimeLimit(
        EnvironmentModel(
            environment,
            ConstantReward(observation_space, action_space, 0.0),
            ConstantFalseTermination(observation_space),
            create_uniform_initial_state_distribution(observation_space),
        ),
        trajectory_length,
    )

    collect_policy = RandomTFPolicy(ts_spec, action_space)
    replay_buffer_capacity = 1001
    policy_training_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        collect_policy.trajectory_spec, batch_size=1, max_length=replay_buffer_capacity
    )

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        wrapped_environment,
        collect_policy,
        observers=[policy_training_buffer.add_batch],
        num_episodes=1,
    )
    collect_driver.run()

    trajectories = policy_training_buffer.gather_all()

    assert trajectories.step_type.shape == (1, trajectory_length + 1)
