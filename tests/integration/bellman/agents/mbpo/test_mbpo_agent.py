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
import tensorflow as tf
from tf_agents.drivers.tf_driver import TFDriver
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from bellman.agents.background_planning.model_free_agent_types import ModelFreeAgentType
from bellman.agents.components import EnvironmentModelComponents
from bellman.agents.mbpo.mbpo_agent import MbpoAgent
from bellman.environments.transition_model.keras_model.trajectory_sampler_types import (
    TrajectorySamplerType,
)
from bellman.environments.transition_model.keras_model.transition_model_types import (
    TransitionModelType,
)
from bellman.training.background_planning_agent_trainer import BackgroundPlanningAgentTrainer
from examples.utils.classic_control import MountainCarInitialState, MountainCarReward


@pytest.mark.parametrize("transition_model", [e for e in TransitionModelType])
@pytest.mark.parametrize("trajectory_sampler", [e for e in TrajectorySamplerType])
@pytest.mark.parametrize(
    "model_free_agent_type",
    [ModelFreeAgentType.Ddpg, ModelFreeAgentType.Sac, ModelFreeAgentType.Td3],
)
def test_all_mepo_variants_work(transition_model, trajectory_sampler, model_free_agent_type):
    """
    Mbpo Agent has prespecified transition model, trajectory sampler and model-free agent
    types. Here we check that all combinations execute without errors.
    """

    # setup the environment and a prespecified model components
    py_env = suite_gym.load("MountainCarContinuous-v0")
    tf_env = TFPyEnvironment(py_env)
    time_step_spec = tf_env.time_step_spec()
    observation_spec = tf_env.observation_spec()
    action_spec = tf_env.action_spec()
    reward_model = MountainCarReward(observation_spec, action_spec)
    initial_state_distribution_model = MountainCarInitialState(observation_spec)

    # some parameters need to be set correctly
    ensemble_size = 2
    num_elites = 10
    population_size = num_elites + 10
    horizon = 1

    # define agent, many transition model and trajectory optimiser parameters can
    # be arbitrary
    agent = MbpoAgent(
        time_step_spec,
        action_spec,
        transition_model,
        1,
        10,
        tf.nn.relu,
        ensemble_size,
        False,
        1,
        1,
        [tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)],
        reward_model,
        initial_state_distribution_model,
        trajectory_sampler,
        horizon,
        population_size,
        model_free_agent_type,
        1,
        10,
        tf.nn.relu,
        2,
        1,
    )

    # we need some training data
    random_policy = RandomTFPolicy(
        time_step_spec,
        action_spec,
        info_spec=agent.collect_policy.info_spec,
    )
    model_training_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        random_policy.trajectory_spec, batch_size=1, max_length=1000
    )
    collect_driver_random_policy = TFDriver(
        tf_env,
        random_policy,
        observers=[model_training_buffer.add_batch],
        max_steps=10,
        disable_tf_function=True,
    )
    initial_time_step = tf_env.reset()
    collect_driver_random_policy.run(initial_time_step)
    pets_agent_trainer = BackgroundPlanningAgentTrainer(10, 10)
    tf_training_scheduler = pets_agent_trainer.create_training_scheduler(
        agent, model_training_buffer
    )
    training_losses = tf_training_scheduler.maybe_train(tf.constant(10, dtype=tf.int64))
    assert EnvironmentModelComponents.TRANSITION in training_losses

    # test the agent
    collect_driver_planning_policy = TFDriver(
        tf_env,
        agent.collect_policy,
        observers=[model_training_buffer.add_batch],
        max_steps=10,
        disable_tf_function=True,
    )
    time_step = tf_env.reset()
    collect_driver_planning_policy.run(time_step)
