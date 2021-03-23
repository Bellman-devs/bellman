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
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment

from bellman.agents.background_planning.model_free_agent_types import ModelFreeAgentType
from bellman.agents.mbpo.mbpo_agent import MbpoAgent
from bellman.environments.transition_model.keras_model.trajectory_sampler_types import (
    TrajectorySamplerType,
)
from bellman.environments.transition_model.keras_model.transition_model_types import (
    TransitionModelType,
)
from examples.utils.classic_control import MountainCarInitialState, MountainCarReward


def test_unknown_transition_model():
    """
    Mepo Agent has prespecified transition model, RuntimeError should raise on unknown model.
    """

    # setup the environment and a prespecified model components
    py_env = suite_gym.load("MountainCarContinuous-v0")
    tf_env = TFPyEnvironment(py_env)
    time_step_spec = tf_env.time_step_spec()
    observation_spec = tf_env.observation_spec()
    action_spec = tf_env.action_spec()
    reward_model = MountainCarReward(observation_spec, action_spec)
    initial_state_distribution_model = MountainCarInitialState(observation_spec)

    # transition model and model-free agent
    transition_model_type = "unknown_model"
    trajectory_sampler_type = TrajectorySamplerType.TS1
    model_free_agent_type = ModelFreeAgentType.Ppo

    # some parameters need to be set correctly
    ensemble_size = 2
    num_elites = 10
    population_size = num_elites + 10
    horizon = 1

    with pytest.raises(RuntimeError) as excinfo:
        MbpoAgent(
            time_step_spec,
            action_spec,
            transition_model_type,
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
            trajectory_sampler_type,
            horizon,
            population_size,
            model_free_agent_type,
            1,
            10,
            tf.nn.relu,
            2,
            1,
        )

    assert "Unknown transition model" in str(excinfo.value)


def test_ensemble_size_set_correctly():
    """
    For ensemble transition models ensemble size needs to be larger than 1.
    """

    # setup the environment and a prespecified model components
    py_env = suite_gym.load("MountainCarContinuous-v0")
    tf_env = TFPyEnvironment(py_env)
    time_step_spec = tf_env.time_step_spec()
    observation_spec = tf_env.observation_spec()
    action_spec = tf_env.action_spec()
    reward_model = MountainCarReward(observation_spec, action_spec)
    initial_state_distribution_model = MountainCarInitialState(observation_spec)

    # transition model and model-free agent
    transition_model_type = TransitionModelType.DeterministicEnsemble
    trajectory_sampler_type = TrajectorySamplerType.TS1
    model_free_agent_type = ModelFreeAgentType.Ppo

    # some parameters need to be set correctly
    ensemble_size = 1
    population_size = 10
    horizon = 1

    # define agent, many transition model and trajectory optimiser parameters can
    # be arbitrary
    with pytest.raises(AssertionError) as excinfo:
        MbpoAgent(
            time_step_spec,
            action_spec,
            transition_model_type,
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
            trajectory_sampler_type,
            horizon,
            population_size,
            model_free_agent_type,
            1,
            10,
            tf.nn.relu,
            2,
            1,
        )

    assert "ensemble_size should be > 1" in str(excinfo.value)
