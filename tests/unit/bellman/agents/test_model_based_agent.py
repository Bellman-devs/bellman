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
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.random_tf_policy import RandomTFPolicy

from bellman.agents.model_based_agent import ModelBasedAgent
from bellman.environments.transition_model.keras_model.keras import KerasTransitionModel
from bellman.environments.transition_model.keras_model.linear import LinearTransitionNetwork
from examples.utils.classic_control import (
    MountainCarInitialState,
    MountainCarReward,
    MountainCarTermination,
)


def test_incorrect_termination_model():
    """
    The generic model-based agent should only allow a ConstantFalseTermination model.
    """

    # setup arguments for the model-based agent constructor
    py_env = suite_gym.load("MountainCarContinuous-v0")
    tf_env = TFPyEnvironment(py_env)
    time_step_spec = tf_env.time_step_spec()
    observation_spec = tf_env.observation_spec()
    action_spec = tf_env.action_spec()
    network = LinearTransitionNetwork(observation_spec)
    transition_model = KerasTransitionModel([network], observation_spec, action_spec)
    reward_model = MountainCarReward(observation_spec, action_spec)
    initial_state_distribution_model = MountainCarInitialState(observation_spec)
    termination_model = MountainCarTermination(observation_spec)
    policy = RandomTFPolicy(time_step_spec, action_spec)

    with pytest.raises(AssertionError) as excinfo:
        ModelBasedAgent(
            time_step_spec,
            action_spec,
            transition_model,
            reward_model,
            termination_model,
            initial_state_distribution_model,
            policy,
            policy,
        )

    assert "Only constant false termination supported" in str(excinfo.value)
