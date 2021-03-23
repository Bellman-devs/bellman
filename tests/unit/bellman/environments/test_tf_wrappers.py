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
from tf_agents.trajectories.time_step import StepType

from bellman.distributions.utils import create_uniform_distribution_from_spec
from bellman.environments.environment_model import EnvironmentModel
from bellman.environments.initial_state_distribution_model import (
    DeterministicInitialStateModel,
)
from bellman.environments.termination_model import ConstantFalseTermination
from bellman.environments.tf_wrappers import TFTimeLimit
from bellman.environments.transition_model.keras_model.keras import KerasTransitionModel
from tests.tools.bellman.environments.reward_model import ConstantReward
from tests.tools.bellman.environments.transition_model.keras_models.dummy_ensemble import (
    DummyEnsembleTransitionNetwork,
)


@pytest.fixture(name="wrapped_environment_and_action")
def _wrapped_environment_fixture(observation_space, action_space, batch_size):
    observation = create_uniform_distribution_from_spec(observation_space).sample()
    network = DummyEnsembleTransitionNetwork(observation_space)
    model = KerasTransitionModel([network], observation_space, action_space)
    env_model = EnvironmentModel(
        transition_model=model,
        reward_model=ConstantReward(observation_space, action_space, -1.0),
        termination_model=ConstantFalseTermination(observation_space),
        initial_state_distribution_model=DeterministicInitialStateModel(observation),
        batch_size=batch_size,
    )
    wrapped_environment_model = TFTimeLimit(env_model, 2)

    action = create_uniform_distribution_from_spec(action_space).sample((batch_size,))

    return wrapped_environment_model, action


def test_tf_time_limit_reset_num_steps(wrapped_environment_and_action):
    """
    Ensure that the number of steps after a termination are reset to 0.
    """
    wrapped_environment_model, action = wrapped_environment_and_action

    time_step = wrapped_environment_model.reset()
    np.testing.assert_array_equal(time_step.step_type, StepType.FIRST)

    next_time_step = wrapped_environment_model.step(action)
    np.testing.assert_array_equal(next_time_step.step_type, StepType.MID)

    next_time_step = wrapped_environment_model.step(action)
    np.testing.assert_array_equal(next_time_step.step_type, StepType.LAST)

    next_time_step = wrapped_environment_model.step(action)
    np.testing.assert_array_equal(next_time_step.step_type, StepType.FIRST)

    next_time_step = wrapped_environment_model.step(action)
    np.testing.assert_array_equal(next_time_step.step_type, StepType.MID)

    next_time_step = wrapped_environment_model.step(action)
    np.testing.assert_array_equal(next_time_step.step_type, StepType.LAST)


def test_tf_wrapper_reset_method_resets_num_steps(wrapped_environment_and_action):
    """
    Ensure that the number of steps after a reset are reset to 0.
    """
    wrapped_environment_model, action = wrapped_environment_and_action

    time_step = wrapped_environment_model.reset()
    np.testing.assert_array_equal(time_step.step_type, StepType.FIRST)

    next_time_step = wrapped_environment_model.step(action)
    np.testing.assert_array_equal(next_time_step.step_type, StepType.MID)

    next_time_step = wrapped_environment_model.reset()
    np.testing.assert_array_equal(next_time_step.step_type, StepType.FIRST)

    next_time_step = wrapped_environment_model.step(action)
    np.testing.assert_array_equal(next_time_step.step_type, StepType.MID)

    next_time_step = wrapped_environment_model.step(action)
    np.testing.assert_array_equal(next_time_step.step_type, StepType.LAST)


def test_tf_wrapper_set_initial_observation_resets_num_steps(wrapped_environment_and_action):
    """
    Ensure that the number of steps after setting the initial observation are reset to 0.
    """
    wrapped_environment_model, action = wrapped_environment_and_action

    time_step = wrapped_environment_model.reset()
    np.testing.assert_array_equal(time_step.step_type, StepType.FIRST)

    next_time_step = wrapped_environment_model.step(action)
    np.testing.assert_array_equal(next_time_step.step_type, StepType.MID)

    next_time_step = wrapped_environment_model.set_initial_observation(
        next_time_step.observation
    )
    np.testing.assert_array_equal(next_time_step.step_type, StepType.FIRST)

    next_time_step = wrapped_environment_model.step(action)
    np.testing.assert_array_equal(next_time_step.step_type, StepType.MID)

    next_time_step = wrapped_environment_model.step(action)
    np.testing.assert_array_equal(next_time_step.step_type, StepType.LAST)
