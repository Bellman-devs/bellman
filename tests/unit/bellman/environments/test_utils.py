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

from itertools import chain, repeat

import numpy as np
import pytest
import tensorflow as tf
from gym import Env, register
from gym.envs.registration import registry
from gym.spaces import Discrete
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories.time_step import time_step_spec
from tf_agents.utils.common import replicate

from bellman.distributions.utils import create_uniform_distribution_from_spec
from bellman.environments.environment_model import EnvironmentModel
from bellman.environments.initial_state_distribution_model import (
    DeterministicInitialStateModel,
)
from bellman.environments.termination_model import ConstantFalseTermination
from bellman.environments.tf_wrappers import TFTimeLimit
from bellman.environments.transition_model.keras_model.keras import KerasTransitionModel
from bellman.environments.utils import (
    create_real_tf_environment,
    virtual_rollouts_buffer_and_driver,
)
from tests.tools.bellman.environments.reward_model import ConstantReward
from tests.tools.bellman.environments.transition_model.keras_models.dummy_ensemble import (
    DummyEnsembleTransitionNetwork,
)


def test_generate_virtual_rollouts_assert_no_time_limit_wrapper(mocker):
    env_model = TFTimeLimit(mocker.MagicMock(spec=EnvironmentModel), 100)
    policy = mocker.MagicMock(spec=TFPolicy)

    with pytest.raises(AssertionError) as excinfo:
        virtual_rollouts_buffer_and_driver(env_model, policy, 64)

    assert "should not be wrapped" in str(excinfo)


def test_generate_virtual_rollouts(observation_space, action_space, batch_size, horizon):
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
    random_policy = RandomTFPolicy(time_step_spec(observation_space), action_space)

    replay_buffer, driver, wrapped_env_model = virtual_rollouts_buffer_and_driver(
        env_model, random_policy, horizon
    )

    driver.run(wrapped_env_model.reset())
    trajectory = replay_buffer.gather_all()

    mid_steps = repeat(1, horizon - 1)
    expected_step_types = tf.constant(list(chain([0], mid_steps, [2])))
    batched_step_types = replicate(expected_step_types, (batch_size,))
    np.testing.assert_array_equal(batched_step_types, trajectory.step_type)


class DummyEnvironment(Env):

    _seed = None

    @property
    def random_seed(self):
        return self._seed

    def seed(self, seed=None):
        self._seed = seed
        return super().seed(seed)

    def __init__(self):
        self.observation_space = Discrete(1)
        self.action_space = Discrete(1)

    def step(self, action):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def render(self, mode="human"):
        raise NotImplementedError()


@pytest.fixture(name="dummy_environment_name")
def _register_dummy_environment_fixture():
    id_string = "DummyEnvironment-v0"

    register(
        id=id_string, entry_point="tests.unit.bellman.environments.test_utils:DummyEnvironment"
    )

    yield id_string

    del registry.env_specs[id_string]


@pytest.mark.parametrize("environment_seed", [0, 1, 10])
def test_create_real_tf_environment(dummy_environment_name, environment_seed):
    environment = create_real_tf_environment(dummy_environment_name, environment_seed)
    assert environment.pyenv.envs[0].random_seed == environment_seed
