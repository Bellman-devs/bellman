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

import tensorflow as tf
from tf_agents.trajectories.time_step import StepType

from bellman.distributions.utils import create_uniform_distribution_from_spec
from bellman.environments.environment_model import EnvironmentModel
from bellman.environments.initial_state_distribution_model import (
    create_uniform_initial_state_distribution,
)
from bellman.environments.termination_model import ConstantFalseTermination
from bellman.environments.transition_model.keras_model.keras import KerasTransitionModel
from tests.tools.bellman.environments.reward_model import ConstantReward
from tests.tools.bellman.environments.transition_model.keras_models.dummy_ensemble import (
    DummyEnsembleTransitionNetwork,
)


def test_batched_environment_model(observation_space, action_space, batch_size):
    transition_network = DummyEnsembleTransitionNetwork(observation_space)
    transition_model = KerasTransitionModel(
        [transition_network],
        observation_space,
        action_space,
    )
    reward = ConstantReward(observation_space, action_space, 0.0)
    termination = ConstantFalseTermination(observation_space)
    initial_state_sampler = create_uniform_initial_state_distribution(observation_space)

    env_model = EnvironmentModel(
        transition_model, reward, termination, initial_state_sampler, batch_size
    )
    action_distr = create_uniform_distribution_from_spec(action_space)
    single_action = action_distr.sample()
    batch_actions = tf.convert_to_tensor([single_action for _ in range(batch_size)])

    first_step = env_model.reset()
    assert (first_step.step_type == [StepType.FIRST for _ in range(batch_size)]).numpy().all()
    assert first_step.observation.shape == [batch_size] + list(observation_space.shape)

    next_step = env_model.step(batch_actions)
    assert (next_step.step_type == [StepType.MID for _ in range(batch_size)]).numpy().all()
    assert next_step.observation.shape == [batch_size] + list(observation_space.shape)
    assert next_step.reward.shape == [batch_size]
