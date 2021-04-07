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

from bellman.environments.environment_model import EnvironmentModel
from bellman.environments.initial_state_distribution_model import (
    create_uniform_initial_state_distribution,
)
from bellman.environments.termination_model import ConstantFalseTermination
from bellman.environments.tf_wrappers import TFTimeLimit
from bellman.environments.transition_model.keras_model.keras import KerasTransitionModel
from tests.tools.bellman.environments.reward_model import ConstantReward
from tests.tools.bellman.environments.transition_model.keras_models.dummy_ensemble import (
    DummyEnsembleTransitionNetwork,
)
from tests.tools.bellman.policies.utils import replay_actions_across_batch_transition_models


def test_replay_actions_across_batches(observation_space, action_space, horizon, batch_size):
    transition_network = DummyEnsembleTransitionNetwork(observation_space)
    transition_model = KerasTransitionModel(
        [transition_network],
        observation_space,
        action_space,
    )
    reward = ConstantReward(observation_space, action_space, 0.0)
    termination = ConstantFalseTermination(observation_space)
    initial_state_sampler = create_uniform_initial_state_distribution(observation_space)

    env_model = TFTimeLimit(
        EnvironmentModel(
            transition_model, reward, termination, initial_state_sampler, batch_size
        ),
        horizon,
    )

    actions_distribution = create_uniform_initial_state_distribution(observation_space)
    actions = actions_distribution.sample((horizon,))
    trajectory = replay_actions_across_batch_transition_models(env_model, actions)

    assert (
        trajectory.observation.shape
        == (
            batch_size,
            horizon,
        )
        + observation_space.shape
    )
