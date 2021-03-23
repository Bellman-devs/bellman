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

from bellman.distributions.utils import create_uniform_distribution_from_spec
from bellman.environments.environment_model import EnvironmentModel
from bellman.environments.initial_state_distribution_model import (
    DeterministicInitialStateModel,
)
from bellman.environments.termination_model import ConstantFalseTermination
from bellman.environments.transition_model.keras_model.keras import KerasTransitionModel
from tests.tools.bellman.environments.reward_model import ConstantReward
from tests.tools.bellman.environments.transition_model.keras_models.dummy_ensemble import (
    DummyEnsembleTransitionNetwork,
)


def get_optimiser_and_environment_model(
    time_step_space,
    observation_space,
    action_space,
    population_size,
    number_of_particles,
    horizon,
    optimiser_policy_trajectory_optimiser_factory,
    sample_shape=(),
):
    reward = ConstantReward(observation_space, action_space, -1.0)

    batched_transition_network = DummyEnsembleTransitionNetwork(observation_space)
    batched_transition_model = KerasTransitionModel(
        [batched_transition_network],
        observation_space,
        action_space,
    )

    observation = create_uniform_distribution_from_spec(observation_space).sample(
        sample_shape=sample_shape
    )
    environment_model = EnvironmentModel(
        transition_model=batched_transition_model,
        reward_model=reward,
        termination_model=ConstantFalseTermination(observation_space),
        initial_state_distribution_model=DeterministicInitialStateModel(observation),
        batch_size=population_size,
    )
    trajectory_optimiser = optimiser_policy_trajectory_optimiser_factory(
        time_step_space, action_space, horizon, population_size, number_of_particles
    )
    return trajectory_optimiser, environment_model
