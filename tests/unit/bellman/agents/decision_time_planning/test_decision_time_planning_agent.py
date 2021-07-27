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
from tf_agents.agents import DdpgAgent, SacAgent, Td3Agent
from tf_agents.utils import common

from bellman.agents.components import EnvironmentModelComponents, ModelFreeAgentComponent
from bellman.agents.decision_time_planning.decision_time_planning_agent import (
    DecisionTimePlanningAgent,
    ModelFreeSupportedDecisionTimePlanningAgent,
)
from bellman.environments.initial_state_distribution_model import (
    create_uniform_initial_state_distribution,
)
from bellman.environments.transition_model.keras_model.keras import KerasTransitionModel
from bellman.environments.transition_model.keras_model.linear import LinearTransitionNetwork
from bellman.environments.transition_model.transition_model import TransitionModelTrainingSpec
from bellman.training.utils import TRAIN_ARGSPEC_COMPONENT_ID
from bellman.trajectory_optimisers.random_shooting import (
    random_shooting_trajectory_optimisation,
)
from tests.tools.bellman.environments.reward_model import ConstantReward
from tests.tools.bellman.environments.transition_model.transition_model import (
    StubTransitionModel,
)
from tests.tools.bellman.optimisers.optimisers import create_mock_model_free_agent
from tests.tools.bellman.specs.tensor_spec import ACTION_SPEC, OBSERVATION_SPEC, TIMESTEP_SPEC
from tests.tools.bellman.trajectories.trajectory import generate_dummy_trajectories


def test_train_method_increments_counter():
    """
    The docstring for the `_train` method of a TFAgent requires that the implementation increments
    the `train_step_counter`.
    """
    population_size = 1
    horizon = 10
    number_of_particles = 1

    trajectory_optimiser = random_shooting_trajectory_optimisation(
        TIMESTEP_SPEC, ACTION_SPEC, horizon, population_size, number_of_particles
    )
    network = LinearTransitionNetwork(OBSERVATION_SPEC)
    transition_model = KerasTransitionModel([network], OBSERVATION_SPEC, ACTION_SPEC)
    reward_model = ConstantReward(OBSERVATION_SPEC, ACTION_SPEC)
    initial_state_model = create_uniform_initial_state_distribution(OBSERVATION_SPEC)

    train_step_counter = common.create_variable("train_step_counter", shape=())
    agent = DecisionTimePlanningAgent(
        TIMESTEP_SPEC,
        ACTION_SPEC,
        (transition_model, TransitionModelTrainingSpec(1, 1)),
        reward_model,
        initial_state_model,
        trajectory_optimiser,
        train_step_counter=train_step_counter,
    )

    dummy_trajectories = generate_dummy_trajectories(
        OBSERVATION_SPEC, ACTION_SPEC, batch_size=population_size, trajectory_length=horizon
    )
    train_kwargs = {TRAIN_ARGSPEC_COMPONENT_ID: EnvironmentModelComponents.TRANSITION.value}
    agent.train(dummy_trajectories, **train_kwargs)

    assert train_step_counter.value() == 1


@pytest.mark.parametrize("agent_class", [DdpgAgent, SacAgent, Td3Agent])
@pytest.mark.parametrize(
    "train_component",
    [EnvironmentModelComponents.TRANSITION, ModelFreeAgentComponent.MODEL_FREE_AGENT],
)
def test_train_method_increments_counter_for_model_free_supported_agents(
    mocker, agent_class, train_component
):
    """
    The docstring for the `_train` method of a TFAgent requires that the implementation increments
    the `train_step_counter`.
    """
    population_size = 1
    number_of_particles = 1
    horizon = 10

    mf_agent = create_mock_model_free_agent(mocker, TIMESTEP_SPEC, ACTION_SPEC, agent_class)
    trajectory_optimiser = random_shooting_trajectory_optimisation(
        TIMESTEP_SPEC, ACTION_SPEC, horizon, population_size, number_of_particles
    )
    network = LinearTransitionNetwork(OBSERVATION_SPEC)
    transition_model = KerasTransitionModel([network], OBSERVATION_SPEC, ACTION_SPEC)
    reward_model = ConstantReward(OBSERVATION_SPEC, ACTION_SPEC)
    initial_state_model = create_uniform_initial_state_distribution(OBSERVATION_SPEC)

    train_step_counter = common.create_variable("train_step_counter", shape=())
    agent = ModelFreeSupportedDecisionTimePlanningAgent(
        TIMESTEP_SPEC,
        ACTION_SPEC,
        (transition_model, TransitionModelTrainingSpec(1, 1)),
        reward_model,
        initial_state_model,
        trajectory_optimiser,
        mf_agent,
        train_step_counter=train_step_counter,
    )

    dummy_trajectories = generate_dummy_trajectories(
        OBSERVATION_SPEC, ACTION_SPEC, batch_size=population_size, trajectory_length=horizon
    )
    train_kwargs = {TRAIN_ARGSPEC_COMPONENT_ID: train_component.value}
    agent.train(dummy_trajectories, **train_kwargs)

    assert train_step_counter.value() == 1


def test_train_oracle_transition_model():
    """
    Ensure that a non-trainable oracle transition model does not cause the agent `train` method to
    fail.
    """
    population_size = 1
    number_of_particles = 1
    horizon = 10

    trajectory_optimiser = random_shooting_trajectory_optimisation(
        TIMESTEP_SPEC, ACTION_SPEC, horizon, population_size, number_of_particles
    )
    transition_model = StubTransitionModel(OBSERVATION_SPEC, ACTION_SPEC)
    reward_model = ConstantReward(OBSERVATION_SPEC, ACTION_SPEC)
    initial_state_model = create_uniform_initial_state_distribution(OBSERVATION_SPEC)

    train_step_counter = common.create_variable("train_step_counter", shape=())
    with pytest.warns(RuntimeWarning):
        agent = DecisionTimePlanningAgent(
            TIMESTEP_SPEC,
            ACTION_SPEC,
            transition_model,
            reward_model,
            initial_state_model,
            trajectory_optimiser,
            train_step_counter=train_step_counter,
        )

    dummy_trajectories = generate_dummy_trajectories(
        OBSERVATION_SPEC, ACTION_SPEC, batch_size=population_size, trajectory_length=horizon
    )
    train_kwargs = {TRAIN_ARGSPEC_COMPONENT_ID: EnvironmentModelComponents.TRANSITION.value}
    loss_info = agent.train(dummy_trajectories, **train_kwargs)

    assert loss_info.loss is None
    assert loss_info.extra is None
