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

"""
A decision-time planning agent which uses trajectories sampled from an approximate MDP in accordance
with Sutton and Barto 2017. Note that decision-time planning refers to any algorithm that specifies
and solves a planning problem in order to identify an optimal action for an encountered state.
Here, we assume "simple" planning algorithms that roll out trajectories with action proposals, and
eventually execute the first action from the best trajectory (measured via cumulative reward).
"""

from typing import Optional, Tuple, Union

import tensorflow as tf
from tf_agents.agents import DdpgAgent, SacAgent, Td3Agent, tf_agent
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.typing.types import NestedTensorSpec

from bellman.agents.components import ModelFreeAgentComponent
from bellman.agents.model_based_agent import ModelBasedAgent
from bellman.environments.environment_model import EnvironmentModel
from bellman.environments.initial_state_distribution_model import InitialStateDistributionModel
from bellman.environments.reward_model import RewardModel
from bellman.environments.termination_model import ConstantFalseTermination
from bellman.environments.transition_model.transition_model import (
    TrainableTransitionModel,
    TransitionModel,
    TransitionModelTrainingSpec,
)
from bellman.policies.planning_policy import PlanningPolicy
from bellman.trajectory_optimisers.trajectory_optimisers import TrajectoryOptimiser


class DecisionTimePlanningAgent(ModelBasedAgent):
    """
    An agent for decision-time planning. The agent is assumed to have access to models for all
    components of the model of the environment (i.e. MDP) - transition dynamics, rewards,
    termination state and initial state distribution - and a trajectory optimiser that optimizes
    actions using an internally assembled  environment model. The trajectory optimizer together
    with the environment model determine a planning policy.
    """

    def __init__(
        self,
        time_step_spec: TimeStep,
        action_spec: NestedTensorSpec,
        transition_model: Union[
            Tuple[TrainableTransitionModel, TransitionModelTrainingSpec], TransitionModel
        ],
        reward_model: RewardModel,
        initial_state_distribution_model: InitialStateDistributionModel,
        trajectory_optimiser: TrajectoryOptimiser,
        debug_summaries: bool = False,
        train_step_counter: Optional[tf.Variable] = None,
    ):
        """
        :param time_step_spec: A nest of tf.TypeSpec representing the time_steps.
        :param action_spec: A nest of BoundedTensorSpec representing the actions.
        :param transition_model: A component of the environment model that describes the
            transition dynamics. Either a tuple containing a trainable transition model together
            with training specs, or a pre-specified transition model.
        :param reward_model: A component of the environment model that describes the
            rewards. At the moment only pre-specified reward models are allowed, i.e. the agent
            assumes the reward function is known.
        :param initial_state_distribution_model: A component of the environment model that
            describes the initial state distribution (can be both deterministic or
            probabilistic). At the moment only pre-specified initial state distribution models
            are allowed, i.e. the agent assumes the initial state distribution is known.
        :param trajectory_optimiser: A TrajectoryOptimiser which takes an environment model and
            optimises a sequence of actions over a given horizon.
        :param debug_summaries: A bool; if true, subclasses should gather debug summaries.
        :param train_step_counter: An optional counter to increment every time the train op is run.
            Defaults to the global_step.
        """

        # setting up the environment model and policy
        if isinstance(transition_model, tuple):
            _transition_model, _ = transition_model
        else:
            _transition_model = transition_model  # type: ignore
        environment_model = EnvironmentModel(
            transition_model=_transition_model,
            reward_model=reward_model,
            termination_model=ConstantFalseTermination(
                _transition_model.observation_space_spec
            ),
            initial_state_distribution_model=initial_state_distribution_model,
        )
        planning_policy = PlanningPolicy(environment_model, trajectory_optimiser)

        super().__init__(
            time_step_spec,
            action_spec,
            transition_model,
            reward_model,
            environment_model.termination_model,
            initial_state_distribution_model,
            planning_policy,
            planning_policy,
            debug_summaries=debug_summaries,
            train_step_counter=train_step_counter,
        )


class ModelFreeSupportedDecisionTimePlanningAgent(DecisionTimePlanningAgent):
    """
    An agent that uses a state-conditioned policy for decision-time planning. The state-conditioned
    policy comes from an off-policy model-free agent that is trained concurrently on real data.
    """

    def __init__(
        self,
        time_step_spec: TimeStep,
        action_spec: NestedTensorSpec,
        transition_model: Union[
            Tuple[TrainableTransitionModel, TransitionModelTrainingSpec], TransitionModel
        ],
        reward_model: RewardModel,
        initial_state_distribution_model: InitialStateDistributionModel,
        trajectory_optimiser: TrajectoryOptimiser,
        model_free_agent: tf_agent.TFAgent,
        debug_summaries: bool = False,
        train_step_counter: Optional[tf.Variable] = None,
    ):
        """
        Initializes the agent.

        :param time_step_spec: A nest of tf.TypeSpec representing the time_steps.
        :param action_spec: A nest of BoundedTensorSpec representing the actions.
        :param transition_model: A component of the environment model that describes the
            transition dynamics. Either a tuple containing a trainable transition model together
            with training specs, or a pre-specified transition model.
        :param reward_model: A component of the environment model that describes the
            rewards. At the moment only pre-specified reward models are allowed, i.e. the agent
            assumes the reward function is known.
        :param initial_state_distribution_model: A component of the environment model that
            describes the initial state distribution (can be both deterministic or
            probabilistic). At the moment only pre-specified initial state distribution models
            are allowed, i.e. the agent assumes the initial state distribution is known.
        :param trajectory_optimiser: A TrajectoryOptimiser which takes an environment model and
            optimises a sequence of actions over a given horizon.
        :param model_free_agent: An off-policy model-free agent that is trained on real data but
            only virtually used for planning. Asserted to be explicitly DDPG, SAC or TD3.
        :param debug_summaries: A bool; if true, subclasses should gather debug summaries.
        :param train_step_counter: An optional counter to increment every time the train op is run.
            Defaults to the global_step.
        """
        assert time_step_spec == model_free_agent.time_step_spec
        assert action_spec == model_free_agent.action_spec
        assert model_free_agent.__class__ in {DdpgAgent, SacAgent, Td3Agent}

        super().__init__(
            time_step_spec,
            action_spec,
            transition_model,
            reward_model,
            initial_state_distribution_model,
            trajectory_optimiser,
            debug_summaries,
            train_step_counter,
        )

        self._trainable_components[ModelFreeAgentComponent.MODEL_FREE_AGENT.value.numpy()] = (
            model_free_agent,
            None,
        )
