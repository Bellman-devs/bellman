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
A model-based agent superclass from which decision-time planners and background planners inherit.
"""

from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union
from warnings import warn

import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.specs import TensorSpec
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.typing import types
from tf_agents.typing.types import NestedTensor, NestedTensorSpec, Tensor

from bellman.agents.components import EnvironmentModelComponents
from bellman.environments.initial_state_distribution_model import InitialStateDistributionModel
from bellman.environments.reward_model import RewardModel
from bellman.environments.termination_model import ConstantFalseTermination, TerminationModel
from bellman.environments.transition_model.transition_model import (
    TrainableTransitionModel,
    TransitionModel,
    TransitionModelTrainingSpec,
)
from bellman.training.utils import TRAIN_ARGSPEC_COMPONENT_ID


class ModelBasedAgent(tf_agent.TFAgent):
    """
    An abstract class for model-based agents. This agent is assumed to have access to models for
    all components of the model of the environment (i.e. MDP) - transition dynamics, rewards,
    termination state and initial state distribution. It is also assumed to have a policy and a
    collect policy that are specified by subclasses before calling init of the model-based agent.
    """

    def __init__(
        self,
        time_step_spec: TimeStep,
        action_spec: NestedTensorSpec,
        transition_model: Union[
            Tuple[TrainableTransitionModel, TransitionModelTrainingSpec], TransitionModel
        ],
        reward_model: RewardModel,
        termination_model: TerminationModel,
        initial_state_distribution_model: InitialStateDistributionModel,
        policy: TFPolicy,
        collect_policy: TFPolicy,
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
        :param termination_model: A component of the environment model that describes the
            termination of the episode. At the moment only pre-specified termination models are
            allowed, i.e. the agent assumes the termination function is known.
        :param initial_state_distribution_model: A component of the environment model that
            describes the initial state distribution (can be both deterministic or
            probabilistic). At the moment only pre-specified initial state distribution models
            are allowed, i.e. the agent assumes the initial state distribution is known.
        :param policy: An instance of `tf_policy.TFPolicy` representing the agent's current policy.
        :param collect_policy: An instance of `tf_policy.TFPolicy` representing the agent's current
            data collection policy (used to set `self.step_spec`).
        :param debug_summaries: A bool; if true, subclasses should gather debug summaries.
        :param train_step_counter: An optional counter to increment every time the train op is run.
            Defaults to the global_step.
        """

        assert isinstance(
            termination_model, ConstantFalseTermination
        ), "Only constant false termination supported"

        # unpack and create a dictionary with trainable models
        self._trainable_components: Dict[Enum, Any] = dict()
        if isinstance(transition_model, tuple):
            self._transition_model, self._transition_model_spec = transition_model
            self._trainable_components[
                EnvironmentModelComponents.TRANSITION.value.numpy()
            ] = transition_model
        else:
            self._transition_model = transition_model  # type: ignore
        self._reward_model = reward_model
        self._termination_model = termination_model
        self._initial_state_distribution_model = initial_state_distribution_model
        if not self._trainable_components:
            warn("No trainable model specified!", RuntimeWarning)

        # additional input for the _train method
        train_argspec = {TRAIN_ARGSPEC_COMPONENT_ID: TensorSpec(shape=(), dtype=tf.string)}

        super().__init__(
            time_step_spec,
            action_spec,
            policy,
            collect_policy,
            train_sequence_length=None,
            train_argspec=train_argspec,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=False,
            train_step_counter=train_step_counter,
            validate_args=True,
        )

    @property
    def transition_model(self):
        """
        return the transition model.
        """
        return self._transition_model

    @property
    def reward_model(self):
        """
        return the reward model.
        """
        return self._reward_model

    @property
    def termination_model(self):
        """
        return the termination model.
        """
        return self._termination_model

    @property
    def initial_state_distribution_model(self):
        """
        return the initial state distribution model.
        """
        return self._initial_state_distribution_model

    def _loss(
        self, experience: types.NestedTensor, weights: types.Tensor
    ) -> Optional[LossInfo]:
        raise ValueError("A single loss is not well defined.")

    def _train(  # pylint: disable=arguments-differ
        self, experience: NestedTensor, weights: Optional[Tensor] = None, **kwargs
    ) -> LossInfo:
        """
        Train one or more of the models composing the environment model. Models need to be of a
        trainable type.

        :param experience: A batch of experience data in the form of a `Trajectory`.
            All tensors in `experience` must be shaped `[batch, time, ...]`.
        :param weights: Optional scalar or element-wise (per-batch-entry) importance
            weights. Not used at the moment.
        :param kwargs: A dictionary that contains a key with a string tensor value indicating
            which model should be trained.

        :return: A `LossInfo` tuples containing loss and info tensors of a trained model.
        """
        # TODO: TFAgent class has an error in _train method, missing kwargs, probably it will be
        # fixed in due time, until then we disable linting in def
        trainable_component_name = kwargs[TRAIN_ARGSPEC_COMPONENT_ID].numpy()

        self._train_step_counter.assign_add(1)

        if trainable_component_name not in self._trainable_components:
            warn(
                f"Trainable component {trainable_component_name} name not in trainable components"
                f" {self._trainable_components}, no train!",
                RuntimeWarning,
            )
            return LossInfo(None, None)

        model, model_training_spec = self._trainable_components[trainable_component_name]
        if model_training_spec is not None:
            history = model.train(experience, model_training_spec)
            return LossInfo(history.history["loss"], None)
        else:
            return model.train(experience)
