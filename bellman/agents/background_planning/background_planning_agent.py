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
A background planning agent which trains a model-free agent virtually with samples from an
approximate MDP in accordance with Sutton and Barto 2017. Note that the definition of background
planning is usually more general and refers to any algorithm that improves / trains a policy
based on model predictions and is not a decision-time planner. Here, we are explicitly referring to
the setting where any model-free agent can be trained virtually inside of a model.
"""

from typing import Optional, Tuple, Union

import tensorflow as tf
from tf_agents.agents import DdpgAgent, SacAgent, Td3Agent, TFAgent
from tf_agents.agents.ppo.ppo_clip_agent import PPOClipAgent
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.typing.types import NestedTensor, Tensor

from bellman.agents.components import ModelFreeAgentComponent
from bellman.agents.model_based_agent import ModelBasedAgent
from bellman.agents.trpo.trpo_agent import TRPOAgent
from bellman.environments.environment_model import EnvironmentModel
from bellman.environments.initial_state_distribution_model import InitialStateDistributionModel
from bellman.environments.reward_model import RewardModel
from bellman.environments.termination_model import ConstantFalseTermination
from bellman.environments.transition_model.transition_model import (
    TrainableTransitionModel,
    TransitionModel,
    TransitionModelTrainingSpec,
)
from bellman.environments.utils import virtual_rollouts_buffer_and_driver
from bellman.training.utils import TRAIN_ARGSPEC_COMPONENT_ID


class BackgroundPlanningAgent(ModelBasedAgent):
    """
    An agent for background planning. The agent is assumed to have an internal environment model
    and an internal model-free agent that can both be trained through a single train method.
    """

    def __init__(
        self,
        transition_model: Union[
            Tuple[TrainableTransitionModel, TransitionModelTrainingSpec], TransitionModel
        ],
        reward_model: RewardModel,
        initial_state_distribution_model: InitialStateDistributionModel,
        model_free_agent: TFAgent,
        planner_batch_size: int,
        planning_horizon: int,
        model_free_training_iterations: int,
        debug_summaries=False,
        train_step_counter=None,
    ):
        """
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
        :param model_free_agent: model-free agent that is trained virtually with samples from the
            model. Could be for example PPO or TRPO which are on-policy, as well as DDPG, SAC or
            TD3 which are off-policy.
        :param planner_batch_size: Number of parallel virtual trajectories.
        :param planning_horizon: Number of steps taken in the environment in each virtual rollout.
        :param model_free_training_iterations: Number of model-free training iterations per each
            train-call.
        :param debug_summaries: A bool; if true, subclasses should gather debug summaries.
        :param train_step_counter: An optional counter to increment every time the train op is run.
            Defaults to the global_step.
        """
        assert planner_batch_size > 0, "Planner batch size must be positive."
        assert planning_horizon > 0, "Planning horizon must be positive."
        assert (
            model_free_training_iterations > 0
        ), "Model-free train iterations must be positive."

        self._assert_model_free_agent(model_free_agent)

        # for setting up the environment model and policy
        if isinstance(transition_model, tuple):
            _transition_model, _ = transition_model
        else:
            _transition_model = transition_model  # type: ignore

        assert (
            _transition_model.observation_space_spec
            == model_free_agent.time_step_spec.observation
        ), "Transition model observation spec needs to match the model-free agent's one."
        assert (
            _transition_model.action_space_spec == model_free_agent.action_spec
        ), "Transition model action spec needs to match the model-free agent's one."

        super().__init__(
            model_free_agent.time_step_spec,
            model_free_agent.action_spec,
            transition_model,
            reward_model,
            ConstantFalseTermination(_transition_model.observation_space_spec),
            initial_state_distribution_model,
            model_free_agent.policy,
            model_free_agent.collect_policy,
            debug_summaries=debug_summaries,
            train_step_counter=train_step_counter,
        )

        environment_model = EnvironmentModel(
            transition_model=_transition_model,
            reward_model=reward_model,
            termination_model=self.termination_model,
            initial_state_distribution_model=initial_state_distribution_model,
            batch_size=planner_batch_size,
        )

        (
            self._virtual_rollouts_replay_buffer,
            self._virtual_rollouts_driver,
            self._environment_model,
        ) = virtual_rollouts_buffer_and_driver(
            environment_model, model_free_agent.collect_policy, planning_horizon
        )

        self._model_free_agent = model_free_agent
        self._model_free_training_iterations = model_free_training_iterations

    def _assert_model_free_agent(self, model_free_agent: TFAgent):
        """
        Assert that the model-free agent is of the correct type.
        :param model_free_agent: model-free agent that is trained virtually with samples from the
            model. Could be for example PPO or TRPO which are on-policy, as well as DDPG, SAC or
            TD3 which are off-policy.
        """
        pass

    def _train_model_free_agent(self, experience: NestedTensor) -> LossInfo:
        """
        Train the model-free agent virtually for multiple iterations.
        :param experience: A batch of experience data in the form of a `Trajectory`.
            All tensors in `experience` must be shaped `[batch, time, ...]`. Importantly,
            this is real-world experience and the agent needs to decide how to leverage this
            real-world experience for virtual training using the environment model.
        :return: A `LossInfo` tuples containing loss and info tensors of a trained model.
        """
        pass

    def _train(  # pylint: disable=arguments-differ
        self, experience: NestedTensor, weights: Optional[Tensor] = None, **kwargs
    ) -> LossInfo:
        """
        Train one or more of the models composing the environment model.

        :param experience: A batch of experience data in the form of a `Trajectory`.
            All tensors in `experience` must be shaped `[batch, time, ...]`.
        :param weights: Optional scalar or element-wise (per-batch-entry) importance
            weights. Not used at the moment.
        :param kwargs: A dictionary that contains a key with a string tensor value indicating
            which model should be trained.

        :return: A `LossInfo` tuples containing loss and info tensors of a trained model.
        """
        trainable_component_name = kwargs[TRAIN_ARGSPEC_COMPONENT_ID].numpy()

        if trainable_component_name == ModelFreeAgentComponent.MODEL_FREE_AGENT.value.numpy():
            self._train_step_counter.assign_add(1)
            return self._train_model_free_agent(experience)
        else:
            return super()._train(experience, weights, **kwargs)


class OnPolicyBackgroundPlanningAgent(BackgroundPlanningAgent):
    """
    An agent for background planning with an internal on-policy model-free agent.
    """

    def _assert_model_free_agent(self, model_free_agent: TFAgent):
        """
        Assert that the model-free agent is of the correct type.
        :param model_free_agent: model-free agent that is trained virtually with samples from the
            model. For now PPO and TRPO are supported.
        """
        assert isinstance(model_free_agent, (PPOClipAgent, TRPOAgent))

    def _train_model_free_agent(self, experience: NestedTensor) -> LossInfo:
        """
        Train the model-free agent virtually for multiple iterations.
        :param experience: A batch of experience data in the form of a `Trajectory`.
            All tensors in `experience` must be shaped `[batch, time, ...]`. Importantly,
            this is real-world experience and the agent needs to decide how to leverage this
            real-world experience for virtual training using the environment model.
        :return: A `LossInfo` tuples containing loss and info tensors of a trained model.
        """
        assert tf.keras.backend.ndim(experience.observation) >= 3
        assert experience.observation.shape[0] == 1, "The real environment has batch size 1."

        mask = ~experience.is_boundary()  # [batch, time, ...]
        masked_observation = tf.boolean_mask(
            experience.observation, mask
        )  # [reduced batch, ...]

        model_free_losses = []
        for _ in range(self._model_free_training_iterations):
            random_indexes = tf.random.uniform(
                shape=(self._environment_model.batch_size,),
                maxval=masked_observation.shape[0],
                dtype=tf.int32,
            )
            initial_observation = tf.gather(
                masked_observation, random_indexes
            )  # [env model batch, ...]

            initial_time_step = self._environment_model.set_initial_observation(
                initial_observation
            )
            self._virtual_rollouts_driver.run(initial_time_step)

            policy_experience = self._virtual_rollouts_replay_buffer.gather_all()
            model_free_losses.append(self._model_free_agent.train(policy_experience))

            self._virtual_rollouts_replay_buffer.clear()

        loss_info = LossInfo(loss=model_free_losses[0].loss, extra=model_free_losses)
        return loss_info


class OffPolicyBackgroundPlanningAgent(BackgroundPlanningAgent):
    """
    An agent for off-policy background planning with an internal off-policy model-free agent.
    """

    def __init__(
        self,
        transition_model: Union[
            Tuple[TrainableTransitionModel, TransitionModelTrainingSpec], TransitionModel
        ],
        reward_model: RewardModel,
        initial_state_distribution_model: InitialStateDistributionModel,
        model_free_agent: TFAgent,
        planner_batch_size: int,
        planning_horizon: int,
        model_free_training_iterations: int,
        virtual_sample_batch_size: int,
        debug_summaries=False,
        train_step_counter=None,
    ):
        """
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
        :param model_free_agent: off-policy model-free agent that is trained virtually with samples
            from the model. Could be for example DDPG, TD3 or SAC.
        :param planner_batch_size: Number of parallel virtual trajectories.
        :param planning_horizon: Number of steps taken in the environment in each virtual rollout.
        :param model_free_training_iterations: Number of model-free training iterations per each
            train-call.
        :param virtual_sample_batch_size: sample batch size for virtually training the off-policy
            model-free agent.
        :param debug_summaries: A bool; if true, subclasses should gather debug summaries.
        :param train_step_counter: An optional counter to increment every time the train op is run.
            Defaults to the global_step.
        """
        assert virtual_sample_batch_size > 0

        super().__init__(
            transition_model,
            reward_model,
            initial_state_distribution_model,
            model_free_agent,
            planner_batch_size,
            planning_horizon,
            model_free_training_iterations,
            debug_summaries=debug_summaries,
            train_step_counter=train_step_counter,
        )

        def _not_boundary(trajectories, _):
            condition_1 = ~trajectories.is_boundary()[0]
            condition_2 = ~(
                tf.logical_and(~trajectories.is_last()[0], trajectories.is_first()[1])
            )  # to be on the safe side...
            return tf.logical_and(condition_1, condition_2)

        dataset = (
            self._virtual_rollouts_replay_buffer.as_dataset(num_steps=2)
            .filter(_not_boundary)
            .batch(virtual_sample_batch_size)
        )
        self._iterator = iter(dataset)

    def _assert_model_free_agent(self, model_free_agent: TFAgent):
        """
        Assert that the model-free agent is of the correct type.
        :param model_free_agent: model-free agent that is trained virtually with samples from the
            model. For now, DDPG, SAC and TD3 are supported.
        """
        assert isinstance(model_free_agent, (DdpgAgent, SacAgent, Td3Agent))

    def _train_model_free_agent(self, experience: NestedTensor) -> LossInfo:
        """
        Train the model-free agent virtually for multiple iterations.
        :param experience: A batch of experience data in the form of a `Trajectory`.
            All tensors in `experience` must be shaped `[batch, time, ...]`. Importantly,
            this is real-world experience and the agent needs to decide how to leverage this
            real-world experience for virtual training using the environment model.
        :return: A `LossInfo` tuples containing loss and info tensors of a trained model.
        """
        assert tf.keras.backend.ndim(experience.observation) >= 3
        assert experience.observation.shape[0] == 1, "The real environment has batch size 1."

        mask = ~experience.is_boundary()  # [batch, time, ...]
        masked_observation = tf.boolean_mask(
            experience.observation, mask
        )  # [reduced batch, ...]

        ridx = tf.random.uniform(
            shape=(self._environment_model.batch_size,),
            maxval=masked_observation.shape[0],
            dtype=tf.int32,
        )
        initial_observation = tf.gather(masked_observation, ridx)  # [env batch, ...]

        initial_time_step = self._environment_model.set_initial_observation(
            initial_observation
        )
        self._virtual_rollouts_driver.run(initial_time_step)

        model_free_losses = []
        for _ in range(self._model_free_training_iterations):
            policy_experience, _ = next(self._iterator)
            model_free_losses.append(self._model_free_agent.train(policy_experience))

        loss_info = LossInfo(loss=model_free_losses[0].loss, extra=model_free_losses)
        return loss_info
