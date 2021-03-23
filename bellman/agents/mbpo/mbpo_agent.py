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

""" A Model-Based Policy Optimization Agent.

Implements the Model-Based Policy Optimization algorithm from:

Janner et al. (2019) "When to Trust Your Model: Model-Based Policy Optimization",
which can be found at https://arxiv.org/pdf/1906.08253.pdf

"""

from typing import Callable, List, Optional

import gin
import tensorflow as tf
from tf_agents.agents.ddpg.actor_network import ActorNetwork
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents.ddpg.ddpg_agent import DdpgAgent
from tf_agents.agents.sac.sac_agent import SacAgent
from tf_agents.agents.td3.td3_agent import Td3Agent
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from bellman.agents.background_planning.background_planning_agent import (
    OffPolicyBackgroundPlanningAgent,
)
from bellman.agents.background_planning.model_free_agent_types import ModelFreeAgentType
from bellman.environments.initial_state_distribution_model import InitialStateDistributionModel
from bellman.environments.reward_model import RewardModel
from bellman.environments.transition_model.keras_model.factory_methods import (
    build_trajectory_sampler_from_type,
    build_transition_model_and_training_spec_from_type,
)
from bellman.environments.transition_model.keras_model.trajectory_sampler_types import (
    TrajectorySamplerType,
)
from bellman.environments.transition_model.keras_model.trajectory_sampling import (
    TrajectorySamplingStrategy,
)
from bellman.environments.transition_model.keras_model.transition_model_types import (
    TransitionModelType,
)


@gin.configurable
class MbpoAgent(OffPolicyBackgroundPlanningAgent):
    """
    An MBPO agent.

    This implementation follows the algorithm proposed in the original article, but importantly
    allowing for not only SAC but also DDPG and TD3 as model-free off-policy algorithms. It assumes
    a trainable transition model, while the reward model, termination model and initial state
    distribution are fixed. The transition model structure is pre-specified as a fully connected
    multi-layer neural network implemented in Keras - one can specify parameters such as the number
    of layers and hidden nodes in each layer. Several types of transition models are available
    (e.g. ensemble and non-ensemble versions), as well as several types of trajectory samplers like
    TS1, TSinf or TSmean. The actor and critic network structure of the model-free agent is assumed
    to be the same, but can be configured in terms of the number of hidden layers and nodes per
    hidden layer. If more flexibility is desired, one should use the
    `OffPolicyBackgroundPlanningAgent` class instead.
    """

    def __init__(
        self,
        time_step_spec: ts.TimeStep,
        action_spec: types.NestedTensorSpec,
        transition_model_type: TransitionModelType,
        num_hidden_layers_model: int,
        num_hidden_nodes_model: int,
        activation_function_model: Callable,
        ensemble_size: int,
        predict_state_difference: bool,
        epochs: int,
        training_batch_size: int,
        callbacks: List[tf.keras.callbacks.Callback],
        reward_model: RewardModel,
        initial_state_distribution_model: InitialStateDistributionModel,
        trajectory_sampler_type: TrajectorySamplerType,
        horizon: int,
        population_size: int,
        model_free_agent_type: ModelFreeAgentType,
        num_hidden_layers_agent: int,
        num_hidden_nodes_agent: int,
        activation_function_agent: Callable,
        model_free_training_iterations: int,
        virtual_sample_batch_size: int,
        debug_summaries: bool = False,
        train_step_counter: Optional[tf.Variable] = None,
    ):
        """
        Initializes the agent

        :param time_step_spec: A nest of tf.TypeSpec representing the time_steps.
        :param action_spec: A nest of BoundedTensorSpec representing the actions.
        :param transition_model_type: An indicator which of the available transition models
            should be used - list can be found in `TransitionModelType`. A component of the
            environment model that describes the transition dynamics.
        :param num_hidden_layers_model: A transition model parameter, used for constructing a neural
            network. A number of hidden layers in the neural network.
        :param num_hidden_nodes_model: A transition model parameter, used for constructing a neural
            network. A number of nodes in each hidden layer. Parameter is shared across all layers.
        :param activation_function_model: A transition model parameter, used for constructing a
            neural network. An activation function of the hidden nodes.
        :param ensemble_size: A transition model parameter, used for constructing a neural
            network. The number of networks in the ensemble.
        :param predict_state_difference: A transition model parameter, used for constructing a
            neural network. A boolean indicating whether transition model will be predicting a
            difference between current and a next state or the next state directly.
        :param epochs: A transition model parameter, used by Keras fit method. A number of epochs
            used for training the neural network.
        :param training_batch_size: A transition model parameter, used by Keras fit method. A
            batch size used for training the neural network.
        :param callbacks: A transition model parameter, used by Keras fit method. A list of Keras
            callbacks used for training the neural network.
        :param reward_model: A component of the environment model that describes the
            rewards. At the moment only pre-specified reward models are allowed, i.e. agent
            assumes reward function is known.
        :param initial_state_distribution_model: A component of the environment model that
            describes the initial state distribution (can be both deterministic or
            probabilistic). At the moment only pre-specified initial state distribution models
            are allowed, i.e. agent assumes initial state distribution is known.
        :param trajectory_sampler_type: An indicator which of the available trajectory samplers
            should be used - list can be found in `TrajectorySamplerType`. Trajectory sampler
            determines how predictions from an ensemble of neural networks that model the
            transition dynamics are sampled. Works only with ensemble type of transition models.
        :param horizon: A trajectory optimiser parameter. The number of steps taken in the
            environment in each virtual rollout.
        :param population_size: A trajectory optimiser parameter. The number of virtual rollouts
            that are simulated in each iteration during trajectory optimization.
        :param model_free_agent_type: Type of model-free agent, e.g. SAC, DDPG or TD3.
        :param num_hidden_layers_agent: A model-free agent parameter, used for constructing neural
            networks for actor and critic. A number of hidden layers in the neural network.
        :param num_hidden_nodes_agent: A model-free agent parameter, used for constructing neural
            networks for actor and critic. A number of nodes in each hidden layer. Parameter is
            shared across all layers.
        :param activation_function_agent: A model-free agent parameter, used for constructing a
            neural network. An activation function of the hidden nodes.
        :param model_free_training_iterations: Number of model-free training iterations per each
            train-call.
        :param virtual_sample_batch_size: sample batch size for virtually training the off-policy
            model-free agent.
        :param debug_summaries: A bool; if true, subclasses should gather debug summaries.
        :param train_step_counter: An optional counter to increment every time the train op is run.
            Defaults to the global_step.
        """

        assert ensemble_size > 0, "ensemble_size must be an integer > 0"
        assert num_hidden_layers_agent >= 0
        if num_hidden_layers_agent > 0:
            assert num_hidden_nodes_agent > 0

        self._ensemble_size = ensemble_size
        observation_spec = time_step_spec.observation

        # trajectory sampler (meaningful only for ensemble models)
        trajectory_sampler: Optional[TrajectorySamplingStrategy] = None
        if transition_model_type in [
            TransitionModelType.DeterministicEnsemble,
            TransitionModelType.ProbabilisticEnsemble,
        ]:
            trajectory_sampler = build_trajectory_sampler_from_type(
                ensemble_size=ensemble_size,
                trajectory_sampler_type=trajectory_sampler_type,
                batch_size=population_size,
            )

        # transition dynamics model plus training spec
        transition_model, training_spec = build_transition_model_and_training_spec_from_type(
            observation_spec=observation_spec,
            action_spec=action_spec,
            transition_model_type=transition_model_type,
            num_hidden_layers=num_hidden_layers_model,
            num_hidden_nodes=num_hidden_nodes_model,
            activation_function=activation_function_model,
            ensemble_size=ensemble_size,
            predict_state_difference=predict_state_difference,
            epochs=epochs,
            training_batch_size=training_batch_size,
            callbacks=callbacks,
            trajectory_sampler=trajectory_sampler,
        )

        # model-free agent
        value_net = CriticNetwork(
            input_tensor_spec=(observation_spec, action_spec),
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=[num_hidden_nodes_agent] * num_hidden_layers_agent,
            activation_fn=activation_function_agent,
        )
        if model_free_agent_type == ModelFreeAgentType.Sac:
            actor_net = ActorDistributionNetwork(
                input_tensor_spec=observation_spec,
                output_tensor_spec=action_spec,
                fc_layer_params=[num_hidden_nodes_agent] * num_hidden_layers_agent,
                activation_fn=activation_function_agent,
            )
            model_free_agent = SacAgent(
                time_step_spec=time_step_spec,
                action_spec=action_spec,
                critic_network=value_net,
                actor_network=actor_net,
                actor_optimizer=tf.compat.v1.train.AdamOptimizer(),
                critic_optimizer=tf.compat.v1.train.AdamOptimizer(),
                alpha_optimizer=tf.compat.v1.train.AdamOptimizer(),
            )
        else:
            actor_net = ActorNetwork(
                input_tensor_spec=observation_spec,
                output_tensor_spec=action_spec,
                fc_layer_params=[num_hidden_nodes_agent] * num_hidden_layers_agent,
                activation_fn=activation_function_agent,
            )
            if model_free_agent_type == ModelFreeAgentType.Ddpg:
                model_free_agent = DdpgAgent(
                    time_step_spec=time_step_spec,
                    action_spec=action_spec,
                    critic_network=value_net,
                    actor_network=actor_net,
                    actor_optimizer=tf.compat.v1.train.AdamOptimizer(),
                    critic_optimizer=tf.compat.v1.train.AdamOptimizer(),
                )
            elif model_free_agent_type == ModelFreeAgentType.Td3:
                model_free_agent = Td3Agent(
                    time_step_spec=time_step_spec,
                    action_spec=action_spec,
                    critic_network=value_net,
                    actor_network=actor_net,
                    actor_optimizer=tf.compat.v1.train.AdamOptimizer(),
                    critic_optimizer=tf.compat.v1.train.AdamOptimizer(),
                )
            else:
                raise RuntimeError("Unknown or unsupported agent type")

        super().__init__(
            (transition_model, training_spec),
            reward_model,
            initial_state_distribution_model,
            model_free_agent,
            population_size,
            horizon,
            model_free_training_iterations,
            virtual_sample_batch_size,
            debug_summaries,
            train_step_counter,
        )
