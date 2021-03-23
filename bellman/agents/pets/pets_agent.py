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

""" A Probabilistic Ensembles with Trajectory Sampling Agent.

Implements the Probabilistic Ensembles with Trajectory Sampling (PETS) algorithm from:

Chua et al. (2018) "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics
Models", which can be found at https://arxiv.org/abs/1805.12114

"""

from typing import Callable, List, Optional, cast

import gin
import tensorflow as tf
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from bellman.agents.decision_time_planning.decision_time_planning_agent import (
    DecisionTimePlanningAgent,
)
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
from bellman.trajectory_optimisers.cross_entropy_method import (
    cross_entropy_method_trajectory_optimisation,
)
from bellman.trajectory_optimisers.random_shooting import (
    random_shooting_trajectory_optimisation,
)
from bellman.trajectory_optimisers.trajectory_optimization_types import (
    TrajectoryOptimizationType,
)


@gin.configurable
class PetsAgent(DecisionTimePlanningAgent):
    """
    A PETS agent.

    This implementation strictly follows the algorithm proposed in the original article. It
    assumes a trainable transition model, while the reward model, termination model and initial
    state distribution are fixed. The transition model structure is pre-specified as a fully
    connected multi-layer neural network implemented in Keras - one can specify parameters such
    as the number of layers and hidden nodes in each layer. Same as in the original article,
    several types of transition models are available (e.g. ensemble and non-ensemble versions), as
    well as several types of trajectory optimizers and samplers. Regarding the sampler types, note
    that we did not implement moment matching and distribution sampling. If more flexibility
    is desired, one should use the `DecisionTimePlanningAgent` class instead.
    """

    def __init__(
        self,
        time_step_spec: ts.TimeStep,
        action_spec: types.NestedTensorSpec,
        transition_model_type: TransitionModelType,
        num_hidden_layers: int,
        num_hidden_nodes: int,
        activation_function: Callable,
        ensemble_size: int,
        predict_state_difference: bool,
        epochs: int,
        training_batch_size: int,
        callbacks: List[tf.keras.callbacks.Callback],
        reward_model: RewardModel,
        initial_state_distribution_model: InitialStateDistributionModel,
        trajectory_sampler_type: TrajectorySamplerType,
        trajectory_optimization_type: TrajectoryOptimizationType,
        horizon: int,
        population_size: int,
        number_of_particles: int,
        num_elites: Optional[int] = None,
        learning_rate: Optional[float] = None,
        max_iterations: Optional[int] = None,
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
        :param num_hidden_layers: A transition model parameter, used for constructing a neural
            network. A number of hidden layers in the neural network.
        :param num_hidden_nodes: A transition model parameter, used for constructing a neural
            network. A number of nodes in each hidden layer. Parameter is shared across all layers.
        :param activation_function: A transition model parameter, used for constructing a neural
            network. An activation function of the hidden nodes.
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
        :param trajectory_optimization_type: An indicator which of the available trajectory
            optimisers should be used - list can be found in `TrajectoryOptimizationType`.
            Trajectory optimiser optimises a sequence of actions over a given horizon and an
            environment model.
        :param horizon: A trajectory optimiser parameter. The number of steps taken in the
            environment in each virtual rollout.
        :param population_size: A trajectory optimiser parameter. The number of virtual rollouts
            that are simulated in each iteration during trajectory optimization.
        :param number_of_particles: Number of monte-carlo rollouts of each action trajectory.
        :param num_elites: A trajectory optimiser parameter, required only for the Cross entropy
            method. The number of elite trajectories used for updating the parameters of
            distribution for each action. This should be a proportion of `population_size`
            rollouts.
        :param learning_rate: A trajectory optimiser parameter, required only for the Cross
            entropy method. The learning rate for updating the distribution parameters.
        :param max_iterations: A trajectory optimiser parameter, required only for the Cross
            entropy method. The maximum number of iterations to use for optimisation.
        :param debug_summaries: A bool; if true, subclasses should gather debug summaries.
        :param train_step_counter: An optional counter to increment every time the train op is run.
            Defaults to the global_step.
        """

        assert ensemble_size > 0, "ensemble_size must be an integer > 0"

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
            num_hidden_layers=num_hidden_layers,
            num_hidden_nodes=num_hidden_nodes,
            activation_function=activation_function,
            ensemble_size=ensemble_size,
            predict_state_difference=predict_state_difference,
            epochs=epochs,
            training_batch_size=training_batch_size,
            callbacks=callbacks,
            trajectory_sampler=trajectory_sampler,
        )

        # set the trajectory optimizer
        if trajectory_optimization_type == TrajectoryOptimizationType.CrossEntropyMethod:
            num_elites = cast(int, num_elites)
            learning_rate = cast(float, learning_rate)
            max_iterations = cast(int, max_iterations)
            trajectory_optimiser = cross_entropy_method_trajectory_optimisation(
                time_step_spec,
                action_spec,
                horizon,
                population_size,
                number_of_particles,
                num_elites,
                learning_rate,
                max_iterations,
            )
        elif trajectory_optimization_type == TrajectoryOptimizationType.RandomShooting:
            trajectory_optimiser = random_shooting_trajectory_optimisation(
                time_step_spec,
                action_spec,
                horizon,
                population_size,
                number_of_particles,
            )
        else:
            raise RuntimeError("Unknown trajectory optimiser")

        super().__init__(
            time_step_spec,
            action_spec,
            (transition_model, training_spec),
            reward_model,
            initial_state_distribution_model,
            trajectory_optimiser,
            debug_summaries,
            train_step_counter,
        )
