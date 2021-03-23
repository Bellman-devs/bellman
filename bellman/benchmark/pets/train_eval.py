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
This module provides a function to train and evaluate a PETS agent.
"""

import gin
import tensorflow as tf

from bellman.agents.pets.pets_agent import PetsAgent
from bellman.environments.utils import create_real_tf_environment
from bellman.harness.harness import ExperimentHarness
from bellman.training.decision_time_planning_agent_trainer import (
    DecisionTimePlanningAgentTrainer,
)


@gin.configurable
def train_eval(
    # harness
    # tensorboard files
    root_dir,
    # Params for collect
    num_environment_steps,
    # Params for eval
    num_eval_episodes,
    eval_interval,
    # Params for summaries
    summary_interval,
    # environment
    env_name,
    gym_random_seed,
    reward_model_class,
    initial_state_distribution_model_class,
    # agent
    random_seed,
    transition_model_type,
    num_hidden_layers,
    num_hidden_nodes,
    activation_function,
    ensemble_size,
    predict_state_difference,
    epochs,
    training_batch_size,
    trajectory_sampler_type,
    trajectory_optimization_type,
    horizon,
    population_size,
    number_of_particles,
    num_elites,
    learning_rate,
    max_iterations,
    # agent trainer
    steps_per_transition_model_update,
    # agent specific harness parameters
    replay_buffer_capacity,
    number_of_initial_random_policy_steps,
    use_tf_function,
):
    """
    This function will train and evaluate a PETS agent.

    :param root_dir: Root directory where all experiments are stored.
    :param num_environment_steps: The number of environment steps to run the
            experiment for.
    :param num_eval_episodes: Number of episodes at each evaluation point.
    :param eval_interval: Interval for evaluation points.
    :param summary_interval: Interval for summaries.
    :param env_name: Name for the environment to load.
    :param gym_random_seed: Value to use as seed for the environment.
    :param reward_model_class: A component of the environment model that describes the
            rewards. At the moment only pre-specified reward models are allowed, i.e. agent
            assumes reward function is known.
    :param initial_state_distribution_model_class: A component of the environment model that
            describes the initial state distribution (can be both deterministic or
            probabilistic). At the moment only pre-specified initial state distribution models
            are allowed, i.e. agent assumes initial state distribution is known.
    :param random_seed: A component of the environment model that describes the
            rewards. At the moment only pre-specified reward models are allowed, i.e. agent
            assumes reward function is known.
    :param transition_model_type: An indicator which of the available transition models
            should be used - list can be found in `TransitionModelType`. A component of the
            environment model that describes the transition dynamics.
    :param num_hidden_layers: A transition model parameter, used for constructing a neural
            network. A number of hidden layers in the neural network.
    :param num_hidden_nodes: A transition model parameter, used for constructing a neural
            network. A number of nodes in each hidden layer. Parameter is shared across all layers.
    :param activation_function: A transition model parameter, used for constructing a
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
    :param number_of_particles: A trajectory optimiser parameter. The number of monte-carlo
            rollouts of each action trajectory.
    :param num_elites: A trajectory optimiser parameter, required only for the Cross entropy
            method. The number of elite trajectories used for updating the parameters of
            distribution for each action. This should be a proportion of `population_size`
            rollouts.
    :param learning_rate: A trajectory optimiser parameter, required only for the Cross
            entropy method. The learning rate for updating the distribution parameters.
    :param max_iterations: A trajectory optimiser parameter, required only for the Cross
            entropy method. The maximum number of iterations to use for optimisation.
    :param steps_per_transition_model_update: steps between transition model updates.
    :param replay_buffer_capacity: Capacity of the buffer collecting real samples.
    :param number_of_initial_random_policy_steps: If > 0, some initial training data is
            gathered by running a random policy on the real environment.
    :param use_tf_function: If `True`, use a `tf.function` for data collection.
    """
    tf.compat.v1.set_random_seed(random_seed)

    environment = create_real_tf_environment(env_name, gym_random_seed)
    evaluation_environment = create_real_tf_environment(env_name, gym_random_seed)

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)]
    reward_model = reward_model_class(
        environment.observation_spec(), environment.action_spec()
    )
    initial_state_distribution_model = initial_state_distribution_model_class(
        environment.observation_spec()
    )
    global_step = tf.compat.v1.train.get_or_create_global_step()

    agent = PetsAgent(
        environment.time_step_spec(),
        environment.action_spec(),
        transition_model_type,
        num_hidden_layers,
        num_hidden_nodes,
        activation_function,
        ensemble_size,
        predict_state_difference,
        epochs,
        training_batch_size,
        callbacks,
        reward_model,
        initial_state_distribution_model,
        trajectory_sampler_type.TSinf,
        trajectory_optimization_type,
        horizon,
        population_size,
        number_of_particles,
        num_elites,
        learning_rate,
        max_iterations,
        train_step_counter=global_step,
    )

    agent_trainer = DecisionTimePlanningAgentTrainer(steps_per_transition_model_update)

    experiment_harness = ExperimentHarness(
        root_dir,
        environment,
        evaluation_environment,
        agent,
        agent_trainer,
        replay_buffer_capacity,
        num_environment_steps,
        summary_interval,
        eval_interval,
        num_eval_episodes,
        number_of_initial_random_policy_steps,
        use_tf_function,
    )
    experiment_harness.run()
