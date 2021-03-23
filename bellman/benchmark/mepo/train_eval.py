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
This module provides a function to train and evaluate an MEPO agent.
"""

import gin
import tensorflow as tf

from bellman.agents.mepo.mepo_agent import MepoAgent
from bellman.environments.utils import create_real_tf_environment
from bellman.harness.harness import ExperimentHarness
from bellman.training.background_planning_agent_trainer import BackgroundPlanningAgentTrainer


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
    num_hidden_layers_model,
    num_hidden_nodes_model,
    activation_function_model,
    ensemble_size,
    predict_state_difference,
    epochs,
    training_batch_size,
    trajectory_sampler_type,
    horizon,
    population_size,
    model_free_agent_type,
    num_hidden_layers_agent,
    num_hidden_nodes_agent,
    activation_function_agent,
    model_free_training_iterations,
    debug_summaries,
    # agent trainer
    steps_per_transition_model_update,
    steps_per_model_free_agent_update,
    # agent specific harness parameters
    replay_buffer_capacity,
    number_of_initial_random_policy_steps,
    use_tf_function,
):
    """
    This function will train and evaluate an MEPO agent.

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
    :param trajectory_sampler_type: An indicator which of the available trajectory samplers
            should be used - list can be found in `TrajectorySamplerType`. Trajectory sampler
            determines how predictions from an ensemble of neural networks that model the
            transition dynamics are sampled. Works only with ensemble type of transition models.
    :param horizon: A trajectory optimiser parameter. The number of steps taken in the
            environment in each virtual rollout.
    :param population_size: A trajectory optimiser parameter. The number of virtual rollouts
            that are simulated in each iteration during trajectory optimization.
    :param model_free_agent_type: Type of model-free agent, e.g. PPO or TRPO.
    :param num_hidden_layers_agent: A model-free agent parameter, used for constructing neural
            networks for actor and critic. A number of hidden layers in the neural network.
    :param num_hidden_nodes_agent: A model-free agent parameter, used for constructing neural
            networks for actor and critic. A number of nodes in each hidden layer. Parameter is
            shared across all layers.
    :param activation_function_agent: A model-free agent parameter, used for constructing a
            neural network. An activation function of the hidden nodes.
    :param model_free_training_iterations: Number of model-free training iterations per each
            train-call.
    :param debug_summaries: A bool; if true, subclasses should gather debug summaries.
    :param steps_per_transition_model_update: steps between transition model updates.
    :param steps_per_model_free_agent_update: steps between model-free agent updates.
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

    agent = MepoAgent(
        environment.time_step_spec(),
        environment.action_spec(),
        transition_model_type,
        num_hidden_layers_model,
        num_hidden_nodes_model,
        activation_function_model,
        ensemble_size,
        predict_state_difference,
        epochs,
        training_batch_size,
        callbacks,
        reward_model,
        initial_state_distribution_model,
        trajectory_sampler_type,
        horizon,
        population_size,
        model_free_agent_type,
        num_hidden_layers_agent,
        num_hidden_nodes_agent,
        activation_function_agent,
        model_free_training_iterations,
        debug_summaries=debug_summaries,
        train_step_counter=global_step,
    )

    agent_trainer = BackgroundPlanningAgentTrainer(
        steps_per_transition_model_update, steps_per_model_free_agent_update
    )

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
