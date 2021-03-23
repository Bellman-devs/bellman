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
This module provides a function to train and evaluate a TRPO agent.
"""

import gin
import tensorflow as tf
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork

from bellman.agents.trpo.trpo_agent import TRPOAgent
from bellman.environments.utils import create_real_tf_environment
from bellman.harness.harness import ExperimentHarness
from bellman.training.model_free_agent_trainer import OnPolicyModelFreeAgentTrainer


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
    # agent
    random_seed,
    num_hidden_layers_agent,
    num_hidden_nodes_agent,
    discount_factor,
    lambda_value,
    max_kl,
    backtrack_coefficient,
    backtrack_iters,
    cg_iters,
    reward_normalizer,
    reward_norm_clipping,
    log_prob_clipping,
    value_train_iters,
    value_optimizer,
    gradient_clipping,
    debug,
    # agent trainer
    steps_per_policy_update,
    # agent specific harness parameters
    replay_buffer_capacity,
    use_tf_function,
):
    """
    This function will train and evaluate a TRPO agent.

    :param root_dir: Root directory where all experiments are stored.
    :param num_environment_steps: The number of environment steps to run the
            experiment for.
    :param num_eval_episodes: Number of episodes at each evaluation point.
    :param eval_interval: Interval for evaluation points.
    :param summary_interval: Interval for summaries.
    :param env_name: Name for the environment to load.
    :param gym_random_seed: Value to use as seed for the environment.
    :param random_seed: A component of the environment model that describes the
            rewards. At the moment only pre-specified reward models are allowed, i.e. agent
            assumes reward function is known.
    :param num_hidden_layers_agent: A model-free agent parameter, used for constructing neural
            networks for actor and critic. A number of hidden layers in the neural network.
    :param num_hidden_nodes_agent: A model-free agent parameter, used for constructing neural
            networks for actor and critic. A number of nodes in each hidden layer. Parameter is
            shared across all layers.
    :param discount_factor: discount factor in [0, 1]
    :param lambda_value: trace decay used by the GAE critic in [0, 1]
    :param max_kl: maximum KL distance between updated and old policy
    :param backtrack_coefficient: coefficient used in step size search
    :param backtrack_iters: number of iterations to performa in line search
    :param cg_iters: number of conjugate gradient iterations to approximate natural gradient
    :param reward_normalizer: TensorNormalizer applied to rewards
    :param reward_norm_clipping: value to clip rewards
    :param log_prob_clipping: clip value for log probs in policy gradient , None for no clipping
    :param value_train_iters: number of gradient steps to perform on value estimator
            for every policy update
    :param value_optimizer: optimizer used to train value_function (default: Adam)
    :param gradient_clipping: clip born value gradient (None for no clipping)
    :param debug: debug flag to check computations for Nans
    :param steps_per_policy_update: steps between policy updates
    :param replay_buffer_capacity: Capacity of the buffer collecting real samples.
    :param use_tf_function: If `True`, use a `tf.function` for data collection.
    """
    tf.compat.v1.set_random_seed(random_seed)

    environment = create_real_tf_environment(env_name, gym_random_seed)
    evaluation_environment = create_real_tf_environment(env_name, gym_random_seed)

    network_architecture = (num_hidden_nodes_agent,) * num_hidden_layers_agent
    actor_net = ActorDistributionNetwork(
        environment.observation_spec(),
        environment.action_spec(),
        fc_layer_params=network_architecture,
    )
    value_net = ValueNetwork(
        environment.observation_spec(), fc_layer_params=network_architecture
    )
    global_step = tf.compat.v1.train.get_or_create_global_step()

    agent = TRPOAgent(
        environment.time_step_spec(),
        environment.action_spec(),
        actor_net,
        value_net,
        discount_factor,
        lambda_value,
        max_kl,
        backtrack_coefficient,
        backtrack_iters,
        cg_iters,
        reward_normalizer,
        reward_norm_clipping,
        log_prob_clipping,
        value_train_iters,
        value_optimizer,
        gradient_clipping,
        debug,
        train_step_counter=global_step,
    )

    agent_trainer = OnPolicyModelFreeAgentTrainer(steps_per_policy_update)

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
        number_of_initial_random_policy_steps=0,
        use_tf_function=use_tf_function,
    )
    experiment_harness.run()
