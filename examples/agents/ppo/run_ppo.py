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

r"""
This module trains and evaluates PPO according to
Schulman et al. (2017) "Proximal Policy Optimization Algorithms",
which can be found at https://arxiv.org/pdf/1707.06347.pdf

To run:

```bash
tensorboard --logdir $HOME/tmp/ppo/CartPole-v1/ --port 2223

python run_ppo.py --root_dir=$HOME/tmp/ppo/CartPole-v1/
```
"""

import argparse

import tensorflow as tf
from absl import logging
from tf_agents.agents.ppo.ppo_clip_agent import PPOClipAgent
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork

from bellman.harness.harness import ExperimentHarness
from bellman.training.model_free_agent_trainer import OnPolicyModelFreeAgentTrainer


def train_eval(
    # tensorboard files
    root_dir,
    # environment
    env_name="CartPole-v1",
    random_seed=0,
    # Params for collect
    num_environment_steps=100000,
    replay_buffer_capacity=1001,  # Per-environment
    # Params for eval
    num_eval_episodes=30,
    eval_interval=200,
    # Params for summaries
    summary_interval=50,
):
    tf.compat.v1.set_random_seed(random_seed)

    environment = TFPyEnvironment(suite_gym.load(env_name))
    evaluation_environment = TFPyEnvironment(suite_gym.load(env_name))

    actor_net = ActorDistributionNetwork(
        environment.observation_spec(), environment.action_spec(), fc_layer_params=(200, 100)
    )
    value_net = ValueNetwork(environment.observation_spec(), fc_layer_params=(200, 100))
    global_step = tf.compat.v1.train.get_or_create_global_step()

    agent = PPOClipAgent(  # should be closer to the paper than PPOAgent...
        environment.time_step_spec(),
        environment.action_spec(),
        optimizer=tf.compat.v1.train.AdamOptimizer(),  # default None does not work
        actor_net=actor_net,
        value_net=value_net,
        importance_ratio_clipping=0.2,
        normalize_observations=False,
        normalize_rewards=False,
        use_gae=True,
        lambda_value=0.5,
        discount_factor=0.95,
        train_step_counter=global_step,
    )

    agent_trainer = OnPolicyModelFreeAgentTrainer(400)

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
        use_tf_function=True,
    )
    experiment_harness.run()


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", type=str, required=True, help="Set a root directory for saving reports"
    )
    args = parser.parse_args()

    train_eval(args.root_dir)
