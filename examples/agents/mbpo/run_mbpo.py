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
This module trains and evaluates MBPO according to
Janner et al. (2019) "When to Trust Your Model: Model-Based Policy Optimization",
which can be found at https://arxiv.org/abs/1906.08253

To run:

```bash
tensorboard --logdir $HOME/tmp/mbpo/MountainCarContinuous-v0/ --port 2223 &

python run_mbpo.py --root_dir=$HOME/tmp/mbpo/MountainCarContinuous-v0/
```
"""
import argparse

import tensorflow as tf
from absl import logging
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment

from bellman.agents.background_planning.model_free_agent_types import ModelFreeAgentType
from bellman.agents.mbpo.mbpo_agent import MbpoAgent
from bellman.environments.transition_model.keras_model.trajectory_sampler_types import (
    TrajectorySamplerType,
)
from bellman.environments.transition_model.keras_model.transition_model_types import (
    TransitionModelType,
)
from bellman.harness.harness import ExperimentHarness
from bellman.training.background_planning_agent_trainer import BackgroundPlanningAgentTrainer
from examples.utils.classic_control import MountainCarInitialState, MountainCarReward


def train_eval(
    # tensorboard files
    root_dir,
    # environment
    env_name="MountainCarContinuous-v0",
    random_seed=0,
    # Params for collect
    num_environment_steps=10000,
    replay_buffer_capacity=10001,  # Per-environment
    # Params for eval
    num_eval_episodes=1,
    eval_interval=2000,
    # Params for summaries
    summary_interval=50,
):
    tf.compat.v1.set_random_seed(random_seed)

    environment = TFPyEnvironment(suite_gym.load(env_name))
    evaluation_environment = TFPyEnvironment(suite_gym.load(env_name))

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)]
    reward_model = MountainCarReward(environment.observation_spec(), environment.action_spec())
    initial_state_distribution_model = MountainCarInitialState(environment.observation_spec())
    global_step = tf.compat.v1.train.get_or_create_global_step()

    agent = MbpoAgent(
        environment.time_step_spec(),
        environment.action_spec(),
        transition_model_type=TransitionModelType.DeterministicEnsemble,
        num_hidden_layers_model=1,
        num_hidden_nodes_model=100,
        activation_function_model=tf.nn.relu,
        ensemble_size=5,
        predict_state_difference=True,
        epochs=100,
        training_batch_size=32,
        callbacks=callbacks,
        reward_model=reward_model,
        initial_state_distribution_model=initial_state_distribution_model,
        trajectory_sampler_type=TrajectorySamplerType.TS1,
        horizon=5,
        population_size=400,
        model_free_agent_type=ModelFreeAgentType.Sac,
        num_hidden_layers_agent=1,
        num_hidden_nodes_agent=256,
        activation_function_agent=tf.nn.relu,
        model_free_training_iterations=40,
        virtual_sample_batch_size=64,
        train_step_counter=global_step,
    )

    agent_trainer = BackgroundPlanningAgentTrainer(1000, 1)

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
        number_of_initial_random_policy_steps=1000,
        use_tf_function=False,
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
