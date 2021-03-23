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
This module provides a harness for running experiments.
"""

import datetime
import os
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import gin
import tensorflow as tf
from absl import logging
from gin.tf import GinConfigSaverHook
from tf_agents.agents import TFAgent
from tf_agents.drivers.driver import Driver
from tf_agents.drivers.tf_driver import TFDriver
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.eval import metric_utils
from tf_agents.metrics.tf_metric import TFStepMetric
from tf_agents.metrics.tf_metrics import (
    AverageEpisodeLengthMetric,
    AverageReturnMetric,
    EnvironmentSteps,
    NumberOfEpisodes,
)
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.utils import common

from bellman.harness.utils import (
    EVALUATION_METRICS_DIR,
    GIN_CONFIG,
    TIME_METRIC,
    TRAIN_METRICS_DIR,
)
from bellman.training.agent_trainer import AgentTrainer
from bellman.training.model_free_agent_trainer import OnPolicyModelFreeAgentTrainer


class ExperimentHarness:
    """
    A harness for running experiments. The `run` method will run the experiment.
    """

    def __init__(
        self,
        root_dir: str,
        environment: TFEnvironment,
        evaluation_environment: TFEnvironment,
        agent: TFAgent,
        agent_trainer: AgentTrainer,
        real_replay_buffer_capacity: int,
        total_number_of_environment_steps: int,
        summary_interval: int,
        evaluation_interval: int,
        number_of_evaluation_episodes: int,
        number_of_initial_random_policy_steps: int = 0,
        use_tf_function: bool = False,
    ):
        """
        :param root_dir: Root directory where all experiments are stored.
        :param environment: The training environment the agent is stepping through.
        :param evaluation_environment: The environment for evaluating the performance of the agent.
        :param agent: The TF-Agent agent to train.
        :param agent_trainer: The trainer which will produce a training schedule for the components
            of the agent.
        :param real_replay_buffer_capacity: Capacity of the buffer collecting real samples.
        :param total_number_of_environment_steps: The number of environment steps to run the
            experiment for.
        :param summary_interval: Interval for summaries.
        :param evaluation_interval: Interval for evaluation points.
        :param number_of_evaluation_episodes: Number of episodes at each evaluation point.
        :param number_of_initial_random_policy_steps: If > 0, some initial training data is
            gathered by running a random policy on the real environment.
        :param use_tf_function: If `True`, use a `tf.function` for data collection.
        """
        assert real_replay_buffer_capacity > 0
        assert total_number_of_environment_steps > 0
        assert summary_interval > 0
        assert evaluation_interval > 0
        assert number_of_evaluation_episodes > 0
        assert 0 <= number_of_initial_random_policy_steps <= total_number_of_environment_steps
        assert number_of_initial_random_policy_steps == 0 or not isinstance(
            agent_trainer, OnPolicyModelFreeAgentTrainer
        )  # model-free on-policy agents must always execute their own policy!

        self._root_dir = root_dir
        self._base_dirname = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self._environment = environment
        self._evaluation_environment = evaluation_environment
        self._agent = agent
        self._agent_trainer = agent_trainer
        self._real_replay_buffer_capacity = real_replay_buffer_capacity
        self._total_number_of_environment_steps = total_number_of_environment_steps
        self._summary_interval = summary_interval
        self._evaluation_interval = evaluation_interval
        self._number_of_evaluation_episodes = number_of_evaluation_episodes
        self._number_of_initial_random_policy_steps = number_of_initial_random_policy_steps
        self._use_tf_function = use_tf_function
        self._max_steps: Optional[int] = None

    @property
    def agent(self) -> TFAgent:
        """
        :return: The agent which is trained by this harness.
        """
        return self._agent

    @property
    def base_dirname(self) -> str:
        """
        :return: The directory name in the root directory where results will be stored.
        """
        return self._base_dirname

    def define_base_experiment_directory(self) -> str:
        """
        Define the path for the base directory for the experiment.
        """
        base_dir = os.path.join(
            os.path.expanduser(self._root_dir),
            self._base_dirname,
        )
        return base_dir

    @staticmethod
    def serialise_config(base_dir: str):
        """
        This method creates a single gin config file in the `base_dir`. This file can be used to
        reproduce the experiment. This "operative" config file will contain all of the parameters
        which have been used to parameterise a function decorated with `@gin.configurable'.

        The config parameter values are also written to a TensorBoard events file in the
        `base_dir`. This ensures that the parameter values can be viewed in TensorBoard.

        Note: The `GinConfigSaverHook` can create a TensorBoard events file. Unfortunately it does
        not seem to work with TensorFlow 2.0, so this is done manually.
        """
        config_saver_hooks = GinConfigSaverHook(base_dir, summarize_config=False)
        config_saver_hooks.after_create_session()

        base_dir_summary_writer = tf.summary.create_file_writer(base_dir)
        with base_dir_summary_writer.as_default():
            tf.summary.text(GIN_CONFIG, gin.operative_config_str(), 0)

    @staticmethod
    def define_tensorboard_directories(base_dir: str) -> Tuple[str, str]:
        """
        Define the paths of the tensorboard directories.
        """
        train_dir = os.path.join(base_dir, TRAIN_METRICS_DIR)
        eval_dir = os.path.join(base_dir, EVALUATION_METRICS_DIR)

        return train_dir, eval_dir

    @staticmethod
    def create_summary_writers(
        train_dir: str, eval_dir: str
    ) -> Tuple[tf.summary.SummaryWriter, tf.summary.SummaryWriter]:
        """
        Create and return the training time summary writer and the evaluation time summary writer.
        """
        # Summary writers
        train_summary_writer = tf.summary.create_file_writer(train_dir, flush_millis=1000)
        eval_summary_writer = tf.summary.create_file_writer(eval_dir, flush_millis=1000)

        return train_summary_writer, eval_summary_writer

    @staticmethod
    def create_train_metrics() -> List[TFStepMetric]:
        """
        Create a list of metrics to capture during training.
        """
        return [
            NumberOfEpisodes(),
            EnvironmentSteps(),
            AverageReturnMetric(buffer_size=1),
            AverageEpisodeLengthMetric(buffer_size=1),
        ]

    @staticmethod
    def create_evaluation_metrics() -> List[TFStepMetric]:
        """
        Create a list of metrics to capture during policy evaluation.
        """
        return [
            AverageReturnMetric(buffer_size=1),
            AverageEpisodeLengthMetric(buffer_size=1),
        ]

    def create_real_replay_buffer(self) -> ReplayBuffer:
        """
        Create the replay buffer for storing data from the real environment.
        """
        return TFUniformReplayBuffer(
            self._agent.collect_policy.trajectory_spec,
            batch_size=1,
            max_length=self._real_replay_buffer_capacity,
        )

    def create_real_drivers(
        self,
        real_replay_buffer: ReplayBuffer,
        train_metrics: List[TFStepMetric],
    ) -> Tuple[Driver, Driver]:
        """
        Create the drivers for interacting with the real environment.

        This method creates two drivers: one uses the agent's "collect" policy, the other uses a
        uniform random policy.

        Note that the random policy is defined with the same `info_spec` as the agent's "collect"
        policy. The `info_spec` of the trajectories generated by the random policy must conform to
        the expectations of the agent when the data is used for training.
        """
        agent_collect_driver = TFDriver(
            self._environment,
            self._agent.collect_policy,
            observers=[real_replay_buffer.add_batch] + train_metrics,
            max_steps=self._max_steps,
            disable_tf_function=not self._use_tf_function,
        )
        random_policy = RandomTFPolicy(
            self._environment.time_step_spec(),
            self._environment.action_spec(),
            info_spec=self._agent.collect_policy.info_spec,
        )
        random_policy_collect_driver = TFDriver(
            self._environment,
            random_policy,
            observers=[real_replay_buffer.add_batch] + train_metrics,
            max_steps=self._max_steps,
            disable_tf_function=not self._use_tf_function,
        )

        return agent_collect_driver, random_policy_collect_driver

    @staticmethod
    def write_summary_scalar(
        metric_name: str,
        metric_value: Union[List[float], float],
        step: int,
        summary_writer: tf.summary.SummaryWriter,
    ):
        """Write a scalar summary statistic to a tensorboard directory."""
        with summary_writer.as_default():
            if isinstance(metric_value, list):
                value = metric_value[-1]
                tf.compat.v2.summary.scalar(name=metric_name, data=value, step=step)
            else:
                tf.compat.v2.summary.scalar(name=metric_name, data=metric_value, step=step)

    def run(self):
        """
        This method runs an experiment. It creates a loop that steps through the environment,
        using the agent to collect actions in each step. Note that loop only seemingly goes
        step-by-step, drivers actually collect multiple steps in each call, governed by the number
        of initial random policy steps and maximum number of steps - the greatest common divisor of
        all the trainable components' training intervals, obtained through the
        `TFTrainingScheduler` method `environment_steps_between_maybe_train`.

        While running, this will collect metrics (during training and also separate evaluation
        metrics). These metrics are periodically logged to `stdout`, and also recorded by summary
        writers. Note that `max_steps` takes into account various reporting intervals computing a
        greatest common denominator among all intervals to determine the step size. Hence, it
        would be good to use intervals that are multiples of each other, if possible.

        Note that this method also creates an environment step counter metric, which is used
        throughout to monitor the progress of the experiment. This counter ensures that periodic
        tasks, such as logging and training, happen at the correct intervals.
        """
        logging.info("Initialising the experiment.")
        base_dir = self.define_base_experiment_directory()
        self.serialise_config(base_dir)
        train_dir, eval_dir = self.define_tensorboard_directories(base_dir)

        train_summary_writer, eval_summary_writer = self.create_summary_writers(
            train_dir, eval_dir
        )
        train_summary_writer.set_as_default()

        train_metrics = self.create_train_metrics()
        environment_steps_metric = EnvironmentSteps()
        train_metrics.extend([environment_steps_metric])
        evaluation_metrics = self.create_evaluation_metrics()

        real_replay_buffer = self.create_real_replay_buffer()
        training_scheduler = self._agent_trainer.create_training_scheduler(
            self._agent,
            real_replay_buffer,
        )
        self._max_steps = training_scheduler.environment_steps_between_maybe_train(
            additional_intervals=[
                self._summary_interval,
                self._evaluation_interval,
                self._number_of_initial_random_policy_steps,
            ]
        )
        agent_collect_driver, random_policy_collect_driver = self.create_real_drivers(
            real_replay_buffer,
            train_metrics,
        )

        # Reset the real environment
        time_step = self._environment.reset()

        # executing the experiment
        logging.info("Experiment started running.")
        self.write_summary_scalar(
            TIME_METRIC, 0.0, environment_steps_metric.result(), train_summary_writer
        )

        # step-by-step
        while tf.math.less(
            environment_steps_metric.result(), self._total_number_of_environment_steps
        ):

            # Initial transitions with random policy to bootstrap training
            if environment_steps_metric.result() < self._number_of_initial_random_policy_steps:
                logging.info(
                    "Step = %d, collecting initial transitions with random policy, "
                    + "%d steps in total.",
                    environment_steps_metric.result(),
                    self._number_of_initial_random_policy_steps,
                )
                time_step, _ = random_policy_collect_driver.run(time_step)
            # Collecting data with the agent's "collect" policy
            else:
                logging.info(
                    "Step = %d, collecting regular transitions with agent policy, "
                    + "%d steps in total.",
                    environment_steps_metric.result(),
                    self._max_steps,
                )
                time_step, _ = agent_collect_driver.run(time_step)

            # potentially train certain component in current `self._max_steps` iteration
            training_info = training_scheduler.maybe_train(environment_steps_metric.result())
            for component, loss_info in training_info.items():
                self.write_summary_scalar(
                    "TrainingLoss/" + component.name,
                    loss_info.loss,
                    environment_steps_metric.result(),
                    train_summary_writer,
                )
                if isinstance(loss_info.loss, list):
                    loss = loss_info.loss[-1]
                else:
                    loss = loss_info.loss
                logging.info(
                    "Step = %d, training of the %s component, loss (at final epoch) = %s",
                    environment_steps_metric.result(),
                    component.name,
                    str(loss),
                )

            # training summary and logs
            if environment_steps_metric.result() % self._summary_interval == 0:
                logging.info(
                    "Step = %d, ongoing performance of the policy, summary of the results:",
                    environment_steps_metric.result(),
                )
                for metric in train_metrics:
                    # The last metric in the `train_metrics` list is the environment steps counter
                    # which is an implementation detail of this class rather than a user-specified
                    # metric to report.
                    metric.tf_summaries(
                        train_step=environment_steps_metric.result(),
                        step_metrics=train_metrics[:-1],
                    )
                metric_utils.log_metrics(train_metrics)

            # Evaluate the policy
            if environment_steps_metric.result() % self._evaluation_interval == 0:
                logging.info(
                    "Step = %d, evaluation of the policy, summary of the results:",
                    environment_steps_metric.result(),
                )
                metric_utils.eager_compute(
                    evaluation_metrics,
                    self._evaluation_environment,
                    self._agent.policy,
                    num_episodes=self._number_of_evaluation_episodes,
                    train_step=environment_steps_metric.result(),
                    summary_writer=eval_summary_writer,
                    summary_prefix="Metrics",
                    use_function=self._use_tf_function,
                )
                metric_utils.log_metrics(evaluation_metrics)

        self.write_summary_scalar(
            "Time", 0.0, environment_steps_metric.result(), train_summary_writer
        )
        logging.info("Experiment completed.")
