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

import os
from pathlib import Path

import pytest
import tensorflow as tf
from absl import logging
from tf_agents.agents.tf_agent_test import MyAgent
from tf_agents.benchmark.utils import extract_event_log_values, find_event_log
from tf_agents.environments.random_tf_environment import RandomTFEnvironment

from bellman.harness.harness import ExperimentHarness
from bellman.harness.utils import (
    EVALUATION_METRICS_DIR,
    TIME_METRIC,
    TRAIN_METRICS_DIR,
    get_metric_values,
    get_tag_names,
)
from tests.tools.bellman.specs.tensor_spec import ACTION_SPEC, TIMESTEP_SPEC
from tests.tools.bellman.training.agent_trainer import SingleComponentAgentTrainer
from tests.tools.bellman.training.schedule import SingleComponentAgent

_REAL_REPLAY_BUFFER_CAPACITY = 1000
_MAX_STEPS = 10


@pytest.fixture(name="experiment_harness")
def _experiment_harness_fixture(tmpdir) -> ExperimentHarness:
    root_dir = str(tmpdir / "root_dir")

    environment = RandomTFEnvironment(TIMESTEP_SPEC, ACTION_SPEC)
    evaluation_environment = RandomTFEnvironment(TIMESTEP_SPEC, ACTION_SPEC)
    agent = MyAgent(
        time_step_spec=environment.time_step_spec(), action_spec=environment.action_spec()
    )
    agent_trainer = SingleComponentAgentTrainer()

    return ExperimentHarness(
        root_dir=root_dir,
        environment=environment,
        evaluation_environment=evaluation_environment,
        agent=agent,
        agent_trainer=agent_trainer,
        real_replay_buffer_capacity=_REAL_REPLAY_BUFFER_CAPACITY,
        total_number_of_environment_steps=_MAX_STEPS,
        summary_interval=1,
        evaluation_interval=_MAX_STEPS,
        number_of_evaluation_episodes=1,
        number_of_initial_random_policy_steps=0,
    )


def test_run(experiment_harness):
    experiment_harness.run()


@pytest.mark.parametrize("summary_interval", [1, 3])
@pytest.mark.parametrize("evaluation_interval", [1, 3])
@pytest.mark.parametrize("number_of_initial_random_policy_steps", [0, 1, 3, _MAX_STEPS * 2])
def test_experiment_harness_summaries_and_logs(
    caplog,
    tmpdir,
    summary_interval,
    evaluation_interval,
    number_of_initial_random_policy_steps,
):
    root_dir = str(tmpdir / "root_dir")
    train_interval = _MAX_STEPS
    total_steps = train_interval * 2
    caplog.set_level(logging.INFO)

    # define a simple agent
    environment = RandomTFEnvironment(TIMESTEP_SPEC, ACTION_SPEC)
    evaluation_environment = RandomTFEnvironment(TIMESTEP_SPEC, ACTION_SPEC)
    agent = MyAgent(
        time_step_spec=environment.time_step_spec(), action_spec=environment.action_spec()
    )
    agent_trainer = SingleComponentAgentTrainer(train_interval)

    # execute the experiment
    harness = ExperimentHarness(
        root_dir=root_dir,
        environment=environment,
        evaluation_environment=evaluation_environment,
        agent=agent,
        agent_trainer=agent_trainer,
        real_replay_buffer_capacity=_REAL_REPLAY_BUFFER_CAPACITY,
        total_number_of_environment_steps=total_steps,
        summary_interval=summary_interval,
        evaluation_interval=evaluation_interval,
        number_of_evaluation_episodes=1,
        number_of_initial_random_policy_steps=number_of_initial_random_policy_steps,
    )
    harness.run()

    # get correct paths
    experiment_id = os.listdir(root_dir)[0]

    # check wall clock time
    wallclock_time = get_metric_values(
        root_dir,
        TRAIN_METRICS_DIR,
        TIME_METRIC,
        [experiment_id],
        True,
    )
    assert experiment_id in wallclock_time and isinstance(wallclock_time[experiment_id], float)

    # check train and evaluation summary
    tag_name = "Metrics/AverageReturn"
    train_metric_values = get_metric_values(
        root_dir, TRAIN_METRICS_DIR, tag_name, [experiment_id]
    )
    eval_metric_values = get_metric_values(
        root_dir, EVALUATION_METRICS_DIR, tag_name, [experiment_id]
    )
    assert [*train_metric_values[experiment_id].keys()] == [
        i
        for i in range(summary_interval, total_steps + summary_interval, summary_interval)
        if i <= total_steps
    ]
    assert [*eval_metric_values[experiment_id].keys()] == [
        i
        for i in range(
            evaluation_interval, total_steps + summary_interval, evaluation_interval
        )
        if i <= total_steps
    ]

    # check record of training the models or agents
    tag_name = "TrainingLoss/" + SingleComponentAgent.COMPONENT.name
    train_metric_values = get_metric_values(
        root_dir, TRAIN_METRICS_DIR, tag_name, [experiment_id]
    )
    assert [*train_metric_values[experiment_id].keys()] == [
        i for i in range(train_interval, total_steps + 1, train_interval) if i <= total_steps
    ]

    # check logs for accurate random and regular transition collection intervals
    random_start_steps = []
    regular_start_steps = []
    for record in caplog.records:
        if hasattr(record, "message") and "initial transitions" in record.message:
            random_start_steps.append(int("".join(filter(str.isdigit, record.message[0:15]))))
        if hasattr(record, "message") and "regular transitions" in record.message:
            regular_start_steps.append(int("".join(filter(str.isdigit, record.message[0:15]))))
    if number_of_initial_random_policy_steps > 0:
        assert random_start_steps == [
            i
            for i in range(0, number_of_initial_random_policy_steps, harness._max_steps)
            if i <= number_of_initial_random_policy_steps and i <= total_steps
        ]
    if number_of_initial_random_policy_steps < total_steps:
        assert regular_start_steps == [
            i
            for i in range(
                number_of_initial_random_policy_steps, total_steps, harness._max_steps
            )
            if i <= total_steps
        ]


def test_metric_values_can_be_written_to_a_file_and_read_back(experiment_harness):
    base_dir = experiment_harness.define_base_experiment_directory()
    train_dir, eval_dir = experiment_harness.define_tensorboard_directories(base_dir)
    root_path = Path(train_dir).parent.parent
    experiment_id = Path(train_dir).parent.name
    train_summary_writer, eval_summary_writer = experiment_harness.create_summary_writers(
        train_dir, eval_dir
    )

    tag_name = "tag"

    with train_summary_writer.as_default():
        with tf.summary.record_if(True):
            data_tensor = tf.zeros(shape=())
            step_tensor = tf.constant(0, dtype=tf.int64)
            tf.compat.v2.summary.scalar(name=tag_name, data=data_tensor, step=step_tensor)

    with eval_summary_writer.as_default():
        with tf.summary.record_if(True):
            data_tensor = tf.ones(shape=())
            step_tensor = tf.constant(0, dtype=tf.int64)
            tf.compat.v2.summary.scalar(name=tag_name, data=data_tensor, step=step_tensor)

    train_metric_values = get_metric_values(
        str(root_path), TRAIN_METRICS_DIR, tag_name, [experiment_id]
    )
    eval_metric_values = get_metric_values(
        str(root_path), EVALUATION_METRICS_DIR, tag_name, [experiment_id]
    )

    assert {experiment_id: {0: 0.0}} == train_metric_values
    assert {experiment_id: {0: 1.0}} == eval_metric_values
