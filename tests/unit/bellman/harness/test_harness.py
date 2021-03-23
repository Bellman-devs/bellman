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

from pathlib import Path
from typing import Tuple

import gin
import pytest
from tf_agents.agents.tf_agent_test import MyAgent
from tf_agents.benchmark.utils import extract_event_log_values, find_event_log
from tf_agents.environments.random_tf_environment import RandomTFEnvironment
from tf_agents.environments.tf_environment import TFEnvironment

from bellman.harness.harness import ExperimentHarness
from bellman.harness.utils import EVALUATION_METRICS_DIR, GIN_CONFIG, TRAIN_METRICS_DIR
from tests.tools.bellman.specs.tensor_spec import ACTION_SPEC, TIMESTEP_SPEC
from tests.tools.bellman.training.agent_trainer import SingleComponentAgentTrainer

_REAL_REPLAY_BUFFER_CAPACITY = 1000
_MAX_STEPS = 10


@pytest.fixture(name="experiment_setup")
def _experiment_harness_fixture(tmpdir) -> Tuple[ExperimentHarness, TFEnvironment]:
    root_dir = str(tmpdir / "root_dir")

    environment = RandomTFEnvironment(TIMESTEP_SPEC, ACTION_SPEC, episode_end_probability=0.0)
    evaluation_environment = RandomTFEnvironment(TIMESTEP_SPEC, ACTION_SPEC)
    agent = MyAgent(
        time_step_spec=environment.time_step_spec(), action_spec=environment.action_spec()
    )
    agent_trainer = SingleComponentAgentTrainer()

    harness = ExperimentHarness(
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
    )

    return harness, environment


def test_serialise_config_empty_operational_config_tensorboard_events_file(experiment_setup):
    experiment_harness, _ = experiment_setup
    base_dir = experiment_harness.define_base_experiment_directory()
    experiment_harness.serialise_config(base_dir)

    assert not gin.operative_config_str()

    event_file = find_event_log(base_dir)
    values = extract_event_log_values(event_file, GIN_CONFIG)

    assert not values[0][0]


@pytest.fixture(name="dummy_gin_global_config")
def _dummy_gin_config_file_fixture(tmpdir):
    dummy_config_file_path = tmpdir / "dummy_config.gin"
    dummy_config_file_path.write_text("test_fn.test_arg = 1", encoding="utf-8")

    @gin.configurable
    def test_fn(test_arg):
        pass

    gin.parse_config_file(dummy_config_file_path)

    # call the function so the `test_arg` is added to the "operative" config.
    test_fn()  # pylint: disable=no-value-for-parameter

    yield

    gin.clear_config()


def test_serialise_config_operational_config_tensorboard_events_file(
    experiment_setup, dummy_gin_global_config
):
    experiment_harness, _ = experiment_setup
    base_dir = experiment_harness.define_base_experiment_directory()
    experiment_harness.serialise_config(base_dir)

    event_file = find_event_log(base_dir)
    values = extract_event_log_values(event_file, GIN_CONFIG)

    assert "test_arg" in str(values[0][0])


def test_define_tensorboard_directories(experiment_setup):
    experiment_harness, _ = experiment_setup
    base_dir = experiment_harness.define_base_experiment_directory()
    train_dir, eval_dir = experiment_harness.define_tensorboard_directories(base_dir)
    train_dir_path = Path(train_dir)
    eval_dir_path = Path(eval_dir)

    assert str(train_dir_path.parent) == base_dir
    assert str(eval_dir_path.parent) == base_dir
    assert train_dir_path.name == TRAIN_METRICS_DIR
    assert eval_dir_path.name == EVALUATION_METRICS_DIR


def test_create_summary_writers_parameters(tmpdir, experiment_setup):
    experiment_harness, _ = experiment_setup
    train_dir = str(tmpdir / TRAIN_METRICS_DIR)
    eval_dir = str(tmpdir / EVALUATION_METRICS_DIR)
    train_summary_writer, eval_summary_writer = experiment_harness.create_summary_writers(
        train_dir, eval_dir
    )
    assert train_summary_writer._metadata["logdir"] == train_dir
    assert eval_summary_writer._metadata["logdir"] == eval_dir


def test_real_replay_buffer_capacity(experiment_setup):
    experiment_harness, _ = experiment_setup
    real_replay_buffer = experiment_harness.create_real_replay_buffer()
    assert real_replay_buffer.capacity == _REAL_REPLAY_BUFFER_CAPACITY


@pytest.mark.parametrize("steps_to_collect", [1, 3, 5, 7, 9])
def test_real_driver_and_real_replay_buffer(experiment_setup, steps_to_collect):
    experiment_harness, environment = experiment_setup
    real_replay_buffer = experiment_harness.create_real_replay_buffer()
    experiment_harness._max_steps = steps_to_collect
    agent_collect_driver, _ = experiment_harness.create_real_drivers(real_replay_buffer, [])
    agent_collect_driver.run(environment.reset())
    trajectories = real_replay_buffer.gather_all()

    assert trajectories.step_type.shape == (1, steps_to_collect)


@pytest.mark.parametrize("steps_to_collect", [1, 3, 5, 7, 9])
def test_random_policy_driver_and_real_replay_buffer(experiment_setup, steps_to_collect):
    experiment_harness, environment = experiment_setup
    real_replay_buffer = experiment_harness.create_real_replay_buffer()
    experiment_harness._max_steps = steps_to_collect
    _, random_policy_collect_driver = experiment_harness.create_real_drivers(
        real_replay_buffer, []
    )
    random_policy_collect_driver.run(environment.reset())
    trajectories = real_replay_buffer.gather_all()

    assert trajectories.step_type.shape == (1, steps_to_collect)
