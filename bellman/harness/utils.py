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
Utilities for extracting results from experiments run through the harness.
"""

import datetime
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from warnings import warn

import numpy as np
import tensorflow as tf
from absl import logging
from tf_agents.benchmark.utils import (
    extract_event_log_values,
    find_event_log,
    summary_iterator,
)

GIN_CONFIG = "gin/operative_config"

TRAIN_METRICS_DIR = "train"
EVALUATION_METRICS_DIR = "eval"
TIME_METRIC = "Time"


def get_metric_values(
    root_dir: str,
    experiment_phase: str,
    tag_name: str,
    experiment_dirs: Optional[List[str]] = None,
    return_time: bool = False,
) -> Dict[str, Dict[int, np.generic]]:
    """
    When running an experiment through the experiment harness, tensorboard event files are created
    with metrics recorded during the experiment. Each experiment creates a separate directory
    under the `root_dir`. Metrics gathered during each phase of the experiment (e.g. training and
    policy evaluation) are stored in separate subdirectories of the experiment directory:

        `root_dir` / experiment id / `experiment_phase`

    This function will collect the values for a named metric gathered in a particular phase, from
    all of the listed experiments.

    :param root_dir: The root directory used by the experiment harness.
    :param experiment_phase: The phase of the experiment in which the trace was recorded.
    :param tag_name: The "tag" of the metric (usually defined by the metric object).
    :param experiment_dirs: A list of experiment ids which have been recorded in the root
                            directory. The default behaviour is to collect the specified metric
                            from all of the experiments in the root directory. This argument can be
                            used to specify a subset of these, if desired.
    :param return_time: If set to to `True` a dictionary of wallclock times for `tag_name`
        events are returned instead of metric values.

    :return: A dictionary mapping experiment ids to either metric values or wallclock times.
    """
    if not experiment_dirs:
        experiment_dirs_search_pattern = os.path.join(root_dir, "*/")
        experiment_dirs_full_path = tf.io.gfile.glob(experiment_dirs_search_pattern)
        experiment_dirs = [Path(full_path).name for full_path in experiment_dirs_full_path]

    all_values = {}
    for experiment_dir in experiment_dirs:
        summary_dir = os.path.join(root_dir, experiment_dir, experiment_phase)

        if not os.path.isdir(summary_dir):
            warn(f"{summary_dir} does not exist.")
            continue

        event_file = find_event_log(summary_dir)

        # we use TF-Agents' extract_event_log_values to extract wallclock time
        if return_time:
            metric_values = extract_event_log_values(event_file, tag_name)[1]
            all_values[experiment_dir] = metric_values
        # we use internal simplified version for metric values
        else:
            metric_values = _extract_event_log_values(event_file, tag_name)
        all_values[experiment_dir] = metric_values

    return all_values


def get_tag_names(
    root_dir: str,
    experiment_phase: str,
    experiment_dir: str,
) -> Set[str]:
    """
    When running an experiment through the experiment harness, tensorboard event files are created
    with metrics recorded during the experiment. Each metric gets a unique tag name that has to be
    used to retrieve metric values with `get_metric_values`. Tensorflow generates a lots of
    metrics that are recorded and this function can be used to get tag names associated with each
    metric.

    :param root_dir: The root directory used by the experiment harness.
    :param experiment_phase: The phase of the experiment in which the trace was recorded.
    :param experiment_dir: An experiment id which has been recorded in the root directory.

    :return: A set of tag names.
    """
    summary_dir = os.path.join(root_dir, experiment_dir, experiment_phase)

    if not os.path.isdir(summary_dir):
        warn(f"{summary_dir} does not exist.")

    tags: Set[str] = set()
    event_file = find_event_log(summary_dir)
    for summary in summary_iterator(event_file):
        for value in summary.summary.value:
            tags.add(value.tag)

    return tags


def _extract_event_log_values(
    event_file: str,
    event_tag: str,
) -> Dict[int, np.generic]:
    """Simplification of TF-Agents' `extract_event_log_values`, since it requires 0th step to
    be written, for the purpose of computing the wall clock time. It often does not make sense to
    write a summary at step 0 and since we do not need wall clock time computation we simplify
    things here and extract only the event values for the `event_tag`.

    An issue was filed to TF-Agents to remove requirement of writing 0th step
    (see https://github.com/tensorflow/agents/issues/560), if this behaviour is modified
    accordingly we will be able to drop this function and use TF-Agents'
    `extract_event_log_values`.

    :param event_file: Path to the event log.
    :param event_tag: Event to extract from the logs.

    :return: A dictionary with (step: event value) structure.
    """
    event_values = {}
    for summary in summary_iterator(event_file):
        for value in summary.summary.value:
            if value.tag == event_tag:
                ndarray = tf.make_ndarray(value.tensor)
                event_values[summary.step] = ndarray.item(0)

    return event_values
