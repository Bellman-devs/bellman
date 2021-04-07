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

import pytest
import tensorflow as tf

from bellman.harness.utils import get_metric_values

TAG_NAME = "tag"


def _record_one_metric_step_value(summary_writer, data_tensor):
    with summary_writer.as_default():
        with tf.summary.record_if(True):
            step_tensor = tf.constant(0, dtype=tf.int64)
            tf.compat.v2.summary.scalar(name=TAG_NAME, data=data_tensor, step=step_tensor)


def test_one_step_get_metric_values(tmpdir):
    summary_dir = tmpdir / "experiment_id" / "summary_dir"
    summary_writer = tf.summary.create_file_writer(str(summary_dir), flush_millis=0)
    _record_one_metric_step_value(summary_writer, tf.zeros(shape=()))

    metric_values = get_metric_values(str(tmpdir), "summary_dir", TAG_NAME, ["experiment_id"])

    assert {"experiment_id": {0: 0.0}} == metric_values


def test_multi_step_get_metric_values(tmpdir):
    summary_dir = tmpdir / "experiment_id" / "summary_dir"
    summary_writer = tf.summary.create_file_writer(str(summary_dir), flush_millis=0)
    with summary_writer.as_default():
        with tf.summary.record_if(True):
            data_tensor = tf.zeros(shape=())
            step_tensor = tf.constant(0, dtype=tf.int64)
            tf.compat.v2.summary.scalar(name=TAG_NAME, data=data_tensor, step=step_tensor)

            data_tensor = tf.ones(shape=())
            step_tensor = tf.constant(1, dtype=tf.int64)
            tf.compat.v2.summary.scalar(name=TAG_NAME, data=data_tensor, step=step_tensor)

            data_tensor = 2 * tf.ones(shape=())
            step_tensor = tf.constant(2, dtype=tf.int64)
            tf.compat.v2.summary.scalar(name=TAG_NAME, data=data_tensor, step=step_tensor)

    metric_values = get_metric_values(str(tmpdir), "summary_dir", TAG_NAME, ["experiment_id"])

    assert {"experiment_id": {0: 0.0, 1: 1.0, 2: 2.0}} == metric_values


def test_get_metric_values_read_from_several_experiment_runs(tmpdir):
    summary_dir = tmpdir / "experiment_id" / "summary_dir"
    summary_writer = tf.summary.create_file_writer(str(summary_dir), flush_millis=0)
    _record_one_metric_step_value(summary_writer, tf.zeros(shape=()))

    later_summary_dir = tmpdir / "experiment_id_2" / "summary_dir"
    later_summary_writer = tf.summary.create_file_writer(
        str(later_summary_dir), flush_millis=0
    )
    _record_one_metric_step_value(later_summary_writer, tf.ones(shape=()))

    metric_values = get_metric_values(
        str(tmpdir), "summary_dir", TAG_NAME, ["experiment_id", "experiment_id_2"]
    )

    assert {"experiment_id": {0: 0.0}, "experiment_id_2": {0: 1.0}} == metric_values


def test_get_metric_values_read_from_subset_of_several_experiment_runs(tmpdir):
    summary_dir = tmpdir / "experiment_id" / "summary_dir"
    summary_writer = tf.summary.create_file_writer(str(summary_dir), flush_millis=0)
    _record_one_metric_step_value(summary_writer, tf.zeros(shape=()))

    later_summary_dir = tmpdir / "experiment_id_2" / "summary_dir"
    later_summary_writer = tf.summary.create_file_writer(
        str(later_summary_dir), flush_millis=0
    )
    _record_one_metric_step_value(later_summary_writer, tf.ones(shape=()))

    metric_values = get_metric_values(
        str(tmpdir), "summary_dir", TAG_NAME, ["experiment_id_2"]
    )

    assert {"experiment_id_2": {0: 1.0}} == metric_values


def test_default_is_all_experiment_ids(tmpdir):
    summary_dir = tmpdir / "experiment_id" / "summary_dir"
    summary_writer = tf.summary.create_file_writer(str(summary_dir), flush_millis=0)
    _record_one_metric_step_value(summary_writer, tf.zeros(shape=()))

    later_summary_dir = tmpdir / "experiment_id_2" / "summary_dir"
    later_summary_writer = tf.summary.create_file_writer(
        str(later_summary_dir), flush_millis=0
    )
    _record_one_metric_step_value(later_summary_writer, tf.ones(shape=()))

    metric_values = get_metric_values(str(tmpdir), "summary_dir", TAG_NAME)

    assert {"experiment_id": {0: 0.0}, "experiment_id_2": {0: 1.0}} == metric_values


def test_root_dir_is_empty_with_default_experiment_ids(tmpdir):
    metric_values = get_metric_values(str(tmpdir), "summary_dir", TAG_NAME)
    assert not metric_values


def test_root_dir_is_empty_with_explicit_experiment_ids(tmpdir):
    with pytest.warns(UserWarning):
        metric_values = get_metric_values(
            str(tmpdir), "summary_dir", TAG_NAME, ["experiment_id"]
        )

    assert not metric_values


def test_experiment_phase_name_is_wrong(tmpdir):
    summary_dir = tmpdir / "experiment_id" / "summary_dir"
    summary_writer = tf.summary.create_file_writer(str(summary_dir), flush_millis=0)
    _record_one_metric_step_value(summary_writer, tf.zeros(shape=()))

    with pytest.warns(UserWarning):
        metric_values = get_metric_values(
            str(tmpdir), "WRONG_SUMMARY_DIR_NAME", TAG_NAME, ["experiment_id"]
        )

    assert not metric_values
