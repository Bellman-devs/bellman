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

import tensorflow as tf
from tf_agents.agents.tf_agent import LossInfo

from bellman.training.schedule import TFTrainingScheduler, TrainingDefinition
from tests.tools.bellman.training.schedule import (
    IdentifiableComponentTrainer,
    MultiComponentAgent,
    SingleComponentAgent,
)


def test_training_scheduler_calls_train():
    """
    Ensure that the training scheduler trains a single component agent.
    """
    schedule = {
        SingleComponentAgent.COMPONENT: TrainingDefinition(
            1, IdentifiableComponentTrainer(SingleComponentAgent.COMPONENT.name)
        )
    }

    training_scheduler = TFTrainingScheduler(schedule)
    loss_dictionary = training_scheduler.maybe_train(tf.ones(tuple(), dtype=tf.int64))

    assert (
        loss_dictionary[SingleComponentAgent.COMPONENT].extra
        == SingleComponentAgent.COMPONENT.name
    )


def test_training_scheduler_calls_more_than_one_train():
    """
    Ensure the training scheduler trains each component in a multi-component agent.
    """
    schedule = {
        MultiComponentAgent.COMPONENT_1: TrainingDefinition(
            1, IdentifiableComponentTrainer(MultiComponentAgent.COMPONENT_1.name)
        ),
        MultiComponentAgent.COMPONENT_2: TrainingDefinition(
            1, IdentifiableComponentTrainer(MultiComponentAgent.COMPONENT_2.name)
        ),
        MultiComponentAgent.COMPONENT_3: TrainingDefinition(
            1, IdentifiableComponentTrainer(MultiComponentAgent.COMPONENT_3.name)
        ),
    }

    training_scheduler = TFTrainingScheduler(schedule)
    loss_dictionary = training_scheduler.maybe_train(tf.ones(tuple(), dtype=tf.int64))

    assert (
        loss_dictionary[MultiComponentAgent.COMPONENT_1].extra
        == MultiComponentAgent.COMPONENT_1.name
    )
    assert (
        loss_dictionary[MultiComponentAgent.COMPONENT_2].extra
        == MultiComponentAgent.COMPONENT_2.name
    )
    assert (
        loss_dictionary[MultiComponentAgent.COMPONENT_3].extra
        == MultiComponentAgent.COMPONENT_3.name
    )


def test_training_scheduler_resets_step_counter():
    """
    Ensure the training scheduler trains a single component agent at the specified schedule.
    """
    schedule = {
        SingleComponentAgent.COMPONENT: TrainingDefinition(
            2, IdentifiableComponentTrainer(SingleComponentAgent.COMPONENT.name)
        )
    }

    training_scheduler = TFTrainingScheduler(schedule)
    loss_dictionary_1 = training_scheduler.maybe_train(1 * tf.ones(tuple(), dtype=tf.int64))
    loss_dictionary_2 = training_scheduler.maybe_train(2 * tf.ones(tuple(), dtype=tf.int64))
    loss_dictionary_3 = training_scheduler.maybe_train(3 * tf.ones(tuple(), dtype=tf.int64))
    loss_dictionary_4 = training_scheduler.maybe_train(4 * tf.ones(tuple(), dtype=tf.int64))

    assert not loss_dictionary_1
    assert not loss_dictionary_3
    assert (
        loss_dictionary_2[SingleComponentAgent.COMPONENT].extra
        == SingleComponentAgent.COMPONENT.name
    )
    assert (
        loss_dictionary_4[SingleComponentAgent.COMPONENT].extra
        == SingleComponentAgent.COMPONENT.name
    )


def test_training_scheduler_resets_one_step_counter_of_several():
    """
    Ensure that the training scheduler trains each component of a multi-component agent at the
    specified schedule.
    """
    schedule = {
        MultiComponentAgent.COMPONENT_1: TrainingDefinition(
            1, IdentifiableComponentTrainer(MultiComponentAgent.COMPONENT_1.name)
        ),
        MultiComponentAgent.COMPONENT_2: TrainingDefinition(
            2, IdentifiableComponentTrainer(MultiComponentAgent.COMPONENT_2.name)
        ),
        MultiComponentAgent.COMPONENT_3: TrainingDefinition(
            3, IdentifiableComponentTrainer(MultiComponentAgent.COMPONENT_3.name)
        ),
    }

    training_scheduler = TFTrainingScheduler(schedule)
    loss_dictionary_1 = training_scheduler.maybe_train(1 * tf.ones(tuple(), dtype=tf.int64))
    loss_dictionary_2 = training_scheduler.maybe_train(2 * tf.ones(tuple(), dtype=tf.int64))
    loss_dictionary_3 = training_scheduler.maybe_train(3 * tf.ones(tuple(), dtype=tf.int64))

    assert len(loss_dictionary_1) == 1
    assert (
        loss_dictionary_1[MultiComponentAgent.COMPONENT_1].extra
        == MultiComponentAgent.COMPONENT_1.name
    )

    assert len(loss_dictionary_2) == 2
    assert (
        loss_dictionary_2[MultiComponentAgent.COMPONENT_1].extra
        == MultiComponentAgent.COMPONENT_1.name
    )
    assert (
        loss_dictionary_2[MultiComponentAgent.COMPONENT_2].extra
        == MultiComponentAgent.COMPONENT_2.name
    )

    assert len(loss_dictionary_3) == 2
    assert (
        loss_dictionary_3[MultiComponentAgent.COMPONENT_1].extra
        == MultiComponentAgent.COMPONENT_1.name
    )
    assert (
        loss_dictionary_3[MultiComponentAgent.COMPONENT_3].extra
        == MultiComponentAgent.COMPONENT_3.name
    )


def test_training_info_does_not_contain_none_losses():
    def _none_returning_train_step():
        return LossInfo(loss=None, extra=None)

    schedule = {SingleComponentAgent.COMPONENT: TrainingDefinition(1, _none_returning_train_step)}
    training_scheduler = TFTrainingScheduler(schedule)

    loss_dictionary = training_scheduler.maybe_train(tf.ones(tuple(), dtype=tf.int64))
    assert not loss_dictionary  # loss_dictionary should be empty
