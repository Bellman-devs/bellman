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
This module provides a training scheduler for agents which are composed of one or more trainable
components.
"""

from enum import Enum
from functools import reduce
from math import gcd
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import tensorflow as tf
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.utils import common

from bellman.agents.components import EnvironmentModelComponents


class TrainingDefinition(NamedTuple):
    """
    A named tuple which contains the information necessary to train a component of an agent.

    When the agent is trained using the `TFTrainingSchedule` the `train_step` function will be
    called after `interval` real environment steps have elapsed.
    """

    interval: int
    train_step: Callable[[], LossInfo]


class TFTrainingScheduler:
    """
    A class for scheduling the training steps for components of agents.

    Training an agent is achieved by training the constituent components. For each component this
    class must define:
    1) the frequency at which the component should be trained, and
    2) a function which performs a single training step.

    This class requires a training "schedule" for the agent. This schedule is specified as a
    dictionary mapping between the component identifiers and `TrainingDefinition` named tuples.
    One or more enumerations should be defined for an agent. It may be convenient to combine common
    enumerations, for example an enumeration for the components of an approximate MDP together with
    a separate enumeration for the components of a model-free agent which will be used to generate
    virtual rollouts. The schedule should provide a `TrainingDefinition` for each of the components
    in the agent.
    """

    def __init__(self, schedule: Dict[Enum, TrainingDefinition]):
        """
        :param schedule: A dictionary mapping between the trainable components of the agent and
                         `TrainingDefinition`s.

        """
        self._schedule = schedule
        self._counters = {k: common.create_variable(k.name) for k, _ in schedule.items()}
        self._lagged_environment_steps = tf.zeros(shape=(), dtype=tf.int64)

    def environment_steps_between_maybe_train(
        self, additional_intervals: Optional[List[int]] = None
    ) -> int:
        """
        The schedule defines a list of intervals for each trainable component, which represent the
        number of real environment steps that should elapse between training steps for that
        component. This method calculates the number of environment steps that can happen between
        training steps for all trainable components. This allows the training loop to avoid calling
        the `maybe_train` method when on environment steps when no trainable components are due to
        be trained.

        :param additional_intervals: Optional list of additional intervals to take into account
            when generating the number of steps for collecting data.

        :return: The greatest common divisor of all the trainable components' training intervals.
        """

        training_intervals = [
            training_def.interval for _, training_def in self._schedule.items()
        ]
        if isinstance(additional_intervals, List):
            training_intervals.extend(additional_intervals)
        return reduce(gcd, training_intervals)

    def maybe_train(self, environment_steps: tf.Tensor) -> Dict[Enum, LossInfo]:
        """
        This method may train one or more of the trainable components of the agent, if a sufficient
        number of environment steps have elapsed since the last call.

        This method resets the component specific counters when the associated `train_step`
        function is called. As such, even if the number of elapsed environment steps is a multiple
        of the training interval the train step method will only be called once.

        :param environment_steps: the total number of elapsed environment steps.

        :return: A dictionary mapping between the component identifiers which were trained in this
                 method call and the `LossInfo` returned from training that
                 component.
        """
        delta_steps = environment_steps - self._lagged_environment_steps
        self._lagged_environment_steps = environment_steps

        training_info = {}
        for component, training_definition in self._schedule.items():
            step_counter = self._counters[component]
            step_counter.assign_add(delta_steps)
            if step_counter >= training_definition.interval:
                step_counter.assign(0)
                loss_info = training_definition.train_step()
                training_info[component] = loss_info

        return training_info
