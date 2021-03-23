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
This module defines environment model components which provide easy and specific control over the
`step_type`s of unrolled trajectories.

This is accomplished by using the `step_type` as the observation, and defining a transition model
with a pre-specified list of batches of observations to predict. The termination model can then use
the observed step type to mark `LAST` steps as terminal.
"""

from typing import Iterator, List

import tensorflow as tf
from tf_agents.specs import BoundedTensorSpec
from tf_agents.trajectories.time_step import StepType

from bellman.environments.termination_model import TerminationModel
from bellman.environments.transition_model.transition_model import TransitionModel

OBSERVATION_SPACE_SPEC = BoundedTensorSpec((), tf.int32, StepType.FIRST, StepType.LAST)


class TrajectoryOptimiserTransitionModel(TransitionModel):
    def __init__(
        self, action_space_spec: BoundedTensorSpec, observations: Iterator[tf.Tensor]
    ):
        self._action_space_spec = action_space_spec
        self._observations = observations

    @property
    def observation_space_spec(self) -> BoundedTensorSpec:
        return OBSERVATION_SPACE_SPEC

    @property
    def action_space_spec(self) -> BoundedTensorSpec:
        return self._action_space_spec

    def step(self, observation: tf.Tensor, action: tf.Tensor) -> tf.Tensor:
        next_observation = next(self._observations)
        return next_observation


class TrajectoryOptimiserTerminationModel(TerminationModel):
    def _terminates(self, observation: tf.Tensor) -> tf.Tensor:
        return tf.equal(observation, StepType.LAST)
