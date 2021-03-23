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
from tf_agents.specs import BoundedTensorSpec
from tf_agents.trajectories.trajectory import Trajectory

from bellman.environments.transition_model.transition_model import (
    TS,
    T,
    TrainableTransitionModel,
    TransitionModel,
)


class StubTransitionModel(TransitionModel):
    def __init__(
        self,
        observation_space_spec: BoundedTensorSpec,
        action_space_spec: BoundedTensorSpec,
    ) -> None:
        self._observation_space_spec = observation_space_spec
        self._action_space_spec = action_space_spec

    @property
    def observation_space_spec(self) -> BoundedTensorSpec:
        return self._observation_space_spec

    @property
    def action_space_spec(self) -> BoundedTensorSpec:
        return self._action_space_spec

    def step(self, observation: tf.Tensor, action: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()


class StubTrainableTransitionModel(TrainableTransitionModel):
    def _step(self, latent_observation: tf.Tensor, action: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def _train(self, latent_trajectories: Trajectory, training_spec: TS) -> T:
        raise NotImplementedError()
