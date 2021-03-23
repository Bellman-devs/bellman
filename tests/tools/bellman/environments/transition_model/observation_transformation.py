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

from typing import Optional

import tensorflow as tf
from tf_agents.specs import BoundedTensorSpec

from bellman.environments.transition_model.observation_transformation import (
    ObservationTransformation,
)


class GoalStateObservationTransformation(ObservationTransformation):
    def __init__(
        self,
        latent_observation_space_spec: BoundedTensorSpec,
        goal_state_start_index: int,
    ) -> None:
        self._goal_state_start_index = goal_state_start_index
        super().__init__(latent_observation_space_spec)

    def forward_observation(self, observation: tf.Tensor) -> tf.Tensor:
        latent_observation = observation[..., : self._goal_state_start_index]
        return latent_observation

    def invert_observation(
        self,
        latent_observation: tf.Tensor,
        previous_observation: tf.Tensor,
    ) -> tf.Tensor:
        observation = tf.concat(
            [latent_observation, previous_observation[..., self._goal_state_start_index :]],
            axis=-1,
        )
        return observation
