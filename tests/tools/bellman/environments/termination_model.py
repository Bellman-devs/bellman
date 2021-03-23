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

from bellman.environments.termination_model import TerminationModel


class MutableBatchConstantTermination(TerminationModel):
    def __init__(self, observation_spec: tf.TensorSpec, batch_size: int):
        super().__init__(observation_spec)
        self._should_terminate = tf.zeros(shape=(batch_size,), dtype=tf.dtypes.bool)

    @property
    def should_terminate(self):
        return self._should_terminate

    @should_terminate.setter
    def should_terminate(self, should_terminate):
        self._should_terminate = should_terminate

    def _terminates(self, observation: tf.Tensor) -> tf.Tensor:
        return self.should_terminate
