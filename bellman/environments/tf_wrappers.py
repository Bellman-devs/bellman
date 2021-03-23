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
Wrappers for TF environments.
"""

import tensorflow as tf
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.environments.tf_wrappers import TFEnvironmentBaseWrapper
from tf_agents.trajectories.time_step import StepType, TimeStep
from tf_agents.utils import common


class TFTimeLimit(TFEnvironmentBaseWrapper):
    """
    A wrapper for TF environments which terminates trajectories after a defined number of steps.
    Some GPUs don't support int64, this class addresses this issue.
    """

    def __init__(self, env: TFEnvironment, duration: int):
        """
        :param env: the environment to wrap.
        :param duration: number of steps after which to terminate the trajectories.
        """
        super().__init__(env)
        self._duration = duration

        # DK: Some GPUs don't support int64
        self._num_steps = common.create_variable(
            "TimeLimitTermination", initial_value=0, shape=(env.batch_size,), dtype=tf.int32
        )

    def set_initial_observation(self, observation):
        """
        Set initial observation of the environment model.

        :param observation: A batch of observations, one for each batch element (the batch size is
            the first dimension)
        """

        self._num_steps.assign(tf.zeros_like(self._num_steps))
        return self._env.set_initial_observation(observation)

    def _reset(self):
        self._num_steps.assign(tf.zeros_like(self._num_steps))
        return super()._reset()

    def _step(self, action):
        self._num_steps.assign_add(tf.ones_like(self._num_steps))

        time_step = super()._step(action)

        time_limit_terminations = tf.math.greater_equal(self._num_steps, self._duration)
        step_types = tf.where(
            condition=time_limit_terminations, x=StepType.LAST, y=time_step.step_type
        )
        discounts = tf.where(condition=time_limit_terminations, x=0, y=time_step.discount)
        new_time_step = TimeStep(
            step_types, time_step.reward, discounts, time_step.observation
        )
        self._env._time_step = new_time_step  # pylint: disable=protected-access

        # We convert the TF Tensors to numpy first for performance reasons.
        if any(new_time_step.is_last().numpy()):
            terminates = step_types == StepType.LAST
            termination_indexes = tf.where(terminates)
            number_terminations = tf.math.count_nonzero(terminates)
            # we use dtype tf.int32 because this avoids a GPU bug detected by Dongho
            self._num_steps.scatter_nd_update(
                termination_indexes,
                tf.constant(-1, shape=(number_terminations,), dtype=tf.int32),
            )

        return new_time_step
