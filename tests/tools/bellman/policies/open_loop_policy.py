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
This module provides an open loop policy.
"""
from typing import Optional

import tensorflow as tf
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common, nest_utils


class TFOpenLoopPolicy(TFPolicy):
    """
    This policy chooses actions by replaying a sequence of actions from a `Trajectory`.
    """

    def __init__(self, time_step_spec, action_spec, actions: tf.Tensor):
        super().__init__(time_step_spec, action_spec)

        tf.ensure_shape(actions, [None] + action_spec.shape)  # [time_step, features...]
        self._actions = actions

        self._action_index = tf.constant(0, shape=())
        self._next_action = common.create_variable(
            "next_action",
            initial_value=self._actions[self._action_index],
            shape=action_spec.shape,
            dtype=action_spec.dtype,
        )

    def _action(
        self,
        time_step: ts.TimeStep,
        policy_state: types.NestedTensor,
        seed: Optional[types.Seed] = None,
    ) -> policy_step.PolicyStep:
        outer_shape = nest_utils.get_outer_shape(time_step, self._time_step_spec)
        action = common.replicate(self._next_action, outer_shape)

        self._action_index += 1
        self._action_index %= self._actions.shape[0]
        self._next_action.assign(self._actions[self._action_index])

        return policy_step.PolicyStep(action, policy_state, info=())

    def _distribution(self, time_step, policy_state):
        raise NotImplementedError()
