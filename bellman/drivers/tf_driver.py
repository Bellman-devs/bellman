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
A similar driver to the TF-Agents `TFDriver`, designed for batches of virtual rollouts. Virtual
rollouts should all exceed a minimum length across the batch, and so this driver reverses the
conditions of the TF-Agents driver.
"""

from math import inf
from typing import Any, Callable, Optional, Sequence, Tuple

import tensorflow as tf
from tf_agents.drivers.driver import Driver
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories.trajectory import Trajectory, Transition, from_transition
from tf_agents.typing.types import Int, NestedTensor, TimeStep
from tf_agents.utils import common


class TFBatchDriver(Driver):
    """
    A driver that runs a TF policy in a TF environment. In contrast to the TF-Agents `TFDriver`,
    this driver will ensure that a minimum number of time steps and/or episodes have been produced
    from each of the elements in the batch.
    """

    def __init__(
        self,
        env: TFEnvironment,
        policy: TFPolicy,
        observers: Sequence[Callable[[Trajectory], Any]],
        transition_observers: Optional[Sequence[Callable[[Transition], Any]]] = None,
        min_steps: Optional[Int] = None,
        min_episodes: Optional[Int] = None,
        disable_tf_function: bool = False,
    ):
        """

        :param env: A TFEnvironment environment.
        :param policy: A TFPolicy policy.
        :param observers: A list of observers that are notified after every step in the environment.
                          Each observer is a callable(trajectory.Trajectory).
        :param transition_observers: A list of observers that are updated after every step in the
                                     environment. Each observer is a callable((TimeStep, PolicyStep,
                                     NextTimeStep)). The transition is shaped just as trajectories
                                     are for regular observers.
        :param min_steps: Optional minimum number of steps for each run() call. For batched or
                          parallel environments, this is the minimum number of steps for each
                          environments. Also see below. Default: inf.
        :param min_episodes: Optional minimum number of episodes for each run() call. For batched or
                             parallel environments, this is the minimum number of episodes for each
                             of the environments. At least one of min_steps or min_episodes must be
                             provided. If both are set, run() terminates when at least one of the
                             conditions is satisfied.  Default: inf.
        :param disable_tf_function: If True the use of tf.function for the run method is disabled.
        """
        assert (min_steps is not None) or (
            min_episodes is not None
        ), "One of min_steps and min_episodes must be specified."
        super().__init__(env, policy, observers, transition_observers)

        self._min_steps = min_steps or inf
        self._min_episodes = min_episodes or inf

        if not disable_tf_function:
            self._run_fn = common.function(self._run, autograph=True)  # type: ignore
        else:
            self._run_fn = self._run

    def _run(  # pylint: disable=W0221
        self, time_step: TimeStep, policy_state: NestedTensor = ()
    ) -> Tuple[TimeStep, NestedTensor]:
        num_steps = tf.zeros(shape=(self._env.batch_size,))
        num_episodes = tf.zeros(shape=(self._env.batch_size,))

        while (
            tf.math.reduce_min(num_steps) < self._min_steps
            and tf.math.reduce_min(num_episodes) < self._min_episodes
        ):
            action_step = self.policy.action(time_step, policy_state)
            next_time_step = self.env.step(action_step.action)

            traj = from_transition(time_step, action_step, next_time_step)
            for observer in self._transition_observers:
                observer((time_step, action_step, next_time_step))
            for observer in self.observers:
                observer(traj)

            num_episodes += tf.cast(traj.is_boundary(), tf.float32)
            num_steps += tf.cast(~traj.is_boundary(), tf.float32)

            time_step = next_time_step
            policy_state = action_step.state

        return time_step, policy_state

    def run(  # pylint: disable=W0221
        self, time_step: TimeStep, policy_state: NestedTensor = ()
    ) -> Tuple[TimeStep, NestedTensor]:
        return self._run_fn(time_step, policy_state)
