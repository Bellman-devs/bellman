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
This module provides a policy which can be used with the `PolicyTrajectoryOptimiser` to optimise
trajectories using the cross entropy method.

To use the cross entropy method for trajectory optimisation, create an optimiser with the helper
function `cross_entropy_method_trajectory_optimisation` in the `trajectory_optimisation` package.
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.typing import types
from tf_agents.typing.types import TimeStep


class CrossEntropyMethodPolicy(tf_policy.TFPolicy):
    """
    A cross entropy policy for use with trajectory optimisation.
    """

    def __init__(
        self,
        time_step_spec: TimeStep,
        action_spec: types.NestedTensorSpec,
        horizon: int,
        population_size: int,
    ):
        """
        Initializes the policy.

        :param time_step_spec: A `TimeStep` spec of the expected time_steps.
        :param action_spec: A nest of BoundedTensorSpec representing the actions.
        :param horizon: Number of steps taken in the environment in each virtual rollout.
        :param population_size: The number of candidate sequences of actions at each iteration.
        """
        assert (
            not action_spec.dtype.is_integer
        ), "CrossEntropyMethodPolicy currently does not support discrete actions"

        sample_shape = (horizon + 1,) + action_spec.shape
        super().__init__(
            time_step_spec,
            action_spec,
            policy_state_spec=(
                tf.TensorSpec(sample_shape, action_spec.dtype),  # mean
                tf.TensorSpec(sample_shape, action_spec.dtype),  # variance
                tf.TensorSpec(sample_shape, action_spec.dtype),  # distribution lower bound
                tf.TensorSpec(sample_shape, action_spec.dtype),  # distribution upper bound
                tf.TensorSpec(
                    (population_size,) + sample_shape, action_spec.dtype
                ),  # sampled actions
                tf.TensorSpec((population_size,), tf.int32),  # step index
            ),
            automatic_state_reset=False,  # do not reset the policy state automatically.
        )

        self._low = tf.broadcast_to(tf.convert_to_tensor(action_spec.minimum), sample_shape)
        self._high = tf.broadcast_to(tf.convert_to_tensor(action_spec.maximum), sample_shape)
        self._horizon = horizon
        self._population_size = population_size

    def _distribution(
        self, time_step: TimeStep, policy_state: types.NestedTensorSpec
    ) -> policy_step.PolicyStep:
        """
        This method returns a distribution over actions for a particular step within the planner
        horizon. The cross entropy method attempts to find the optimal set of actions up to the
        planner horizon, by maintaining a distribution over actions for each time step, up to the
        planning horizon.

        This policy uses a batched `step_index` counter to track which time step index has been
        reached in each of the batched trajectories.

        If a trajectory in the batch terminates within the planning horizon the policy has to
        choose an action for the final time step. This action is never used. In this case we
        decrement the step counter before returning the action distribution. As such the action
        distribution for the final time step in a trajectory will be identical to the action
        distribution of the previous time step. This is done because, for each termination time
        step along a trajectory, the total number of time steps in the element of the batch
        increases by one. This is a problem for the cross-entropy policy which optimises "planning
        horizon" action distributions.
        """
        # mean, var, low and high shapes = (horizon + 1,) + action_space.shape
        mean, var, low, high, batch_actions, step_index = policy_state

        assert tf.reduce_all(
            step_index <= self._horizon
        ), f"Max step index {max(step_index)} is out of range (> {self._horizon})"

        actions = tf.gather_nd(batch_actions, step_index[:, None], batch_dims=1)

        distribution = tfp.distributions.Deterministic(actions)

        step_index_increment = tf.where(time_step.is_last(), 0, 1)

        policy_state = tf.nest.pack_sequence_as(
            self._policy_state_spec,
            [mean, var, low, high, batch_actions, step_index + step_index_increment],
        )
        return policy_step.PolicyStep(distribution, policy_state)

    def _get_initial_state(self, batch_size: int) -> types.NestedTensor:
        initial_mean = self._low + (self._high - self._low) / 2.0
        lb_dist, ub_dist = initial_mean - self._low, self._high - initial_mean
        initial_var = tf.minimum(tf.math.square(lb_dist / 2), tf.math.square(ub_dist / 2))
        step_index = tf.zeros((self._population_size,), tf.int32)

        initial_actions = sample_action_batch(
            initial_mean, initial_var, self._low, self._high, self._population_size
        )

        return tf.nest.pack_sequence_as(
            self._policy_state_spec,
            [initial_mean, initial_var, self._low, self._high, initial_actions, step_index],
        )


def sample_action_batch(
    mean: tf.Tensor, var: tf.Tensor, low: tf.Tensor, high: tf.Tensor, batch_size: int
) -> tf.Tensor:
    """
    Sample a batch of trajectories of actions from a truncated normal distribution with parameters
    `mean` and `var`.

    :param mean: The mean of the truncated normal distribution.
    :param var: The variance of the truncated normal distribution.
    :param low: The lower bound on the distribution's support.
    :param high: The upper bound on the distribution's support.
    :param batch_size: The environment batch size.
    :return: A batch of trajectories of actions.
    """
    distribution = tfp.distributions.TruncatedNormal(
        loc=mean,
        scale=tf.math.sqrt(var),
        low=low,
        high=high,
        validate_args=True,
        allow_nan_stats=False,
    )
    return distribution.sample((batch_size,))
