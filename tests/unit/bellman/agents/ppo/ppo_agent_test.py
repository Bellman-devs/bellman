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
This module contains tests of the PPOAgent from TF-Agents. These tests are closely
modelled on the unit tests in TF-Agents.
"""

import tensorflow as tf
from absl.testing import parameterized
from tf_agents.agents.ppo import ppo_agent
from tf_agents.agents.ppo.ppo_agent_test import DummyActorNet, DummyValueNet
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common, test_utils


class PPOAgentTest(parameterized.TestCase, test_utils.TestCase):
    def setUp(self):
        super(PPOAgentTest, self).setUp()
        tf.compat.v1.enable_resource_variables()
        self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
        self._time_step_spec = ts.time_step_spec(self._obs_spec)
        self._action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, -1, 1)

    @parameterized.named_parameters(
        [
            ("OneEpoch", 1),
            ("FiveEpochs", 5),
        ]
    )
    def testStepCounterBug(self, num_epochs):
        """
        This test checks whether the PPO step counter bug is still present in TF-Agents:
        https://github.com/tensorflow/agents/issues/357

        The `train_step_counter` is supposed to be incremented once per call to `train`, however
        the PPO train method actually has an inner loop of length `num_epochs`, and increments the
        counter in that inner loop. As such the counter in incremented by `num_epochs` each time.

        If this test fails with a new version of TF-Agents, we should update out code which works
        around this issue.
        """
        # Mock the build_train_op to return an op for incrementing this counter.
        counter = common.create_variable("test_train_counter")
        agent = ppo_agent.PPOAgent(
            self._time_step_spec,
            self._action_spec,
            tf.compat.v1.train.AdamOptimizer(),
            actor_net=DummyActorNet(
                self._obs_spec,
                self._action_spec,
            ),
            value_net=DummyValueNet(self._obs_spec),
            normalize_observations=False,
            num_epochs=num_epochs,
            train_step_counter=counter,
        )
        observations = tf.constant(
            [
                [[1, 2], [3, 4], [5, 6]],
                [[1, 2], [3, 4], [5, 6]],
            ],
            dtype=tf.float32,
        )

        time_steps = ts.TimeStep(
            step_type=tf.constant([[1] * 3] * 2, dtype=tf.int32),
            reward=tf.constant([[1] * 3] * 2, dtype=tf.float32),
            discount=tf.constant([[1] * 3] * 2, dtype=tf.float32),
            observation=observations,
        )
        actions = tf.constant([[[0], [1], [1]], [[0], [1], [1]]], dtype=tf.float32)

        action_distribution_parameters = {
            "dist_params": {
                "loc": tf.constant([[[0.0]] * 3] * 2, dtype=tf.float32),
                "scale": tf.constant([[[1.0]] * 3] * 2, dtype=tf.float32),
            }
        }

        policy_info = action_distribution_parameters

        experience = trajectory.Trajectory(
            time_steps.step_type,
            observations,
            actions,
            policy_info,
            time_steps.step_type,
            time_steps.reward,
            time_steps.discount,
        )

        if tf.executing_eagerly():
            loss = lambda: agent.train(experience)
        else:
            loss = agent.train(experience)

        # Assert that counter starts out at zero.
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertEqual(0, self.evaluate(counter))
        self.evaluate(loss)
        # Assert that train_op ran increment_counter num_epochs times.
        self.assertEqual(num_epochs, self.evaluate(counter))
