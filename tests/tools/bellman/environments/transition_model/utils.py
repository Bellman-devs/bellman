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

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies.random_tf_policy import RandomTFPolicy

from bellman.environments.environment_model import EnvironmentModel
from bellman.environments.initial_state_distribution_model import (
    DeterministicInitialStateModel,
)
from bellman.environments.termination_model import ConstantFalseTermination
from bellman.environments.tf_wrappers import TFTimeLimit
from examples.utils.classic_control import PendulumReward, create_pendulum_environment
from tests.tools.bellman.eval.evaluation_utils import policy_evaluation
from tests.tools.bellman.policies.utils import replay_actions_across_batch_transition_models


def assert_rollouts_are_close_to_actuals(model, max_steps):
    tf_env = tf_py_environment.TFPyEnvironment(create_pendulum_environment(max_steps))
    collect_policy = RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())

    test_trajectory = policy_evaluation(
        tf_env, collect_policy, num_episodes=1, max_buffer_capacity=200, use_function=True
    )

    start_state = test_trajectory.observation[0, 0, :]

    env_model = TFTimeLimit(
        EnvironmentModel(
            model,
            PendulumReward(tf_env.observation_spec(), tf_env.action_spec()),
            ConstantFalseTermination(tf_env.observation_spec()),
            DeterministicInitialStateModel(start_state),
            batch_size=30,
        ),
        max_steps + 1,
    )

    replayed_trajectories = replay_actions_across_batch_transition_models(
        env_model, test_trajectory.action[0]
    )

    prediction_mean = tf.reduce_mean(replayed_trajectories.observation, axis=0)
    np.testing.assert_allclose(
        prediction_mean, test_trajectory.observation[0], atol=1e-1, rtol=2e-1
    )
