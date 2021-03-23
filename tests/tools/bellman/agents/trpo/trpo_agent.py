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
from tf_agents.agents.ppo.ppo_agent_test import DummyActorNet, DummyValueNet
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.trajectory import Trajectory

from bellman.agents.trpo.trpo_agent import TRPOAgent
from tests.tools.bellman.specs.tensor_spec import ACTION_SPEC, OBSERVATION_SPEC, TIMESTEP_SPEC


def _dummy_actor_net():
    return DummyActorNet(OBSERVATION_SPEC, ACTION_SPEC)


def _dummy_value_net():
    return DummyValueNet(OBSERVATION_SPEC)


def create_trpo_agent_factory():
    default_args = {
        "discount_factor": 0.99,
        "lambda_value": 0.5,
        "max_kl": 0.01,
        "cg_iters": 10,
        "log_prob_clipping": 0.0,
        "value_train_iters": 80,
        "gradient_clipping": None,
        "debug": False,
        "train_step_counter": None,
    }

    def _create_agent(**kwargs):
        """agenet factory"""
        agent_args = default_args.copy()
        agent_args.update(kwargs)
        return TRPOAgent(
            TIMESTEP_SPEC,
            ACTION_SPEC,
            actor_net=_dummy_actor_net(),
            value_net=_dummy_value_net(),
            **agent_args,
        )

    return _create_agent


def dummy_trajectory_batch(batch_size=2, n_steps=5, obs_dim=2):
    observations = tf.reshape(
        tf.constant(np.arange(batch_size * n_steps * obs_dim), dtype=tf.float32),
        (batch_size, n_steps, obs_dim),
    )

    time_steps = TimeStep(
        step_type=tf.constant([[1] * (n_steps - 2) + [2] * 2] * batch_size, dtype=tf.int32),
        reward=tf.constant([[1] * n_steps] * batch_size, dtype=tf.float32),
        discount=tf.constant([[1.0] * n_steps] * batch_size, dtype=tf.float32),
        observation=observations,
    )
    actions = tf.ones((batch_size, n_steps, 1), dtype=tf.float32)

    action_distribution_parameters = {
        "dist_params": {
            "loc": tf.constant([[[10.0]] * n_steps] * batch_size, dtype=tf.float32),
            "scale": tf.constant([[[10.0]] * n_steps] * batch_size, dtype=tf.float32),
        },
        "value_prediction": tf.constant([[0.0] * n_steps] * batch_size, dtype=tf.float32),
    }

    policy_info = action_distribution_parameters

    return Trajectory(
        time_steps.step_type,
        observations,
        actions,
        policy_info,
        time_steps.step_type,
        time_steps.reward,
        time_steps.discount,
    )
