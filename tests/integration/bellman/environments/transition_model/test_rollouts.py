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
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy

from bellman.distributions.utils import create_uniform_distribution_from_spec
from bellman.environments.environment_model import EnvironmentModel
from bellman.environments.initial_state_distribution_model import (
    create_uniform_initial_state_distribution,
)
from bellman.environments.termination_model import ConstantFalseTermination
from bellman.environments.transition_model.keras_model.keras import KerasTransitionModel
from bellman.environments.transition_model.keras_model.linear import LinearTransitionNetwork
from tests.tools.bellman.environments.reward_model import ConstantReward


class _RecordLastLogProbTransitionObserver:
    def __init__(self):
        self._action = None
        self._last_log_prob = None

    @property
    def action(self):
        return self._action

    @property
    def last_log_probability(self):
        return self._last_log_prob

    def __call__(self, transition):
        _, policy_step, _ = transition

        self._action = policy_step.action
        self._last_log_prob = policy_step.info.log_probability


def test_random_shooting_with_dynamic_step_driver(observation_space, action_space):
    """
    This test uses the environment wrapper as an adapter so that a driver from TF-Agents can be used
    to generate a rollout. This also serves as an example of how to construct "random shooting"
    rollouts from an environment model.

    The assertion in this test is that selected action has the expected log_prob value consistent
    with optimisers from a uniform distribution. All this is really checking is that the preceeding
    code has run successfully.
    """

    network = LinearTransitionNetwork(observation_space)
    environment = KerasTransitionModel([network], observation_space, action_space)
    wrapped_environment = EnvironmentModel(
        environment,
        ConstantReward(observation_space, action_space, 0.0),
        ConstantFalseTermination(observation_space),
        create_uniform_initial_state_distribution(observation_space),
    )

    random_policy = RandomTFPolicy(
        wrapped_environment.time_step_spec(), action_space, emit_log_probability=True
    )

    transition_observer = _RecordLastLogProbTransitionObserver()

    driver = DynamicStepDriver(
        env=wrapped_environment,
        policy=random_policy,
        transition_observers=[transition_observer],
    )
    driver.run()

    last_log_prob = transition_observer.last_log_probability

    uniform_distribution = create_uniform_distribution_from_spec(action_space)
    action_log_prob = uniform_distribution.log_prob(transition_observer.action)
    expected = np.sum(action_log_prob.numpy().astype(np.float32))
    actual = np.sum(last_log_prob.numpy())

    np.testing.assert_array_almost_equal(actual, expected, decimal=4)
