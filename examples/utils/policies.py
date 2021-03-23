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
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories.time_step import transition

from bellman.distributions.utils import create_uniform_distribution_from_spec


def sample_uniformly_distributed_observations_and_get_actions(
    policy: TFPolicy, number_of_samples: int
):
    """
    Sample observations from a uniform distribution over the space of observations, and then get
    corresponding actions from the policy.

    :param policy: A policy, instance of `TFPolicy`, from which observations and actions are
                   sampled.
    :param number_of_samples: Number of observation action pairs that will be sampled.

    :return: Dictionary (`dict`) consisting of 'observations' and 'actions'.
    """
    observation_distribution = create_uniform_distribution_from_spec(
        policy.time_step_spec.observation
    )

    observations = observation_distribution.sample((number_of_samples,))
    rewards = tf.zeros((number_of_samples,), dtype=tf.float32)

    time_step = transition(observations, rewards)

    actions = policy.action(time_step).action

    return {"observations": observations, "actions": actions}
