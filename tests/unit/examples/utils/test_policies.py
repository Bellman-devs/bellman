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

from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.trajectories.time_step import time_step_spec

from examples.utils.policies import sample_uniformly_distributed_observations_and_get_actions


def test_get_batch_of_actions(observation_space, action_space, batch_size):
    policy = RandomTFPolicy(time_step_spec(observation_space), action_space)

    samples = sample_uniformly_distributed_observations_and_get_actions(policy, batch_size)

    for i in range(batch_size):
        assert action_space.is_compatible_with(samples["actions"][i, ...])
