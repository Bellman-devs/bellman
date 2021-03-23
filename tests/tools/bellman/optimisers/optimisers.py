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

from tf_agents.agents.tf_agent import LossInfo
from tf_agents.policies.random_tf_policy import RandomTFPolicy

from bellman.trajectory_optimisers.trajectory_optimisers import TrajectoryOptimiser


def create_mock_trajectory_optimiser(mocker, population_size):
    mock_trajectory_optimiser = mocker.MagicMock(spec=TrajectoryOptimiser)
    mock_trajectory_optimiser.population_size = mocker.MagicMock(return_value=population_size)

    return mock_trajectory_optimiser


def create_mock_model_free_agent(mocker, time_step_spec, action_spec, model_free_agent_class):
    model_free_agent = mocker.MagicMock(spec=model_free_agent_class)
    model_free_agent.policy = RandomTFPolicy(time_step_spec, action_spec)
    model_free_agent.collect_policy = model_free_agent.policy
    model_free_agent.time_step_spec = time_step_spec
    model_free_agent.action_spec = action_spec
    model_free_agent.train.return_value = mocker.MagicMock(spec=LossInfo)

    return model_free_agent
