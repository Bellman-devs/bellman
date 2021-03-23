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

from tf_agents.trajectories.time_step import restart, time_step_spec
from tf_agents.utils.nest_utils import get_outer_shape, unstack_nested_tensors

from bellman.distributions.utils import create_uniform_distribution_from_spec
from bellman.policies.planning_policy import PlanningPolicy
from tests.tools.bellman.samplers.samplers import get_optimiser_and_environment_model


def test_planning_policy_action_shape(
    observation_space, action_space, optimiser_policy_trajectory_optimiser_factory
):
    """
    Ensure action shape of the planning policy is correct.
    """
    population_size = 10
    number_of_particles = 1
    horizon = 7
    time_step_space = time_step_spec(observation_space)
    trajectory_optimiser, environment_model = get_optimiser_and_environment_model(
        time_step_space,
        observation_space,
        action_space,
        population_size=population_size,
        number_of_particles=number_of_particles,
        horizon=horizon,
        optimiser_policy_trajectory_optimiser_factory=optimiser_policy_trajectory_optimiser_factory,
    )

    # remember the time step comes from the real environment with batch size 1
    observation = create_uniform_distribution_from_spec(observation_space).sample(
        sample_shape=(1,)
    )
    time_step = restart(observation, batch_size=1)

    planning_policy = PlanningPolicy(environment_model, trajectory_optimiser)

    policy_step = planning_policy.action(time_step)
    action = policy_step.action
    assert get_outer_shape(action, action_space) == (1,)
    assert action_space.is_compatible_with(action[0])
