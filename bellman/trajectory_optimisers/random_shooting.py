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
This module defines a simple uniform random shooting method which can be used to optimise
trajectories. One example implementation is described in:

Nagabandi, A., Kahn, G., Fearing, R. S., & Levine, S. (2018). Neural network dynamics for
model-based deep reinforcement learning with model-free fine-tuning. In 2018 IEEE International
Conference on Robotics and Automation (ICRA) (pp. 7559-7566).
"""

from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.typing import types

from bellman.environments.environment_model import EnvironmentModel
from bellman.trajectory_optimisers.trajectory_optimisers import (
    PolicyTrajectoryOptimiser,
    TrajectoryOptimiser,
)


def random_shooting_trajectory_optimisation(
    time_step_spec: TimeStep,
    action_spec: types.NestedTensorSpec,
    horizon: int,
    population_size: int,
    number_of_particles: int,
) -> TrajectoryOptimiser:
    """
    Construct a trajectory optimiser which uses the random shooting method. This method relies
    on `RandomTFPolicy` as a uniformly random policy.

    :param time_step_spec: A `TimeStep` spec of the expected time_steps.
    :param action_spec: A nest of BoundedTensorSpec representing the actions.
    :param horizon: Number of steps taken in the environment in each virtual rollout.
    :param population_size: The number of candidate sequences of actions at each iteration.
    :param number_of_particles: Number of monte-carlo rollouts of each action trajectory.

    :return: A `TrajectoryOptimiser` object which uses the random shooting method.
    """
    policy = RandomTFPolicy(time_step_spec, action_spec)
    trajectory_optimiser = PolicyTrajectoryOptimiser(
        policy, horizon, population_size, number_of_particles
    )
    return trajectory_optimiser
