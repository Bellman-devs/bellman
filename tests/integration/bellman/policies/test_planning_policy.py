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

from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.random_tf_policy import RandomTFPolicy

from bellman.environments.environment_model import EnvironmentModel
from bellman.environments.transition_model.keras_model.keras import KerasTransitionModel
from bellman.environments.transition_model.keras_model.linear import LinearTransitionNetwork
from bellman.environments.transition_model.keras_model.trajectory_sampling import (
    OneStepTrajectorySampling,
)
from bellman.policies.planning_policy import PlanningPolicy
from bellman.trajectory_optimisers.random_shooting import (
    random_shooting_trajectory_optimisation,
)
from bellman.trajectory_optimisers.trajectory_optimisers import PolicyTrajectoryOptimiser
from examples.utils.classic_control import (
    MountainCarInitialState,
    MountainCarReward,
    MountainCarTermination,
)


def test_planning_policy_batch_environment_model():
    """
    Ensure that planning policy is operational.
    """

    # number of trajectories for planning and planning horizon
    population_size = 3
    planner_horizon = 5
    number_of_particles = 1

    # setup the environment and a model of it
    py_env = suite_gym.load("MountainCar-v0")
    tf_env = TFPyEnvironment(py_env)
    reward = MountainCarReward(tf_env.observation_spec(), tf_env.action_spec())
    terminates = MountainCarTermination(tf_env.observation_spec())
    network = LinearTransitionNetwork(tf_env.observation_spec())
    transition_model = KerasTransitionModel(
        [network],
        tf_env.observation_spec(),
        tf_env.action_spec(),
    )
    initial_state = MountainCarInitialState(tf_env.observation_spec())
    environment_model = EnvironmentModel(
        transition_model=transition_model,
        reward_model=reward,
        termination_model=terminates,
        initial_state_distribution_model=initial_state,
    )

    # setup the trajectory optimiser
    random_policy = RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())
    trajectory_optimiser = PolicyTrajectoryOptimiser(
        random_policy, planner_horizon, population_size, number_of_particles
    )
    planning_policy = PlanningPolicy(environment_model, trajectory_optimiser)

    # test whether it runs
    collect_driver_planning_policy = DynamicEpisodeDriver(
        tf_env, planning_policy, num_episodes=1
    )
    time_step = tf_env.reset()
    collect_driver_planning_policy.run(time_step)
