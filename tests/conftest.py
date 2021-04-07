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

from warnings import warn

import gin
import gym
import numpy as np
import pytest
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.environments.gym_wrapper import spec_from_gym_space
from tf_agents.specs import tensor_spec

from bellman.environments.transition_model.keras_model.linear import LinearTransitionNetwork
from bellman.environments.transition_model.keras_model.multilayer import (
    MultilayerFcTransitionNetwork,
)
from bellman.environments.transition_model.keras_model.probabilistic import (
    DiagonalGaussianTransitionNetwork,
    GaussianTransitionNetwork,
)
from bellman.environments.transition_model.keras_model.trajectory_sampling import (
    InfiniteHorizonTrajectorySampling,
    MeanTrajectorySamplingStrategy,
    OneStepTrajectorySampling,
    SingleFunction,
)
from bellman.trajectory_optimisers.cross_entropy_method import (
    cross_entropy_method_trajectory_optimisation,
)
from bellman.trajectory_optimisers.random_shooting import (
    random_shooting_trajectory_optimisation,
)


@pytest.fixture(name="gym_space_bound", params=[1.0, 2.5], scope="session")
def _gym_space_bound_fixture(request):
    return request.param


@pytest.fixture(name="gym_space_shape", params=[tuple(), (1,), (5, 10)], scope="session")
def _gym_space_shape_fixture(request):
    return request.param


@pytest.fixture(name="gym_space_shape_latent_obs", params=[(2, 3), (5, 10)], scope="session")
def _gym_space_shape_latent_obs_fixture(request):
    return request.param


@pytest.fixture(name="observation_space", scope="session")
def _observation_space_fixture(gym_space_bound, gym_space_shape):
    gym_space = gym.spaces.Box(
        low=-gym_space_bound, high=gym_space_bound, shape=gym_space_shape, dtype=np.float32
    )
    return tensor_spec.from_spec(spec_from_gym_space(gym_space, name="observation"))


@pytest.fixture(name="action_space", scope="session")
def _action_space_fixture(gym_space_bound, gym_space_shape):
    gym_space = gym.spaces.Box(
        low=-gym_space_bound, high=gym_space_bound, shape=gym_space_shape, dtype=np.float32
    )
    return tensor_spec.from_spec(spec_from_gym_space(gym_space, name="action"))


@pytest.fixture(name="observation_space_latent_obs", scope="session")
def _observation_space_latent_obs_fixture(gym_space_bound, gym_space_shape_latent_obs):
    gym_space = gym.spaces.Box(
        low=-gym_space_bound,
        high=gym_space_bound,
        shape=gym_space_shape_latent_obs,
        dtype=np.float32,
    )
    return tensor_spec.from_spec(spec_from_gym_space(gym_space, name="observation"))


@pytest.fixture(name="action_space_latent_obs", scope="session")
def _action_space_latent_obs_fixture(gym_space_bound, gym_space_shape_latent_obs):
    gym_space = gym.spaces.Box(
        low=-gym_space_bound,
        high=gym_space_bound,
        shape=gym_space_shape_latent_obs,
        dtype=np.float32,
    )
    return tensor_spec.from_spec(spec_from_gym_space(gym_space, name="action"))


@pytest.fixture(name="batch_size", params=[1, 5])
def _batch_size_fixture(request):
    return request.param


@pytest.fixture(name="mountain_car_environment", scope="session")
def _environment_fixture():
    gym_env = suite_gym.load("MountainCar-v0")
    tf_env = tf_py_environment.TFPyEnvironment(gym_env)

    yield tf_env

    tf_env.close()


@pytest.fixture(name="trajectory_length", params=[1, 50])
def _trajectory_length_fixture(request):
    return request.param


def _create_random_shooting_trajectory_optimiser(
    time_step_space, action_space, horizon, population_size, number_of_particles
):
    return random_shooting_trajectory_optimisation(
        time_step_space, action_space, horizon, population_size, number_of_particles
    )


def _create_cross_entropy_trajectory_optimiser(
    time_step_space, action_space, horizon, population_size, number_of_particles
):
    return cross_entropy_method_trajectory_optimisation(
        time_step_space,
        action_space,
        horizon,
        population_size,
        number_of_particles,
        num_elites=5,
        learning_rate=0.5,
        max_iterations=5,
    )


@pytest.fixture(
    name="optimiser_policy_trajectory_optimiser_factory",
    params=[
        _create_random_shooting_trajectory_optimiser,
        _create_cross_entropy_trajectory_optimiser,
    ],
)
def _optimiser_fixture(request):
    return request.param


@pytest.fixture(name="ensemble_size", params=[1, 10])
def _ensemble_size_fixture(request):
    return request.param


TRAJECTORY_SAMPLING_STRATEGY_FACTORIES = [
    OneStepTrajectorySampling,
    InfiniteHorizonTrajectorySampling,
    lambda _, e: MeanTrajectorySamplingStrategy(e),
    lambda _1, _2: SingleFunction(),
]


@pytest.fixture(
    name="trajectory_sampling_strategy_factory", params=TRAJECTORY_SAMPLING_STRATEGY_FACTORIES
)
def _trajectory_sampling_strategy_factory_fixture(request):
    return request.param


_TRANSITION_NETWORK_CLASSES = [
    LinearTransitionNetwork,
    DiagonalGaussianTransitionNetwork,
    GaussianTransitionNetwork,
    MultilayerFcTransitionNetwork,
]


@pytest.fixture(name="transition_network", params=_TRANSITION_NETWORK_CLASSES)
def _transition_network_fixture(request):
    return request.param


@pytest.fixture(name="bootstrap_data", params=[False, True])
def _bootstrap_data_fixture(request):
    return request.param


@pytest.fixture(name="predict_state_difference", params=[False, True])
def _predict_state_difference_fixture(request):
    return request.param


@pytest.fixture(name="population_size", params=[1, 2, 10])
def _population_size_fixture(request):
    return request.param


@pytest.fixture(name="number_of_particles", params=[1, 2, 10])
def _number_of_particles_fixture(request):
    return request.param


#
# Copied from the pytest documentation
#
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="only run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # only run slow tests
        skip_non_slow = pytest.mark.skip(reason="running slow tests")
        for item in items:
            if "slow" not in item.keywords:
                item.add_marker(skip_non_slow)
    else:
        # only run quick tests
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


#
# End of copied block
#


@pytest.fixture(autouse=True)
def _auto_clear_gin_config_fixture():
    yield

    if gin.operative_config_str():
        warn(f"Automatically clearing gin config.")
        gin.clear_config()
