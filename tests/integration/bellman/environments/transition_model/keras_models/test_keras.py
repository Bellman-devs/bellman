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

import pytest
import tensorflow as tf
from tf_agents.specs import BoundedTensorSpec

from bellman.distributions.utils import create_uniform_distribution_from_spec
from bellman.environments.transition_model.keras_model.keras import (
    KerasTrainingSpec,
    KerasTransitionModel,
)
from bellman.environments.transition_model.keras_model.trajectory_sampling import (
    OneStepTrajectorySampling,
)
from tests.tools.bellman.environments.transition_model.observation_transformation import (
    GoalStateObservationTransformation,
)


@pytest.mark.skip("Stochastic failure to investigate.")
def test_fit_mountain_car_data(
    mountain_car_data, transition_network, bootstrap_data, batch_size, ensemble_size
):
    tf_env, trajectories = mountain_car_data

    network_list = [
        transition_network(tf_env.observation_spec(), bootstrap_data=bootstrap_data)
        for _ in range(ensemble_size)
    ]
    transition_model = KerasTransitionModel(
        network_list,
        tf_env.observation_spec(),
        tf_env.action_spec(),
        predict_state_difference=False,
        trajectory_sampling_strategy=OneStepTrajectorySampling(batch_size, ensemble_size),
    )

    training_spec = KerasTrainingSpec(
        epochs=10,
        training_batch_size=256,
        callbacks=[],
    )

    history = transition_model.train(trajectories, training_spec)

    assert history.history["loss"][-1] < history.history["loss"][0]


def test_step_call_shape(
    transition_network,
    observation_space,
    action_space,
    batch_size,
    ensemble_size,
):
    network_list = [
        transition_network(observation_space, bootstrap_data=True)
        for _ in range(ensemble_size)
    ]
    transition_model = KerasTransitionModel(
        network_list,
        observation_space,
        action_space,
        predict_state_difference=True,
        trajectory_sampling_strategy=OneStepTrajectorySampling(batch_size, ensemble_size),
    )
    observation_distribution = create_uniform_distribution_from_spec(observation_space)
    observations = observation_distribution.sample((batch_size,))
    action_distribution = create_uniform_distribution_from_spec(action_space)
    actions = action_distribution.sample((batch_size,))

    next_observations = transition_model.step(observations, actions)

    assert next_observations.shape == (batch_size,) + observation_space.shape
    assert observation_space.is_compatible_with(next_observations[0])


def test_step_call_goal_state_transform(
    transition_network,
    observation_space_latent_obs,
    action_space_latent_obs,
    batch_size,
    ensemble_size,
):
    latent_observation_space_spec = BoundedTensorSpec(
        shape=observation_space_latent_obs.shape[:-1]
        + [observation_space_latent_obs.shape[-1] - 1],
        dtype=observation_space_latent_obs.dtype,
        minimum=observation_space_latent_obs.minimum,
        maximum=observation_space_latent_obs.maximum,
        name=observation_space_latent_obs.name,
    )
    network_list = [
        transition_network(latent_observation_space_spec, bootstrap_data=True)
        for _ in range(ensemble_size)
    ]
    observation_transformation = GoalStateObservationTransformation(
        latent_observation_space_spec=latent_observation_space_spec,
        goal_state_start_index=-1,
    )
    transition_model = KerasTransitionModel(
        network_list,
        observation_space_latent_obs,
        action_space_latent_obs,
        predict_state_difference=True,
        trajectory_sampling_strategy=OneStepTrajectorySampling(batch_size, ensemble_size),
        observation_transformation=observation_transformation,
    )
    observation_distribution = create_uniform_distribution_from_spec(
        observation_space_latent_obs
    )
    observations = observation_distribution.sample((batch_size,))
    action_distribution = create_uniform_distribution_from_spec(action_space_latent_obs)
    actions = action_distribution.sample((batch_size,))

    next_observations = transition_model.step(observations, actions)

    assert next_observations.shape == (batch_size,) + observation_space_latent_obs.shape
    assert observation_space_latent_obs.is_compatible_with(next_observations[0])
    tf.assert_equal(next_observations[..., -1], observations[..., -1])
