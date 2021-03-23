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

from bellman.environments.transition_model.keras_model.keras import (
    KerasTrainingSpec,
    KerasTransitionModel,
)
from bellman.environments.transition_model.keras_model.linear import LinearTransitionNetwork
from bellman.environments.transition_model.keras_model.trajectory_sampling import (
    InfiniteHorizonTrajectorySampling,
    OneStepTrajectorySampling,
)
from tests.tools.bellman.environments.transition_model.utils import (
    assert_rollouts_are_close_to_actuals,
)

_ENSEMBLE_SIZE = 3


def _test_ensemble_model_close_to_actuals(trajectories, tf_env, trajectory_sampling_strategy):
    keras_transition_networks = [
        LinearTransitionNetwork(tf_env.observation_spec(), True) for _ in range(_ENSEMBLE_SIZE)
    ]
    model = KerasTransitionModel(
        keras_transition_networks,
        tf_env.observation_spec(),
        tf_env.action_spec(),
        predict_state_difference=False,
        trajectory_sampling_strategy=trajectory_sampling_strategy,
    )

    training_spec = KerasTrainingSpec(
        epochs=1000,
        training_batch_size=256,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=20)],
    )

    model.train(trajectories, training_spec)

    assert_rollouts_are_close_to_actuals(model, max_steps=1)


def test_ensemble_model_one_step_resampling_close_to_actuals(pendulum_training_data):
    trajectories, tf_env = pendulum_training_data

    _test_ensemble_model_close_to_actuals(
        trajectories, tf_env, OneStepTrajectorySampling(30, _ENSEMBLE_SIZE)
    )


def test_ensemble_model_infinite_horizon_close_to_actuals(pendulum_training_data):
    trajectories, tf_env = pendulum_training_data

    _test_ensemble_model_close_to_actuals(
        trajectories, tf_env, InfiniteHorizonTrajectorySampling(30, _ENSEMBLE_SIZE)
    )
