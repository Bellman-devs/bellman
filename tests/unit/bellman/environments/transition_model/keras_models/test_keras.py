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

from bellman.environments.transition_model.keras_model.keras import KerasTransitionModel
from bellman.environments.transition_model.keras_model.linear import LinearTransitionNetwork
from bellman.environments.transition_model.keras_model.trajectory_sampling import (
    SingleFunction,
)
from bellman.environments.transition_model.keras_model.utils import create_concatenated_inputs


def test_transition_network_output_shape(observation_space, action_space, transition_network):
    concatenated_network_inputs, _ = create_concatenated_inputs(
        [observation_space, action_space], ""
    )
    network = transition_network(observation_space)

    network_output = network.build_model(concatenated_network_inputs)

    assert observation_space.is_compatible_with(network_output[0].type_spec)


def test_mismatch_ensemble_size(
    observation_space, action_space, trajectory_sampling_strategy_factory, batch_size
):
    """
    Ensure that the ensemble size specified in the trajectory sampling strategy is equal to the
    number of networks in the models.
    """
    strategy = trajectory_sampling_strategy_factory(batch_size, 2)
    if isinstance(strategy, SingleFunction):
        pytest.skip("SingleFunction strategy is not an ensemble strategy.")

    with pytest.raises(AssertionError):
        KerasTransitionModel(
            [LinearTransitionNetwork(observation_space)],
            observation_space,
            action_space,
            trajectory_sampling_strategy=strategy,
        )
