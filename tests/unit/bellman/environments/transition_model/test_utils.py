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

import numpy as np
import pytest
from tf_agents.specs import TensorSpec
from tf_agents.utils.nest_utils import is_batched_nested_tensors

from bellman.environments.reward_model import RewardSpec
from bellman.environments.transition_model.utils import (
    extract_transitions_from_trajectories,
    size,
)
from tests.tools.bellman.trajectories.trajectory import generate_dummy_trajectories


def test_extract_transitions_from_trajectories(
    observation_space, action_space, batch_size, trajectory_length, predict_state_difference
):
    trajectories = generate_dummy_trajectories(
        observation_space, action_space, batch_size, trajectory_length
    )
    transitions = extract_transitions_from_trajectories(
        trajectories, observation_space, action_space, predict_state_difference
    )

    observation = transitions.observation
    action = transitions.action
    reward = transitions.reward
    next_observation = transitions.next_observation

    assert is_batched_nested_tensors(
        tensors=[observation, action, reward, next_observation],
        specs=[observation_space, action_space, RewardSpec, observation_space],
    )

    assert (
        observation.shape[0]
        == action.shape[0]
        == reward.shape[0]
        == next_observation.shape[0]
        == (batch_size * (trajectory_length - 1))
    )


@pytest.mark.parametrize("n_dims", list(range(10)))
def test_size(n_dims):
    shape = np.random.randint(1, 10, (n_dims,))

    tensor = np.random.randint(0, 1, shape)
    tensor_spec = TensorSpec(shape)

    assert size(tensor_spec) == np.size(tensor)
