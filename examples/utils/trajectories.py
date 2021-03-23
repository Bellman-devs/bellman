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
This module contains utility functions for working with trajectories.
"""

from bellman.distributions.utils import create_uniform_distribution_from_spec
from bellman.environments.reward_model import RewardModel
from bellman.environments.transition_model.transition_model import TransitionModel
from bellman.environments.transition_model.utils import Transition


def sample_uniformly_distributed_transitions(
    model: TransitionModel, number_of_transitions: int, reward_model: RewardModel
) -> Transition:
    """
    Sample `number_of_transitions` transitions from the model. Draw observations and actions from a
    uniform distribution over the respective spaces. Get corresponding rewards from a reward model.
    """

    observation_distribution = create_uniform_distribution_from_spec(
        model.observation_space_spec
    )
    action_distribution = create_uniform_distribution_from_spec(model.action_space_spec)

    observations = observation_distribution.sample((number_of_transitions,))
    actions = action_distribution.sample((number_of_transitions,))
    next_observations = model.step(observations, actions)
    rewards = reward_model.step_reward(observations, actions, next_observations)

    return Transition(
        observation=observations,
        action=actions,
        reward=rewards,
        next_observation=next_observations,
    )
