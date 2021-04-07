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
This module contains helper functions which are used for visualisation of MBRL over the mountain car
environment.
"""

import matplotlib.pyplot as plt
import numpy as np

# LEFT: blue
# NOOP: red
# RIGHT: green
COLOUR_MAP = {0: "b", 1: "r", 2: "g"}


def _mountain_car_bounds(plot):
    plot.xlim(-1.2, 0.6)
    plot.ylim(-0.07, 0.07)


def plot_mountain_car_transitions(
    observations: np.ndarray, actions: np.ndarray, next_observations: np.ndarray
):
    plt.figure()
    for i in range(observations.shape[0]):
        start_coords = observations[i]
        action = actions[i]
        end_coords = next_observations[i]

        if action.dtype == np.float32:
            colour_index = 2 if action > 0 else 0
            colour = COLOUR_MAP[colour_index]
        else:
            colour = COLOUR_MAP[action]

        plt.arrow(
            start_coords[0],
            start_coords[1],
            end_coords[0] - start_coords[0],
            end_coords[1] - start_coords[1],
            color=colour,
        )
    _mountain_car_bounds(plt)
    plt.show()


def plot_mountain_car_policy_decisions(observations: np.ndarray, actions: np.ndarray):
    plt.figure()

    colours = map(lambda action: COLOUR_MAP[action], actions)
    plt.scatter(x=observations[:, 0], y=observations[:, 1], c=list(colours))

    _mountain_car_bounds(plt)
    plt.show()
