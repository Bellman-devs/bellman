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

import matplotlib.pyplot as plt
import tensorflow as tf


def plot_open_loop_trajectories(
    rollouts: tf.Tensor, title: str = "", real_trajectory: tf.Tensor = None
):
    """
    Produce a compound plot which shows the rolled out trajectories, split by dimension and
    optionally compared to the ground truth.

    :param rollouts: the `observation` tensor from a `Trajectory` object produced by rolling out
                     the environment model.
    :param title: string to use as the title of the plot.
    :param real_trajectory: a single element of the batch `observation` tensor, produced from the
                            true environment.

    :return: A plot
    """
    n_dims = rollouts.shape[-1]

    _, axes = plt.subplots(n_dims, 1, sharex=True, figsize=(10, 12))
    axes[0].set_title(title)
    plt.xlabel("timestep")
    for d in range(n_dims):
        if real_trajectory is not None:
            axes[d].plot(real_trajectory[:, d], linewidth=3, linestyle=":", color="black")
        axes[d].set_ylabel("dimension %d" % d)
        axes[d].plot(rollouts[:, :, d].numpy().T)
