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
This module provides the interface for defining the individual networks in the keras ensemble
transition model.
"""

from abc import ABC, abstractmethod

import tensorflow as tf

from bellman.environments.transition_model.utils import Transition


def sample_with_replacement(transition: Transition) -> Transition:
    """
    Create a new `Transition` object with data sampled with replacement from `transition`. This
    function is useful for creating bootstrap samples of data for training ensembles.

    :param transition: A `Transition` object, consisting of states, actions and successor states.

    :return: Return a (new) `Transition` object.
    """
    # transition.observation has shape [batch_size,] + observation_space_spec.shape
    n_rows = transition.observation.shape[0]

    index_tensor = tf.random.uniform(
        (n_rows,), maxval=n_rows, dtype=tf.dtypes.int64
    )  # pylint: disable=all

    observations = tf.gather(transition.observation, index_tensor)  # pylint: disable=all
    actions = tf.gather(transition.action, index_tensor)  # pylint: disable=all
    rewards = tf.gather(transition.reward, index_tensor)  # pylint: disable=all
    next_observations = tf.gather(
        transition.next_observation, index_tensor
    )  # pylint: disable=all

    return Transition(
        observation=observations,
        action=actions,
        reward=rewards,
        next_observation=next_observations,
    )


class KerasTransitionNetwork(ABC):
    """
    This class defines the structure and essential methods for a transition network. The transition
    network is a sequential, feed forward neural network. It also makes it easy to create networks
    where data is bootstrapped for each new network in an ensemble.

    Subclasses of this class should define the structure of the network by implementing the
    `build_model` method. The output layer should be reshaped to match the observation tensors from
    the environment. The loss function of the network should be specified by implementing the
    `loss` method. The training data can be manipulated by overriding the `transform_training_data`
    method with the appropriate transformation.
    """

    def __init__(self, bootstrap_data: bool = False):
        """
        :param bootstrap_data: Create an ensemble version of the network where data is resampled with
                               replacement.
        """
        self._bootstrap_data = bootstrap_data

    @abstractmethod
    def build_model(self, inputs: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        """
        Define the layers of the sequential, feed-forward neural network from the layer
        `input_layer`.

        :param inputs: Input layer.

        :return: outputs layer.
        """
        pass

    @abstractmethod
    def loss(self) -> tf.keras.losses.Loss:
        """
        Allows a custom definition of the loss function, for continuous state-space MDP's this
        would typically be mean squared error.

        :return: Return the loss function for this network.
        """
        pass

    def metrics(self) -> tf.keras.metrics.Metric:
        """
        Defines metrics for monitoring the training of the network. Method should be overwritten
        for custom metrics.

        :return: Return the metrics for this network.
        """
        metrics = [
            tf.keras.metrics.RootMeanSquaredError(name="Transition model RMSE"),
            tf.keras.metrics.MeanAbsoluteError(name="Transition model MAE"),
        ]
        return metrics

    def transform_training_data(self, transition: Transition) -> Transition:
        """
        This network will be trained on the data in `Transition`. This method ensures the training
        data can be transformed before it is used in training. Also, when ensembles are used
        this method can use the `bootstrap_data` flag to use bootstrap samples of the data for
        each model in the ensemble.

        :param transition: A `Transition` object, consisting of states, actions and successor
                           states.

        :return: Return a (new) `Transition` object.
        """
        if self._bootstrap_data:
            return sample_with_replacement(transition)
        else:
            return transition
