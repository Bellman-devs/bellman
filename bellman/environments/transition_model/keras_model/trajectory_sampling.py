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
This module provides trajectory sampling strategies for ensemble transition models implemented
using Keras.
"""

from abc import ABC, abstractmethod
from itertools import chain, repeat
from typing import List, Union

import tensorflow as tf
from tf_agents.utils import common

from bellman.environments.mixins import BatchSizeUpdaterMixin


class TrajectorySamplingStrategy(BatchSizeUpdaterMixin):
    """
    Interface for trajectory sampling strategies for ensemble transition models.
    """

    def __init__(self, ensemble_size: int):
        """
        :param ensemble_size: Number of functions in the ensemble.
        """
        self._ensemble_size = ensemble_size

    @property
    def ensemble_size(self) -> int:
        """
        Return the size of the ensemble.
        """
        return self._ensemble_size

    @abstractmethod
    def transform_step_inputs(self, inputs: List[tf.Tensor]) -> List[tf.Tensor]:
        """
        This method should be called on the list of tensors which will be passed to the Keras model
        in the `KerasTransitionModel`. This method transforms the input tensors into a list
        corresponding to the input layers of each function in the ensemble.
        """
        pass

    @abstractmethod
    def _transform_step_outputs(self, outputs: List[tf.Tensor]) -> tf.Tensor:
        """
        This method should be called on the list of tensors which is returned from the Keras model
        in the `KerasTransitionModel`. This method transforms the output tensors from each of the
        functions in the ensemble into a single tensor.
        """
        pass

    def transform_step_outputs(self, outputs: Union[tf.Tensor, List[tf.Tensor]]) -> tf.Tensor:
        """
        This method should be called on the output from the Keras model in the
        `KerasTransitionModel`.
        """
        if isinstance(outputs, tf.Tensor):
            return outputs

        return self._transform_step_outputs(outputs)

    def train_model(self) -> None:
        """
        This method should be implemented by subclasses which need to be take some actions at the
        end of a trial.
        """
        pass

    def update_batch_size(self, batch_size: int) -> None:
        """
        :param batch_size: New value for batch size
        """
        pass


class FunctionSampling(TrajectorySamplingStrategy):
    """
    Base class for trajectory sampling strategies for ensemble transition models.

    Subclasses of this class propagate each element of a batch of states with a function chosen
    from the ensemble.
    """

    def __init__(self, batch_size: int, ensemble_size: int):
        """
        :param batch_size: The batch size expected for the actions and observations.
        :param ensemble_size: Number of functions in the ensemble.
        """
        super().__init__(ensemble_size)

        self._batch_size = batch_size

        self._indices = tf.Variable(
            0, name="TS_bootstrap_indices", dtype=tf.int32, shape=tf.TensorShape(None)
        )

        self._resample_indices()

    def _resample_indices(self):
        self._indices.assign(
            tf.random.uniform(
                shape=(self._batch_size,),
                maxval=self._ensemble_size,
                dtype=tf.int32,  # pylint: disable=all
            )
        )

    def transform_step_inputs(self, inputs: List[tf.Tensor]) -> List[tf.Tensor]:
        """
        This method should be called on the list of tensors which will be passed to the Keras model
        in the `KerasTransitionModel`. This method partitions the input tensors along the batch
        dimension, and returns a list of tensors which should be passed to the Keras model `call`
        method.

        This partition along the batch dimension is to ensure that each element in the batch is
        propagated by exactly one member of the ensemble of neural networks.
        """
        partitioned_inputs = [
            tf.dynamic_partition(tensor, self._indices, self._ensemble_size)
            for tensor in inputs
        ]
        input_tensors = []  # type: List[tf.Tensor]
        for partitions in zip(*partitioned_inputs):
            input_tensors.extend(partitions)

        return input_tensors

    def _transform_step_outputs(self, outputs: List[tf.Tensor]) -> tf.Tensor:
        """
        This method should be called on the list of tensors which is returned from the Keras model
        in the `KerasTransitionModel`. This method stitches the output tensors together along the
        batch dimension in the same order as the input tensors of the `transform_step_inputs`
        method.
        """
        merge_indices = tf.dynamic_partition(
            tf.range(self._batch_size), self._indices, self._ensemble_size
        )
        output = tf.dynamic_stitch(merge_indices, outputs)
        return output

    def train_model(self) -> None:
        """
        This method should be implemented by subclasses which need to change the indices of the
        partitions at the end of a trial.
        """
        pass

    def update_batch_size(self, batch_size: int) -> None:
        """
        :param batch_size: New value for batch size
        """
        self._batch_size = batch_size
        self._resample_indices()


class OneStepTrajectorySampling(FunctionSampling):
    """
    This strategy should be used with transition models that are implemented in terms of an ensemble
    of functions.

    The transition model accepts batches of states. For each element in the batch, at each time
    step, we choose a function from the ensemble to predict the next state.

    Reference::
    This approach to trajectory sampling is referred to as TS1 in::

    Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models
    Chua K., Calandra R., McAllister R., Levine S.
    """

    def transform_step_inputs(self, inputs: List[tf.Tensor]) -> List[tf.Tensor]:
        self._resample_indices()

        return super().transform_step_inputs(inputs)


class InfiniteHorizonTrajectorySampling(FunctionSampling):
    """
    This strategy should be used with transition models that are implemented in terms of an
    ensemble of functions.

    The transition model accepts batches of states. For each element in the batch, for each trial,
    we choose a function from the ensemble to predict the next state.

    Reference::
    This approach to trajectory sampling is referred to as TS∞ in::

    Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models
    Chua K., Calandra R., McAllister R., Levine S.
    """

    def train_model(self) -> None:
        self._resample_indices()


class MeanTrajectorySamplingStrategy(TrajectorySamplingStrategy):
    """
    This strategy should be used with transition models that are implemented in terms of an
    ensemble of functions.

    The transition model accepts batches of states. For each element in the batch, return the mean
    of the predictions of the next state from all of the functions in the ensemble.

    Reference::

    Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models
    Chua K., Calandra R., McAllister R., Levine S.
    """

    def transform_step_inputs(self, inputs: List[tf.Tensor]) -> List[tf.Tensor]:
        return list(chain.from_iterable(repeat(inputs, self._ensemble_size)))

    def _transform_step_outputs(self, outputs: List[tf.Tensor]) -> tf.Tensor:
        return tf.reduce_mean(tf.stack(outputs, axis=0), axis=0)


class SingleFunction(TrajectorySamplingStrategy):
    """
    This strategy should be used with transition models that are implemented in terms of a single
     function.
    """

    def __init__(self):
        super().__init__(1)

    def transform_step_inputs(self, inputs: List[tf.Tensor]) -> List[tf.Tensor]:
        """
        Pass through the list of input tensors. There is only one function, so there is no need to
        modify or duplicate these tensors.
        """
        return inputs

    def _transform_step_outputs(self, outputs: List[tf.Tensor]) -> tf.Tensor:
        """
        Return the first element of the list. There is only one function, so the list should only
        contain a single output tensor.
        """
        assert len(outputs) == 1

        return outputs[0]
