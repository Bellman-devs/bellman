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
This module provides a policy for decision time planning.
"""

from typing import Optional, Text

import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.typing import types

from bellman.distributions.utils import create_uniform_distribution_from_spec
from bellman.environments.environment_model import EnvironmentModel
from bellman.trajectory_optimisers.trajectory_optimisers import TrajectoryOptimiser


class PlanningPolicy(tf_policy.TFPolicy):
    """
    Planning policies take a model of the environment and use it to sample virtual
    trajectories, which are used to select the optimal action at the current state. Note that
    the computational burden, compared to a parametric policy, is high.

    Note that for simulating multiple rollouts in the environment we use batches as set by a
    `batch_size` parameter. This is in fact the `population_size` parameter common to most
    trajectory optimizers. This class makes sure that `batch_size` is set according to
    `population_size` parameter.
    """

    def __init__(
        self,
        environment_model: EnvironmentModel,
        trajectory_optimiser: TrajectoryOptimiser,
        clip: bool = True,
        emit_log_probability: bool = False,
        automatic_state_reset: bool = True,
        observation_and_action_constraint_splitter: Optional[types.Splitter] = None,
        validate_args: bool = True,
        name: Optional[Text] = None,
    ):
        """
        Initializes the class.

        :param environment_model: An `EnvironmentModel` is a model of the MDP that represents
            the environment, consisting of a transition, reward, termination and initial state
            distribution model, of which some are trainable and some are fixed.
        :param trajectory_optimiser: A `TrajectoryOptimiser` takes an environment model and
            optimises a sequence of actions over a given horizon using virtual rollouts.
        :param clip: Whether to clip actions to spec before returning them. By default True.
        :param emit_log_probability: Emit log-probabilities of actions, if supported. If
            True, policy_step.info will have CommonFields.LOG_PROBABILITY set.
            Please consult utility methods provided in policy_step for setting and
            retrieving these. When working with custom policies, either provide a
            dictionary info_spec or a namedtuple with the field 'log_probability'.
        :param automatic_state_reset:  If `True`, then `get_initial_policy_state` is used
            to clear state in `action()` and `distribution()` for for time steps
            where `time_step.is_first()`.
        :param observation_and_action_constraint_splitter: A function used to process
            observations with action constraints. These constraints can indicate,
            for example, a mask of valid/invalid actions for a given state of the
            environment. The function takes in a full observation and returns a
            tuple consisting of 1) the part of the observation intended as input to
            the network and 2) the constraint. An example
            `observation_and_action_constraint_splitter` could be as simple as: ```
            def observation_and_action_constraint_splitter(observation): return
              observation['network_input'], observation['constraint'] ```
            *Note*: when using `observation_and_action_constraint_splitter`, make
              sure the provided `q_network` is compatible with the network-specific
              half of the output of the
              `observation_and_action_constraint_splitter`. In particular,
              `observation_and_action_constraint_splitter` will be called on the
              observation before passing to the network. If
              `observation_and_action_constraint_splitter` is None, action
              constraints are not applied.
        :param validate_args: Python bool.  Whether to verify inputs to, and outputs of,
            functions like `action` and `distribution` against spec structures,
            dtypes, and shapes. Research code may prefer to set this value to `False`
            to allow iterating on input and output structures without being hamstrung
            by overly rigid checking (at the cost of harder-to-debug errors). See also
            `TFAgent.validate_args`.
        :param name: A name for this module. Defaults to the class name.
        """

        self.trajectory_optimiser = trajectory_optimiser
        self._environment_model = environment_model

        # making sure the batch_size in environment_model is correctly set
        self._environment_model.batch_size = self.trajectory_optimiser.batch_size

        super(PlanningPolicy, self).__init__(
            time_step_spec=environment_model.time_step_spec(),
            action_spec=environment_model.action_spec(),
            policy_state_spec=(),
            info_spec=(),
            clip=clip,
            emit_log_probability=emit_log_probability,
            automatic_state_reset=automatic_state_reset,
            observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
            validate_args=validate_args,
            name=name,
        )

    def _distribution(self, time_step, policy_state):
        # Planning subroutine outputs nested structure of distributions or actions. The
        # trajectory optimizer will return a sequence of actions, one for each virtual planning
        # step. While planners do not currently support batched environments, the TFEnvironment
        # expects a batch dimension with length one in this case.

        # This condition catches initial states for the optimizer that are terminal
        # (TF Agents drivers query policies for these states as well), we return random
        # sample here (NaN cannot work for both continuous and discrete actions), otherwise
        # we call the optimizer.
        actions_or_distributions = tf.cond(
            tf.equal(time_step.is_last(), tf.constant(True)),
            lambda: tf.reshape(
                create_uniform_distribution_from_spec(self.action_spec).sample(),
                1 + self.action_spec.shape,
            ),
            lambda: self.trajectory_optimiser.optimise(time_step, self._environment_model)[
                None, 0
            ],
        )

        def _to_distribution(action_or_distribution):
            if isinstance(action_or_distribution, tf.Tensor):
                # This is an action tensor, so wrap it in a deterministic distribution.
                return tfp.distributions.Deterministic(loc=action_or_distribution)
            return action_or_distribution

        distributions = tf.nest.map_structure(_to_distribution, actions_or_distributions)
        return policy_step.PolicyStep(distributions, policy_state)
