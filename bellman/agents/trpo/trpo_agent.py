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

""" A TRPO agent

Implements TRPO agent as described in:

Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015, June).
Trust region policy optimization. In International conference on machine learning (pp. 1889-1897).

Uses a generalised advantage estimator (GAE) critic to estimate policy gradient advantages.
See following paper for details:

Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015).
High-dimensional continuous control using generalized advantage estimation.
arXiv preprint arXiv:1506.02438.

TRPO performs a policy gradient update while enforcing a maximum KL constraint between the current
policy and the updated policy. It does this by approximating the natural gradient direction for the
policy wrt to parameters and optimising the stepsize in that direction. The natural gradient is
approximated by using the conjugate gradient method to approximately solving the system  "H ng = g".
Here H is the hessian of the KL, ng is the natural gradient and g is the gradient.
"""

import collections
from typing import Optional

import gin
import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.agents.ppo import ppo_utils
from tf_agents.agents.ppo.ppo_policy import PPOPolicy
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.networks import network
from tf_agents.specs import distribution_spec
from tf_agents.trajectories import trajectory
from tf_agents.typing.types import NestedTensor, Tensor
from tf_agents.utils import common, eager_utils, nest_utils, value_ops

from bellman.agents.trpo.utils import (
    conjugate_gradient,
    flatten_tensors,
    hessian_vector_product,
    unflatten_tensor,
)

EPS = 1e-5

TRPOLossInfo = collections.namedtuple(
    "TRPOLossInfo", ("policy_gradient_loss", "value_estimation_loss")
)


# Copied from the TF-Agents 0.6.0 `ppo_agent.py` module.
def _normalize_advantages(advantages, axes=(0,), variance_epsilon=1e-8):
    adv_mean, adv_var = tf.nn.moments(x=advantages, axes=axes, keepdims=True)
    normalized_advantages = (advantages - adv_mean) / (tf.sqrt(adv_var) + variance_epsilon)
    return normalized_advantages


def compute_return_and_advantage(
    discount_factor, lambda_, rewards, next_time_steps, value_preds
):
    """Compute the TD-lambda return and GAE(lambda) advantages.
    Normalization will be applied to the advantages.

    :param discount_factor: discount in [0,1]
    :param lambda_: trace_decay in [0,1]
    :param rewards: next_step rewards (possibly normalized)
    :param next_time_steps: batched tensor of TimeStep tuples after action is taken.
    :param value_preds: Batched value prediction tensor. Should have one more entry
        in time index than time_steps, with the final value corresponding to the
        value prediction of the final state.

    :return: tuple of (return, normalized_advantage), both are batched tensors.
    """
    discounts = next_time_steps.discount * tf.constant(discount_factor, dtype=tf.float32)

    # Make discount 0.0 at end of each episode to restart cumulative sum
    #   end of each episode.
    episode_mask = common.get_episode_mask(next_time_steps)
    discounts *= episode_mask

    # Arg value_preds was appended with final next_step value. Make tensors
    #   next_value_preds by stripping first and last elements respectively.
    final_value_pred = value_preds[:, -1]
    value_preds = value_preds[:, :-1]

    # Compute advantages.
    advantages = value_ops.generalized_advantage_estimation(
        values=value_preds,
        final_value=final_value_pred,
        rewards=rewards,
        discounts=discounts,
        td_lambda=lambda_,
        time_major=False,
    )
    normalized_advantages = _normalize_advantages(advantages, axes=(0, 1))

    # compute TD-Lambda returns.
    returns = tf.add(advantages, value_preds, name="td_lambda_returns")

    return returns, normalized_advantages


@gin.configurable
class TRPOAgent(tf_agent.TFAgent):
    """Create a TRPO agent."""

    def __init__(
        self,
        time_step_spec,
        action_spec,
        actor_net=None,
        value_net=None,
        discount_factor=0.99,
        lambda_value=0.5,
        max_kl=0.01,
        backtrack_coefficient=0.8,
        backtrack_iters=10,
        cg_iters=10,
        reward_normalizer=None,
        reward_norm_clipping=10.0,
        log_prob_clipping=None,
        value_train_iters=80,
        value_optimizer=None,
        gradient_clipping=None,
        debug=False,
        train_step_counter=None,
    ):
        """
        Initializes the agent

        :param time_step_spec:
        :param action_spec:
        :param actor_net: ActorNet implementing policy distribution
        :param value_net: Network approximating value function
        :param discount_factor: discount factor in [0, 1]
        :param lambda_value: trace decay used by the GAE critic in [0, 1]
        :param max_kl: maximum KL distance between updated and old policy
        :param backtrack_coefficient: coefficient used in step size search
        :param backtrack_iters: number of iterations to performa in line search
        :param cg_iters: number of conjugate gradient iterations to approximate natural gradient
        :param reward_normalizer: TensorNormalizer applied to rewards
        :param reward_norm_clipping: value to clip rewards
        :param log_prob_clipping: clip value for log probs in policy gradient , None for no clipping
        :param value_train_iters: number of gradient steps to perform on value estimator
            for every policy update
        :param value_optimizer: optimizer used to train value_function (default: Adam)
        :param gradient_clipping: clip born value gradient (None for no clipping)
        :param debug: debug flag to check computations for Nans
        :param train_step_counter: counter for optimizer
        """
        if not isinstance(actor_net, network.DistributionNetwork):
            raise ValueError("actor_net must be an instance of a DistributionNetwork.")

        self._optimizer = value_optimizer or tf.compat.v1.train.AdamOptimizer()
        self._actor_net = actor_net
        self._value_net = value_net
        self._discount_factor = discount_factor
        self._lambda = lambda_value
        self._backtrack_coeff = backtrack_coefficient
        self._backtrack_iters = backtrack_iters
        self._cg_iters = cg_iters
        self._log_prob_clipping = log_prob_clipping or 0.0
        self._max_kl = max_kl
        self._value_train_iters = value_train_iters
        self._reward_normalizer = reward_normalizer
        self._reward_norm_clipping = reward_norm_clipping
        self._gradient_clipping = gradient_clipping or 0.0

        policy = PPOPolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=actor_net,
            value_network=value_net,
        )

        super(TRPOAgent, self).__init__(
            time_step_spec,
            action_spec,
            policy,
            policy,
            train_sequence_length=None,
            debug_summaries=debug,
            summarize_grads_and_vars=False,
            train_step_counter=train_step_counter,
        )

        self._value_net = value_net
        self._action_distribution_spec = self._actor_net.output_spec

        # copy of the policy used in optimisation
        opt_actor_net = actor_net.copy()
        self._opt_policy = PPOPolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=opt_actor_net,
            value_network=value_net.copy(),
        )
        self._opt_policy_parameters = opt_actor_net.trainable_variables

    def _initialize(self):
        pass

    def policy_gradient_loss(
        self,
        time_steps,
        actions,
        sample_action_log_probs,
        advantages,
        current_policy_distribution,
        weights,
    ):
        """Create tensor for policy gradient loss.

        All tensors should have a single batch dimension.

        Args:
          time_steps: TimeSteps with observations for each timestep.
          actions: Tensor of actions for timesteps, aligned on index.
          sample_action_log_probs: Tensor of sample probability of each action.
          advantages: Tensor of advantage estimate for each timestep, aligned on
            index. Works better when advantage estimates are normalized.
          current_policy_distribution: The policy distribution, evaluated on all
            time_steps.
          weights: Optional scalar or element-wise (per-batch-entry) importance
            weights.  Includes a mask for invalid timesteps.

        Returns:
          policy_gradient_loss: A tensor that will contain policy gradient loss for
            the on-policy experience.
        """
        tf.nest.assert_same_structure(time_steps, self.time_step_spec)
        action_log_prob = common.log_probability(
            current_policy_distribution, actions, self._action_spec
        )

        action_log_prob = tf.cast(action_log_prob, tf.float32)
        if self._log_prob_clipping > 0.0:
            action_log_prob = tf.clip_by_value(
                action_log_prob, -self._log_prob_clipping, self._log_prob_clipping
            )

        tf.debugging.check_numerics(action_log_prob, "action_log_prob")

        tf.debugging.check_numerics(sample_action_log_probs, "sample_action_log_probs")

        # Prepare unclipped importance ratios.
        importance_ratio = tf.exp(action_log_prob - sample_action_log_probs)

        tf.debugging.check_numerics(
            importance_ratio, "importance_ratio", name="importance_ratio"
        )

        per_timestep_objective = importance_ratio * advantages
        policy_gradient_loss = -per_timestep_objective

        policy_gradient_loss = tf.reduce_mean(input_tensor=policy_gradient_loss * weights)

        tf.debugging.check_numerics(
            policy_gradient_loss, "Policy Loss divergence", name="policy_check"
        )

        return policy_gradient_loss

    def _kl_divergence(
        self, time_steps, action_distribution_parameters, current_policy_distribution
    ):
        """Compute mean KL divergence for 2 policies on given batch of timesteps"""
        outer_dims = list(range(nest_utils.get_outer_rank(time_steps, self.time_step_spec)))

        old_actions_distribution = distribution_spec.nested_distributions_from_specs(
            self._action_distribution_spec, action_distribution_parameters["dist_params"]
        )

        kl_divergence = ppo_utils.nested_kl_divergence(
            old_actions_distribution, current_policy_distribution, outer_dims=outer_dims
        )
        return kl_divergence

    def value_estimation_loss(self, time_steps, returns, weights):
        """Computes the value estimation loss for actor-critic training.
        All tensors should have a single batch dimension.
        Args:
          time_steps: A batch of timesteps.
          returns: Per-timestep returns for value function to predict. (Should come
            from TD-lambda computation.)
          weights: Optional scalar or element-wise (per-batch-entry) importance
            weights.  Includes a mask for invalid timesteps.
        Returns:
          value_estimation_loss: A scalar value_estimation_loss loss.
        """
        batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
        value_state = self._collect_policy.get_initial_value_state(batch_size=batch_size)

        value_preds, _ = self._collect_policy.apply_value_network(
            time_steps.observation, time_steps.step_type, value_state=value_state
        )

        value_estimation_error = tf.math.squared_difference(returns, value_preds)
        value_estimation_error *= weights

        value_estimation_loss = tf.reduce_mean(input_tensor=value_estimation_error)

        tf.debugging.check_numerics(
            value_estimation_loss, "Value loss diverged", name="Value_check"
        )

        return value_estimation_loss

    def policy_gradient(self, time_steps, policy_steps_, advantages, weights):
        """
        Compute policy gradient wrt actor_net parameters.

        :param time_steps: batch of TimeSteps with observations for each timestep
        :param policy_steps_: policy info for time step sampling policy
        :param advantages: Tensor of advantage estimate for each timestep, aligned on index.
        :param weights: mask for invalid timesteps
        :return: list of gradient tensors, policy loss computer on timesteps
        """
        batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
        actions = policy_steps_.action

        # get policy info before update
        action_distribution_parameters = policy_steps_.info

        # Reconstruct per-timestep policy distribution
        old_actions_distribution = distribution_spec.nested_distributions_from_specs(
            self._action_distribution_spec, action_distribution_parameters["dist_params"]
        )

        # Log probability of actions taken during data collection
        act_log_probs = common.log_probability(
            old_actions_distribution, actions, self._action_spec
        )

        with tf.GradientTape() as tape:
            # current policy distribution
            policy_state = self._collect_policy.get_initial_state(batch_size)
            distribution_step = self._collect_policy.distribution(time_steps, policy_state)
            current_policy_distribution = distribution_step.action

            policy_gradient_loss = self.policy_gradient_loss(
                time_steps,
                actions,
                tf.stop_gradient(act_log_probs),
                tf.stop_gradient(advantages),
                current_policy_distribution,
                weights,
            )

        trainable = self._actor_net.trainable_weights

        grads = tape.gradient(policy_gradient_loss, trainable)

        for g in grads:
            tf.debugging.check_numerics(g, "Gradient divergence", name="grad_check")

        return policy_gradient_loss, grads

    def natural_policy_gradient(self, time_steps, policy_steps_, gradient, weights):
        """
        Compute natural policy gradient wrt actor_net parameters.

        :param time_steps: batch of TimeSteps with observations for each timestep
        :param policy_steps_: policy info for time step sampling policy
        :param gradient: vanilla policy gradient computed on batch
        :param weights: mask for invalid timesteps
        :return: natural gradient as single flattened vector, lagrange coefficient for updating
        parameters with KL constraint
        """

        batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]

        # get policy info before update
        action_distribution_parameters = policy_steps_.info

        def _kl(params):
            """ Compute KL between old policy and policy using given params"""
            unflatten_tensor(params, self._opt_policy_parameters)
            opt_policy_state = self._opt_policy.get_initial_state(batch_size)
            dists = self._opt_policy.distribution(time_steps, opt_policy_state)
            policy_distribution = dists.action
            kl = self._kl_divergence(
                time_steps, action_distribution_parameters, policy_distribution
            )
            return tf.reduce_mean(kl)

        def _hv(vector: tf.Tensor) -> tf.Tensor:
            """Compute product of vector with Hessian of KL divergence"""
            return hessian_vector_product(_kl, self._opt_policy_parameters, vector)

        flat_grads = flatten_tensors(gradient)

        # sync optimisation policy with current policy
        common.soft_variables_update(
            self.policy.variables(),
            self._opt_policy.variables(),  # pylint: disable=not-callable
            tau=1.0,
        )

        # approximate natural gradient by approximately solving  grad = H @ nat_grad
        nat_grad = conjugate_gradient(_hv, flat_grads, max_iter=self._cg_iters)

        # lagrange coefficient for solving the constrained maximisation
        coeff = tf.sqrt(2.0 * self._max_kl / (tf.transpose(nat_grad) @ _hv(nat_grad) + EPS))

        tf.debugging.check_numerics(nat_grad, "Natural gradient", name="natgrad_check")

        tf.debugging.check_numerics(
            coeff, "NatGrad lagrange multiplier", name="multiplier_check"
        )

        return nat_grad, coeff

    def _line_search(
        self, time_steps, policy_steps_, advantages, natural_gradient, coeff, weights
    ):
        """Find new policy parameters by line search in natural gradient direction"""

        batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]

        # old policy distribution
        action_distribution_parameters = policy_steps_.info
        actions = policy_steps_.action
        actions_distribution = distribution_spec.nested_distributions_from_specs(
            self._action_distribution_spec, action_distribution_parameters["dist_params"]
        )
        act_log_probs = common.log_probability(
            actions_distribution, actions, self._action_spec
        )

        # loss for the old policy
        loss_threshold = self.policy_gradient_loss(
            time_steps,
            actions,
            tf.stop_gradient(act_log_probs),
            tf.stop_gradient(advantages),
            actions_distribution,
            weights,
        )

        policy_params = flatten_tensors(self._actor_net.trainable_variables)

        # try different steps_sizes, accept first one that improves loss and satisfies KL constraint
        for it in range(self._backtrack_iters):
            new_params = policy_params - self._backtrack_coeff ** it * coeff * natural_gradient

            unflatten_tensor(new_params, self._opt_policy_parameters)
            opt_policy_state = self._opt_policy.get_initial_state(batch_size)
            dists = self._opt_policy.distribution(time_steps, opt_policy_state)
            new_policy_distribution = dists.action

            kl = tf.reduce_mean(
                self._kl_divergence(
                    time_steps, action_distribution_parameters, new_policy_distribution
                )
            )
            loss = self.policy_gradient_loss(
                time_steps,
                actions,
                tf.stop_gradient(act_log_probs),
                tf.stop_gradient(advantages),
                new_policy_distribution,
                weights,
            )
            if kl < self._max_kl and loss < loss_threshold:
                return new_params

        # no improvement found
        return policy_params

    def _update_policy(self, time_steps, policy_steps_, advantages, weights):
        """Update policy parameters by computing natural gradient and step_size"""

        policy_gradient_loss, policy_grad = self.policy_gradient(
            time_steps, policy_steps_, advantages, weights
        )

        natural_gradient, coeff = self.natural_policy_gradient(
            time_steps, policy_steps_, policy_grad, weights
        )

        # find best step size in natural gradient direction
        new_params = self._line_search(
            time_steps, policy_steps_, advantages, natural_gradient, coeff, weights
        )

        tf.debugging.check_numerics(
            new_params, "Updated policy parameters", name="new_params_check"
        )
        unflatten_tensor(new_params, self._actor_net.trainable_variables)

        return policy_gradient_loss

    def _update_values(self, time_steps, returns, weights):
        """Update value function estimate by performing gradient descent on value loss"""
        variables_to_train = self._value_net.trainable_weights

        value_loss = 0.0
        for _ in range(self._value_train_iters):
            with tf.GradientTape() as tape:
                value_loss = self.value_estimation_loss(time_steps, returns, weights)

            grads = tape.gradient(value_loss, variables_to_train)

            # Tuple is used for py3, where zip is a generator producing values once.
            grads_and_vars = tuple(zip(grads, variables_to_train))
            if self._gradient_clipping > 0:
                grads_and_vars = eager_utils.clip_gradient_norms(
                    grads_and_vars, self._gradient_clipping
                )

            self._optimizer.apply_gradients(
                grads_and_vars, global_step=self.train_step_counter
            )
        return value_loss

    def _loss(self, experience: NestedTensor, weights: Tensor) -> Optional[LossInfo]:
        raise ValueError("A single loss is not well defined.")

    def _train(self, experience, weights=None):
        # unpack trajectories
        (time_steps, policy_steps_, next_time_steps) = trajectory.to_transition(experience)

        batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
        value_state = self._collect_policy.get_initial_value_state(batch_size=batch_size)

        weights = ppo_utils.make_timestep_mask(next_time_steps)

        value_preds, _ = self._collect_policy.apply_value_network(
            experience.observation, experience.step_type, value_state=value_state
        )
        value_preds = tf.stop_gradient(value_preds)

        rewards = next_time_steps.reward

        # normalize rewards
        if self._reward_normalizer is not None:
            rewards = self._reward_normalizer.normalize(
                rewards, center_mean=False, clip_value=self._reward_norm_clipping
            )

        returns, normalized_advantages = compute_return_and_advantage(
            self._discount_factor, self._lambda, rewards, next_time_steps, value_preds
        )

        policy_loss = self._update_policy(
            time_steps, policy_steps_, normalized_advantages, weights
        )

        value_loss = self._update_values(time_steps, returns, weights)

        return tf_agent.LossInfo(
            loss=value_loss + policy_loss,
            extra=TRPOLossInfo(
                value_estimation_loss=value_loss, policy_gradient_loss=policy_loss
            ),
        )
