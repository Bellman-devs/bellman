# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Trajectory Optimisation

Decision-time planning agents solve a virtual planning problem from an initial, real, state. The solution to the planning problem is an optimal action which is executed in the real environment. This toolbox solves the planning problem by sampling a sequence of actions from a given policy and estimating the value of those actions using the transition and reward models.

## Mathematical details

We have an approximate MDP:
$$\hat{\mathcal{M}} = (S, A, \hat{r}, \hat{P}, \rho_0, \gamma)$$
and a policy $\pi$. This policy may choose actions conditional on the state ($\pi(a|s)$), or independent of the state ($\pi(a)$).

A single virtual trajectory is unrolled, starting from the current state of the real environment $s_0$, by repeatedly applying the transition model and the policy. The lengths of virtual rollouts are constrained by a hyper-parameter $h$:
$$a_i = \pi(s_i); s_{i + 1} = \hat{P}(a_i)$$
for $i < h$.

The value of a trajectory is the sum of the rewards at each time step, where the step rewards are provided by the approximate reward model $\hat{r}$.

Some trajectory optimisation methods use a parameterised policy where the parameters are learned for each time step (such as the cross-entropy method which is implemented in this toolbox). These methods repeat this process for a number of iterations.

## Implementation details

This toolbox provides a flexible and extensible framework for trajectory optimisation methods. There is a single class for policy base trajectory optimisation (`PolicyTrajectoryOptimiser` in the `trajectory_optimiser` package). This class is composed of a set of objects which can be combined in different ways to produce different algorithms. Two algorithms have been implemented: the cross-entropy method and random shooting.
"""

# %%
