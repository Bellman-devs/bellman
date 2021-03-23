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
Utilities to support environment models.
"""

from typing import Any, Dict, Optional, Tuple, cast

import gym
import numpy as np
from tf_agents.drivers.driver import Driver
from tf_agents.environments.suite_gym import load
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer

from bellman.drivers.tf_driver import TFBatchDriver
from bellman.environments.environment_model import EnvironmentModel
from bellman.environments.tf_wrappers import TFTimeLimit


def virtual_rollouts_buffer_and_driver(
    environment_model: EnvironmentModel, policy: TFPolicy, duration: int
) -> Tuple[ReplayBuffer, Driver, EnvironmentModel]:
    """
    Generating rollouts from an environment model requires creating a coupled replay buffer and
    driver. This function returns the replay buffer and driver which are configured correctly.

    The replay buffer is a FIFO buffer which has a `capacity` of 2 × (`horizon` + 1). This is
    because the `TFDriver` does not count boundary transitions towards the total number of steps
    collected and stored in the replay buffer. The shortest possible trajectory is of length 2
    ([StepType.FIRST, StepType.LAST]). The `TFDriver` counts one step in that trajectory, and so
    the replay buffer must have enough space to avoid losing the initial `time_step`.

    :param environment_model: virtual environment model. This should not be wrapped in a
                              `TFTimeLimit`.
    :param policy: policy to use to generate rollouts.
    :param duration: Maximum number of steps in a generated trajectory.

    :return: A replay buffer and a driver to use for generating virtual rollouts, and the
             environment model wrapped in a time limit wrapper.
    """
    assert isinstance(
        environment_model, EnvironmentModel
    ), f"{environment_model} should not be wrapped in ."

    wrapped_environment_model = TFTimeLimit(environment_model, duration)

    replay_buffer = TFUniformReplayBuffer(
        policy.trajectory_spec,
        batch_size=wrapped_environment_model.batch_size,
        max_length=duration + 1,
    )

    driver = TFBatchDriver(
        wrapped_environment_model,
        policy,
        observers=[replay_buffer.add_batch],
        min_episodes=1,
        disable_tf_function=True,
    )

    return replay_buffer, driver, cast(EnvironmentModel, wrapped_environment_model)


def create_real_tf_environment(
    env_name: str,
    random_seed: Optional[int] = None,
    spec_dtype_map: Optional[Dict[gym.spaces.space.Space, np.dtype]] = None,
    gym_kwargs: Optional[Dict[str, Any]] = None,
) -> TFEnvironment:
    """
    This function creates a gym environment object by loading the spec by name from gym registry,
    and wrapping it in a `TFPyEnvironment` for compatibility with TF-Agents.

    :param env_name: Name for the environment to load.
    :param random_seed: Value to use as seed for the environment.
    :param spec_dtype_map: Maps gym spaces to np dtypes to use as the
        default dtype for the arrays.
    :param gym_kwargs: Optional kwargs to pass to the Gym environment class.

    :return: A `TFEnvironment` which wraps the real gym environment.
    """
    py_environment = load(env_name, spec_dtype_map=spec_dtype_map, gym_kwargs=gym_kwargs)
    if random_seed is not None:
        py_environment.seed(random_seed)

    return TFPyEnvironment(py_environment)
