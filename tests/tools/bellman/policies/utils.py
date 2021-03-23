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

from tf_agents.drivers.tf_driver import TFDriver
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories.trajectory import Trajectory

from tests.tools.bellman.policies.open_loop_policy import TFOpenLoopPolicy


def replay_actions_across_batch_transition_models(env_model, actions) -> Trajectory:
    """
    Use an open loop policy to apply a sequence of actions to the environment model. This returns
    at least one episode per environment batch (the same action sequence is applied to each batch).
    """
    open_loop_policy = TFOpenLoopPolicy(
        env_model.time_step_spec(), env_model.action_spec(), actions
    )
    buffer = TFUniformReplayBuffer(
        open_loop_policy.trajectory_spec, batch_size=env_model.batch_size, max_length=1000
    )
    driver = TFDriver(
        env_model,
        open_loop_policy,
        observers=[buffer.add_batch],
        max_steps=env_model.batch_size * actions.shape[0],
        disable_tf_function=True,
    )
    driver.run(env_model.reset())

    trajectories = buffer.gather_all()

    return trajectories
