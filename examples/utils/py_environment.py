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
This module offers useful classes and functions for rendering the environment.
"""

import os

import gym
import imageio
import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.wrappers import TimeLimit


class RenderPyEnvironment:
    def __init__(self, py_env: gym.Env):
        self._py_env = py_env

    @tf.autograph.experimental.do_not_convert()
    def __call__(self, *args, **kwargs):
        self._py_env.render()


def save_environment_agent_video(
    filename: str,
    agent: tf_agent.TFAgent,
    tf_env: TFPyEnvironment,
    py_env: TimeLimit,
    num_episodes: int = 1,
) -> None:
    """
    Save a video of an agent acting in the environment. Render method needs to be available in the
    python version of the environment.
    TODO:
    - how to prevent opening a window when saving a video?
    - sometimes nothing is saved?
    - gym wrappers monitoring VideoRecorder

    :param filename: A valid path to which a file with the video will be saved.
    :param agent: An agent whose policy will be evaluated.
    :param tf_env: A TensorFlow environment used for interaction with the agent.
    :param py_env: A Python OpenAI Gym environment used for rendering the video. Environment has
        to provide `render` method.
    :param num_episodes: A number of episodes to evaluate.

    :return: A video is saved to filename.
    """
    with imageio.get_writer(filename, fps=60) as video:
        for _ in range(num_episodes):
            time_step = tf_env.reset()
            video.append_data(py_env.render())
            while not time_step.is_last():
                action_step = agent.policy.action(time_step)
                time_step = tf_env.step(action_step.action)
                video.append_data(py_env.render())
    py_env.close()
