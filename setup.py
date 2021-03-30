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


from setuptools import find_packages, setup

package_data = {"": ["*"]}

install_requires = \
['gym==0.17.2',
 'imageio-ffmpeg==0.4.2',
 'imageio==2.8.0',
 'matplotlib==3.2.1',
 'tensorflow-probability==0.12.1',
 'tensorflow==2.4.0',
 'tf-agents==0.7.1']

extras_require = {"mujoco-py": ["mujoco-py>=2.0,<2.1"]}

setup_kwargs = {
    "name": "bellman",
    "version": "0.1.0",
    "description": "Model Based Reinforcement Learning",
    "long_description": None,
    "author": "The Bellman Contributors",
    "author_email": "bellman-devs@protonmail.com",
    "maintainer": None,
    "maintainer_email": None,
    "url": None,
    "packages": find_packages(include=("bellman*",)),
    "package_data": package_data,
    "install_requires": install_requires,
    "extras_require": extras_require,
    "python_requires": ">=3.7,<4.0",
}


setup(**setup_kwargs)
