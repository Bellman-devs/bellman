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
This package provides helper wrappers for training RL agents which have trainable components. The
TF-Agents `Agent` class has a `train` method, which in the model-free setting updates the policy
parameters. In the model-based setting there may be more than one trainable "component", for
example a transition model and a parameterised policy.

The wrappers provide a consistent interface for training the components of RL agents on a defined
schedule. An agent specifies a list of names (implemented in the toolbox by a set of enumerations)
of components which will be trained, and an `AgentTrainer` object defines the schedules at which
those components should be trained as well as which "real" data (from the real environment) should
be used to train each component.
"""
