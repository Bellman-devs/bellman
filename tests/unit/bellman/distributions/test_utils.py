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

from tf_agents.specs import BoundedTensorSpec

from bellman.distributions.utils import create_uniform_distribution_from_spec


def test_sample_from_uniform_distribution_from_tensor_spec_by_dtype(gym_space_shape, dtype):
    tensor_spec = BoundedTensorSpec(gym_space_shape, dtype, 0, 1)
    uniform_distribution = create_uniform_distribution_from_spec(tensor_spec)

    sample = uniform_distribution.sample()
    assert sample.dtype == dtype


def test_sample_shape_from_uniform_distribution_from_tensor_spec_by_dtype(
    gym_space_shape, dtype
):
    tensor_spec = BoundedTensorSpec(gym_space_shape, dtype, 0, 1)
    uniform_distribution = create_uniform_distribution_from_spec(tensor_spec)

    sample = uniform_distribution.sample()
    assert sample.shape == gym_space_shape
