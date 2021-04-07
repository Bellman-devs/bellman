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

import pytest
import tensorflow as tf


@pytest.fixture(name="n_dims", params=[1, 5])
def _n_dims_fixture(request):
    return request.param


@pytest.fixture(name="n_features", params=[1, 30])
def _n_features_fixture(request):
    return request.param


@pytest.fixture(name="dtype", params=[tf.float64, tf.int64], scope="session")
def _dtype_fixture(request):
    return request.param
