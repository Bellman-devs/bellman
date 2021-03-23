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
This module defines trajectory sampler types.
"""

from enum import Enum

import gin


@gin.constants_from_enum
class TrajectorySamplerType(Enum):
    """
    This class defines types of trajectory samplers.
    """

    TS1 = "TS1"
    TSinf = "TSinf"
    Mean = "Mean"
