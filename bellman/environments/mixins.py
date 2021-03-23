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
This module defines mixins for environment model components.
"""

from abc import ABC, abstractmethod


class BatchSizeUpdaterMixin(ABC):
    """
    Mixin for environment model components that use the batch size as part of their internal state.
    """

    @abstractmethod
    def update_batch_size(self, batch_size: int) -> None:
        """
        :param batch_size: New value for batch size
        """
