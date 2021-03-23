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

r"""
This module trains and evaluates PETS algorithm. PETS algorithm can be found in
Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine (2018) "Deep
Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models",
which can be found at https://arxiv.org/abs/1805.12114

To run:

```bash
tensorboard --logdir $HOME/tmp/pets/MountainCarContinuous-v0/ --port 2223

python run_pets.py --root_dir=$HOME/tmp/pets/MountainCarContinuous-v0/
```
"""

import argparse
import os

import gin
from absl import logging

from bellman.benchmark.pets.train_eval import train_eval

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", type=str, required=True, help="Set a root directory for saving reports"
    )
    args = parser.parse_args()

    gin.parse_config_files_and_bindings(
        [
            "agent.gin",
            "experiment.gin",
            "../../environments/mountain_car_continuous.gin",
            "../../environments/mountain_car_continuous_models.gin",
        ],
        None,
    )

    train_eval(args.root_dir)  # pylint: disable=no-value-for-parameter
