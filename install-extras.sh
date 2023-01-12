#!/bin/bash
# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Installs meltingpot extras on Linux/macOS.

set -euxo pipefail


function check_setup() {
  echo -e "\nChecking meltingpot is installed..."
  python -c 'import meltingpot'
}


function install_extras() {
  echo -e "\nInstalling meltingpot extras..."
  pip install .[rllib,pettingzoo]
}


function test_extras() {
  echo -e "\nTesting meltingpot extras..."
  # Test RLLib and Petting Zoo training scripts.
  # TODO(b/265139141): Add PettingZoo test.
  test_rllib
}


function test_rllib() {
  echo -e "\nTesting RLLib example..."
  python <<EOF
from examples.rllib import self_play_train
from examples.rllib import utils

import ray
from ray import tune
from ray import air

config = self_play_train.get_config(
    num_rollout_workers=1,
    rollout_fragment_length=10,
    train_batch_size=20,
    sgd_minibatch_size=20,
    fcnet_hiddens=(4,),
    post_fcnet_hiddens=(4,),
    lstm_cell_size=2)

tune.register_env("meltingpot", utils.env_creator)
ray.init()
stop = {
    "training_iteration": 1,
}
results = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(stop=stop, verbose=1),
).fit()
assert results.num_errors == 0
EOF
}


function main() {
  check_setup
  install_extras
  test_extras
}


main "$@"
