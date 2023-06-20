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

# Tests dmlab2d installed by install-dmlab2d.sh.

set -euxo pipefail


function test_dmlab2d() {
  echo -e "\nTesting dmlab2d..."
  python lab2d/dmlab2d/dmlab2d_test.py
}


function main() {
  test_dmlab2d
}


main "$@"
