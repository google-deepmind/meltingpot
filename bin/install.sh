#!/bin/bash
#
# Copyright 2024 DeepMind Technologies Limited.
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
#
# Install meltingpot.
set -euxo pipefail
cd "$(dirname "$0")/.."

echo 'Installing requirements...'
pip install --no-deps --require-hashes --requirement requirements.txt
echo
echo

echo 'Installing Melting Pot...'
pip install --no-deps --no-index --no-build-isolation --editable .
echo
echo

pip list