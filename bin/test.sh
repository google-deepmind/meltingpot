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
# Test meltingpot.
set -euxo pipefail
cd "$(dirname "$0")/.."
FAILURES=false

echo "pytest meltingpot..."
pytest meltingpot || FAILURES=true
echo
echo

echo "pytype meltingpot..."
pytype meltingpot || FAILURES=true
echo
echo

echo "pylint meltingpot..."
pylint --errors-only meltingpot || FAILURES=true
echo
echo

if "${FAILURES}"; then
  echo -e '\033[0;31mFAILURE\033[0m' && exit 1
else
  echo -e '\033[0;32mSUCCESS\033[0m'
fi
