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

# Installs meltingpot on Linux/macOS.

set -e  # Fail on any error.
set -u  # Fail on unset variables.

readonly MELTINGPOT_ROOT="$(dirname "$(realpath $0)")"

# Set this to where you would like create the virtual env.
readonly VENV_PATH="${MELTINGPOT_ROOT}/venv"

# Build dmlab2d wheel from src if desired.
readonly BUILD_DMLAB2D=0
if (( ${BUILD_DMLAB2D} )); then
  echo "Building dmlab2d wheel..."
  sudo apt-get update
  sudo apt-get install bazel
  cd "${MELTINGPOT_ROOT}"
  git clone https://github.com/deepmind/lab2d
  cd lab2d
  bazel build -c opt --define=lua=5_1 //dmlab2d:dmlab2d_wheel
  readonly DMLAB_WHEEL_ROOT="${MELTINGPOT_ROOT}/lab2d/bazel-bin/dmlab2d"
else
  readonly DMLAB_WHEEL_ROOT='https://github.com/deepmind/lab2d/releases/download/release_candidate_2021-07-13'
fi

echo "Creating venv..."
python3 -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

echo "Installing dmlab2d..."
readonly VERSION="$(python --version | sed 's/^Python \([0-9]\)\.\([0-9]\+\)\..*$/\1\2/g')"
readonly PYTHON_VERSION="$(python --version | sed 's/^Python \([0-9]\)\.\([0-9]\+\)\..*$/\1\2/g')"
if [[ "$(uname -s)" == "Darwin" ]]; then
  readonly DMLAB_WHEEL="${DMLAB_WHEEL_ROOT}/dmlab2d-1.0-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-macosx_10_15_x86_64.whl"
else
  readonly DMLAB_WHEEL="${DMLAB_WHEEL_ROOT}/dmlab2d-1.0-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux_2_31_x86_64.whl"
fi
pip install "${DMLAB_WHEEL}"

echo "Testing dmlab2d..."
cd "${HOME}"
python - <<'____HERE'
import dmlab2d
import dmlab2d.runfiles_helper

lab = dmlab2d.Lab2d(dmlab2d.runfiles_helper.find(), {"levelName": "chase_eat"})
env = dmlab2d.Environment(lab, ["WORLD.RGB"])
env.step({})
____HERE

echo "Installing meltingpot..."
cd "${MELTINGPOT_ROOT}"
pip install .

echo "Testing meltingpot..."
python - <<'____HERE'
from meltingpot.python import scenario

for name in list(scenario.AVAILABLE_SCENARIOS)[:3]:
  config = scenario.get_config(name)
  with scenario.build(config) as env:
    env.reset()
    action_spec = env.action_spec()
    action = [spec.generate_value() for spec in action_spec]
    env.step(action)
    print(f'{name} OK')
____HERE

cat <<____HERE

Ran a few tests and everything seems OK.

You can now run all the tests, using:
    pip install nose
    nosetests meltingpot

However, this may take some time.

____HERE
read -n1 -r -p "Would you like to run all the tests now [y/N]? " response
echo
if [[ "${response}" =~ ^[Yy]$ ]]; then
  pip install nose
  nosetests meltingpot
fi
