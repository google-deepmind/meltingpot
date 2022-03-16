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

set -euxo pipefail


function check_setup() {
  echo -e "\nChecking OS is Linux or macOS..."
  uname -s
  [[ "$(uname -s)" =~ (Linux|Darwin) ]] || exit 1

  echo -e "\nChecking python version is >= 3.7..."
  python --version
  python -c 'import sys; exit(0) if sys.version_info.major == 3 and sys.version_info.minor >= 7 else exit(1)' || exit 1

  echo -e "\nChecking bazel installed..."
  bazel --version
}


function install_dmlab2d() {
  echo -e "\nCloning dmlab2d..."
  git clone https://github.com/deepmind/lab2d

  echo -e "\nInstalling dmlab2d requirements..."
  pip install --upgrade pip packaging

  echo -e "\nBuilding dmlab2d wheel..."
  if [[ "$(uname -s)" == 'Linux' ]]; then
    # readonly LUA_VERSION=luajit  # PASSES
    # readonly LUA_VERSION=lua5_2  # TESTS FAIL
    readonly LUA_VERSION=lua5_1  # TESTS PASS
  else
    # readonly LUA_VERSION=luajit  # BUILD FAILS
    # readonly LUA_VERSION=lua5_2  # TESTS FAIL
    readonly LUA_VERSION=lua5_1  # TESTS PASS
  fi
  pushd lab2d
  bazel build \
      --compilation_mode=opt \
      --config="${LUA_VERSION}" \
      --verbose_failures \
      --experimental_ui_max_stdouterr_bytes=-1 \
      --toolchain_resolution_debug \
      --sandbox_debug \
      //dmlab2d:dmlab2d_wheel
  popd

  echo -e "\nInstalling dmlab2d..."
  pip install lab2d/bazel-bin/dmlab2d/dmlab2d-*.whl
}


function test_dmlab2d() {
  echo -e "\nTesting dmlab2d..."
  python - <<'____HERE'
import dmlab2d
import dmlab2d.runfiles_helper

lab = dmlab2d.Lab2d(dmlab2d.runfiles_helper.find(), {"levelName": "chase_eat"})
env = dmlab2d.Environment(lab, ["WORLD.RGB"])
env.step({})
____HERE
}


function install_meltingpot() {
  echo -e "\nInstalling meltingpot..."
  pip install --upgrade pip setuptools
  pip install .
}


function test_meltingpot() {
  echo -e "\nTesting meltingpot..."
  pip install pytest-xdist
  pytest -n auto -ra --durations=10 meltingpot
}


function main() {
  check_setup
  install_dmlab2d
  test_dmlab2d
  install_meltingpot
  test_meltingpot
}


main "$@"
