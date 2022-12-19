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

# Installs dmlab2d on Linux/macOS.

set -euxo pipefail


function check_version_gt() {
  local required="$1"
  local input lowest
  input="$(grep -Eo '[0-9]+\.[0-9]+' /dev/stdin | head -n 1)"
  lowest="$(printf "${required}\n${input}" | sort -V | head -n 1)"
  [[ "${lowest}" == "${required}" ]]
}


function check_setup() {
  echo -e "\nChecking OS is Linux or macOS..."
  uname -a
  [[ "$(uname -s)" =~ (Linux|Darwin) ]]

  echo -e "\nChecking python version..."
  python --version
  python --version | check_version_gt '3.9'

  echo -e "\nChecking pip version..."
  pip --version
  pip --version | check_version_gt '20.3'

  echo -e "\nChecking gcc version ..."
  gcc --version
  gcc --version | check_version_gt '8'

  echo -e "\nChecking bazel version..."
  bazel --version
  bazel --version | check_version_gt '5.2'
}


function install_dmlab2d() {
  echo -e "\nCloning dmlab2d..."
  git clone https://github.com/deepmind/lab2d

  echo -e "\nInstalling dmlab2d requirements..."
  pip install --upgrade pip packaging

  echo -e "\nBuilding dmlab2d wheel..."
  if [[ "$(uname -s)" == 'Linux' ]]; then
    local -r LUA_VERSION=luajit
  elif [[ "$(uname -s)" == 'Darwin' ]]; then
    # luajit not available on macOS
    local -r LUA_VERSION=lua5_2
  else
    exit 1
  fi
  pushd lab2d
  C=clang CXX=clang++ bazel build \
      --compilation_mode=opt \
      --dynamic_mode=off \
      --config="${LUA_VERSION}" \
      --subcommands \
      --verbose_failures \
      --experimental_ui_max_stdouterr_bytes=-1 \
      --sandbox_debug \
      //dmlab2d:dmlab2d_wheel
  popd

  echo -e "\nInstalling dmlab2d..."
  pip install -vvv --find-links=lab2d/bazel-bin/dmlab2d dmlab2d
}


function test_dmlab2d() {
  echo -e "\nTesting dmlab2d..."
  python lab2d/dmlab2d/dmlab2d_test.py
}


function main() {
  check_setup
  install_dmlab2d
  test_dmlab2d
}


main "$@"
