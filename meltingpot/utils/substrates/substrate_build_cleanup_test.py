# Copyright 2026 DeepMind Technologies Limited.
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
"""Regression tests for substrate construction cleanup."""

from unittest import mock

from absl.testing import absltest
from meltingpot.utils.substrates import substrate


class SubstrateBuildCleanupTest(absltest.TestCase):

  def _build(self):
    return substrate.build_substrate(
        lab2d_settings=mock.sentinel.settings,
        individual_observations=(),
        global_observations=(),
        action_table=(),
    )

  def test_closes_raw_environment_if_first_wrapper_fails(self):
    raw_env = mock.Mock()
    with mock.patch.object(
        substrate.builder, 'builder', return_value=raw_env
    ), mock.patch.object(
        substrate.observables_wrapper,
        'ObservablesWrapper',
        side_effect=RuntimeError('wrapper failed'),
    ):
      with self.assertRaisesRegex(RuntimeError, 'wrapper failed'):
        self._build()

    raw_env.close.assert_called_once_with()

  def test_closes_latest_owner_if_later_wrapper_fails(self):
    raw_env = mock.Mock()
    observable_env = mock.Mock()
    with mock.patch.object(
        substrate.builder, 'builder', return_value=raw_env
    ), mock.patch.object(
        substrate.observables_wrapper,
        'ObservablesWrapper',
        return_value=observable_env,
    ), mock.patch.object(
        substrate.multiplayer_wrapper,
        'Wrapper',
        side_effect=RuntimeError('multiplayer wrapper failed'),
    ):
      with self.assertRaisesRegex(RuntimeError, 'multiplayer wrapper failed'):
        self._build()

    observable_env.close.assert_called_once_with()


if __name__ == '__main__':
  absltest.main()
