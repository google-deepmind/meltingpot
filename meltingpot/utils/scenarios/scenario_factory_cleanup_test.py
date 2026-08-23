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
"""Regression tests for ScenarioFactory failure cleanup."""

from unittest import mock

from absl.testing import absltest
from meltingpot.utils.scenarios import scenario_factory


class ScenarioFactoryCleanupTest(absltest.TestCase):

  def _make_factory(self, substrate_factory, bots):
    return scenario_factory.ScenarioFactory(
        substrate=substrate_factory,
        bots=bots,
        bots_by_role={'role': tuple(bots)},
        roles=('role',),
        is_focal=(False,),
        permitted_observations={'RGB'},
    )

  def test_build_closes_completed_resources_when_bot_build_fails(self):
    substrate = mock.Mock()
    substrate_factory = mock.Mock()
    substrate_factory.build.return_value = substrate

    built_bot = mock.Mock()
    good_factory = mock.Mock()
    good_factory.build.return_value = built_bot
    bad_factory = mock.Mock()
    bad_factory.build.side_effect = RuntimeError('bot build failed')
    factory = self._make_factory(
        substrate_factory, {'good': good_factory, 'bad': bad_factory})

    with self.assertRaisesRegex(RuntimeError, 'bot build failed'):
      factory.build()

    built_bot.close.assert_called_once_with()
    substrate.close.assert_called_once_with()

  def test_build_closes_resources_when_scenario_construction_fails(self):
    substrate = mock.Mock()
    substrate_factory = mock.Mock()
    substrate_factory.build.return_value = substrate

    built_bot = mock.Mock()
    bot_factory = mock.Mock()
    bot_factory.build.return_value = built_bot
    factory = self._make_factory(substrate_factory, {'bot': bot_factory})

    with mock.patch.object(
        scenario_factory.scenario_lib,
        'build_scenario',
        side_effect=RuntimeError('scenario build failed')):
      with self.assertRaisesRegex(RuntimeError, 'scenario build failed'):
        factory.build()

    built_bot.close.assert_called_once_with()
    substrate.close.assert_called_once_with()

  def test_build_transformed_closes_substrate_when_transform_fails(self):
    substrate = mock.Mock()
    substrate_factory = mock.Mock()
    substrate_factory.build.return_value = substrate
    factory = self._make_factory(substrate_factory, {})

    transform = mock.Mock(side_effect=RuntimeError('transform failed'))
    with self.assertRaisesRegex(RuntimeError, 'transform failed'):
      factory.build_transformed(transform)

    substrate.close.assert_called_once_with()


if __name__ == '__main__':
  absltest.main()
