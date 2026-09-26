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
"""Tests for populations without background roles."""

from absl.testing import absltest
import dm_env
from meltingpot.utils.scenarios import population as population_lib


class EmptyPopulationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.population = population_lib.Population(
        policies={}, names_by_role={}, roles=()
    )
    self.addCleanup(self.population.close)

  def test_empty_population_round_trip(self):
    actions = []
    names = []
    timesteps = []
    observables = self.population.observables()
    observables.action.subscribe(actions.append)
    observables.names.subscribe(names.append)
    observables.timestep.subscribe(timesteps.append)
    timestep = dm_env.restart(observation=())

    self.population.reset()
    self.population.send_timestep(timestep)
    action = self.population.await_action()

    self.assertEmpty(action)
    self.assertEqual(names, [[]])
    self.assertEqual(actions, [()])
    self.assertEqual(timesteps, [timestep])

  def test_requires_empty_action_to_be_awaited(self):
    self.population.reset()
    timestep = dm_env.restart(observation=())
    self.population.send_timestep(timestep)

    with self.assertRaisesRegex(RuntimeError, 'Previous action'):
      self.population.send_timestep(timestep)

    self.assertEmpty(self.population.await_action())
    with self.assertRaisesRegex(RuntimeError, 'No timestep'):
      self.population.await_action()


if __name__ == '__main__':
  absltest.main()
