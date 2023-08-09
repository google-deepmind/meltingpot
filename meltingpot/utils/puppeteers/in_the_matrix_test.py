# Copyright 2022 DeepMind Technologies Limited.
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
"""Tests for in_the_matrix Puppeteers."""

import itertools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
import immutabledict
from meltingpot.testing import puppeteers
from meltingpot.utils.puppeteers import in_the_matrix
import numpy as np

_RESOURCE_0 = in_the_matrix.Resource(
    index=0,
    collect_goal=mock.sentinel.collect_0,
    interact_goal=mock.sentinel.interact_0,
)
_RESOURCE_1 = in_the_matrix.Resource(
    index=1,
    collect_goal=mock.sentinel.collect_1,
    interact_goal=mock.sentinel.interact_1,
)
_RESOURCE_2 = in_the_matrix.Resource(
    index=2,
    collect_goal=mock.sentinel.collect_2,
    interact_goal=mock.sentinel.interact_2,
)

_INTERACTION = (np.array([1, 1, 1]), np.array([1, 1, 1]))
_NO_INTERACTION = (np.array([-1, -1, -1]), np.array([-1, -1, -1]))


class HelperFunctionTest(parameterized.TestCase):

  @parameterized.parameters(
      [_INTERACTION, _INTERACTION[1]],
      [_NO_INTERACTION, None],
  )
  def test_get_partner_interaction_inventory(self, interaction, expected):
    timestep = dm_env.restart(_observation(None, interaction))
    actual = in_the_matrix.get_partner_interaction_inventory(timestep)
    np.testing.assert_equal(actual, expected)

  @parameterized.parameters(
      [_INTERACTION, True],
      [_NO_INTERACTION, False],
  )
  def test_has_interaction(self, interaction, expected):
    timestep = dm_env.restart(_observation(None, interaction))
    actual = in_the_matrix.has_interaction(timestep)
    self.assertEqual(actual, expected)

  @parameterized.parameters(
      [(0, 0, 0), (mock.ANY, 0)],
      [(1, 0, 0), (0, 1)],
      [(0, 3, 1), (1, 2)],
      [(3, 0, 7), (2, 4)],
  )
  def test_max_resource_and_margin(self, inventory, expected):
    actual = in_the_matrix.max_resource_and_margin(np.array(inventory))
    self.assertEqual(actual, expected)

  @parameterized.parameters(
      [(1, 2, 3), 0, 1, False],
      [(1, 2, 3), 1, 1, False],
      [(1, 2, 3), 2, 1, True],
      [(1, 2, 3), 2, 2, False],
      [(1, 2, 5), 2, 1, True],
      [(1, 2, 5), 2, 3, True],
  )
  def test_has_sufficient(self, inventory, target, margin, expected):
    actual = in_the_matrix.has_collected_sufficient(
        np.array(inventory), target, margin)
    self.assertEqual(actual, expected)

  @parameterized.parameters(
      [(1, 2, 3), 2],
      [(1, 2, 5), 2],
      [(1, 0, 0), 0],
      [(1, 2, 0), 1],
  )
  def test_partner_max_resource(self, inventory, expected):
    timestep = dm_env.restart({
        'INTERACTION_INVENTORIES': (None, np.array(inventory)),
    })
    actual = in_the_matrix.partner_max_resource(timestep)
    self.assertEqual(actual, expected)

  @parameterized.parameters(
      [(1, 2, 3), _RESOURCE_0, 1, _RESOURCE_0.collect_goal],
      [(1, 2, 3), _RESOURCE_1, 1, _RESOURCE_1.collect_goal],
      [(1, 2, 3), _RESOURCE_2, 1, _RESOURCE_2.interact_goal],
      [(1, 2, 3), _RESOURCE_2, 2, _RESOURCE_2.collect_goal],
      [(1, 2, 5), _RESOURCE_2, 1, _RESOURCE_2.interact_goal],
      [(1, 2, 5), _RESOURCE_2, 3, _RESOURCE_2.interact_goal],
  )
  @mock.patch.object(immutabledict, 'immutabledict', dict)
  def test_collect_or_interact_puppet_timestep(
      self, inventory, target, margin, goal):
    timestep = dm_env.restart({'INVENTORY': np.array(inventory)})
    actual = in_the_matrix.collect_or_interact_puppet_timestep(
        timestep, target, margin)
    expected = dm_env.restart({'INVENTORY': np.array(inventory), 'GOAL': goal})
    np.testing.assert_equal(actual, expected)


def _observation(inventory, interaction=None):
  if interaction is None:
    interaction = _NO_INTERACTION
  return {
      'INVENTORY': np.array(inventory),
      'INTERACTION_INVENTORIES': np.array(interaction),
  }


def _goals_from_observations(puppeteer,
                             inventories,
                             interactions=(),
                             state=None):
  observations = []
  for inventory, interaction in itertools.zip_longest(inventories,
                                                      interactions):
    observations.append(_observation(inventory, interaction))
  return puppeteers.goals_from_observations(puppeteer, observations, state)


class SpecialistTest(parameterized.TestCase):

  def test(self):
    puppeteer = in_the_matrix.Specialist(
        target=_RESOURCE_1,
        margin=1,
    )
    inventories = [(1, 1, 1), (1, 2, 1), (1, 2, 2), (1, 3, 2)]
    expected = [
        _RESOURCE_1.collect_goal,
        _RESOURCE_1.interact_goal,
        _RESOURCE_1.collect_goal,
        _RESOURCE_1.interact_goal,
    ]
    actual, _ = _goals_from_observations(puppeteer, inventories)
    self.assertEqual(actual, expected)


class ScheduledFlipTest(parameterized.TestCase):

  def test(self):
    puppeteer = in_the_matrix.ScheduledFlip(
        threshold=1,
        initial_target=_RESOURCE_1,
        final_target=_RESOURCE_2,
        initial_margin=1,
        final_margin=2,
    )
    inventories = [(1, 1, 1), (1, 2, 1), (1, 2, 2), (1, 2, 4)]
    interactions = [
        _NO_INTERACTION, _NO_INTERACTION, _INTERACTION, _INTERACTION
    ]
    expected = [
        _RESOURCE_1.collect_goal,
        _RESOURCE_1.interact_goal,
        _RESOURCE_2.collect_goal,
        _RESOURCE_2.interact_goal,
    ]
    actual, _ = _goals_from_observations(puppeteer, inventories, interactions)
    self.assertEqual(actual, expected)


class GrimTriggerTest(parameterized.TestCase):

  def test_trigger(self):
    puppeteer = in_the_matrix.GrimTrigger(
        threshold=2,
        cooperate_resource=_RESOURCE_1,
        defect_resource=_RESOURCE_0,
        margin=1,
    )
    inventories = [
        (1, 1),
        (1, 2),
        (2, 2),
        (2, 2),
        (3, 2),
        (3, 2),
    ]
    interactions = [
        ((-1, -1), (-1, -1)),  # neither
        ((-1, -1), (1, 0)),  # defection
        ((-1, -1), (0, 1)),  # cooperation
        ((-1, -1), (1, 0)),  # defection
        ((-1, -1), (0, 1)),  # cooperation
        ((-1, -1), (0, 1)),  # cooperation
    ]
    expected = [
        _RESOURCE_1.collect_goal,  # cooperate
        _RESOURCE_1.interact_goal,  # cooperate
        _RESOURCE_1.collect_goal,  # cooperate
        _RESOURCE_0.collect_goal,  # defect
        _RESOURCE_0.interact_goal,  # defect
        _RESOURCE_0.interact_goal,  # defect
    ]
    actual, _ = _goals_from_observations(puppeteer, inventories, interactions)
    self.assertEqual(actual, expected)

  def test_not_grim_after_reset(self):
    puppeteer = in_the_matrix.GrimTrigger(
        threshold=2,
        cooperate_resource=_RESOURCE_1,
        defect_resource=_RESOURCE_0,
        margin=1,
    )
    inventories = [
        (1, 1),
        (1, 2),
        (2, 2),
        (2, 2),
        (3, 2),
        (3, 2),
    ]
    interactions = [
        ((-1, -1), (-1, -1)),  # neither
        ((-1, -1), (1, 0)),  # defection
        ((-1, -1), (0, 1)),  # cooperation
        ((-1, -1), (1, 0)),  # defection
        ((-1, -1), (0, 1)),  # cooperation
        ((-1, -1), (0, 1)),  # cooperation
    ]
    expected = [
        _RESOURCE_1.collect_goal,  # cooperate
        _RESOURCE_1.interact_goal,  # cooperate
        _RESOURCE_1.collect_goal,  # cooperate
        _RESOURCE_0.collect_goal,  # defect
        _RESOURCE_0.interact_goal,  # defect
        _RESOURCE_0.interact_goal,  # defect
    ]
    _, state = _goals_from_observations(puppeteer, inventories, interactions)
    actual, _ = _goals_from_observations(puppeteer, inventories, interactions,
                                         state)
    self.assertEqual(actual, expected)


class TitForTatTest(parameterized.TestCase):

  def test(self):
    puppeteer = in_the_matrix.TitForTat(
        cooperate_resource=_RESOURCE_1,
        defect_resource=_RESOURCE_0,
        margin=1,
        tremble_probability=0,
    )
    inventories = [
        (1, 1, 0),  # not ready to interact
        (1, 2, 0),  # ready to interact if cooperating
        (3, 2, 0),  # ready to interact if defecting
        (2, 3, 0),  # ready to interact if cooperating
        (3, 2, 0),  # ready to interact if defecting
        (2, 2, 0),  # not ready to interact
    ]
    interactions = [
        ((-1, -1, -1), (2, 2, 0)),  # coplayer cooperates and defects
        ((-1, -1, -1), (1, 0, 0)),  # coplayer defects
        ((-1, -1, -1), (0, 0, 1)),  # coplayer plays other
        ((-1, -1, -1), (0, 1, 0)),  # coplayer cooperates
        ((-1, -1, -1), (-1, -1, -1)),  # no interaction
        ((-1, -1, -1), (2, 1, 1)),  # coplayer defects
    ]
    expected = [
        _RESOURCE_1.collect_goal,  # cooperate
        _RESOURCE_0.collect_goal,  # defect
        _RESOURCE_0.interact_goal,  # continue defecting
        _RESOURCE_1.interact_goal,  # cooperate
        _RESOURCE_1.collect_goal,  # continue cooperating
        _RESOURCE_0.collect_goal,  # defect
    ]
    actual, _ = _goals_from_observations(puppeteer, inventories, interactions)
    self.assertEqual(actual, expected)

  def test_with_tremble(self):
    puppeteer = in_the_matrix.TitForTat(
        cooperate_resource=_RESOURCE_1,
        defect_resource=_RESOURCE_0,
        margin=1,
        tremble_probability=1,
    )
    inventories = [
        (1, 1, 0),  # not ready to interact
        (1, 2, 0),  # ready to interact if cooperating
        (3, 2, 0),  # ready to interact if defecting
        (2, 3, 0),  # ready to interact if cooperating
        (3, 2, 0),  # ready to interact if defecting
        (2, 2, 0),  # not ready to interact
    ]
    interactions = [
        ((-1, -1, -1), (2, 2, 0)),  # coplayer cooperates and defects
        ((-1, -1, -1), (1, 0, 0)),  # coplayer defects
        ((-1, -1, -1), (0, 0, 1)),  # coplayer plays other
        ((-1, -1, -1), (0, 1, 0)),  # coplayer cooperates
        ((-1, -1, -1), (-1, -1, -1)),  # no interaction
        ((-1, -1, -1), (2, 1, 1)),  # coplayer defects
    ]
    expected = [
        _RESOURCE_0.collect_goal,  # cooperate but tremble and defect
        _RESOURCE_1.interact_goal,  # defect but tremble and cooperate
        _RESOURCE_1.collect_goal,  # continue cooperating
        _RESOURCE_0.collect_goal,  # cooperate but tremble and defect
        _RESOURCE_0.interact_goal,  # continue defecting
        _RESOURCE_1.collect_goal,  # defect but tremble and cooperate
    ]
    actual, _ = _goals_from_observations(puppeteer, inventories, interactions)
    self.assertEqual(actual, expected)


class CorrigibleTest(parameterized.TestCase):

  def test(self):
    puppeteer = in_the_matrix.Corrigible(
        threshold=1,
        cooperate_resource=_RESOURCE_1,
        defect_resource=_RESOURCE_0,
        margin=2,
        tremble_probability=0,
    )
    inventories = [
        (2, 0, 1),  # not ready to interact
        (4, 1, 0),  # ready to interact if defecting
        (2, 1, 0),  # not ready to interact
        (1, 2, 0),  # not ready to interact
        (2, 1, 0),  # not ready to interact
        (1, 4, 0),  # ready to interact if cooperating
        (3, 1, 0),  # ready to interact if defecting
    ]
    interactions = [
        ((-1, -1, -1), (0, 1, 0)),  # coplayer cooperates
        ((-1, -1, -1), (0, 0, 1)),  # coplayer plays other
        ((-1, -1, -1), (1, 0, 0)),  # coplayer defects
        ((-1, -1, -1), (0, 1, 0)),  # coplayer cooperates
        ((-1, -1, -1), (-1, -1, -1)),  # no interaction
        ((-1, -1, -1), (1, 1, 0)),  # coplayer cooperates and defects
        ((-1, -1, -1), (2, 0, 1)),  # coplayer defects
    ]
    expected = [
        _RESOURCE_0.collect_goal,  # defect
        _RESOURCE_0.interact_goal,  # continue defecting
        _RESOURCE_1.collect_goal,  # cooperate
        _RESOURCE_1.collect_goal,  # cooperate
        _RESOURCE_1.collect_goal,  # continue cooperating
        _RESOURCE_1.interact_goal,  # continue cooperating
        _RESOURCE_0.interact_goal,  # defect
    ]
    actual, _ = _goals_from_observations(puppeteer, inventories, interactions)
    self.assertEqual(actual, expected)

  def test_tremble(self):
    puppeteer = in_the_matrix.Corrigible(
        threshold=1,
        cooperate_resource=_RESOURCE_1,
        defect_resource=_RESOURCE_0,
        margin=2,
        tremble_probability=1,
    )
    inventories = [
        (2, 0, 1),  # not ready to interact
        (4, 1, 0),  # ready to interact if defecting
        (2, 1, 0),  # not ready to interact
        (1, 2, 0),  # not ready to interact
        (2, 1, 0),  # not ready to interact
        (1, 4, 0),  # ready to interact if cooperating
        (3, 1, 0),  # ready to interact if defecting
    ]
    interactions = [
        ((-1, -1, -1), (0, 1, 0)),  # coplayer cooperates
        ((-1, -1, -1), (0, 0, 1)),  # coplayer plays other
        ((-1, -1, -1), (1, 0, 0)),  # coplayer defects
        ((-1, -1, -1), (0, 1, 0)),  # coplayer cooperates
        ((-1, -1, -1), (-1, -1, -1)),  # no interaction
        ((-1, -1, -1), (1, 1, 0)),  # coplayer cooperates and defects
        ((-1, -1, -1), (2, 0, 1)),  # coplayer defects
    ]
    expected = [
        _RESOURCE_0.collect_goal,  # defect
        _RESOURCE_0.interact_goal,  # continue defecting
        _RESOURCE_0.collect_goal,  # cooperate but tremble and defect
        _RESOURCE_0.collect_goal,  # cooperate but tremble and defect
        _RESOURCE_0.collect_goal,  # continue defecting
        _RESOURCE_0.collect_goal,  # continue defecting
        _RESOURCE_1.collect_goal,  # defect but tremble and cooperate
    ]
    actual, _ = _goals_from_observations(puppeteer, inventories, interactions)
    self.assertEqual(actual, expected)


class RespondToPreviousTest(parameterized.TestCase):

  def test(self):
    puppeteer = in_the_matrix.RespondToPrevious(
        responses={
            _RESOURCE_0: _RESOURCE_2,
            _RESOURCE_1: _RESOURCE_0,
            _RESOURCE_2: _RESOURCE_1,
        },
        margin=1,
    )
    inventories = [
        (1, 1, 1),
        (1, 2, 1),
        (1, 2, 3),
        (2, 3, 1),
        (3, 2, 1),
        (3, 2, 1),
        (2, 3, 1),
    ]
    interactions = [
        ((-1, -1, -1), (-1, -1, -1)),  # no interaction
        ((-1, -1, -1), (1, 0, 0)),  # opponent plays 0
        ((-1, -1, -1), (-1, -1, -1)),  # no interaction
        ((-1, -1, -1), (0, 1, 0)),  # opponent plays 1
        ((-1, -1, -1), (1, 1, 1)),  # no clear interaction
        ((-1, -1, -1), (0, 0, 1)),  # opponent plays 2
        ((-1, -1, -1), (-1, -1, -1)),  # no interaction
    ]
    expected = [
        mock.ANY,  # random
        _RESOURCE_2.collect_goal,
        _RESOURCE_2.interact_goal,
        _RESOURCE_0.collect_goal,
        _RESOURCE_0.interact_goal,
        _RESOURCE_1.collect_goal,
        _RESOURCE_1.interact_goal,
    ]
    actual, _ = _goals_from_observations(puppeteer, inventories, interactions)
    self.assertEqual(actual, expected)


class AlternatingSpecialistTest(parameterized.TestCase):

  def testOneInteractionPerOption(self):
    puppeteer = in_the_matrix.AlternatingSpecialist(
        targets=[_RESOURCE_0, _RESOURCE_1, _RESOURCE_2],
        interactions_per_target=1,
        margin=1,
    )
    inventories = [
        (1, 1, 1),
        (1, 2, 1),
        (1, 2, 3),
        (2, 3, 1),
        (3, 2, 1),
        (3, 2, 1),
        (2, 3, 1),
    ]
    interactions = [
        ((-1, -1, -1), (-1, -1, -1)),  # no interaction
        ((-1, -1, -1), (1, 0, 0)),  # opponent plays 0
        ((-1, -1, -1), (-1, -1, -1)),  # no interaction
        ((-1, -1, -1), (0, 1, 0)),  # opponent plays 1
        ((-1, -1, -1), (2, 2, 2)),  # no clear interaction
        ((-1, -1, -1), (0, 0, 1)),  # opponent plays 2
        ((-1, -1, -1), (-1, -1, -1)),  # no interaction
    ]
    expected = [
        _RESOURCE_0.collect_goal,
        _RESOURCE_1.interact_goal,
        _RESOURCE_1.collect_goal,
        _RESOURCE_2.collect_goal,
        _RESOURCE_0.interact_goal,
        _RESOURCE_1.collect_goal,
        _RESOURCE_1.interact_goal,
    ]
    actual, _ = _goals_from_observations(
        puppeteer, inventories, interactions)
    self.assertEqual(actual, expected)

  def testTwoInteractionsPerOption(self):
    puppeteer = in_the_matrix.AlternatingSpecialist(
        targets=[_RESOURCE_0, _RESOURCE_1, _RESOURCE_2],
        interactions_per_target=2,
        margin=1,
    )
    inventories = [
        (1, 1, 1),
        (1, 1, 1),
        (1, 2, 1),
        (1, 2, 1),
        (1, 2, 3),
        (1, 2, 3),
        (2, 3, 1),
        (2, 3, 1),
        (3, 2, 1),
        (3, 2, 1),
        (3, 2, 1),
        (3, 2, 1),
        (2, 3, 1),
        (2, 3, 1),
    ]
    interactions = [
        ((-1, -1, -1), (-1, -1, -1)),  # no interaction
        ((-1, -1, -1), (-1, -1, -1)),  # no interaction
        ((-1, -1, -1), (1, 0, 0)),  # opponent plays 0
        ((-1, -1, -1), (1, 0, 0)),  # opponent plays 0
        ((-1, -1, -1), (-1, -1, -1)),  # no interaction
        ((-1, -1, -1), (-1, -1, -1)),  # no interaction
        ((-1, -1, -1), (0, 1, 0)),  # opponent plays 1
        ((-1, -1, -1), (0, 1, 0)),  # opponent plays 1
        ((-1, -1, -1), (2, 2, 2)),  # no clear interaction
        ((-1, -1, -1), (2, 2, 2)),  # no clear interaction
        ((-1, -1, -1), (0, 0, 1)),  # opponent plays 2
        ((-1, -1, -1), (0, 0, 1)),  # opponent plays 2
        ((-1, -1, -1), (-1, -1, -1)),  # no interaction
        ((-1, -1, -1), (-1, -1, -1)),  # no interaction
    ]
    expected = [
        _RESOURCE_0.collect_goal,
        _RESOURCE_0.collect_goal,
        _RESOURCE_0.collect_goal,
        _RESOURCE_1.interact_goal,
        _RESOURCE_1.collect_goal,
        _RESOURCE_1.collect_goal,
        _RESOURCE_1.interact_goal,
        _RESOURCE_2.collect_goal,
        _RESOURCE_2.collect_goal,
        _RESOURCE_0.interact_goal,
        _RESOURCE_0.interact_goal,
        _RESOURCE_1.collect_goal,
        _RESOURCE_1.interact_goal,
        _RESOURCE_1.interact_goal,
    ]
    actual, _ = _goals_from_observations(
        puppeteer, inventories, interactions)
    self.assertEqual(actual, expected)


if __name__ == '__main__':
  absltest.main()
