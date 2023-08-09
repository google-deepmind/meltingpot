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
"""Tests for gift_refinements puppeteers."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meltingpot.testing import puppeteers
from meltingpot.utils.puppeteers import gift_refinements

_COLLECT = mock.sentinel.collect
_CONSUME = mock.sentinel.consume
_GIFT = mock.sentinel.gift


class GiftRefinementsCooperatorTest(parameterized.TestCase):

  @parameterized.parameters(
      [(0, 0, 0), _COLLECT],
      [(0, 0, 1), _CONSUME],
      [(1, 0, 0), _GIFT],
      [(2, 0, 0), _GIFT],
      [(2, 0, 2), _CONSUME],
      [(1, 1, 0), _CONSUME],
      [(1, 1, 1), _CONSUME],
      [(5, 2, 0), _CONSUME],
      [(5, 5, 5), _CONSUME],
  )
  def test(self, inventory, expected):
    puppeteer = gift_refinements.GiftRefinementsCooperator(
        collect_goal=_COLLECT,
        consume_goal=_CONSUME,
        gift_goal=_GIFT,
    )
    (actual,), _ = puppeteers.goals_from_observations(
        puppeteer, [{'INVENTORY': inventory}]
    )
    self.assertEqual(actual, expected)


class GiftRefinementsExtremeCooperatorTest(parameterized.TestCase):

  @parameterized.parameters(
      [(0, 0, 0), _COLLECT],
      [(0, 0, 1), _CONSUME],
      [(1, 0, 0), _GIFT],
      [(2, 0, 2), _CONSUME],
      [(1, 1, 0), _GIFT],
      [(1, 1, 1), _CONSUME],
      [(5, 2, 0), _GIFT],
      [(5, 5, 5), _CONSUME],
  )
  def test(self, inventory, expected):
    puppeteer = gift_refinements.GiftRefinementsExtremeCooperator(
        collect_goal=_COLLECT,
        consume_goal=_CONSUME,
        gift_goal=_GIFT,
    )
    (actual,), _ = puppeteers.goals_from_observations(
        puppeteer, [{'INVENTORY': inventory}]
    )
    self.assertEqual(actual, expected)


if __name__ == '__main__':
  absltest.main()
