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
from absl.testing import absltest
import dm_env
from meltingpot.testing import mocks
from meltingpot.testing import substrates
from meltingpot.utils.substrates import specs as meltingpot_specs
from meltingpot.utils.substrates import substrate
import numpy as np


class MocksTest(substrates.SubstrateTestCase):

  def test_value_from_specs(self):
    specs = (
        {'a': dm_env.specs.Array([1, 2, 3], dtype=np.uint8)},
        {'b': dm_env.specs.Array([1, 2, 3], dtype=np.uint8)},
    )
    actual = mocks._values_from_specs(specs)
    expected = (
        {'a': np.zeros([1, 2, 3], dtype=np.uint8)},
        {'b': np.ones([1, 2, 3], dtype=np.uint8)},
    )
    np.testing.assert_equal(actual, expected)

  def test_mock_substrate(self):
    num_players = 2
    num_actions = 3
    observation_spec = {'a': dm_env.specs.Array([], dtype=np.uint8)}
    mock = mocks.build_mock_substrate(
        num_players=num_players,
        num_actions=num_actions,
        observation_spec=observation_spec)

    expected_observation = (
        {'a': np.zeros([], dtype=np.uint8)},
        {'a': np.ones([], dtype=np.uint8)},
    )
    expected_reward = tuple(float(n) for n in range(num_players))

    with self.subTest('is_substrate'):
      self.assertIsInstance(mock, substrate.Substrate)
    with self.subTest('error_getting_invalid'):
      with self.assertRaises(AttributeError):
        mock.no_such_method()  # pytype: disable=attribute-error
    with self.subTest('error_setting_invalid'):
      with self.assertRaises(AttributeError):
        mock.no_such_method = None

    with self.subTest('can_enter_context'):
      with mock as c:
        self.assertEqual(c.discount_spec(), mock.discount_spec())

    with self.subTest('action_spec'):
      self.assertEqual(mock.action_spec(), (meltingpot_specs.action(3),) * 2)
    with self.subTest('reward_spec'):
      self.assertLen(mock.reward_spec(), num_players)
    with self.subTest('observation_spec'):
      self.assertLen(mock.observation_spec(), num_players)

    with self.subTest('reset'):
      expected = dm_env.TimeStep(
          step_type=dm_env.StepType.FIRST,
          observation=expected_observation,
          reward=(0.,) * num_players,
          discount=0.,
      )
      self.assertEqual(mock.reset(), expected)
    with self.subTest('step'):
      expected = dm_env.transition(expected_reward, expected_observation)
      self.assertEqual(mock.step([0, 0]), expected)
    with self.subTest('events'):
      self.assertEmpty(mock.events())
    with self.subTest('observation'):
      self.assertEqual(mock.observation(), expected_observation)

  def test_mock_substrate_like(self):
    mock = mocks.build_mock_substrate_like('clean_up')
    self.assert_step_matches_specs(mock)

  def test_mock_scenario_like(self):
    mock = mocks.build_mock_scenario_like('clean_up_0')
    self.assert_step_matches_specs(mock)


if __name__ == '__main__':
  absltest.main()
