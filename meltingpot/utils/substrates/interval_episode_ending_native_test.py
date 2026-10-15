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
"""Native regressions for interval episode ending test."""

from pathlib import Path

from absl.testing import absltest
from absl.testing import parameterized
import dmlab2d
from dmlab2d import runfiles_helper
from meltingpot import substrate
from meltingpot.utils.substrates import builder
import numpy as np


class IntervalEpisodeEndingTest(absltest.TestCase):

  def test_native_lua_regressions(self):
    root = Path(__file__).resolve().parents[3]
    lab = dmlab2d.Lab2d(
        runfiles_helper.find(),
        {
            'levelDirectory': str(root),
            'levelName': 'meltingpot/lua/modules/interval_episode_ending_test',
        },
    )
    self.assertEmpty(lab.observation_names())


class NativeIntervalEpisodeLengthTest(parameterized.TestCase):

  def episode_lengths(self, minimum, interval, probability, seed, cap, count):
    config = substrate.get_config('commons_harvest__open')
    settings = config.lab2d_settings_builder(roles=('default',), config=config)
    settings['maxEpisodeLengthFrames'] = cap
    components = settings['simulation']['scene']['components']
    components[:] = [
        item
        for item in components
        if item['component']
        not in ('StochasticEpisodeEnding', 'StochasticIntervalEpisodeEnding')
    ]
    components.append({
        'component': 'StochasticIntervalEpisodeEnding',
        'kwargs': {
            'minimumFramesPerEpisode': minimum,
            'intervalLength': interval,
            'probabilityTerminationPerInterval': probability,
        },
    })
    lengths = []
    with builder.builder(settings, env_seed=seed) as env:
      actions = {
          key: np.zeros(spec.shape, spec.dtype)
          for key, spec in env.action_spec().items()
      }
      for _ in range(count):
        timestep = env.reset()
        self.assertTrue(timestep.first())
        frame = 0
        while not timestep.last():
          timestep = env.step(actions)
          frame += 1
          self.assertLessEqual(frame, cap)
        lengths.append(frame)
    return lengths

  @parameterized.parameters(
      (1, 1),
      (1, 2),
      (1, 5),
      (3, 5),
      (5, 5),
      (7, 5),
      (10, 3),
      (21, 10),
      (1000, 100),
  )
  def test_probability_one_ends_at_first_eligible_boundary(
      self, minimum, interval
  ):
    expected = ((minimum + interval - 1) // interval) * interval
    lengths = self.episode_lengths(
        minimum, interval, 1.0, 17, expected + interval + 2, 2
    )
    self.assertEqual(lengths, [expected, expected])

  @parameterized.parameters(7, 17, 47)
  def test_random_endings_remain_aligned_and_seed_reproducible(self, seed):
    first = self.episode_lengths(7, 5, 0.4, seed, 500, 3)
    second = self.episode_lengths(7, 5, 0.4, seed, 500, 3)
    self.assertEqual(first, second)
    for length in first:
      self.assertGreaterEqual(length, 7)
      self.assertEqual(length % 5, 0)

  def test_hard_frame_limit_still_takes_precedence(self):
    self.assertEqual(self.episode_lengths(5, 5, 1.0, 17, 3, 2), [3, 3])


if __name__ == '__main__':
  absltest.main()
