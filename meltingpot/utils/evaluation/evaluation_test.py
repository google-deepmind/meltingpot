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
"""Tests for evaluation utilities."""

from unittest import mock

from absl.testing import absltest
import dm_env
from meltingpot.utils.evaluation import evaluation


class RunEpisodeTest(absltest.TestCase):

  def test_does_not_await_action_after_terminal_timestep(self):
    population = mock.Mock()
    substrate = mock.Mock()
    first = dm_env.restart(observation=())
    last = dm_env.termination(reward=0.0, observation=())
    substrate.reset.return_value = first
    substrate.step.return_value = last
    population.await_action.return_value = mock.sentinel.action

    evaluation.run_episode(population, substrate)

    population.reset.assert_called_once_with()
    substrate.reset.assert_called_once_with()
    substrate.step.assert_called_once_with(mock.sentinel.action)
    self.assertEqual(
        population.send_timestep.call_args_list,
        [mock.call(first), mock.call(last)],
    )
    population.await_action.assert_called_once_with()


if __name__ == '__main__':
  absltest.main()
