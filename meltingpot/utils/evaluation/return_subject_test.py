# Copyright 2024 DeepMind Technologies Limited.
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
from meltingpot.utils.evaluation import return_subject
import numpy as np


def _send_timesteps_to_subject(subject, timesteps):
  results = []
  subject.subscribe(on_next=results.append)

  for n, timestep in enumerate(timesteps):
    subject.on_next(timestep)
    if results:
      return n, results.pop()
  return None, None


class ReturnSubjectTest(absltest.TestCase):

  def test(self):
    timesteps = [
        dm_env.restart(observation=[{}])._replace(reward=[0, 0]),
        dm_env.transition(observation=[{}], reward=[2, 4]),
        dm_env.termination(observation=[{}], reward=[1, 3]),
    ]
    subject = return_subject.ReturnSubject()
    step_written, episode_returns = _send_timesteps_to_subject(
        subject, timesteps
    )

    with self.subTest('written_on_final_step'):
      self.assertEqual(step_written, 2)

    with self.subTest('returns'):
      np.testing.assert_equal(episode_returns, [3, 7])

if __name__ == '__main__':
  absltest.main()
