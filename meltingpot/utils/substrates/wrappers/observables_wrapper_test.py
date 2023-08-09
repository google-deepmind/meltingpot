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
"""Tests for observables_wrapper."""

import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dmlab2d
from meltingpot.utils.substrates.wrappers import observables_wrapper


class ObservablesWrapperTest(parameterized.TestCase):

  def test_observables(self):
    base = mock.create_autospec(
        dmlab2d.Environment, instance=True, spec_set=True)
    with observables_wrapper.ObservablesWrapper(base) as env:
      received = []
      observables = env.observables()
      for field in dataclasses.fields(observables):
        getattr(observables, field.name).subscribe(
            on_next=received.append,
            on_error=lambda e: received.append(type(e)),
            on_completed=lambda: received.append('DONE'),
        )

      base.reset.return_value = mock.sentinel.timestep_0
      base.events.return_value = [mock.sentinel.events_0]
      env.reset()
      base.step.return_value = mock.sentinel.timestep_1
      base.events.return_value = [mock.sentinel.events_1]
      env.step(mock.sentinel.action_1)
      base.step.return_value = mock.sentinel.timestep_2
      base.events.return_value = [mock.sentinel.events_2]
      env.step(mock.sentinel.action_2)

    self.assertSequenceEqual(received, [
        mock.sentinel.timestep_0,
        mock.sentinel.events_0,
        mock.sentinel.action_1,
        mock.sentinel.timestep_1,
        mock.sentinel.events_1,
        mock.sentinel.action_2,
        mock.sentinel.timestep_2,
        mock.sentinel.events_2,
        'DONE',
        'DONE',
        'DONE',
    ])


if __name__ == '__main__':
  absltest.main()
