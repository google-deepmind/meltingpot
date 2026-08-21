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
"""Validation tests for running-with-scissors puppeteers."""

from unittest import mock

from absl.testing import absltest
from meltingpot.utils.puppeteers import in_the_matrix
from meltingpot.utils.puppeteers import running_with_scissors_in_the_matrix


def _resource(index):
  return in_the_matrix.Resource(
      index=index,
      collect_goal=mock.sentinel.collect,
      interact_goal=mock.sentinel.interact,
  )


class CounterPreviousValidationTest(absltest.TestCase):

  def test_rejects_duplicate_resource_indices(self):
    with self.assertRaisesRegex(ValueError, 'must have distinct indices'):
      running_with_scissors_in_the_matrix.CounterPrevious(
          rock_resource=_resource(0),
          paper_resource=_resource(0),
          scissors_resource=_resource(2),
          margin=1,
      )


if __name__ == '__main__':
  absltest.main()
