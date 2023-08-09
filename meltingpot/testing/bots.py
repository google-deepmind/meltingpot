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
"""Utilities for testing bots."""

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from meltingpot.utils.policies import policy as policy_lib
import tree


class BotTestCase(parameterized.TestCase):
  """Base test case for bots."""

  def assert_compatible(
      self,
      policy: policy_lib.Policy,
      timestep_spec: dm_env.TimeStep,
      action_spec: dm_env.specs.DiscreteArray,
  ) -> None:
    """Asserts that policy matches the given spec.

    Args:
      policy: policy to check.
      timestep_spec: the timestep spec to check the policy against.
      action_spec: the action spec to check the policy against.

    Raises:
      AssertionError: the env doesn't match its spec.
    """
    timestep = tree.map_structure(
        lambda spec: spec.generate_value(), timestep_spec)
    prev_state = policy.initial_state()
    try:
      action, _ = policy.step(timestep, prev_state)
    except Exception:  # pylint: disable=broad-except
      self.fail(f'Failed step with timestep matching spec {timestep_spec!r}.')
    try:
      action_spec.validate(action)
    except ValueError:
      self.fail(f'Returned action {action!r} does not match action_spec '
                f'{action_spec!r}.')


if __name__ == '__main__':
  absltest.main()
