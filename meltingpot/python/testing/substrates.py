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
"""Test utilities for substrates."""

from absl.testing import parameterized


class SubstrateTestCase(parameterized.TestCase):
  """Base class for tests of substrates."""

  def assert_step_matches_specs(self, env):
    """Asserts that env accepts an action permitted by its spec.

    Args:
      env: environment to check.

    Raises:
      AssertionError: the env doesn't match its spec.
    """
    env.reset()

    action = [spec.maximum for spec in env.action_spec()]
    try:
      timestep = env.step(action)
    except Exception:  # pylint: disable=broad-except
      self.fail(f'Failure when passing action {action!r}.')

    try:
      env.discount_spec().validate(timestep.discount)
    except ValueError:
      self.fail('Discount does not match spec.')

    reward_spec = env.reward_spec()
    if len(reward_spec) != len(timestep.reward):
      self.fail(f'Spec is length {len(reward_spec)} but reward is length '
                f'{len(timestep.reward)}.')
    for n, spec in enumerate(reward_spec):
      try:
        spec.validate(timestep.reward[n])
      except ValueError:
        self.fail(f'Reward {n} does not match spec.')

    observations = timestep.observation
    observation_specs = env.observation_spec()
    if len(observation_specs) != len(observations):
      self.fail(f'Spec is length {len(observation_specs)} but observations '
                f'are length {len(observations)}')
    for n, (observation, spec) in enumerate(
        zip(observations, observation_specs)):
      if set(spec) != set(observation):
        self.fail(f'Observation {n} keys {set(observation)!r} do not match '
                  f'spec keys {set(observation)!r}.')
      for key in spec:
        try:
          spec[key].validate(observation[key])
        except ValueError:
          self.fail(f'Observation {n} key {key!r} does not match spec.')
