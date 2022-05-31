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
"""Tests of bots."""

from absl.testing import absltest
from absl.testing import parameterized
import tree

from meltingpot.python import bot as bot_factory
from meltingpot.python import substrate as substrate_factory
from meltingpot.python.utils.scenarios import substrate_transforms


def _get_specs(substrate):
  config = substrate_factory.get_config(substrate)
  timestep_spec = substrate_transforms.tf1_bot_timestep_spec(
      timestep_spec=config.timestep_spec,
      action_spec=config.action_spec,
      num_players=config.num_players)
  return timestep_spec, config.action_spec


class BotTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (bot, bot) for bot in bot_factory.AVAILABLE_BOTS
  )
  def test_step_without_error(self, bot_name):
    bot_config = bot_factory.get_config(bot_name)
    timestep_spec, action_spec = _get_specs(bot_config.substrate)
    with bot_factory.build(bot_config) as policy:
      self.assert_compatible(policy, timestep_spec, action_spec)

  def assert_compatible(self, policy, timestep_spec, action_spec):
    timestep = tree.map_structure(
        lambda spec: spec.generate_value(), timestep_spec)
    prev_state = policy.initial_state()
    try:
      action, _ = policy.step(timestep, prev_state)
    except Exception:  # pylint: disable=broad-except
      self.fail(f'Step with spec-defined timestep {timestep!r} failed.')
    try:
      action_spec.validate(action)
    except ValueError:
      self.fail(f'Returned action {action!r} does not match action_spec '
                f'{action_spec!r}.')


if __name__ == '__main__':
  absltest.main()
