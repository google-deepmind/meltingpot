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
import dm_env

from meltingpot.python import bot as bot_factory
from meltingpot.python import substrate as substrate_factory
from meltingpot.python.utils.scenarios import substrate_transforms
from meltingpot.python.utils.scenarios.wrappers import base


class _MultiToSinglePlayer(base.Wrapper):
  """Adapter to convert a multiplayer environment to a single player version."""

  def __init__(self, env) -> None:
    """Initializes the environment.

    Args:
      env: multiplayer environment to make single player.
    """
    super().__init__(env)
    self._num_players = len(super().action_spec())

  def reset(self):
    """See base class."""
    timestep = super().reset()
    return dm_env.TimeStep(
        step_type=timestep.step_type,
        reward=timestep.reward[0],
        discount=timestep.discount,
        observation=timestep.observation[0])

  def step(self, action):
    """See base class."""
    actions = [action] * self._num_players
    timestep = super().step(actions)
    return dm_env.TimeStep(
        step_type=timestep.step_type,
        reward=timestep.reward[0],
        discount=timestep.discount,
        observation=timestep.observation[0])

  def observation_spec(self):
    """See base class."""
    return super().observation_spec()[0]

  def action_spec(self):
    """See base class."""
    return super().action_spec()[0]


def build_environment(substrate):
  config = substrate_factory.get_config(substrate)
  env = substrate_factory.build(config)
  env = substrate_transforms.with_tf1_bot_required_observations(env)
  return _MultiToSinglePlayer(env)


class BotTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (bot, bot) for bot in bot_factory.AVAILABLE_BOTS
  )
  def test_step_without_error(self, bot_name):
    bot_config = bot_factory.get_config(bot_name)
    with bot_factory.build(bot_config) as policy:
      prev_state = policy.initial_state()
      with build_environment(bot_config.substrate) as env:
        timestep = env.reset()
        action, _ = policy.step(timestep, prev_state)
        env.step(action)


if __name__ == '__main__':
  absltest.main()
