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
"""Bot factory."""

from ml_collections import config_dict

from meltingpot.python.configs import bots as bot_config
from meltingpot.python.utils.bots import policy

AVAILABLE_BOTS = frozenset(bot_config.BOTS)

# TODO(b/227143834): Remove aliases once internal deps have been removed.
Policy = policy.Policy
PuppetPolicy = policy.PuppetPolicy
SavedModelPolicy = policy.SavedModelPolicy
State = policy.State


def get_config(bot_name: str) -> config_dict.ConfigDict:
  """Returns a config for the specified bot.

  Args:
    bot_name: name of the bot. Must be in AVAILABLE_BOTS.
  """
  if bot_name not in AVAILABLE_BOTS:
    raise ValueError(f'Unknown bot {bot_name!r}.')
  bot = bot_config.BOTS[bot_name]
  config = config_dict.create(
      bot_name=bot_name,
      substrate=bot.substrate,
      puppeteer_fn=bot.puppeteer_fn,
      saved_model_path=bot.model_path,
  )
  return config.lock()


def build(config: config_dict.ConfigDict) -> policy.Policy:
  """Builds a bot policy for the given config.

  Args:
    config: bot config resulting from `get_config`.

  Returns:
    The bot policy.
  """
  if config.puppeteer_fn:
    return policy.PuppetPolicy(config.puppeteer_fn, config.saved_model_path)
  else:
    return policy.SavedModelPolicy(config.saved_model_path)
