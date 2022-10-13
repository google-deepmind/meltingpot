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
from meltingpot.python.utils.policies import policy
from meltingpot.python.utils.policies import puppet_policy
from meltingpot.python.utils.policies import saved_model_policy

AVAILABLE_BOTS = frozenset(bot_config.BOT_CONFIGS)


def get_config(bot_name: str) -> config_dict.ConfigDict:
  """Returns a config for the specified bot.

  Args:
    bot_name: name of the bot. Must be in AVAILABLE_BOTS.
  """
  if bot_name not in AVAILABLE_BOTS:
    raise ValueError(f'Unknown bot {bot_name!r}.')
  bot = bot_config.BOT_CONFIGS[bot_name]
  config = config_dict.create(
      bot_name=bot_name,
      substrate=bot.substrate,
      puppeteer_builder=bot.puppeteer_builder,
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
  saved_model = saved_model_policy.SavedModelPolicy(config.saved_model_path)
  if config.puppeteer_builder:
    puppeteer = config.puppeteer_builder()
    return puppet_policy.PuppetPolicy(puppeteer=puppeteer, puppet=saved_model)
  else:
    return saved_model
