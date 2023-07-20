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
"""Bot factory."""

import functools

from meltingpot import substrate
from meltingpot.configs import bots as bot_configs
from meltingpot.utils.policies import fixed_action_policy
from meltingpot.utils.policies import policy
from meltingpot.utils.policies import policy_factory
from meltingpot.utils.policies import puppet_policy
from meltingpot.utils.policies import saved_model_policy
from meltingpot.utils.substrates import specs

NOOP_BOT_NAME = 'noop_bot'
NOOP_ACTION = 0

BOTS = frozenset(bot_configs.BOT_CONFIGS) | {NOOP_BOT_NAME}


def get_config(bot_name: str) -> bot_configs.BotConfig:
  """Returns the config for the specified bot."""
  return bot_configs.BOT_CONFIGS[bot_name]


def build(name: str) -> policy.Policy:
  """Builds a policy for the specified bot.

  Args:
    name: the name of the bot.

  Returns:
    The bot policy.
  """
  return get_factory(name).build()


def build_from_config(config: bot_configs.BotConfig) -> policy.Policy:
  """Builds a policy from the provided bot config.

  Args:
    config: bot config.

  Returns:
    The bot policy.
  """
  saved_model = saved_model_policy.SavedModelPolicy(config.model_path)
  if config.puppeteer_builder:
    puppeteer = config.puppeteer_builder()
    return puppet_policy.PuppetPolicy(puppeteer=puppeteer, puppet=saved_model)
  else:
    return saved_model


def get_factory(name: str) -> policy_factory.PolicyFactory:
  """Returns a factory for the specified bot."""
  if name == NOOP_BOT_NAME:
    return policy_factory.PolicyFactory(
        timestep_spec=specs.timestep({}),
        action_spec=specs.action(NOOP_ACTION + 1),
        builder=functools.partial(fixed_action_policy.FixedActionPolicy,
                                  NOOP_ACTION))
  else:
    config = bot_configs.BOT_CONFIGS[name]
    return get_factory_from_config(config)


def get_factory_from_config(
    config: bot_configs.BotConfig) -> policy_factory.PolicyFactory:
  """Returns a factory from the provided config."""
  substrate_factory = substrate.get_factory(config.substrate)
  return policy_factory.PolicyFactory(
      timestep_spec=substrate_factory.timestep_spec(),
      action_spec=substrate_factory.action_spec(),
      builder=lambda: build_from_config(config))
