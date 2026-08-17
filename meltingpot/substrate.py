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
"""Substrate builder.

A substrate's configuration describes its observations, actions, and player
roles. Use `get_config(name)` to inspect it before building an environment.
In particular, `config.valid_roles` lists the accepted role names and
`config.default_player_roles` gives a ready-to-use role assignment.

For example::

  config = get_config("predator_prey__open")
  environment = build(
      "predator_prey__open", roles=config.default_player_roles)

To choose a different assignment, pass one role per player, with every role
drawn from `config.valid_roles`.
"""

from collections.abc import Sequence

from meltingpot.configs import substrates as substrate_configs
from meltingpot.utils.substrates import substrate
from meltingpot.utils.substrates import substrate_factory
from ml_collections import config_dict

SUBSTRATES = substrate_configs.SUBSTRATES


def get_config(name: str) -> config_dict.ConfigDict:
  """Returns the locked configuration for the specified substrate.

  The returned configuration includes `valid_roles`, the role names accepted by
  `build`, and `default_player_roles`, a valid default assignment whose length
  determines the default number of players.

  Args:
    name: Name of a substrate in `SUBSTRATES`.

  Returns:
    The locked substrate configuration.
  """
  return substrate_configs.get_config(name).lock()


def build(name: str, *, roles: Sequence[str]) -> substrate.Substrate:
  """Builds an instance of the specified substrate.

  Args:
    name: Name of the substrate.
    roles: One role string per player. Role names must come from
      `get_config(name).valid_roles`; use
      `get_config(name).default_player_roles` for the substrate's standard
      assignment. The length of this sequence determines the number of players.

  Returns:
    The training substrate.
  """
  return get_factory(name).build(roles)


def build_from_config(
    config: config_dict.ConfigDict,
    *,
    roles: Sequence[str],
) -> substrate.Substrate:
  """Builds a substrate from the provided config.

  Args:
    config: Configuration resulting from `get_config`, optionally customized.
    roles: One role string per player. Role names must come from
      `config.valid_roles`; use `config.default_player_roles` for the standard
      assignment. The length of this sequence determines the number of players.

  Returns:
    The training substrate.
  """
  return get_factory_from_config(config).build(roles)


def get_factory(name: str) -> substrate_factory.SubstrateFactory:
  """Returns the factory for the specified substrate."""
  config = substrate_configs.get_config(name)
  return get_factory_from_config(config)


def get_factory_from_config(
    config: config_dict.ConfigDict) -> substrate_factory.SubstrateFactory:
  """Returns a factory from the provided config."""

  def lab2d_settings_builder(roles):
    return config.lab2d_settings_builder(roles=roles, config=config)

  return substrate_factory.SubstrateFactory(
      lab2d_settings_builder=lab2d_settings_builder,
      individual_observations=config.individual_observation_names,
      global_observations=config.global_observation_names,
      action_table=config.action_set,
      timestep_spec=config.timestep_spec,
      action_spec=config.action_spec,
      valid_roles=config.valid_roles,
      default_player_roles=config.default_player_roles)
