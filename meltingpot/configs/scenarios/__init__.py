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
"""Test scenario configurations."""

import collections
import dataclasses
from typing import AbstractSet, Collection, Mapping, Optional, Sequence

import immutabledict


@dataclasses.dataclass(frozen=True)
class ScenarioConfig:
  """Scenario config.

  Attributes:
    description: a description of the scenario.
    tags: tags for the scenario.
    substrate: the substrate the scenario is based on.
    roles: indicates what role the player in the corresponding player slot has.
    is_focal: indicates whether the corresponding player slot is to be filled by
      a focal player or a bot.
    bots_by_role: names of the bots to sample from to fill the bot slots with
      the corresponding role.
  """
  description: str
  tags: AbstractSet[str]
  substrate: str
  roles: Sequence[str]
  is_focal: Sequence[bool]
  bots_by_role: Mapping[str, AbstractSet[str]]

  def __post_init__(self):
    object.__setattr__(self, 'tags', frozenset(self.tags))
    object.__setattr__(self, 'roles', tuple(self.roles))
    object.__setattr__(self, 'is_focal', tuple(self.is_focal))
    bots_by_role = immutabledict.immutabledict({
        role: frozenset(bots) for role, bots in self.bots_by_role.items()
    })
    object.__setattr__(self, 'bots_by_role', bots_by_role)


# Local additions/overrides.
SCENARIO_CONFIGS: Mapping[str, ScenarioConfig] = immutabledict.immutabledict(
    # keep-sorted start numeric=yes block=yes
    allelopathic_harvest__open_0=ScenarioConfig(
        description=(
            'visiting a population where planting green berries is the ' +
            'prevailing convention'),
        tags={
            'visitor',
            'convention_following',
        },
        substrate='allelopathic_harvest__open',
        roles=['player_who_likes_red',] * 8 + ['player_who_likes_green',] * 8,
        is_focal=(True,) * 4 + (False,) * 12,
        bots_by_role={
            # The same bots can play both roles.
            'player_who_likes_red': {
                'allelopathic_harvest__open__bot_that_supports_green_0',
                'allelopathic_harvest__open__bot_that_supports_green_1',
                'allelopathic_harvest__open__bot_that_supports_green_2',
                'allelopathic_harvest__open__bot_that_supports_green_3',
            },
            'player_who_likes_green': {
                'allelopathic_harvest__open__bot_that_supports_green_0',
                'allelopathic_harvest__open__bot_that_supports_green_1',
                'allelopathic_harvest__open__bot_that_supports_green_2',
                'allelopathic_harvest__open__bot_that_supports_green_3',
            },
        },
    ),
    allelopathic_harvest__open_1=ScenarioConfig(
        description=(
            'visiting a population where planting red berries is the ' +
            'prevailing convention'),
        tags={
            'visitor',
            'convention_following',
        },
        substrate='allelopathic_harvest__open',
        roles=['player_who_likes_red',] * 8 + ['player_who_likes_green',] * 8,
        is_focal=(True,) * 4 + (False,) * 12,
        bots_by_role={
            # The same bots can play both roles.
            'player_who_likes_red': {
                'allelopathic_harvest__open__bot_that_supports_red_0',
                'allelopathic_harvest__open__bot_that_supports_red_1',
                'allelopathic_harvest__open__bot_that_supports_red_2',
                'allelopathic_harvest__open__bot_that_supports_red_3',
            },
            'player_who_likes_green': {
                'allelopathic_harvest__open__bot_that_supports_red_0',
                'allelopathic_harvest__open__bot_that_supports_red_1',
                'allelopathic_harvest__open__bot_that_supports_red_2',
                'allelopathic_harvest__open__bot_that_supports_red_3',
            },
        },
    ),
    allelopathic_harvest__open_2=ScenarioConfig(
        description=(
            'focals are resident and visited by bots who plant either red or ' +
            'green'),
        tags={
            'resident',
        },
        substrate='allelopathic_harvest__open',
        roles=['player_who_likes_red',] * 8 + ['player_who_likes_green',] * 8,
        is_focal=(True,) * 14 + (False,) * 2,
        bots_by_role={
            'player_who_likes_green': {
                'allelopathic_harvest__open__bot_that_supports_red_0',
                'allelopathic_harvest__open__bot_that_supports_red_1',
                'allelopathic_harvest__open__bot_that_supports_red_2',
                'allelopathic_harvest__open__bot_that_supports_red_3',
                'allelopathic_harvest__open__bot_that_supports_green_0',
                'allelopathic_harvest__open__bot_that_supports_green_1',
                'allelopathic_harvest__open__bot_that_supports_green_2',
                'allelopathic_harvest__open__bot_that_supports_green_3',
            },
        },
    ),
    allelopathic_harvest__open_3=ScenarioConfig(
        description='focals like red, visited by convention followers',
        tags={
            'convention_following',
        },
        substrate='allelopathic_harvest__open',
        roles=['player_who_likes_red',] * 8 + ['player_who_likes_green',] * 8,
        is_focal=(True,) * 8 + (False,) * 8,
        bots_by_role={
            'player_who_likes_green': {
                'allelopathic_harvest__open__puppet_convention_follower_0',
            },
        },
    ),
    allelopathic_harvest__open_4=ScenarioConfig(
        description=('focals like red, visited by mixture of convention ' +
                     'followers and bots who like red'),
        tags={
            'convention_following',
        },
        substrate='allelopathic_harvest__open',
        roles=['player_who_likes_red',] * 8 + ['player_who_likes_green',] * 8,
        is_focal=(True,) * 8 + (False,) * 8,
        bots_by_role={
            'player_who_likes_green': {
                'allelopathic_harvest__open__bot_that_supports_red_0',
                'allelopathic_harvest__open__puppet_convention_follower_0',
            },
        },
    ),
    allelopathic_harvest__open_5=ScenarioConfig(
        description=('focals like red, visited by mixture of convention ' +
                     'followers and bots who like green'),
        tags={
            'convention_following',
        },
        substrate='allelopathic_harvest__open',
        roles=['player_who_likes_red',] * 8 + ['player_who_likes_green',] * 8,
        is_focal=(True,) * 8 + (False,) * 8,
        bots_by_role={
            'player_who_likes_green': {
                'allelopathic_harvest__open__bot_that_supports_green_0',
                'allelopathic_harvest__open__puppet_convention_follower_0',
            },
        },
    ),
    allelopathic_harvest__open_6=ScenarioConfig(
        description=('not very crowded, focals like red, visited by mixture ' +
                     'of convention followers and bots who like green'),
        tags={
            'convention_following',
        },
        substrate='allelopathic_harvest__open',
        roles=['player_who_likes_red',] * 5 + ['player_who_likes_green',] * 5,
        is_focal=(True,) * 5 + (False,) * 5,
        bots_by_role={
            'player_who_likes_green': {
                'allelopathic_harvest__open__bot_that_supports_green_0',
                'allelopathic_harvest__open__puppet_convention_follower_0',
            },
        },
    ),
    allelopathic_harvest__open_7=ScenarioConfig(
        description=('not very crowded, focals like red, visited by ' +
                     'convention followers'),
        tags={
            'convention_following',
        },
        substrate='allelopathic_harvest__open',
        roles=['player_who_likes_red',] * 5 + ['player_who_likes_green',] * 5,
        is_focal=(True,) * 5 + (False,) * 5,
        bots_by_role={
            'player_who_likes_green': {
                'allelopathic_harvest__open__puppet_convention_follower_0',
            },
        },
    ),
    allelopathic_harvest__open_8=ScenarioConfig(
        description=('very low population, focals like red, meeting ' +
                     'convention followers'),
        tags={
            'convention_following',
        },
        substrate='allelopathic_harvest__open',
        roles=['player_who_likes_red',] * 2 + ['player_who_likes_green',] * 2,
        is_focal=(True,) * 2 + (False,) * 2,
        bots_by_role={
            'player_who_likes_green': {
                'allelopathic_harvest__open__puppet_convention_follower_0',
            },
        },
    ),
    allelopathic_harvest__open_9=ScenarioConfig(
        description=('not very crowded, focals are resident, visited by ' +
                     'a couple of convention followers'),
        tags={
            'convention_following',
        },
        substrate='allelopathic_harvest__open',
        roles=['player_who_likes_red',] * 5 + ['player_who_likes_green',] * 5,
        is_focal=(True,) * 8 + (False,) * 2,
        bots_by_role={
            'player_who_likes_green': {
                'allelopathic_harvest__open__puppet_convention_follower_0',
            },
        },
    ),
    allelopathic_harvest__open_10=ScenarioConfig(
        description=('very low population, focals are resident, and like ' +
                     'red berries, they are visited by a convention follower'),
        tags={
            'convention_following',
        },
        substrate='allelopathic_harvest__open',
        roles=['player_who_likes_red',] * 2 + ['player_who_likes_green',] * 2,
        is_focal=(True,) * 3 + (False,) * 1,
        bots_by_role={
            'player_who_likes_green': {
                'allelopathic_harvest__open__puppet_convention_follower_0',
            },
        },
    ),
    allelopathic_harvest__open_11=ScenarioConfig(
        description=('focals are resident, visited by a couple of convention ' +
                     'followers'),
        tags={
            'convention_following',
        },
        substrate='allelopathic_harvest__open',
        roles=['player_who_likes_red',] * 8 + ['player_who_likes_green',] * 8,
        is_focal=(True,) * 14 + (False,) * 2,
        bots_by_role={
            'player_who_likes_green': {
                'allelopathic_harvest__open__puppet_convention_follower_0',
            },
        },
    ),
    allelopathic_harvest__open_12=ScenarioConfig(
        description=('focals are resident, visited by a couple of convention ' +
                     'followers and bots who plant green'),
        tags={
            'convention_following',
        },
        substrate='allelopathic_harvest__open',
        roles=['player_who_likes_red',] * 8 + ['player_who_likes_green',] * 8,
        is_focal=(True,) * 12 + (False,) * 4,
        bots_by_role={
            'player_who_likes_green': {
                'allelopathic_harvest__open__bot_that_supports_green_0',
                'allelopathic_harvest__open__puppet_convention_follower_0',
            },
        },
    ),
    bach_or_stravinsky_in_the_matrix__arena_0=ScenarioConfig(
        description='visiting background population who picks bach',
        tags={
            'convention_following',
            'visitor',
        },
        substrate='bach_or_stravinsky_in_the_matrix__arena',
        roles=('bach_fan',) * 4 + ('stravinsky_fan',) * 4,
        is_focal=(True,) * 1 + (False,) * 7,
        bots_by_role=immutabledict.immutabledict(
            bach_fan=(
                'bach_or_stravinsky_in_the_matrix__arena__bach_picker_0',
            ),
            stravinsky_fan=(
                'bach_or_stravinsky_in_the_matrix__arena__bach_picker_0',
            ),
        ),
    ),
    bach_or_stravinsky_in_the_matrix__arena_1=ScenarioConfig(
        description='visiting background population who picks stravinsky',
        tags={
            'convention_following',
            'visitor',
        },
        substrate='bach_or_stravinsky_in_the_matrix__arena',
        roles=('bach_fan',) * 4 + ('stravinsky_fan',) * 4,
        is_focal=(True,) * 1 + (False,) * 7,
        bots_by_role=immutabledict.immutabledict(
            bach_fan=(
                'bach_or_stravinsky_in_the_matrix__arena__stravinsky_picker_0',
            ),
            stravinsky_fan=(
                'bach_or_stravinsky_in_the_matrix__arena__stravinsky_picker_0',
            ),
        ),
    ),
    bach_or_stravinsky_in_the_matrix__arena_2=ScenarioConfig(
        description='visited by a pure bot',
        tags={
            'resident',
        },
        substrate='bach_or_stravinsky_in_the_matrix__arena',
        roles=('bach_fan',) * 4 + ('stravinsky_fan',) * 4,
        is_focal=(True,) * 7 + (False,) * 1,
        bots_by_role=immutabledict.immutabledict(
            stravinsky_fan=(
                'bach_or_stravinsky_in_the_matrix__arena__bach_picker_0',
                'bach_or_stravinsky_in_the_matrix__arena__stravinsky_picker_0',
            ),
        ),
    ),
    bach_or_stravinsky_in_the_matrix__arena_3=ScenarioConfig(
        description='visited by three pure bach pickers',
        tags={
            'resident',
        },
        substrate='bach_or_stravinsky_in_the_matrix__arena',
        roles=('bach_fan',) * 4 + ('stravinsky_fan',) * 4,
        is_focal=(True,) * 5 + (False,) * 3,
        bots_by_role=immutabledict.immutabledict(
            stravinsky_fan=(
                'bach_or_stravinsky_in_the_matrix__arena__bach_picker_0',
            ),
        ),
    ),
    bach_or_stravinsky_in_the_matrix__arena_4=ScenarioConfig(
        description='visited by three pure stravinsky pickers',
        tags={
            'resident',
        },
        substrate='bach_or_stravinsky_in_the_matrix__arena',
        roles=('bach_fan',) * 4 + ('stravinsky_fan',) * 4,
        is_focal=(True,) * 5 + (False,) * 3,
        bots_by_role=immutabledict.immutabledict(
            stravinsky_fan=(
                'bach_or_stravinsky_in_the_matrix__arena__stravinsky_picker_0',
            ),
        ),
    ),
    bach_or_stravinsky_in_the_matrix__arena_5=ScenarioConfig(
        description=('visiting background population who alternates, ' +
                     'starting from stravinsky, repeating each twice'),
        tags={
            'visitor',
            'turn_taking',
            'convention_following',
        },
        substrate='bach_or_stravinsky_in_the_matrix__arena',
        roles=('bach_fan',) * 4 + ('stravinsky_fan',) * 4,
        is_focal=(True,) * 1 + (False,) * 7,
        bots_by_role=immutabledict.immutabledict(
            bach_fan=(
                'bach_or_stravinsky_in_the_matrix__arena__turn_taking_initial_stravinsky_0',
            ),
            stravinsky_fan=(
                'bach_or_stravinsky_in_the_matrix__arena__turn_taking_initial_stravinsky_0',
            ),
        ),
    ),
    bach_or_stravinsky_in_the_matrix__arena_6=ScenarioConfig(
        description=('visiting background population who alternates, ' +
                     'starting from bach, repeating each twice'),
        tags={
            'visitor',
            'turn_taking',
            'convention_following',
        },
        substrate='bach_or_stravinsky_in_the_matrix__arena',
        roles=('bach_fan',) * 4 + ('stravinsky_fan',) * 4,
        is_focal=(True,) * 1 + (False,) * 7,
        bots_by_role=immutabledict.immutabledict(
            bach_fan=(
                'bach_or_stravinsky_in_the_matrix__arena__turn_taking_initial_bach_0',
            ),
            stravinsky_fan=(
                'bach_or_stravinsky_in_the_matrix__arena__turn_taking_initial_bach_0',
            ),
        ),
    ),
    bach_or_stravinsky_in_the_matrix__repeated_0=ScenarioConfig(
        description='meeting a stubborn bach picker',
        tags={
            'convention_following',
        },
        substrate='bach_or_stravinsky_in_the_matrix__repeated',
        roles=('stravinsky_fan',) + ('bach_fan',),
        is_focal=(True,) + (False,),
        bots_by_role=immutabledict.immutabledict(
            bach_fan=(
                'bach_or_stravinsky_in_the_matrix__repeated__bach_picker_0',
            ),
        ),
    ),
    bach_or_stravinsky_in_the_matrix__repeated_1=ScenarioConfig(
        description='meeting a bot who plays bach despite not being a fan',
        tags={
            'convention_following',
        },
        substrate='bach_or_stravinsky_in_the_matrix__repeated',
        roles=('bach_fan',) + ('stravinsky_fan',),
        is_focal=(True,) + (False,),
        bots_by_role=immutabledict.immutabledict(
            stravinsky_fan=(
                'bach_or_stravinsky_in_the_matrix__repeated__bach_picker_0',
            ),
        ),
    ),
    bach_or_stravinsky_in_the_matrix__repeated_2=ScenarioConfig(
        description=('meeting a bot who plays stravinsky despite not being a ' +
                     'fan'),
        tags={
            'convention_following',
        },
        substrate='bach_or_stravinsky_in_the_matrix__repeated',
        roles=('stravinsky_fan',) + ('bach_fan',),
        is_focal=(True,) + (False,),
        bots_by_role=immutabledict.immutabledict(
            bach_fan=(
                'bach_or_stravinsky_in_the_matrix__repeated__stravinsky_picker_0',
            ),
        ),
    ),
    bach_or_stravinsky_in_the_matrix__repeated_3=ScenarioConfig(
        description='meeting a stubborn stravinsky picker',
        tags={
            'convention_following',
        },
        substrate='bach_or_stravinsky_in_the_matrix__repeated',
        roles=('bach_fan',) + ('stravinsky_fan',),
        is_focal=(True,) + (False,),
        bots_by_role=immutabledict.immutabledict(
            stravinsky_fan=(
                'bach_or_stravinsky_in_the_matrix__repeated__stravinsky_picker_0',
            ),
        ),
    ),
    bach_or_stravinsky_in_the_matrix__repeated_4=ScenarioConfig(
        description='bach fan focal agent meets an imperfectly copying partner',
        tags={
            'reciprocity',
        },
        substrate='bach_or_stravinsky_in_the_matrix__repeated',
        roles=('bach_fan',) + ('stravinsky_fan',),
        is_focal=(True,) + (False,),
        bots_by_role=immutabledict.immutabledict(
            stravinsky_fan=(
                'bach_or_stravinsky_in_the_matrix__repeated__bach_tft_0',
                'bach_or_stravinsky_in_the_matrix__repeated__bach_tft_tremble_0',
                'bach_or_stravinsky_in_the_matrix__repeated__stravinsky_tft_0',
                'bach_or_stravinsky_in_the_matrix__repeated__stravinsky_tft_tremble_0',
            ),
        ),
    ),
    bach_or_stravinsky_in_the_matrix__repeated_5=ScenarioConfig(
        description=('stravinsky fan focal agent meets an imperfectly ' +
                     'copying partner'),
        tags={
            'reciprocity',
        },
        substrate='bach_or_stravinsky_in_the_matrix__repeated',
        roles=('stravinsky_fan',) + ('bach_fan',),
        is_focal=(True,) + (False,),
        bots_by_role=immutabledict.immutabledict(
            bach_fan=(
                'bach_or_stravinsky_in_the_matrix__repeated__bach_tft_0',
                'bach_or_stravinsky_in_the_matrix__repeated__bach_tft_tremble_0',
                'bach_or_stravinsky_in_the_matrix__repeated__stravinsky_tft_0',
                'bach_or_stravinsky_in_the_matrix__repeated__stravinsky_tft_tremble_0',
            ),
        ),
    ),
    bach_or_stravinsky_in_the_matrix__repeated_6=ScenarioConfig(
        description=('bach fan focal agent meets a turn-taking partner'),
        tags={
            'turn_taking',
        },
        substrate='bach_or_stravinsky_in_the_matrix__repeated',
        roles=('bach_fan',) + ('stravinsky_fan',),
        is_focal=(True,) + (False,),
        bots_by_role=immutabledict.immutabledict(
            stravinsky_fan=(
                'bach_or_stravinsky_in_the_matrix__repeated__turn_taking_initial_bach_0',
                'bach_or_stravinsky_in_the_matrix__repeated__turn_taking_initial_stravinsky_0',
            ),
        ),
    ),
    bach_or_stravinsky_in_the_matrix__repeated_7=ScenarioConfig(
        description=('bach fan focal agent meets a turn-taking partner who ' +
                     'repeats each goal/resource three times before switching'),
        tags={
            'turn_taking',
        },
        substrate='bach_or_stravinsky_in_the_matrix__repeated',
        roles=('bach_fan',) + ('stravinsky_fan',),
        is_focal=(True,) + (False,),
        bots_by_role=immutabledict.immutabledict(
            stravinsky_fan=(
                'bach_or_stravinsky_in_the_matrix__repeated__turn_taking_initial_bach_1',
                'bach_or_stravinsky_in_the_matrix__repeated__turn_taking_initial_stravinsky_1',
            ),
        ),
    ),
    boat_race__eight_races_0=ScenarioConfig(
        description='visiting cooperators',
        tags={
            'visitor',
            'dyadic_trust',
        },
        substrate='boat_race__eight_races',
        roles=('default',) * 6,
        is_focal=(True,) * 1 + (False,) * 5,
        bots_by_role=immutabledict.immutabledict(
            default=('boat_race__eight_races__cooperator_0',),
        ),
    ),
    boat_race__eight_races_1=ScenarioConfig(
        description='visiting defectors',
        tags={
            'visitor',
            'acceptance',
        },
        substrate='boat_race__eight_races',
        roles=('default',) * 6,
        is_focal=(True,) * 1 + (False,) * 5,
        bots_by_role=immutabledict.immutabledict(
            default=('boat_race__eight_races__defector_0',),
        ),
    ),
    boat_race__eight_races_2=ScenarioConfig(
        description='visited by a population of cooperators',
        tags={
            'resident',
            'dyadic_trust',
        },
        substrate='boat_race__eight_races',
        roles=('default',) * 6,
        is_focal=(True,) * 5 + (False,) * 1,
        bots_by_role=immutabledict.immutabledict(
            default=('boat_race__eight_races__cooperator_0',),
        ),
    ),
    boat_race__eight_races_3=ScenarioConfig(
        description='visited by a population of defectors',
        tags={
            'resident',
            'partner_choice',
        },
        substrate='boat_race__eight_races',
        roles=('default',) * 6,
        is_focal=(True,) * 5 + (False,) * 1,
        bots_by_role=immutabledict.immutabledict(
            default=('boat_race__eight_races__defector_0',),
        ),
    ),
    boat_race__eight_races_4=ScenarioConfig(
        description='find the cooperator partner',
        tags={
            'partner_choice',
            'dyadic_trust',
        },
        substrate='boat_race__eight_races',
        roles=('default',) * 5 + ('target',),
        is_focal=(True,) * 1 + (False,) * 5,
        bots_by_role=immutabledict.immutabledict(
            default=('boat_race__eight_races__defector_0',),
            target=('boat_race__eight_races__cooperator_0',),
        ),
    ),
    chemistry__three_metabolic_cycles_0=ScenarioConfig(
        description=('resident focal population meets a small mixture of ' +
                     'background bots'),
        tags={
            'resident',
        },
        substrate='chemistry__three_metabolic_cycles',
        roles=('default',) * 8,
        is_focal=(True,) * 5 + (False,) * 3,
        bots_by_role={
            'default': {
                'chemistry__three_metabolic_cycles__blue_0',
                'chemistry__three_metabolic_cycles__green_0',
                'chemistry__three_metabolic_cycles__yellow_0',
            },
        },
    ),
    chemistry__three_metabolic_cycles_1=ScenarioConfig(
        description='meeting bots running blue and yellow',
        tags={
            'niche_discovery',
        },
        substrate='chemistry__three_metabolic_cycles',
        roles=('default',) * 8,
        is_focal=(True,) * 4 + (False,) * 4,
        bots_by_role={
            'default': {
                'chemistry__three_metabolic_cycles__blue_0',
                'chemistry__three_metabolic_cycles__yellow_0',
            },
        },
    ),
    chemistry__three_metabolic_cycles_2=ScenarioConfig(
        description='meeting one-sided bots running green',
        tags={
            'niche_discovery',
        },
        substrate='chemistry__three_metabolic_cycles',
        roles=('default',) * 8,
        is_focal=(True,) * 4 + (False,) * 4,
        bots_by_role={
            'default': {
                'chemistry__three_metabolic_cycles__green_0',
            },
        },
    ),
    chemistry__three_metabolic_cycles_3=ScenarioConfig(
        description='visit a resident population with mixed specialties',
        tags={
            'visitor',
            'niche_discovery',
        },
        substrate='chemistry__three_metabolic_cycles',
        roles=('default',) * 8,
        is_focal=(True,) * 2 + (False,) * 6,
        bots_by_role={
            'default': {
                'chemistry__three_metabolic_cycles__blue_0',
                'chemistry__three_metabolic_cycles__yellow_0',
                'chemistry__three_metabolic_cycles__green_0',
            },
        },
    ),
    chemistry__three_metabolic_cycles_with_plentiful_distractors_0=ScenarioConfig(
        description=('resident focal population meets a small mixture of ' +
                     'background bots, must avoid distractor molecules'),
        tags={
            'resident',
        },
        substrate=(
            'chemistry__three_metabolic_cycles_with_plentiful_distractors'),
        roles=('default',) * 8,
        is_focal=(True,) * 5 + (False,) * 3,
        bots_by_role={
            'default': {
                'chemistry__three_metabolic_cycles_with_plentiful_distractors__blue_0',
                'chemistry__three_metabolic_cycles_with_plentiful_distractors__green_0',
                'chemistry__three_metabolic_cycles_with_plentiful_distractors__yellow_0',
            },
        },
    ),
    chemistry__three_metabolic_cycles_with_plentiful_distractors_1=ScenarioConfig(
        description='meeting bots running blue, avoid distractors',
        tags={
            'niche_discovery',
        },
        substrate=(
            'chemistry__three_metabolic_cycles_with_plentiful_distractors'),
        roles=('default',) * 8,
        is_focal=(True,) * 4 + (False,) * 4,
        bots_by_role={
            'default': {
                'chemistry__three_metabolic_cycles_with_plentiful_distractors__blue_0',
            },
        },
    ),
    chemistry__three_metabolic_cycles_with_plentiful_distractors_2=ScenarioConfig(
        description='meeting bots running green and yellow, avoid distractors',
        tags={
            'niche_discovery',
        },
        substrate=(
            'chemistry__three_metabolic_cycles_with_plentiful_distractors'),
        roles=('default',) * 8,
        is_focal=(True,) * 4 + (False,) * 4,
        bots_by_role={
            'default': {
                'chemistry__three_metabolic_cycles_with_plentiful_distractors__green_0',
                'chemistry__three_metabolic_cycles_with_plentiful_distractors__yellow_0',
            },
        },
    ),
    chemistry__three_metabolic_cycles_with_plentiful_distractors_3=ScenarioConfig(
        description=('visit a resident population with mixed specialties and ' +
                     'avoid distractor molecules'),
        tags={
            'visitor',
            'niche_discovery',
        },
        substrate=(
            'chemistry__three_metabolic_cycles_with_plentiful_distractors'),
        roles=('default',) * 8,
        is_focal=(True,) * 2 + (False,) * 6,
        bots_by_role={
            'default': {
                'chemistry__three_metabolic_cycles_with_plentiful_distractors__blue_0',
                'chemistry__three_metabolic_cycles_with_plentiful_distractors__yellow_0',
                'chemistry__three_metabolic_cycles_with_plentiful_distractors__green_0',
            },
        },
    ),
    chemistry__two_metabolic_cycles_0=ScenarioConfig(
        description=('resident focal population meets a small mixture of ' +
                     'background bots'),
        tags={
            'resident',
        },
        substrate='chemistry__two_metabolic_cycles',
        roles=('default',) * 8,
        is_focal=(True,) * 6 + (False,) * 2,
        bots_by_role={
            'default': {
                'chemistry__two_metabolic_cycles__blue_0',
                'chemistry__two_metabolic_cycles__green_0',
            },
        },
    ),
    chemistry__two_metabolic_cycles_1=ScenarioConfig(
        description='meeting one-sided bots running blue',
        tags={
            'niche_discovery',
        },
        substrate='chemistry__two_metabolic_cycles',
        roles=('default',) * 8,
        is_focal=(True,) * 4 + (False,) * 4,
        bots_by_role={
            'default': {
                'chemistry__two_metabolic_cycles__blue_0',
            },
        },
    ),
    chemistry__two_metabolic_cycles_2=ScenarioConfig(
        description='meeting one-sided bots running green',
        tags={
            'niche_discovery',
        },
        substrate='chemistry__two_metabolic_cycles',
        roles=('default',) * 8,
        is_focal=(True,) * 4 + (False,) * 4,
        bots_by_role={
            'default': {
                'chemistry__two_metabolic_cycles__green_0',
            },
        },
    ),
    chemistry__two_metabolic_cycles_3=ScenarioConfig(
        description=('visit a resident background population with mixed ' +
                     'specialties'),
        tags={
            'visitor',
            'niche_discovery',
        },
        substrate='chemistry__two_metabolic_cycles',
        roles=('default',) * 8,
        is_focal=(True,) * 2 + (False,) * 6,
        bots_by_role={
            'default': {
                'chemistry__two_metabolic_cycles__blue_0',
                'chemistry__two_metabolic_cycles__green_0',
            },
        },
    ),
    chemistry__two_metabolic_cycles_with_distractors_0=ScenarioConfig(
        description=('resident focal population meets a small mixture of ' +
                     'background bots, must avoid distractor molecules'),
        tags={
            'resident',
        },
        substrate='chemistry__two_metabolic_cycles_with_distractors',
        roles=('default',) * 8,
        is_focal=(True,) * 6 + (False,) * 2,
        bots_by_role={
            'default': {
                'chemistry__two_metabolic_cycles_with_distractors__blue_0',
                'chemistry__two_metabolic_cycles_with_distractors__green_0',
            },
        },
    ),
    chemistry__two_metabolic_cycles_with_distractors_1=ScenarioConfig(
        description=('meeting one-sided bots running blue and avoid ' +
                     'distractor molecules'),
        tags={
            'niche_discovery',
        },
        substrate='chemistry__two_metabolic_cycles_with_distractors',
        roles=('default',) * 8,
        is_focal=(True,) * 4 + (False,) * 4,
        bots_by_role={
            'default': {
                'chemistry__two_metabolic_cycles_with_distractors__blue_0',
            },
        },
    ),
    chemistry__two_metabolic_cycles_with_distractors_2=ScenarioConfig(
        description=('meeting one-sided bots running green and avoid ' +
                     'distractor molecules'),
        tags={
            'niche_discovery',
        },
        substrate='chemistry__two_metabolic_cycles_with_distractors',
        roles=('default',) * 8,
        is_focal=(True,) * 4 + (False,) * 4,
        bots_by_role={
            'default': {
                'chemistry__two_metabolic_cycles_with_distractors__green_0',
            },
        },
    ),
    chemistry__two_metabolic_cycles_with_distractors_3=ScenarioConfig(
        description=('visit a resident background population with mixed ' +
                     'specialties and avoid distractor molecules'),
        tags={
            'visitor',
            'niche_discovery',
        },
        substrate='chemistry__two_metabolic_cycles_with_distractors',
        roles=('default',) * 8,
        is_focal=(True,) * 2 + (False,) * 6,
        bots_by_role={
            'default': {
                'chemistry__two_metabolic_cycles_with_distractors__blue_0',
                'chemistry__two_metabolic_cycles_with_distractors__green_0',
            },
        },
    ),
    chicken_in_the_matrix__arena_0=ScenarioConfig(
        description='visiting unconditional dove players',
        tags={
            'visitor',
            'exploitation',
        },
        substrate='chicken_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'chicken_in_the_matrix__arena__puppet_dove_0',
                'chicken_in_the_matrix__arena__puppet_dove_margin_0',
            },
        },
    ),
    chicken_in_the_matrix__arena_1=ScenarioConfig(
        description=('focals are resident and visitors are unconditional ' +
                     'dove players'),
        tags={
            'resident',
            'prosocial_newcomers',
        },
        substrate='chicken_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 5 + (False,) * 3,
        bots_by_role={
            'default': {
                'chicken_in_the_matrix__arena__puppet_dove_0',
                'chicken_in_the_matrix__arena__puppet_dove_margin_0',
            },
        },
    ),
    chicken_in_the_matrix__arena_2=ScenarioConfig(
        description=('focals are resident and visitors are unconditional' +
                     'hawk players'),
        tags={
            'resident',
            'partner_choice',
            'antisocial_newcomers',
        },
        substrate='chicken_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 5 + (False,) * 3,
        bots_by_role={
            'default': {
                'chicken_in_the_matrix__arena__puppet_hawk_0',
                'chicken_in_the_matrix__arena__puppet_hawk_margin_0',
            },
        },
    ),
    chicken_in_the_matrix__arena_3=ScenarioConfig(
        description=('visiting a population of hair-trigger grim ' +
                     'reciprocator bots who initially cooperate but, if ' +
                     'defected on once, will retaliate by defecting in all ' +
                     'future interactions'),
        tags={
            'visitor',
            'reciprocity',
        },
        substrate='chicken_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'chicken_in_the_matrix__arena__puppet_grim_one_strike_0',
                'chicken_in_the_matrix__arena__puppet_grim_one_strike_margin_0',
            },
        },
    ),
    chicken_in_the_matrix__arena_4=ScenarioConfig(
        description=('visiting a population of two-strikes grim ' +
                     'reciprocator bots who initially cooperate but, if ' +
                     'defected on twice, will retaliate by defecting in all ' +
                     'future interactions'),
        tags={
            'visitor',
            'reciprocity',
        },
        substrate='chicken_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'chicken_in_the_matrix__arena__puppet_grim_two_strikes_0',
                'chicken_in_the_matrix__arena__puppet_grim_two_strikes_margin_0',
            },
        },
    ),
    chicken_in_the_matrix__arena_5=ScenarioConfig(
        description=(
            'visiting a mixed population of k-strikes grim reciprocator bots ' +
            'with k values from 1 to 3, they initially cooperate but, if ' +
            'defected on k times, they retaliate in all future interactions'
        ),
        tags={
            'visitor',
            'reciprocity',
        },
        substrate='chicken_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 3 + (False,) * 5,
        bots_by_role={
            'default': {
                'chicken_in_the_matrix__arena__puppet_grim_one_strike_0',
                'chicken_in_the_matrix__arena__puppet_grim_one_strike_margin_0',
                'chicken_in_the_matrix__arena__puppet_grim_three_strikes_0',
                'chicken_in_the_matrix__arena__puppet_grim_three_strikes_margin_0',
                'chicken_in_the_matrix__arena__puppet_grim_two_strikes_0',
                'chicken_in_the_matrix__arena__puppet_grim_two_strikes_margin_0',
            },
        },
    ),
    chicken_in_the_matrix__arena_6=ScenarioConfig(
        description='visiting a mixture of pure hawk and pure dove players',
        tags={
            'visitor',
            'partner_choice',
        },
        substrate='chicken_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 3 + (False,) * 5,
        bots_by_role={
            'default': {
                'chicken_in_the_matrix__arena__puppet_dove_0',
                'chicken_in_the_matrix__arena__puppet_dove_margin_0',
                'chicken_in_the_matrix__arena__puppet_hawk_0',
                'chicken_in_the_matrix__arena__puppet_hawk_margin_0',
            },
        },
    ),
    chicken_in_the_matrix__repeated_0=ScenarioConfig(
        description='partner may play either hawk or dove',
        tags={
            'flexibility',
        },
        substrate='chicken_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'chicken_in_the_matrix__repeated__puppet_dove_margin_0',
                'chicken_in_the_matrix__repeated__puppet_dove_margin_1',
                'chicken_in_the_matrix__repeated__puppet_hawk_margin_0',
                'chicken_in_the_matrix__repeated__puppet_hawk_margin_1',
            },
        },
    ),
    chicken_in_the_matrix__repeated_1=ScenarioConfig(
        description='partner typically plays dove',
        tags={
            'exploitation',
            'partner_cooperates_unconditionally',
        },
        substrate='chicken_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'chicken_in_the_matrix__repeated__puppet_dove_margin_0',
                'chicken_in_the_matrix__repeated__puppet_dove_margin_1',
            },
        },
    ),
    chicken_in_the_matrix__repeated_2=ScenarioConfig(
        description='partner typically plays hawk',
        tags={
            'acceptance',
            'partner_defects_unconditionally',
        },
        substrate='chicken_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'chicken_in_the_matrix__repeated__puppet_hawk_margin_0',
                'chicken_in_the_matrix__repeated__puppet_hawk_margin_1',
            },
        },
    ),
    chicken_in_the_matrix__repeated_3=ScenarioConfig(
        description=('partner is a hair-trigger grim reciprocator, i.e. one ' +
                     'who initially cooperates but, if defected on once, will' +
                     ' retaliate by defecting forever after'),
        tags={
            'reciprocity',
        },
        substrate='chicken_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'chicken_in_the_matrix__repeated__puppet_grim_one_strike_margin_0',
                'chicken_in_the_matrix__repeated__puppet_grim_one_strike_margin_1',
            },
        },
    ),
    chicken_in_the_matrix__repeated_4=ScenarioConfig(
        description=('partner is a two-strikes grim reciprocator, i.e. one ' +
                     'who initially cooperates, but if defected on twice, ' +
                     'will retaliate by defecting forever after'),
        tags={
            'reciprocity',
        },
        substrate='chicken_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'chicken_in_the_matrix__repeated__puppet_grim_two_strikes_margin_0',
                'chicken_in_the_matrix__repeated__puppet_grim_two_strikes_margin_1',
            },
        },
    ),
    chicken_in_the_matrix__repeated_5=ScenarioConfig(
        description='partner is a tit-for-tat conditional cooperator',
        tags={
            'reciprocity',
        },
        substrate='chicken_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'chicken_in_the_matrix__repeated__puppet_tft_margin_0',
                'chicken_in_the_matrix__repeated__puppet_tft_margin_1',
            },
        },
    ),
    chicken_in_the_matrix__repeated_6=ScenarioConfig(
        description=('partner is a tit-for-tat conditional cooperator who ' +
                     'occasionally plays hawk instead of dove'),
        tags={
            'reciprocity',
            'forgiveness',
        },
        substrate='chicken_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'chicken_in_the_matrix__repeated__puppet_tft_tremble_margin_0',
                'chicken_in_the_matrix__repeated__puppet_tft_tremble_margin_1',
            },
        },
    ),
    chicken_in_the_matrix__repeated_7=ScenarioConfig(
        description='partner plays dove for a while then switches to hawk',
        tags={
            'flexibility',
        },
        substrate='chicken_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'chicken_in_the_matrix__repeated__puppet_flip_0',
            },
        },
    ),
    chicken_in_the_matrix__repeated_8=ScenarioConfig(
        description=('partner tries to take advantage of the focal player ' +
                     'by playing hawk, but if punished, partner then ' +
                     'switches to tit-for-tat conditional cooperation'),
        tags={
            'teaching',
            'reciprocity',
            'forgiveness',
            'sanctioning',
        },
        substrate='chicken_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'chicken_in_the_matrix__repeated__puppet_corrigible_0',
            },
        },
    ),
    chicken_in_the_matrix__repeated_9=ScenarioConfig(
        description=('partner tries to take advantage of the focal player ' +
                     'by playing hawk, but if punished, partner then ' +
                     'switches to noisy tit-for-tat conditional cooperation'),
        tags={
            'teaching',
            'reciprocity',
            'forgiveness',
            'sanctioning',
        },
        substrate='chicken_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'chicken_in_the_matrix__repeated__puppet_corrigible_tremble_0',
            },
        },
    ),
    clean_up_0=ScenarioConfig(
        description='visiting an altruistic population',
        tags={
            'chance_to_benefit_oneself_without_imposing_social_cost',
            'visitor',
        },
        substrate='clean_up',
        roles=('default',) * 7,
        is_focal=(True,) * 3 + (False,) * 4,
        bots_by_role={
            'default': {
                'clean_up__cleaner_0',
                'clean_up__cleaner_1',
            },
        },
    ),
    clean_up_1=ScenarioConfig(
        description='focals are resident and visitors ride free',
        tags={
            'resident',
            'acceptance',
        },
        substrate='clean_up',
        roles=('default',) * 7,
        is_focal=(True,) * 4 + (False,) * 3,
        bots_by_role={
            'default': {
                'clean_up__consumer_0',
                'clean_up__consumer_1',
            },
        },
    ),
    clean_up_2=ScenarioConfig(
        description='visiting a turn-taking population that cleans first',
        tags={
            'turn_taking',
            'visitor',
        },
        substrate='clean_up',
        roles=('default',) * 7,
        is_focal=(True,) * 3 + (False,) * 4,
        bots_by_role={
            'default': {'clean_up__puppet_alternator_first_cleans_0',},
        },
    ),
    clean_up_3=ScenarioConfig(
        description='visiting a turn-taking population that eats first',
        tags={
            'turn_taking',
            'visitor',
        },
        substrate='clean_up',
        roles=('default',) * 7,
        is_focal=(True,) * 3 + (False,) * 4,
        bots_by_role={
            'default': {'clean_up__puppet_alternator_first_eats_0',},
        },
    ),
    clean_up_4=ScenarioConfig(
        description='focals are visited by one reciprocator',
        tags={
            'resident',
            'reciprocity',
        },
        substrate='clean_up',
        roles=('default',) * 7,
        is_focal=(True,) * 6 + (False,) * 1,
        bots_by_role={
            'default': {'clean_up__puppet_low_threshold_reciprocator_0',},
        },
    ),
    clean_up_5=ScenarioConfig(
        description='focals are visited by two suspicious reciprocators',
        tags={
            'resident',
            'reciprocity',
        },
        substrate='clean_up',
        roles=('default',) * 7,
        is_focal=(True,) * 5 + (False,) * 2,
        bots_by_role={
            'default': {'clean_up__puppet_high_threshold_reciprocator_0',},
        },
    ),
    clean_up_6=ScenarioConfig(
        description='focals are visited by one suspicious reciprocator',
        tags={
            'resident',
            'reciprocity',
        },
        substrate='clean_up',
        roles=('default',) * 7,
        is_focal=(True,) * 6 + (False,) * 1,
        bots_by_role={
            'default': {'clean_up__puppet_high_threshold_reciprocator_0',},
        },
    ),
    clean_up_7=ScenarioConfig(
        description='focals visit resident group of suspicious reciprocators',
        tags={
            'visitor',
            'reciprocity',
        },
        substrate='clean_up',
        roles=('default',) * 7,
        is_focal=(True,) * 2 + (False,) * 5,
        bots_by_role={
            'default': {'clean_up__puppet_high_threshold_reciprocator_0',},
        },
    ),
    clean_up_8=ScenarioConfig(
        description='focals are visited by one nice reciprocator',
        tags={
            'resident',
            'reciprocity',
        },
        substrate='clean_up',
        roles=('default',) * 7,
        is_focal=(True,) * 6 + (False,) * 1,
        bots_by_role={
            'default': {'clean_up__puppet_nice_low_threshold_reciprocator_0',},
        },
    ),
    clean_up_9=ScenarioConfig(
        description=(
            'resident focals are visited by an easily corrigible reciprocator'),
        tags={
            'resident',
            'enforcement',
            'reciprocity',
        },
        substrate='clean_up',
        roles=('default',) * 7,
        is_focal=(True,) * 6 + (False,) * 1,
        bots_by_role={
            'default': {'clean_up__puppet_easily_corrigible_reciprocator_0'},
        },
    ),
    clean_up_10=ScenarioConfig(
        description=(
            'resident focals are visited by a bot who rapidly alternates ' +
            'between free riding and zapping'),
        tags={
            'resident',
            'enforcement',
        },
        substrate='clean_up',
        roles=('default',) * 7,
        is_focal=(True,) * 6 + (False,) * 1,
        bots_by_role={
            'default': {'clean_up__puppet_free_rider_to_zapper_alternator_0'},
        },
    ),
    clean_up_11=ScenarioConfig(
        description=(
            'resident focals are visited by a corrigible reciprocator'),
        tags={
            'resident',
            'enforcement',
            'reciprocity',
        },
        substrate='clean_up',
        roles=('default',) * 7,
        is_focal=(True,) * 6 + (False,) * 1,
        bots_by_role={
            'default': {'clean_up__puppet_corrigible_reciprocator_0'},
        },
    ),
    clean_up_12=ScenarioConfig(
        description=(
            'resident focals are visited by a corrigible reciprocator who is ' +
            'difficult to convince to change'),
        tags={
            'resident',
            'enforcement',
            'reciprocity',
        },
        substrate='clean_up',
        roles=('default',) * 7,
        is_focal=(True,) * 6 + (False,) * 1,
        bots_by_role={
            'default': {'clean_up__puppet_corrigible_reciprocator_1',
                        'clean_up__puppet_corrigible_reciprocator_2'},
        },
    ),
    clean_up_13=ScenarioConfig(
        description=(
            'focals visit easily corrigible reciprocators'),
        tags={
            'visitor',
            'enforcement',
            'reciprocity',
        },
        substrate='clean_up',
        roles=('default',) * 7,
        is_focal=(True,) * 3 + (False,) * 4,
        bots_by_role={
            'default': {'clean_up__puppet_easily_corrigible_reciprocator_0',
                        'clean_up__puppet_corrigible_reciprocator_0'},
        },
    ),
    clean_up_14=ScenarioConfig(
        description=(
            'focals visit reciprocators who are corrigible but difficult to ' +
            'convince'),
        tags={
            'visitor',
            'enforcement',
            'reciprocity',
        },
        substrate='clean_up',
        roles=('default',) * 7,
        is_focal=(True,) * 3 + (False,) * 4,
        bots_by_role={
            'default': {'clean_up__puppet_corrigible_reciprocator_1',
                        'clean_up__puppet_corrigible_reciprocator_2'},
        },
    ),
    clean_up_15=ScenarioConfig(
        description=(
            'focals visit easily corrigible reciprocators and also bots who ' +
            'alternate between contributing and free riding'),
        tags={
            'visitor',
            'enforcement',
            'reciprocity',
        },
        substrate='clean_up',
        roles=('default',) * 7,
        is_focal=(False,) * 5 + (True,) * 2,  # Reversed is_focal for variety.
        bots_by_role={
            'default': {'clean_up__puppet_easily_corrigible_reciprocator_0',
                        'clean_up__puppet_corrigible_reciprocator_0',
                        'clean_up__puppet_alternator_first_cleans_0',
                        'clean_up__puppet_alternator_first_eats_0'},
        },
    ),
    clean_up_16=ScenarioConfig(
        description=(
            'focals visit reciprocators who are corrigible but difficult to ' +
            'convince and also bots who alternate between contributing and ' +
            'free riding'),
        tags={
            'visitor',
            'enforcement',
            'reciprocity',
        },
        substrate='clean_up',
        roles=('default',) * 7,
        is_focal=(True,) * 2 + (False,) * 5,
        bots_by_role={
            'default': {'clean_up__puppet_corrigible_reciprocator_1',
                        'clean_up__puppet_corrigible_reciprocator_2',
                        'clean_up__puppet_alternator_first_cleans_0',
                        'clean_up__puppet_alternator_first_eats_0'},
        },
    ),
    clean_up_17=ScenarioConfig(
        description=(
            'not very crowded, focals are resident, they are visited by a ' +
            'an altruist'),
        tags={
            'resident',
        },
        substrate='clean_up',
        roles=('default',) * 4,
        is_focal=(False,) + (True,) * 3,  # Reversed is_focal for variety.
        bots_by_role={
            'default': {
                'clean_up__cleaner_0',
            },
        },
    ),
    clean_up_18=ScenarioConfig(
        description=(
            'not very crowded, focals are resident, they are visited by a ' +
            'a nice reciprocator'),
        tags={
            'resident',
        },
        substrate='clean_up',
        roles=('default',) * 4,
        is_focal=(True,) * 3 + (False,),
        bots_by_role={
            'default': {
                'clean_up__puppet_nice_low_threshold_reciprocator_0',
            },
        },
    ),
    clean_up_19=ScenarioConfig(
        description=(
            'resident focals are visited by a sanctioning alternator who '
            'starts out by cleaning'),
        tags={
            'resident',
        },
        substrate='clean_up',
        roles=('default',) * 7,
        is_focal=(True,) * 6 + (False,) * 1,
        bots_by_role={
            'default': {
                'clean_up__puppet_sanctioning_alternator_nice_0',
            },
        },
    ),
    clean_up_20=ScenarioConfig(
        description=(
            'resident focals are visited by a sanctioning alternator who '
            'starts out by free riding'),
        tags={
            'resident',
        },
        substrate='clean_up',
        roles=('default',) * 7,
        is_focal=(False,) + (True,) * 6,  # Reversed is_focal for variety.
        bots_by_role={
            'default': {
                'clean_up__puppet_sanctioning_alternator_nice_0',
            },
        },
    ),
    clean_up_21=ScenarioConfig(
        description=(
            'resident focals are visited by a couple bots who free ride'),
        tags={
            'resident',
            'acceptance',
        },
        substrate='clean_up',
        roles=('default',) * 7,
        is_focal=(True,) * 5 + (False,) * 2,
        bots_by_role={
            'default': {
                'clean_up__consumer_0',
                'clean_up__consumer_1',
            },
        },
    ),
    clean_up_22=ScenarioConfig(
        description=(
            'not very crowded, resident focals are visited by a couple bots ' +
            'who free ride'),
        tags={
            'resident',
            'acceptance',
        },
        substrate='clean_up',
        roles=('default',) * 6,
        is_focal=(True,) * 4 + (False,) * 2,
        bots_by_role={
            'default': {
                'clean_up__consumer_0',
                'clean_up__consumer_1',
            },
        },
    ),
    coins_0=ScenarioConfig(
        description='partner is either a pure cooperator or a pure defector',
        tags={
            'flexibility',
        },
        substrate='coins',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {'coins__puppet_cooperator_0',
                        'coins__puppet_defector_0',},
        },
    ),
    coins_1=ScenarioConfig(
        description=('partner is a high-threshold (generous) reciprocator'),
        tags={
            'reciprocity',
        },
        substrate='coins',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {'coins__puppet_three_strikes_reciprocator_0',},
        },
    ),
    coins_2=ScenarioConfig(
        description=('partner is a low-threshold (harsh) reciprocator'),
        tags={
            'reciprocity',
        },
        substrate='coins',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {'coins__puppet_one_strike_reciprocator_0',},
        },
    ),
    coins_3=ScenarioConfig(
        description=('partner is a high-threshold (generous) strong ' +
                     'reciprocator'),
        tags={
            'reciprocity',
        },
        substrate='coins',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {'coins__puppet_three_strikes_strong_reciprocator_0',},
        },
    ),
    coins_4=ScenarioConfig(
        description=('partner is a low-threshold (harsh) strong reciprocator'),
        tags={
            'reciprocity',
        },
        substrate='coins',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {'coins__puppet_one_strike_strong_reciprocator_0',},
        },
    ),
    coins_5=ScenarioConfig(
        description='partner is a cooperator',
        tags={
            'exploitation',
            'partner_cooperates_unconditionally',
        },
        substrate='coins',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {'coins__puppet_cooperator_0',},
        },
    ),
    coins_6=ScenarioConfig(
        description='partner is a defector',
        tags={
            'acceptance',
            'partner_defects_unconditionally',
        },
        substrate='coins',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {'coins__puppet_defector_0',},
        },
    ),
    collaborative_cooking__asymmetric_0=ScenarioConfig(
        description='collaborate with a skilled chef',
        tags={
            'imitation',
        },
        substrate='collaborative_cooking__asymmetric',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'collaborative_cooking__asymmetric__chef_0',
                'collaborative_cooking__asymmetric__chef_1',
            },
        },
    ),
    collaborative_cooking__asymmetric_1=ScenarioConfig(
        description='collaborate with a semi-skilled apprentice chef',
        tags={
            'flexibility',
        },
        substrate='collaborative_cooking__asymmetric',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'collaborative_cooking__asymmetric__apprentice_0',
                'collaborative_cooking__asymmetric__apprentice_1',
            },
        },
    ),
    collaborative_cooking__asymmetric_2=ScenarioConfig(
        description='succeed despite an unhelpful partner',
        tags={
            'flexibility',
        },
        substrate='collaborative_cooking__asymmetric',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={'default': {'noop_bot'}},
    ),
    collaborative_cooking__circuit_0=ScenarioConfig(
        description='collaborate with a skilled chef',
        tags={
            'imitation',
        },
        substrate='collaborative_cooking__circuit',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'collaborative_cooking__circuit__chef_0',
                'collaborative_cooking__circuit__chef_1',
            },
        },
    ),
    collaborative_cooking__circuit_1=ScenarioConfig(
        description='collaborate with a semi-skilled apprentice chef',
        tags={
            'flexibility',
        },
        substrate='collaborative_cooking__circuit',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'collaborative_cooking__circuit__apprentice_0',
                'collaborative_cooking__circuit__apprentice_1',
            },
        },
    ),
    collaborative_cooking__circuit_2=ScenarioConfig(
        description='succeed despite an unhelpful partner',
        tags={
            'flexibility',
        },
        substrate='collaborative_cooking__circuit',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={'default': {'noop_bot'}},
    ),
    collaborative_cooking__cramped_0=ScenarioConfig(
        description='collaborate with a skilled chef',
        tags={
            'imitation',
        },
        substrate='collaborative_cooking__cramped',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'collaborative_cooking__cramped__chef_0',
                'collaborative_cooking__cramped__chef_1',
            },
        },
    ),
    collaborative_cooking__cramped_1=ScenarioConfig(
        description='collaborate with a semi-skilled apprentice chef',
        tags={
            'flexibility',
        },
        substrate='collaborative_cooking__cramped',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'collaborative_cooking__cramped__apprentice_0',
                'collaborative_cooking__cramped__apprentice_1',
            },
        },
    ),
    collaborative_cooking__cramped_2=ScenarioConfig(
        description='succeed despite an unhelpful partner',
        tags={
            'flexibility',
        },
        substrate='collaborative_cooking__cramped',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={'default': {'noop_bot'}},
    ),
    collaborative_cooking__crowded_0=ScenarioConfig(
        description=(
            'collaborate with an independent chef who expects others to get ' +
            'out of their way'),
        tags={
            'resident',
            'flexibility',
        },
        substrate='collaborative_cooking__crowded',
        roles=('default',) * 9,
        is_focal=(True,) * 8 + (False,),
        bots_by_role={
            'default': {
                'collaborative_cooking__crowded__independent_chef_0',
            },
        },
    ),
    collaborative_cooking__crowded_1=ScenarioConfig(
        description=(
            'collaborate with several chefs who can work together, but are ' +
            'not very good at doing so'),
        tags={
            'resident',
            'flexibility',
        },
        substrate='collaborative_cooking__crowded',
        roles=('default',) * 9,
        is_focal=(True,) * 6 + (False,) * 3,
        bots_by_role={
            'default': {
                'collaborative_cooking__crowded__robust_chef_0',
            },
        },
    ),
    collaborative_cooking__crowded_2=ScenarioConfig(
        description=(
            'no assistance from an unhelpful visiting noop bot'),
        tags={
            'resident',
            'flexibility',
        },
        substrate='collaborative_cooking__crowded',
        roles=('default',) * 9,
        is_focal=(True,) * 8 + (False,),
        bots_by_role={'default': {'noop_bot'}},
    ),
    collaborative_cooking__figure_eight_0=ScenarioConfig(
        description=(
            'collaborate with an independent chef who expects others to get ' +
            'out of their way'),
        tags={
            'resident',
            'flexibility',
        },
        substrate='collaborative_cooking__figure_eight',
        roles=('default',) * 6,
        is_focal=(True,) * 5 + (False,),
        bots_by_role={
            'default': {
                'collaborative_cooking__figure_eight__independent_chef_0',
            },
        },
    ),
    collaborative_cooking__figure_eight_1=ScenarioConfig(
        description=(
            'collaborate with two chefs who can work together, but are ' +
            'not very good at doing so'),
        tags={
            'resident',
            'flexibility',
        },
        substrate='collaborative_cooking__figure_eight',
        roles=('default',) * 6,
        is_focal=(True,) * 4 + (False,) * 2,
        bots_by_role={
            'default': {
                'collaborative_cooking__figure_eight__robust_chef_0',
            },
        },
    ),
    collaborative_cooking__figure_eight_2=ScenarioConfig(
        description=(
            'no assistance from am unhelpful visiting noop bot'),
        tags={
            'resident',
            'flexibility',
        },
        substrate='collaborative_cooking__figure_eight',
        roles=('default',) * 6,
        is_focal=(True,) * 5 + (False,),
        bots_by_role={'default': {'noop_bot'}},
    ),
    collaborative_cooking__forced_0=ScenarioConfig(
        description='collaborate with a skilled chef',
        tags={
            'flexibility',
            'imitation',
        },
        substrate='collaborative_cooking__forced',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'collaborative_cooking__forced__chef_0',
                'collaborative_cooking__forced__chef_1',
            },
        },
    ),
    collaborative_cooking__forced_1=ScenarioConfig(
        description='collaborate with a semi-skilled apprentice chef',
        tags={
            'flexibility',
        },
        substrate='collaborative_cooking__forced',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'collaborative_cooking__forced__apprentice_0',
                'collaborative_cooking__forced__apprentice_1',
            },
        },
    ),
    collaborative_cooking__ring_0=ScenarioConfig(
        description='collaborate with a skilled chef',
        tags={
            'imitation',
        },
        substrate='collaborative_cooking__ring',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'collaborative_cooking__ring__chef_0',
                'collaborative_cooking__ring__chef_1',
            },
        },
    ),
    collaborative_cooking__ring_1=ScenarioConfig(
        description='collaborate with a semi-skilled apprentice chef',
        tags={
            'flexibility',
        },
        substrate='collaborative_cooking__ring',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'collaborative_cooking__ring__apprentice_0',
                'collaborative_cooking__ring__apprentice_1',
            },
        },
    ),
    commons_harvest__closed_0=ScenarioConfig(
        description='focals visit pacifist bots who harvest unsustainably',
        tags={
            'visitor',
            'ownership',
            'restraint',
            'defense',
        },
        substrate='commons_harvest__closed',
        roles=('default',) * 7,
        is_focal=(True,) * 2 + (False,) * 5,
        bots_by_role={
            'default': {'commons_harvest__closed__pacifist_0',
                        'commons_harvest__closed__pacifist_1',
                        'commons_harvest__closed__pacifist_2',},
        },
    ),
    commons_harvest__closed_1=ScenarioConfig(
        description=('focals are resident and visited by pacifist bots who ' +
                     'harvest unsustainably'),
        tags={
            'resident',
            'ownership',
            'restraint',
            'defense',
            'antisocial_newcomers',
        },
        substrate='commons_harvest__closed',
        roles=('default',) * 7,
        is_focal=(True,) * 5 + (False,) * 2,
        bots_by_role={
            'default': {'commons_harvest__closed__pacifist_0',
                        'commons_harvest__closed__pacifist_1',
                        'commons_harvest__closed__pacifist_2',},
        },
    ),
    commons_harvest__closed_2=ScenarioConfig(
        description=('focals visit bots who zap and harvest sustainably if ' +
                     'they get a chance'),
        tags={
            'visitor',
            'ownership',
            'restraint',
            'defense',
        },
        substrate='commons_harvest__closed',
        roles=('default',) * 7,
        is_focal=(True,) * 2 + (False,) * 5,
        bots_by_role={
            'default': {'commons_harvest__closed__free_0',
                        'commons_harvest__closed__free_1',
                        'commons_harvest__closed__free_2',
                        'commons_harvest__closed__free_3'},
        },
    ),
    commons_harvest__closed_3=ScenarioConfig(
        description=('focals are resident, and are visited by bots who zap ' +
                     'and harvest sustainably if they get a chance'),
        tags={
            'resident',
            'ownership',
            'restraint',
            'defense',
            'prosocial_newcomers',
        },
        substrate='commons_harvest__closed',
        roles=('default',) * 7,
        is_focal=(True,) * 5 + (False,) * 2,
        bots_by_role={
            'default': {'commons_harvest__closed__free_0',
                        'commons_harvest__closed__free_1',
                        'commons_harvest__closed__free_2',
                        'commons_harvest__closed__free_3'},
        },
    ),
    commons_harvest__open_0=ScenarioConfig(
        description=('focals are resident and visited by two bots who zap ' +
                     'and harvest unsustainably'),
        tags={
            'resident',
            'restraint',
            'defense',
            'sanctioning',
            'antisocial_newcomers',
        },
        substrate='commons_harvest__open',
        roles=('default',) * 7,
        is_focal=(True,) * 5 + (False,) * 2,
        bots_by_role={
            'default': {'commons_harvest__open__free_0',
                        'commons_harvest__open__free_1',},
        },
    ),
    commons_harvest__open_1=ScenarioConfig(
        description=('focals are resident and visited by two pacifists who ' +
                     'harvest unsustainably'),
        tags={
            'resident',
            'restraint',
            'sanctioning',
            'antisocial_newcomers',
        },
        substrate='commons_harvest__open',
        roles=('default',) * 7,
        is_focal=(True,) * 5 + (False,) * 2,
        bots_by_role={
            'default': {'commons_harvest__open__pacifist_0',
                        'commons_harvest__open__pacifist_1'},
        },
    ),
    commons_harvest__partnership_0=ScenarioConfig(
        description='meeting good partners',
        tags={
            'visitor',
            'dyadic_trust',
            'ownership',
            'restraint',
            'defense',
        },
        substrate='commons_harvest__partnership',
        roles=('default',) * 7,
        is_focal=(True,) * 1 + (False,) * 6,
        bots_by_role={
            'default': {'commons_harvest__partnership__good_partner_0',
                        'commons_harvest__partnership__good_partner_1',
                        'commons_harvest__partnership__good_partner_2',},
        },
    ),
    commons_harvest__partnership_1=ScenarioConfig(
        description='focals are resident and visitors are good partners',
        tags={
            'resident',
            'dyadic_trust',
            'ownership',
            'restraint',
            'defense',
            'prosocial_newcomers',
        },
        substrate='commons_harvest__partnership',
        roles=('default',) * 7,
        is_focal=(True,) * 5 + (False,) * 2,
        bots_by_role={
            'default': {'commons_harvest__partnership__good_partner_0',
                        'commons_harvest__partnership__good_partner_1',
                        'commons_harvest__partnership__good_partner_2',},
        },
    ),
    commons_harvest__partnership_2=ScenarioConfig(
        description=('focals visit zappers who harvest sustainably but lack ' +
                     'trust'),
        tags={
            'visitor',
            'dyadic_trust',
            'ownership',
            'restraint',
            'defense',
        },
        substrate='commons_harvest__partnership',
        roles=('default',) * 7,
        is_focal=(True,) * 1 + (False,) * 6,
        bots_by_role={
            'default': {'commons_harvest__partnership__sustainable_fighter_0',
                        'commons_harvest__partnership__sustainable_fighter_1',},
        },
    ),
    commons_harvest__partnership_3=ScenarioConfig(
        description=('focals are resident and visited by zappers who harvest ' +
                     'sustainably but lack trust'),
        tags={
            'resident',
            'dyadic_trust',
            'ownership',
            'restraint',
            'defense',
        },
        substrate='commons_harvest__partnership',
        roles=('default',) * 7,
        is_focal=(True,) * 5 + (False,) * 2,
        bots_by_role={
            'default': {'commons_harvest__partnership__sustainable_fighter_0',
                        'commons_harvest__partnership__sustainable_fighter_1',},
        },
    ),
    commons_harvest__partnership_4=ScenarioConfig(
        description='focals visit pacifists who do not harvest sustainably',
        tags={
            'visitor',
            'dyadic_trust',
            'ownership',
            'restraint',
            'sanctioning',
        },
        substrate='commons_harvest__partnership',
        roles=('default',) * 7,
        is_focal=(True,) * 2 + (False,) * 5,
        bots_by_role={
            'default': {'commons_harvest__partnership__pacifist_0',
                        'commons_harvest__partnership__pacifist_1',
                        'commons_harvest__partnership__pacifist_2',},
        },
    ),
    commons_harvest__partnership_5=ScenarioConfig(
        description='focals visit zappers who do not harvest sustainably',
        tags={
            'visitor',
            'dyadic_trust',
            'ownership',
            'restraint',
            'defense',
        },
        substrate='commons_harvest__partnership',
        roles=('default',) * 7,
        is_focal=(True,) * 2 + (False,) * 5,
        bots_by_role={
            'default': {'commons_harvest__partnership__free_0',
                        'commons_harvest__partnership__free_1',
                        'commons_harvest__partnership__free_2',},
        },
    ),
    coop_mining_0=ScenarioConfig(
        description='visiting cooperators',
        tags={
            'visitor',
            'dyadic_trust',
        },
        substrate='coop_mining',
        roles=('default',) * 6,
        is_focal=(True,) * 1 + (False,) * 5,
        bots_by_role={
            'default': {'coop_mining__cooperator_0',},
        },
    ),
    coop_mining_1=ScenarioConfig(
        description='visiting residents that extract both ores',
        tags={
            'visitor',
            'flexibility'
        },
        substrate='coop_mining',
        roles=('default',) * 6,
        is_focal=(True,) * 1 + (False,) * 5,
        bots_by_role={
            'default': {'coop_mining__mixed_0',},
        },
    ),
    coop_mining_2=ScenarioConfig(
        description='visiting defectors',
        tags={
            'visitor',
            'acceptance',
        },
        substrate='coop_mining',
        roles=('default',) * 6,
        is_focal=(True,) * 1 + (False,) * 5,
        bots_by_role={
            'default': {'coop_mining__defector_0'},
        },
    ),
    coop_mining_3=ScenarioConfig(
        description='residents visited by a cooperator',
        tags={
            'resident',
            'partner_choice',
            'dyadic_trust',
        },
        substrate='coop_mining',
        roles=('default',) * 6,
        is_focal=(True,) * 5 + (False,) * 1,
        bots_by_role={
            'default': {'coop_mining__cooperator_0',},
        },
    ),
    coop_mining_4=ScenarioConfig(
        description='residents visited by a defector',
        tags={
            'resident',
            'partner_choice',
        },
        substrate='coop_mining',
        roles=('default',) * 6,
        is_focal=(True,) * 5 + (False,) * 1,
        bots_by_role={
            'default': {'coop_mining__defector_0',},
        },
    ),
    coop_mining_5=ScenarioConfig(
        description='find the cooperator partner',
        tags={
            'partner_choice',
            'visitor',
        },
        substrate='coop_mining',
        roles=('default',) * 5 + ('target',),
        is_focal=(True,) * 1 + (False,) * 5,
        bots_by_role={
            'default': {'coop_mining__defector_0',},
            'target': {'coop_mining__cooperator_0',
                       'coop_mining__mixed_0',},
        },
    ),
    daycare_0=ScenarioConfig(
        description='meeting a helpful parent',
        tags={
            'flexibility',
        },
        substrate='daycare',
        roles=('child',) + ('parent',),
        is_focal=(True,) + (False,),
        bots_by_role={
            'parent': {'daycare__helping_parent_0',},
        },
    ),
    daycare_1=ScenarioConfig(
        description='meeting a child who points to what they want',
        tags={
            'flexibility',
        },
        substrate='daycare',
        roles=('child',) + ('parent',),
        is_focal=(False,) + (True,),
        bots_by_role={
            'child': {'daycare__pointing_child_0',},
        },
    ),
    daycare_2=ScenarioConfig(
        description='meeting an unhelpful parent',
        tags={
            'flexibility',
        },
        substrate='daycare',
        roles=('child',) + ('parent',),
        is_focal=(True,) + (False,),
        bots_by_role={
            'parent': {'daycare__foraging_parent_0',},
        },
    ),
    daycare_3=ScenarioConfig(
        description='meeting an independent child',
        tags={
            'flexibility',
        },
        substrate='daycare',
        roles=('child',) + ('parent',),
        is_focal=(False,) + (True,),
        bots_by_role={
            'child': {'daycare__foraging_child_0',},
        },
    ),
    externality_mushrooms__dense_0=ScenarioConfig(
        description='visiting unconditional hihe (cooperator) players',
        tags={
            'visitor',
        },
        substrate='externality_mushrooms__dense',
        roles=('default',) * 5,
        is_focal=(True,) + (False,) * 4,
        bots_by_role={
            'default': {'externality_mushrooms__dense__puppet_hihe_0',},
        },
    ),
    externality_mushrooms__dense_1=ScenarioConfig(
        description='visiting unconditional fize (defector) players',
        tags={
            'visitor',
        },
        substrate='externality_mushrooms__dense',
        roles=('default',) * 5,
        is_focal=(True,) + (False,) * 4,
        bots_by_role={
            'default': {'externality_mushrooms__dense__puppet_fize_0',},
        },
    ),
    externality_mushrooms__dense_2=ScenarioConfig(
        description=('focals are resident and joined by two unconditional ' +
                     'hihe (cooperator) players'),
        tags={
            'resident',
            'prosocial_newcomers',
        },
        substrate='externality_mushrooms__dense',
        roles=('default',) * 5,
        is_focal=(True,) * 3 + (False,) * 2,
        bots_by_role={
            'default': {'externality_mushrooms__dense__puppet_hihe_0',},
        },
    ),
    externality_mushrooms__dense_3=ScenarioConfig(
        description=('focals are resident and joined by two unconditional ' +
                     'fize (defector) players'),
        tags={
            'resident',
            'antisocial_newcomers',
        },
        substrate='externality_mushrooms__dense',
        roles=('default',) * 5,
        is_focal=(True,) * 3 + (False,) * 2,
        bots_by_role={
            'default': {'externality_mushrooms__dense__puppet_fize_0',},
        },
    ),
    factory_commons__either_or_0=ScenarioConfig(
        description='visiting a sustainable background population',
        tags={
            'visitor',
        },
        substrate='factory_commons__either_or',
        roles=('default',) * 3,
        is_focal=(True,) * 1 + (False,) * 2,
        bots_by_role={
            'default': {'factory_commons__either_or__sustainable_0',
                        'factory_commons__either_or__sustainable_1',
                        'factory_commons__either_or__sustainable_2',},
        },
    ),
    factory_commons__either_or_1=ScenarioConfig(
        description='visiting an unsustainable background population',
        tags={
            'visitor',
        },
        substrate='factory_commons__either_or',
        roles=('default',) * 3,
        is_focal=(True,) * 1 + (False,) * 2,
        bots_by_role={
            'default': {'factory_commons__either_or__unsustainable_0',
                        'factory_commons__either_or__unsustainable_1',
                        'factory_commons__either_or__unsustainable_2',},
        },
    ),
    factory_commons__either_or_2=ScenarioConfig(
        description='resident focal agents are joined by a sustainable visitor',
        tags={
            'resident',
            'prosocial_newcomers',
        },
        substrate='factory_commons__either_or',
        roles=('default',) * 3,
        is_focal=(True,) * 2 + (False,) * 1,
        bots_by_role={
            'default': {'factory_commons__either_or__sustainable_0',
                        'factory_commons__either_or__sustainable_1',
                        'factory_commons__either_or__sustainable_2',},
        },
    ),
    factory_commons__either_or_3=ScenarioConfig(
        description=('resident focal agents are joined by an unsustainable ' +
                     'visitor'),
        tags={
            'resident',
            'antisocial_newcomers',
        },
        substrate='factory_commons__either_or',
        roles=('default',) * 3,
        is_focal=(True,) * 2 + (False,) * 1,
        bots_by_role={
            'default': {'factory_commons__either_or__unsustainable_0',
                        'factory_commons__either_or__unsustainable_1',
                        'factory_commons__either_or__unsustainable_2',},
        },
    ),
    fruit_market__concentric_rivers_0=ScenarioConfig(
        description='all apple farmers are focal',
        tags={
            'convention_following',
        },
        substrate='fruit_market__concentric_rivers',
        roles=('apple_farmer',) * 8 + ('banana_farmer',) * 8,
        is_focal=(True,) * 8 + (False,) * 8,
        bots_by_role={
            'banana_farmer': {
                'fruit_market__concentric_rivers__banana_farmer_0',
                'fruit_market__concentric_rivers__banana_farmer_1',
                'fruit_market__concentric_rivers__banana_farmer_2',
            },
        },
    ),
    fruit_market__concentric_rivers_1=ScenarioConfig(
        description='all banana farmers are focal',
        tags={
            'convention_following'
        },
        substrate='fruit_market__concentric_rivers',
        roles=('apple_farmer',) * 8 + ('banana_farmer',) * 8,
        is_focal=(False,) * 8 + (True,) * 8,
        bots_by_role={
            'apple_farmer': {
                'fruit_market__concentric_rivers__apple_farmer_0',
                'fruit_market__concentric_rivers__apple_farmer_1',
                'fruit_market__concentric_rivers__apple_farmer_2',
            },
        },
    ),
    fruit_market__concentric_rivers_2=ScenarioConfig(
        description='one focal apple farmer visits a background economy',
        tags={
            'visitor',
            'convention_following'
        },
        substrate='fruit_market__concentric_rivers',
        roles=('apple_farmer',) * 8 + ('banana_farmer',) * 8,
        is_focal=(True,) * 1 + (False,) * 15,
        bots_by_role={
            'apple_farmer': {
                'fruit_market__concentric_rivers__apple_farmer_0',
                'fruit_market__concentric_rivers__apple_farmer_1',
                'fruit_market__concentric_rivers__apple_farmer_2',
            },
            'banana_farmer': {
                'fruit_market__concentric_rivers__banana_farmer_0',
                'fruit_market__concentric_rivers__banana_farmer_1',
                'fruit_market__concentric_rivers__banana_farmer_2',
            },
        },
    ),
    fruit_market__concentric_rivers_3=ScenarioConfig(
        description='one focal banana farmer visits a background economy',
        tags={
            'visitor',
            'convention_following'
        },
        substrate='fruit_market__concentric_rivers',
        roles=('banana_farmer',) * 8 + ('apple_farmer',) * 8,
        is_focal=(True,) * 1 + (False,) * 15,
        bots_by_role={
            'apple_farmer': {
                'fruit_market__concentric_rivers__apple_farmer_0',
                'fruit_market__concentric_rivers__apple_farmer_1',
                'fruit_market__concentric_rivers__apple_farmer_2',
            },
            'banana_farmer': {
                'fruit_market__concentric_rivers__banana_farmer_0',
                'fruit_market__concentric_rivers__banana_farmer_1',
                'fruit_market__concentric_rivers__banana_farmer_2',
            },
        },
    ),
    gift_refinements_0=ScenarioConfig(
        description='visiting cooperators',
        tags={
            'visitor',
            'dyadic_trust',
        },
        substrate='gift_refinements',
        roles=('default',) * 6,
        is_focal=(True,) * 1 + (False,) * 5,
        bots_by_role=immutabledict.immutabledict(
            default=('gift_refinements__cooperator_0',),
        ),
    ),
    gift_refinements_1=ScenarioConfig(
        description='visiting defectors',
        tags={
            'visitor',
            'acceptance',
        },
        substrate='gift_refinements',
        roles=('default',) * 6,
        is_focal=(True,) * 1 + (False,) * 5,
        bots_by_role=immutabledict.immutabledict(
            default=('gift_refinements__defector_0',),
        ),
    ),
    gift_refinements_2=ScenarioConfig(
        description='visited by a cooperator',
        tags={
            'resident',
            'partner_choice',
        },
        substrate='gift_refinements',
        roles=('default',) * 6,
        is_focal=(True,) * 5 + (False,) * 1,
        bots_by_role=immutabledict.immutabledict(
            default=('gift_refinements__cooperator_0',),
        ),
    ),
    gift_refinements_3=ScenarioConfig(
        description='visited by a defector',
        tags={
            'resident',
            'parter_choice',
        },
        substrate='gift_refinements',
        roles=('default',) * 6,
        is_focal=(True,) * 5 + (False,) * 1,
        bots_by_role=immutabledict.immutabledict(
            default=('gift_refinements__defector_0',),
        ),
    ),
    gift_refinements_4=ScenarioConfig(
        description='find the cooperator partner',
        tags={
            'partner_choice',
        },
        substrate='gift_refinements',
        roles=('default',) * 5 + ('target',),
        is_focal=(True,) * 1 + (False,) * 5,
        bots_by_role=immutabledict.immutabledict(
            default=('gift_refinements__defector_0',),
            target=('gift_refinements__cooperator_0',),
        ),
    ),
    gift_refinements_5=ScenarioConfig(
        description='visiting extreme cooperators',
        tags={
            'visitor',
        },
        substrate='gift_refinements',
        roles=('default',) * 6,
        is_focal=(True,) * 1 + (False,) * 5,
        bots_by_role=immutabledict.immutabledict(
            default=('gift_refinements__extreme_cooperator_0',),
        ),
    ),
    gift_refinements_6=ScenarioConfig(
        description='visited by an extreme cooperator',
        tags={
            'resident',
        },
        substrate='gift_refinements',
        roles=('default',) * 6,
        is_focal=(True,) * 5 + (False,) * 1,
        bots_by_role=immutabledict.immutabledict(
            default=('gift_refinements__extreme_cooperator_0',),
        ),
    ),
    hidden_agenda_0=ScenarioConfig(
        description=(
            'A focal population is visited by impostor which hunts crewmates'),
        tags={
            'resident',
        },
        substrate='hidden_agenda',
        roles=('crewmate',) * 4 + ('impostor',),
        is_focal=(True,) * 4 + (False,) * 1,
        bots_by_role=immutabledict.immutabledict(
            impostor=('hidden_agenda__hunter_impostor_0',),
        ),
    ),
    hidden_agenda_1=ScenarioConfig(
        description='A focal impostor visits background crew who collect gems',
        tags={
            'visitor',
            'learned_teamwork',
        },
        substrate='hidden_agenda',
        roles=('crewmate',) * 4 + ('impostor',),
        is_focal=(False,) * 4 + (True,) * 1,
        bots_by_role=immutabledict.immutabledict(
            crewmate=('hidden_agenda__collector_crew_0',
                      'hidden_agenda__collector_crew_1'),
        ),
    ),
    hidden_agenda_2=ScenarioConfig(
        description=(
            'Focal crew visits background impostor, and crew who collect gems'),
        tags={
            'ad_hoc_teamwork',
        },
        substrate='hidden_agenda',
        roles=('crewmate',) * 4 + ('impostor',),
        is_focal=(True,) * 2 + (False,) * 3,
        bots_by_role=immutabledict.immutabledict(
            crewmate=('hidden_agenda__collector_crew_0',
                      'hidden_agenda__collector_crew_1'),
            impostor=('hidden_agenda__hunter_impostor_0',),
        ),
    ),
    paintball__capture_the_flag_0=ScenarioConfig(
        description='focal team versus shaped bot team',
        tags={
            'learned_teamwork',
            'defense',
        },
        substrate='paintball__capture_the_flag',
        roles=('default',) * 8,
        is_focal=(True, False) * 4,
        bots_by_role={
            'default': {'paintball__capture_the_flag__shaped_bot_0',
                        'paintball__capture_the_flag__shaped_bot_1',
                        'paintball__capture_the_flag__shaped_bot_2',
                        'paintball__capture_the_flag__shaped_bot_3',},
        },
    ),
    paintball__capture_the_flag_1=ScenarioConfig(
        description='ad hoc teamwork with shaped bots',
        tags={
            'ad_hoc_teamwork',
            'visitor',
            'defense',
        },
        substrate='paintball__capture_the_flag',
        roles=('default',) * 8,
        is_focal=(True,) * 1 + (False,) * 7,
        bots_by_role={
            'default': {'paintball__capture_the_flag__shaped_bot_0',
                        'paintball__capture_the_flag__shaped_bot_1',
                        'paintball__capture_the_flag__shaped_bot_2',
                        'paintball__capture_the_flag__shaped_bot_3',},
        },
    ),
    paintball__king_of_the_hill_0=ScenarioConfig(
        description='focal team versus default bot team',
        tags={
            'learned_teamwork',
            'defense',
        },
        substrate='paintball__king_of_the_hill',
        roles=('default',) * 8,
        is_focal=(True, False) * 4,
        bots_by_role={
            'default': {'paintball__king_of_the_hill__free_0',
                        'paintball__king_of_the_hill__free_1',
                        'paintball__king_of_the_hill__free_2',},
        },
    ),
    paintball__king_of_the_hill_1=ScenarioConfig(
        description='focal team versus shaped bot team',
        tags={
            'learned_teamwork',
            'defense',
        },
        substrate='paintball__king_of_the_hill',
        roles=('default',) * 8,
        is_focal=(True, False) * 4,
        bots_by_role={
            'default': {'paintball__king_of_the_hill__spawn_camper_0',
                        'paintball__king_of_the_hill__spawn_camper_1',
                        'paintball__king_of_the_hill__spawn_camper_2',
                        'paintball__king_of_the_hill__spawn_camper_3',},
        },
    ),
    paintball__king_of_the_hill_2=ScenarioConfig(
        description='ad hoc teamwork with default bots',
        tags={
            'ad_hoc_teamwork',
            'visitor',
            'defense',
        },
        substrate='paintball__king_of_the_hill',
        roles=('default',) * 8,
        is_focal=(True,) * 1 + (False,) * 7,
        bots_by_role={
            'default': {'paintball__king_of_the_hill__free_0',
                        'paintball__king_of_the_hill__free_1',
                        'paintball__king_of_the_hill__free_2',},
        },
    ),
    paintball__king_of_the_hill_3=ScenarioConfig(
        description='ad hoc teamwork with shaped bots',
        tags={
            'ad_hoc_teamwork',
            'visitor',
            'defense',
        },
        substrate='paintball__king_of_the_hill',
        roles=('default',) * 8,
        is_focal=(True,) * 1 + (False,) * 7,
        bots_by_role={
            'default': {'paintball__king_of_the_hill__spawn_camper_0',
                        'paintball__king_of_the_hill__spawn_camper_1',
                        'paintball__king_of_the_hill__spawn_camper_2',
                        'paintball__king_of_the_hill__spawn_camper_3',},
        },
    ),
    predator_prey__alley_hunt_0=ScenarioConfig(
        description='focal prey visited by background predators',
        tags={
            'resident',
        },
        substrate='predator_prey__alley_hunt',
        roles=('predator',) * 5 + ('prey',) * 8,
        is_focal=(False,) * 5 + (True,) * 8,
        bots_by_role={
            'predator': {'predator_prey__alley_hunt__predator_0',
                         'predator_prey__alley_hunt__predator_1',
                         'predator_prey__alley_hunt__predator_2',},
        },
    ),
    predator_prey__alley_hunt_1=ScenarioConfig(
        description=(
            'focal predators aim to eat resident prey'),
        tags={
            'visitor',
        },
        substrate='predator_prey__alley_hunt',
        roles=('predator',) * 5 + ('prey',) * 8,
        is_focal=(True,) * 5 + (False,) * 8,
        bots_by_role={
            'prey': {'predator_prey__alley_hunt__prey_0',
                     'predator_prey__alley_hunt__prey_1',
                     'predator_prey__alley_hunt__prey_2',},
        },
    ),
    predator_prey__alley_hunt_2=ScenarioConfig(
        description=(
            'a focal predator competes with background predators to eat prey'),
        tags={
            'visitor',
        },
        substrate='predator_prey__alley_hunt',
        roles=('predator',) * 5 + ('prey',) * 8,
        is_focal=(True,) + (False,) * 12,
        bots_by_role={
            'prey': {'predator_prey__alley_hunt__prey_0',
                     'predator_prey__alley_hunt__prey_1',
                     'predator_prey__alley_hunt__prey_2',},
            'predator': {'predator_prey__alley_hunt__predator_0',
                         'predator_prey__alley_hunt__predator_1',
                         'predator_prey__alley_hunt__predator_2',},
        },
    ),
    predator_prey__alley_hunt_3=ScenarioConfig(
        description=(
            'one focal prey ad hoc cooperates with background prey to avoid ' +
            'predation'),
        tags={
            'visitor',
        },
        substrate='predator_prey__alley_hunt',
        roles=('prey',) * 8 + ('predator',) * 5,
        is_focal=(True,) + (False,) * 12,
        bots_by_role={
            'prey': {'predator_prey__alley_hunt__prey_0',
                     'predator_prey__alley_hunt__prey_1',
                     'predator_prey__alley_hunt__prey_2',},
            'predator': {'predator_prey__alley_hunt__predator_0',
                         'predator_prey__alley_hunt__predator_1',
                         'predator_prey__alley_hunt__predator_2',},
        },
    ),
    predator_prey__open_0=ScenarioConfig(
        description='focal prey visited by background predators',
        tags={
            'resident',
        },
        substrate='predator_prey__open',
        roles=('predator',) * 3 + ('prey',) * 10,
        is_focal=(False,) * 3 + (True,) * 10,
        bots_by_role={
            'predator': {'predator_prey__open__basic_predator_0',
                         'predator_prey__open__basic_predator_1',},
        },
    ),
    predator_prey__open_1=ScenarioConfig(
        description=(
            'focal predators aim to eat basic resident prey'),
        tags={
            'visitor',
        },
        substrate='predator_prey__open',
        roles=('predator',) * 3 + ('prey',) * 10,
        is_focal=(True,) * 3 + (False,) * 10,
        bots_by_role={
            'prey': {'predator_prey__open__basic_prey_0',
                     'predator_prey__open__basic_prey_1',
                     'predator_prey__open__basic_prey_2',},
        },
    ),
    predator_prey__open_2=ScenarioConfig(
        description=(
            'a focal predator competes with background predators to hunt prey'),
        tags={
            'visitor',
        },
        substrate='predator_prey__open',
        roles=('predator',) * 3 + ('prey',) * 10,
        is_focal=(True,) + (False,) * 12,
        bots_by_role={
            'prey': {'predator_prey__open__basic_prey_0',
                     'predator_prey__open__basic_prey_1',
                     'predator_prey__open__basic_prey_2',},
            'predator': {'predator_prey__open__basic_predator_0',
                         'predator_prey__open__basic_predator_1',},
        },
    ),
    predator_prey__open_3=ScenarioConfig(
        description=(
            'one focal prey ad hoc cooperates with background prey to avoid ' +
            'predation'),
        tags={
            'visitor',
        },
        substrate='predator_prey__open',
        roles=('prey',) * 10 + ('predator',) * 3,
        is_focal=(True,) + (False,) * 12,
        bots_by_role={
            'prey': {'predator_prey__open__basic_prey_0',
                     'predator_prey__open__basic_prey_1',
                     'predator_prey__open__basic_prey_2',},
            'predator': {'predator_prey__open__basic_predator_0',
                         'predator_prey__open__basic_predator_1',},
        },
    ),
    predator_prey__open_4=ScenarioConfig(
        description=(
            'focal predators hunt smarter resident prey'),
        tags={
            'visitor',
        },
        substrate='predator_prey__open',
        roles=('predator',) * 3 + ('prey',) * 10,
        is_focal=(True,) * 3 + (False,) * 10,
        bots_by_role={
            'prey': {'predator_prey__open__smart_prey_0',
                     'predator_prey__open__smart_prey_1',
                     'predator_prey__open__smart_prey_2',},
        },
    ),
    predator_prey__open_5=ScenarioConfig(
        description=(
            'a focal predator competes with background predators to hunt ' +
            'smarter prey'),
        tags={
            'visitor',
        },
        substrate='predator_prey__open',
        roles=('predator',) * 3 + ('prey',) * 10,
        is_focal=(True,) + (False,) * 12,
        bots_by_role={
            'prey': {'predator_prey__open__smart_prey_0',
                     'predator_prey__open__smart_prey_1',
                     'predator_prey__open__smart_prey_2',},
            'predator': {'predator_prey__open__basic_predator_0',
                         'predator_prey__open__basic_predator_1',},
        },
    ),
    predator_prey__open_6=ScenarioConfig(
        description=(
            'one focal prey ad hoc cooperates with background smart prey to ' +
            'avoid predation'),
        tags={
            'visitor',
        },
        substrate='predator_prey__open',
        roles=('prey',) * 10 + ('predator',) * 3,
        is_focal=(True,) + (False,) * 12,
        bots_by_role={
            'prey': {'predator_prey__open__smart_prey_0',
                     'predator_prey__open__smart_prey_1',
                     'predator_prey__open__smart_prey_2',},
            'predator': {'predator_prey__open__basic_predator_0',
                         'predator_prey__open__basic_predator_1',},
        },
    ),
    predator_prey__orchard_0=ScenarioConfig(
        description='focal prey visited by background predators',
        tags={
            'resident',
        },
        substrate='predator_prey__orchard',
        roles=('predator',) * 5 + ('prey',) * 8,
        is_focal=(False,) * 5 + (True,) * 8,
        bots_by_role={
            'predator': {'predator_prey__orchard__basic_predator_0',
                         'predator_prey__orchard__basic_predator_1',
                         'predator_prey__orchard__basic_predator_2',},
        },
    ),
    predator_prey__orchard_1=ScenarioConfig(
        description=(
            'focal predators aim to eat resident population of ' +
            'unspecialized prey'),
        tags={
            'visitor',
        },
        substrate='predator_prey__orchard',
        roles=('predator',) * 5 + ('prey',) * 8,
        is_focal=(True,) * 5 + (False,) * 8,
        bots_by_role={
            'prey': {'predator_prey__orchard__basic_prey_0',
                     'predator_prey__orchard__basic_prey_1',
                     'predator_prey__orchard__basic_prey_2',
                     'predator_prey__orchard__basic_prey_3',
                     'predator_prey__orchard__basic_prey_4',
                     'predator_prey__orchard__basic_prey_5',},
        },
    ),
    predator_prey__orchard_2=ScenarioConfig(
        description=(
            'a focal predator competes with background predators to eat ' +
            'unspecialized prey'),
        tags={
            'visitor',
        },
        substrate='predator_prey__orchard',
        roles=('predator',) * 5 + ('prey',) * 8,
        is_focal=(True,) + (False,) * 12,
        bots_by_role={
            'prey': {'predator_prey__orchard__basic_prey_0',
                     'predator_prey__orchard__basic_prey_1',
                     'predator_prey__orchard__basic_prey_2',
                     'predator_prey__orchard__basic_prey_3',
                     'predator_prey__orchard__basic_prey_4',
                     'predator_prey__orchard__basic_prey_5',},
            'predator': {'predator_prey__orchard__basic_predator_0',
                         'predator_prey__orchard__basic_predator_1',
                         'predator_prey__orchard__basic_predator_2',},
        },
    ),
    predator_prey__orchard_3=ScenarioConfig(
        description=(
            'one focal prey ad hoc cooperates with unspecialized background ' +
            'prey to avoid predation'),
        tags={
            'visitor',
        },
        substrate='predator_prey__orchard',
        roles=('prey',) * 8 + ('predator',) * 5,
        is_focal=(True,) + (False,) * 12,
        bots_by_role={
            'prey': {'predator_prey__orchard__basic_prey_0',
                     'predator_prey__orchard__basic_prey_1',
                     'predator_prey__orchard__basic_prey_2',
                     'predator_prey__orchard__basic_prey_3',
                     'predator_prey__orchard__basic_prey_4',
                     'predator_prey__orchard__basic_prey_5',},
            'predator': {'predator_prey__orchard__basic_predator_0',
                         'predator_prey__orchard__basic_predator_1',
                         'predator_prey__orchard__basic_predator_2',},
        },
    ),
    predator_prey__orchard_4=ScenarioConfig(
        description=(
            'focal predators aim to eat resident population of acorn ' +
            'specialist prey'),
        tags={
            'visitor',
        },
        substrate='predator_prey__orchard',
        roles=('predator',) * 5 + ('prey',) * 8,
        is_focal=(True,) * 5 + (False,) * 8,
        bots_by_role={
            'prey': {'predator_prey__orchard__acorn_specialist_prey_0',
                     'predator_prey__orchard__acorn_specialist_prey_1',
                     'predator_prey__orchard__acorn_specialist_prey_2',
                     'predator_prey__orchard__acorn_specialist_prey_3',
                     'predator_prey__orchard__acorn_specialist_prey_4',},
        },
    ),
    predator_prey__orchard_5=ScenarioConfig(
        description=(
            'a focal predator competes with background predators to eat ' +
            'acorn specialist prey'),
        tags={
            'visitor',
        },
        substrate='predator_prey__orchard',
        roles=('predator',) * 5 + ('prey',) * 8,
        is_focal=(True,) + (False,) * 12,
        bots_by_role={
            'prey': {'predator_prey__orchard__acorn_specialist_prey_0',
                     'predator_prey__orchard__acorn_specialist_prey_1',
                     'predator_prey__orchard__acorn_specialist_prey_2',
                     'predator_prey__orchard__acorn_specialist_prey_3',
                     'predator_prey__orchard__acorn_specialist_prey_4',},
            'predator': {'predator_prey__orchard__basic_predator_0',
                         'predator_prey__orchard__basic_predator_1',
                         'predator_prey__orchard__basic_predator_2',},
        },
    ),
    predator_prey__orchard_6=ScenarioConfig(
        description=(
            'one focal prey ad hoc cooperates with acorn specialized ' +
            'background prey to avoid predation'),
        tags={
            'visitor',
        },
        substrate='predator_prey__orchard',
        roles=('prey',) * 8 + ('predator',) * 5,
        is_focal=(True,) + (False,) * 12,
        bots_by_role={
            'prey': {'predator_prey__orchard__acorn_specialist_prey_0',
                     'predator_prey__orchard__acorn_specialist_prey_1',
                     'predator_prey__orchard__acorn_specialist_prey_2',
                     'predator_prey__orchard__acorn_specialist_prey_3',
                     'predator_prey__orchard__acorn_specialist_prey_4',},
            'predator': {'predator_prey__orchard__basic_predator_0',
                         'predator_prey__orchard__basic_predator_1',
                         'predator_prey__orchard__basic_predator_2',},
        },
    ),
    predator_prey__random_forest_0=ScenarioConfig(
        description='focal prey visited by background predators',
        tags={
            'resident',
        },
        substrate='predator_prey__random_forest',
        roles=('predator',) * 5 + ('prey',) * 8,
        is_focal=(False,) * 5 + (True,) * 8,
        bots_by_role={
            'predator': {'predator_prey__random_forest__basic_predator_0',
                         'predator_prey__random_forest__basic_predator_1',
                         'predator_prey__random_forest__basic_predator_2',},
        },
    ),
    predator_prey__random_forest_1=ScenarioConfig(
        description=(
            'focal predators aim to eat resident prey'),
        tags={
            'visitor',
        },
        substrate='predator_prey__random_forest',
        roles=('predator',) * 5 + ('prey',) * 8,
        is_focal=(True,) * 5 + (False,) * 8,
        bots_by_role={
            'prey': {'predator_prey__random_forest__basic_prey_0',
                     'predator_prey__random_forest__basic_prey_1',
                     'predator_prey__random_forest__basic_prey_2',},
        },
    ),
    predator_prey__random_forest_2=ScenarioConfig(
        description=(
            'a focal predator competes with background predators to eat prey'),
        tags={
            'visitor',
        },
        substrate='predator_prey__random_forest',
        roles=('predator',) * 5 + ('prey',) * 8,
        is_focal=(True,) + (False,) * 12,
        bots_by_role={
            'prey': {'predator_prey__random_forest__basic_prey_0',
                     'predator_prey__random_forest__basic_prey_1',
                     'predator_prey__random_forest__basic_prey_2',},
            'predator': {'predator_prey__random_forest__basic_predator_0',
                         'predator_prey__random_forest__basic_predator_1',
                         'predator_prey__random_forest__basic_predator_2',},
        },
    ),
    predator_prey__random_forest_3=ScenarioConfig(
        description=(
            'one focal prey ad hoc cooperates with background prey to avoid ' +
            'predation'),
        tags={
            'visitor',
        },
        substrate='predator_prey__random_forest',
        roles=('prey',) * 8 + ('predator',) * 5,
        is_focal=(True,) + (False,) * 12,
        bots_by_role={
            'prey': {'predator_prey__random_forest__basic_prey_0',
                     'predator_prey__random_forest__basic_prey_1',
                     'predator_prey__random_forest__basic_prey_2',},
            'predator': {'predator_prey__random_forest__basic_predator_0',
                         'predator_prey__random_forest__basic_predator_1',
                         'predator_prey__random_forest__basic_predator_2',},
        },
    ),
    prisoners_dilemma_in_the_matrix__arena_0=ScenarioConfig(
        description='visiting unconditional cooperators',
        tags={
            'visitor',
            'exploitation',
        },
        substrate='prisoners_dilemma_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__arena__puppet_cooperator_0',
                'prisoners_dilemma_in_the_matrix__arena__puppet_cooperator_margin_0',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__arena_1=ScenarioConfig(
        description=('focals are resident and visited by an unconditional ' +
                     'cooperator'),
        tags={
            'resident',
            'partner_choice',
            'prosocial_newcomers',
        },
        substrate='prisoners_dilemma_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 7 + (False,) * 1,
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__arena__puppet_cooperator_0',
                'prisoners_dilemma_in_the_matrix__arena__puppet_cooperator_margin_0',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__arena_2=ScenarioConfig(
        description='focals are resident and visitors defect unconditionally',
        tags={
            'resident',
            'partner_choice',
            'antisocial_newcomers',
        },
        substrate='prisoners_dilemma_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 6 + (False,) * 2,
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__arena__puppet_defector_0',
                'prisoners_dilemma_in_the_matrix__arena__puppet_defector_margin_0',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__arena_3=ScenarioConfig(
        description=('visiting a population of hair-trigger grim ' +
                     'reciprocator bots who initially cooperate but, if ' +
                     'defected on once, will retaliate by defecting in all ' +
                     'future interactions'),
        tags={
            'visitor',
            'reciprocity',
        },
        substrate='prisoners_dilemma_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__arena__puppet_grim_one_strike_0',
                'prisoners_dilemma_in_the_matrix__arena__puppet_grim_one_strike_margin_0',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__arena_4=ScenarioConfig(
        description=('visiting a population of two-strikes grim ' +
                     'reciprocator bots who initially cooperate but, if ' +
                     'defected on twice, will retaliate by defecting in all ' +
                     'future interactions'),
        tags={
            'visitor',
            'reciprocity',
        },
        substrate='prisoners_dilemma_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__arena__puppet_grim_two_strikes_0',
                'prisoners_dilemma_in_the_matrix__arena__puppet_grim_two_strikes_margin_0',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__arena_5=ScenarioConfig(
        description=(
            'visiting a mixed population of k-strikes grim reciprocator bots ' +
            'with k values from 1 to 3, they initially cooperate but, if ' +
            'defected on k times, they retaliate in all future interactions'
        ),
        tags={
            'visitor',
            'reciprocity',
        },
        substrate='prisoners_dilemma_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 3 + (False,) * 5,
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__arena__puppet_grim_one_strike_0',
                'prisoners_dilemma_in_the_matrix__arena__puppet_grim_one_strike_margin_0',
                'prisoners_dilemma_in_the_matrix__arena__puppet_grim_three_strikes_0',
                'prisoners_dilemma_in_the_matrix__arena__puppet_grim_three_strikes_margin_0',
                'prisoners_dilemma_in_the_matrix__arena__puppet_grim_two_strikes_0',
                'prisoners_dilemma_in_the_matrix__arena__puppet_grim_two_strikes_margin_0',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__arena_6=ScenarioConfig(
        description=(
            'the environment is not as crowded as it could be, focals are ' +
            'resident and are visited by an unconditional cooperator'),
        tags={
            'resident',
            'prosocial_newcomers',
        },
        substrate='prisoners_dilemma_in_the_matrix__arena',
        roles=('default',) * 4,
        is_focal=(True,) * 3 + (False,) * 1,
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__arena__puppet_cooperator_0',
                'prisoners_dilemma_in_the_matrix__arena__puppet_cooperator_margin_0',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__arena_7=ScenarioConfig(
        description=(
            'there are only two players, the focal player and a two-strikes ' +
            'grim reciprocator'),
        tags={
            'reciprocity',
        },
        substrate='prisoners_dilemma_in_the_matrix__arena',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__arena__puppet_grim_two_strikes_margin_0',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__arena_8=ScenarioConfig(
        description=(
            'odd number of players, which creates time pressure to interact ' +
            'in order to avoid being stuck without a partner, focals are ' +
            'resident, and are visited by an unconditionally defecting ' +
            'background bot'),
        tags={
            'resident',
            'partner_choice',
            'time_pressure',
            'ostracism',
        },
        substrate='prisoners_dilemma_in_the_matrix__arena',
        roles=('default',) * 7,
        is_focal=(False,) + (True,) * 6,  # Reversed is_focal order for variety.
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__arena__puppet_defector_margin_0',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__arena_9=ScenarioConfig(
        description=(
            'odd number of players, which creates time pressure to interact ' +
            'in order to avoid being stuck without a partner, focals are ' +
            'resident, and are visited by unconditionally defecting ' +
            'background bots'),
        tags={
            'resident',
            'partner_choice',
            'time_pressure',
            'ostracism',
        },
        substrate='prisoners_dilemma_in_the_matrix__arena',
        roles=('default',) * 5,
        is_focal=(False,) + (True,) * 4,  # Reversed is_focal order for variety.
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__arena__puppet_defector_margin_0',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__arena_10=ScenarioConfig(
        description=(
            'the environment is not as crowded as it could be, focals are ' +
            'resident and are visited by a tit-for-tat reciprocator'),
        tags={
            'resident',
            'reciprocity',
        },
        substrate='prisoners_dilemma_in_the_matrix__arena',
        roles=('default',) * 4,
        is_focal=(True,) * 3 + (False,),
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__arena__puppet_tft_margin_0',
                'prisoners_dilemma_in_the_matrix__arena__puppet_tft_margin_1',
                'prisoners_dilemma_in_the_matrix__arena__puppet_tft_margin_2',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__arena_11=ScenarioConfig(
        description=(
            'there are only two players, the focal player and a tit-for-tat ' +
            'reciprocator'),
        tags={
            'reciprocity',
        },
        substrate='prisoners_dilemma_in_the_matrix__arena',
        roles=('default',) * 2,
        is_focal=(False,) + (True,),  # Reversed is_focal order for variety.
        bots_by_role={
            'default': {
                # In this 2-player case, we use only the *_margin_2 bot since it
                # has the highest margin.
                'prisoners_dilemma_in_the_matrix__arena__puppet_tft_margin_2',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__arena_12=ScenarioConfig(
        description=(
            'the environment is not as crowded as it could be, focals are ' +
            'resident and are visited by a tit-for-tat reciprocator who ' +
            'occasionally chooses to defect instead of to cooperate'),
        tags={
            'resident',
            'reciprocity',
            'forgiveness',
        },
        substrate='prisoners_dilemma_in_the_matrix__arena',
        roles=('default',) * 4,
        is_focal=(False,) + (True,) * 3,  # Reversed is_focal order for variety.
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__arena__puppet_tft_tremble_margin_0',
                'prisoners_dilemma_in_the_matrix__arena__puppet_tft_tremble_margin_1',
                'prisoners_dilemma_in_the_matrix__arena__puppet_tft_tremble_margin_2',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__arena_13=ScenarioConfig(
        description=(
            'there are only two players, the focal player and a tit-for-tat ' +
            'reciprocator who occasionally chooses to defect instead of to ' +
            'cooperate'),
        tags={
            'reciprocity',
        },
        substrate='prisoners_dilemma_in_the_matrix__arena',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                # In this 2-player case, we use only the *_margin_2 bot since it
                # has the highest margin.
                'prisoners_dilemma_in_the_matrix__arena__puppet_tft_tremble_margin_2',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__arena_14=ScenarioConfig(
        description=(
            'focals are resident and are visited by a tit-for-tat reciprocator'
        ),
        tags={
            'resident',
            'reciprocity',
        },
        substrate='prisoners_dilemma_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(False,) + (True,) * 7,  # Reversed is_focal order for variety.
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__arena__puppet_tft_margin_0',
                'prisoners_dilemma_in_the_matrix__arena__puppet_tft_margin_1',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__arena_15=ScenarioConfig(
        description=(
            'visting a population of tit-for-tat reciprocator bots'),
        tags={
            'visitor',
            'reciprocity',
        },
        substrate='prisoners_dilemma_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__arena__puppet_tft_margin_0',
                'prisoners_dilemma_in_the_matrix__arena__puppet_tft_margin_1',
                'prisoners_dilemma_in_the_matrix__arena__puppet_tft_margin_2',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__arena_16=ScenarioConfig(
        description=(
            'focals are resident, a visitor tries to take advantage of them ' +
            'by defecting, but if punished, will give in and switch to ' +
            'tit-for-tat conditional cooperation'
        ),
        tags={
            'resident',
            'teaching',
            'reciprocity',
            'forgiveness',
            'sanctioning',
        },
        substrate='prisoners_dilemma_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 7 + (False,),
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__arena__puppet_corrigible_0',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__arena_17=ScenarioConfig(
        description=(
            'focals are resident, visitors try to take advantage of them ' +
            'by defecting, but if punished, they give in and switch to ' +
            'tit-for-tat conditional cooperation'
        ),
        tags={
            'resident',
            'teaching',
            'reciprocity',
            'forgiveness',
            'sanctioning',
        },
        substrate='prisoners_dilemma_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 5 + (False,) * 3,
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__arena__puppet_corrigible_0',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__arena_18=ScenarioConfig(
        description=(
            'focals are resident, visitors try to take advantage of them ' +
            'by defecting, but if punished, they give in and switch to ' +
            'tit-for-tat conditional cooperation, the visitors occasionally ' +
            'switch their action from cooperate to defect or vice versa'
        ),
        tags={
            'resident',
            'teaching',
            'reciprocity',
            'forgiveness',
            'sanctioning',
        },
        substrate='prisoners_dilemma_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 5 + (False,) * 3,
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__arena__puppet_corrigible_tremble_0',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__arena_19=ScenarioConfig(
        description=(
            'there are only two players, the focal player and a corrigible ' +
            'visitor who initially tries to exploit the focal player by ' +
            'defecting on them, the focals can correct this behavior by ' +
            'punishing the bot in question, once corrected the bot implements' +
            'a conditional cooperation strategy'
        ),
        tags={
            'teaching',
            'sanctioning',
            'reciprocity',
        },
        substrate='prisoners_dilemma_in_the_matrix__arena',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__arena__puppet_corrigible_0',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__arena_20=ScenarioConfig(
        description=(
            'there are only two players, the focal player and a corrigible ' +
            'visitor who initially tries to exploit the focal player by ' +
            'defecting on them, the focals can correct this behavior by ' +
            'punishing the bot in question, once it has been corrected ' +
            'it implements a conditional cooperation strategy'),
        tags={
            'teaching',
            'sanctioning',
            'reciprocity',
            'forgiveness',
        },
        substrate='prisoners_dilemma_in_the_matrix__arena',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__arena__puppet_corrigible_tremble_0',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__arena_21=ScenarioConfig(
        description=(
            'visting a population of tit-for-tat reciprocator bots mixed ' +
            'with pure cooperator and defector bots'),
        tags={
            'visitor',
            'reciprocity',
        },
        substrate='prisoners_dilemma_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__arena__puppet_tft_margin_0',
                'prisoners_dilemma_in_the_matrix__arena__puppet_tft_margin_1',
                'prisoners_dilemma_in_the_matrix__arena__puppet_tft_margin_2',
                'prisoners_dilemma_in_the_matrix__arena__puppet_tft_tremble_margin_1',
                'prisoners_dilemma_in_the_matrix__arena__puppet_cooperator_margin_0',
                'prisoners_dilemma_in_the_matrix__arena__puppet_defector_margin_0',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__repeated_0=ScenarioConfig(
        description='partner may play either cooperate or defect',
        tags={
            'flexibility',
        },
        substrate='prisoners_dilemma_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__repeated__puppet_cooperator_margin_0',
                'prisoners_dilemma_in_the_matrix__repeated__puppet_cooperator_margin_1',
                'prisoners_dilemma_in_the_matrix__repeated__puppet_defector_margin_0',
                'prisoners_dilemma_in_the_matrix__repeated__puppet_defector_margin_1',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__repeated_1=ScenarioConfig(
        description='partner typically plays cooperate',
        tags={
            'exploitation',
            'partner_cooperates_unconditionally',
        },
        substrate='prisoners_dilemma_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__repeated__puppet_cooperator_margin_0',
                'prisoners_dilemma_in_the_matrix__repeated__puppet_cooperator_margin_1',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__repeated_2=ScenarioConfig(
        description='partner typically plays defect',
        tags={
            'acceptance',
            'partner_defects_unconditionally',
        },
        substrate='prisoners_dilemma_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__repeated__puppet_defector_margin_0',
                'prisoners_dilemma_in_the_matrix__repeated__puppet_defector_margin_1',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__repeated_3=ScenarioConfig(
        description=('partner is a hair-trigger grim reciprocator, i.e. one ' +
                     'who initially cooperates but, if defected on once, will' +
                     ' retaliate by defecting forever after'),
        tags={
            'reciprocity',
        },
        substrate='prisoners_dilemma_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__repeated__puppet_grim_one_strike_margin_0',
                'prisoners_dilemma_in_the_matrix__repeated__puppet_grim_one_strike_margin_1',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__repeated_4=ScenarioConfig(
        description=('partner is a two-strikes grim reciprocator, i.e. one ' +
                     'who initially cooperates, but if defected on twice, ' +
                     'will retaliate by defecting forever after'),
        tags={
            'reciprocity',
        },
        substrate='prisoners_dilemma_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__repeated__puppet_grim_two_strikes_margin_0',
                'prisoners_dilemma_in_the_matrix__repeated__puppet_grim_two_strikes_margin_1',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__repeated_5=ScenarioConfig(
        description='partner is a tit-for-tat conditional cooperator',
        tags={
            'reciprocity',
        },
        substrate='prisoners_dilemma_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__repeated__puppet_tft_margin_0',
                'prisoners_dilemma_in_the_matrix__repeated__puppet_tft_margin_1',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__repeated_6=ScenarioConfig(
        description=('partner is a tit-for-tat conditional cooperator who ' +
                     'occasionally plays defect instead of cooperate'),
        tags={
            'reciprocity',
            'forgiveness',
        },
        substrate='prisoners_dilemma_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__repeated__puppet_tft_tremble_margin_0',
                'prisoners_dilemma_in_the_matrix__repeated__puppet_tft_tremble_margin_1',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__repeated_7=ScenarioConfig(
        description=('partner plays cooperate for a while then switches to ' +
                     'defect'),
        tags={
            'flexibility',
        },
        substrate='prisoners_dilemma_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__repeated__puppet_flip_0',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__repeated_8=ScenarioConfig(
        description=('visitor tries to take advantage of the focal player ' +
                     'by playing defect, but if punished, visitor then ' +
                     'switches to tit-for-tat conditional cooperation'),
        tags={
            'teaching',
            'reciprocity',
            'forgiveness',
            'sanctioning',
        },
        substrate='prisoners_dilemma_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__repeated__puppet_corrigible_0',
            },
        },
    ),
    prisoners_dilemma_in_the_matrix__repeated_9=ScenarioConfig(
        description=('visitor tries to take advantage of the focal player ' +
                     'by playing defect, but if punished, visitor then ' +
                     'switches to noisy tit-for-tat conditional cooperation'),
        tags={
            'teaching',
            'reciprocity',
            'forgiveness',
            'sanctioning',
        },
        substrate='prisoners_dilemma_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'prisoners_dilemma_in_the_matrix__repeated__puppet_corrigible_tremble_0',
            },
        },
    ),
    pure_coordination_in_the_matrix__arena_0=ScenarioConfig(
        description=('focals are resident, a single visitor joins who may ' +
                     'prefer any option; whichever option it prefers, it ' +
                     'pursues it greedily'),
        tags={
            'resident',
            'flexibility',
        },
        substrate='pure_coordination_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 7 + (False,) * 1,
        bots_by_role={
            'default': {
                'pure_coordination_in_the_matrix__arena__pure_greedy_a_0',
                'pure_coordination_in_the_matrix__arena__pure_greedy_b_0',
                'pure_coordination_in_the_matrix__arena__pure_greedy_c_0',
            },
        },
    ),
    pure_coordination_in_the_matrix__arena_1=ScenarioConfig(
        description=('focals are resident, three visitors join who always ' +
                     'select the same option as their partner in the previous' +
                     'interaction and do so without being too greedy'),
        tags={
            'resident',
            'flexibility',
        },
        substrate='pure_coordination_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 5 + (False,) * 3,
        bots_by_role={
            'default': {
                'pure_coordination_in_the_matrix__arena__resp2prev_0',
            },
        },
    ),
    pure_coordination_in_the_matrix__arena_2=ScenarioConfig(
        description=('focals are resident, three visitors join who always ' +
                     'select the same option as their partner in the previous' +
                     'interaction and are greedy in doing so'),
        tags={
            'resident',
            'flexibility',
            'scarcity',
        },
        substrate='pure_coordination_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 5 + (False,) * 3,
        bots_by_role={
            'default': {
                'pure_coordination_in_the_matrix__arena__resp2prev_greedy_0',
            },
        },
    ),
    pure_coordination_in_the_matrix__arena_3=ScenarioConfig(
        description=('visiting a background population where all initially ' +
                     'choose option A (without greed) and then switch '+
                     'to either B or C after some time'),
        tags={
            'visitor',
            'convention_following',
            'flexibility',
        },
        substrate='pure_coordination_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'pure_coordination_in_the_matrix__arena__flip_a2b_0',
                'pure_coordination_in_the_matrix__arena__flip_a2c_0',
            },
        },
    ),
    pure_coordination_in_the_matrix__arena_4=ScenarioConfig(
        description=('visiting a background population where all initially ' +
                     'choose option B (without greed) and then switch '+
                     'to either A or C after some time'),
        tags={
            'visitor',
            'convention_following',
            'flexibility',
        },
        substrate='pure_coordination_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'pure_coordination_in_the_matrix__arena__flip_b2a_0',
                'pure_coordination_in_the_matrix__arena__flip_b2c_0',
            },
        },
    ),
    pure_coordination_in_the_matrix__arena_5=ScenarioConfig(
        description=('visiting a background population where all initially ' +
                     'choose option C (without greed) and then switch '+
                     'to either A or B after some time'),
        tags={
            'visitor',
            'convention_following',
            'flexibility',
        },
        substrate='pure_coordination_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'pure_coordination_in_the_matrix__arena__flip_c2a_0',
                'pure_coordination_in_the_matrix__arena__flip_c2b_0',
            },
        },
    ),
    pure_coordination_in_the_matrix__arena_6=ScenarioConfig(
        description=('visiting a background population where all typically ' +
                     'choose option A and do not collect it too greedily'),
        tags={
            'visitor',
            'convention_following',
        },
        substrate='pure_coordination_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'pure_coordination_in_the_matrix__arena__pure_a_0',
            },
        },
    ),
    pure_coordination_in_the_matrix__arena_7=ScenarioConfig(
        description=('visiting a background population where all typically ' +
                     'choose option B and do not collect it too greedily'),
        tags={
            'visitor',
            'convention_following',
        },
        substrate='pure_coordination_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'pure_coordination_in_the_matrix__arena__pure_b_0',
            },
        },
    ),
    pure_coordination_in_the_matrix__arena_8=ScenarioConfig(
        description=('visiting a background population where all typically ' +
                     'choose option C and do not collect it too greedily'),
        tags={
            'visitor',
            'convention_following',
        },
        substrate='pure_coordination_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'pure_coordination_in_the_matrix__arena__pure_c_0',
            },
        },
    ),
    pure_coordination_in_the_matrix__repeated_0=ScenarioConfig(
        description='meeting any pure strategy player',
        tags={
            'flexibility',
        },
        substrate='pure_coordination_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) * 1 + (False,) * 1,
        bots_by_role={
            'default': {
                'pure_coordination_in_the_matrix__repeated__pure_a_margin_0',
                'pure_coordination_in_the_matrix__repeated__pure_b_margin_0',
                'pure_coordination_in_the_matrix__repeated__pure_c_margin_0',
            },
        },
    ),
    pure_coordination_in_the_matrix__repeated_1=ScenarioConfig(
        description=('meeting an agent who plays the best response to ' +
                     'what the focal agent did in the last round.'),
        tags={
            'flexibility',
        },
        substrate='pure_coordination_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) * 1 + (False,) * 1,
        bots_by_role={
            'default': {
                'pure_coordination_in_the_matrix__repeated__resp2prev_margin_0',
            },
        },
    ),
    pure_coordination_in_the_matrix__repeated_2=ScenarioConfig(
        description=('versus mixture of opponents who often flip to other ' +
                     'strategies after some number of interactions'),
        tags={
            'flexibility',
        },
        substrate='pure_coordination_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) * 1 + (False,) * 1,
        bots_by_role={
            'default': {
                'pure_coordination_in_the_matrix__repeated__pure_a_margin_0',
                'pure_coordination_in_the_matrix__repeated__flip_a2b_0',
                'pure_coordination_in_the_matrix__repeated__flip_a2b_1',
                'pure_coordination_in_the_matrix__repeated__flip_a2c_0',
                'pure_coordination_in_the_matrix__repeated__flip_a2c_1',
                'pure_coordination_in_the_matrix__repeated__pure_b_margin_0',
                'pure_coordination_in_the_matrix__repeated__flip_b2c_0',
                'pure_coordination_in_the_matrix__repeated__flip_b2c_1',
                'pure_coordination_in_the_matrix__repeated__flip_b2a_0',
                'pure_coordination_in_the_matrix__repeated__flip_b2a_1',
                'pure_coordination_in_the_matrix__repeated__pure_c_margin_0',
                'pure_coordination_in_the_matrix__repeated__flip_c2a_0',
                'pure_coordination_in_the_matrix__repeated__flip_c2a_1',
                'pure_coordination_in_the_matrix__repeated__flip_c2b_0',
                'pure_coordination_in_the_matrix__repeated__flip_c2b_1'
            },
        },
    ),
    pure_coordination_in_the_matrix__repeated_3=ScenarioConfig(
        description='meeting an agent who almost always chooses resource a',
        tags={
            'convention_following',
        },
        substrate='pure_coordination_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) * 1 + (False,) * 1,
        bots_by_role={
            'default': {
                'pure_coordination_in_the_matrix__repeated__pure_a_margin_0',
            },
        },
    ),
    pure_coordination_in_the_matrix__repeated_4=ScenarioConfig(
        description='meeting an agent who almost always chooses resource b',
        tags={
            'convention_following',
        },
        substrate='pure_coordination_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) * 1 + (False,) * 1,
        bots_by_role={
            'default': {
                'pure_coordination_in_the_matrix__repeated__pure_b_margin_0',
            },
        },
    ),
    pure_coordination_in_the_matrix__repeated_5=ScenarioConfig(
        description='meeting an agent who almost always chooses resource c',
        tags={
            'convention_following',
        },
        substrate='pure_coordination_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) * 1 + (False,) * 1,
        bots_by_role={
            'default': {
                'pure_coordination_in_the_matrix__repeated__pure_c_margin_0',
            },
        },
    ),
    rationalizable_coordination_in_the_matrix__arena_0=ScenarioConfig(
        description=('focals are resident, a single visitor joins who may ' +
                     'prefer any option; whichever option it prefers, it ' +
                     'pursues it greedily'),
        tags={
            'resident',
            'flexibility',
        },
        substrate='rationalizable_coordination_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 7 + (False,) * 1,
        bots_by_role={
            'default': {
                'rationalizable_coordination_in_the_matrix__arena__pure_greedy_a_0',
                'rationalizable_coordination_in_the_matrix__arena__pure_greedy_b_0',
                'rationalizable_coordination_in_the_matrix__arena__pure_greedy_c_0',
            },
        },
    ),
    rationalizable_coordination_in_the_matrix__arena_1=ScenarioConfig(
        description=('focals are resident, three visitors join who always ' +
                     'select the same option as their partner in the previous' +
                     'interaction and do so without being too greedy'),
        tags={
            'resident',
            'flexibility',
        },
        substrate='rationalizable_coordination_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 5 + (False,) * 3,
        bots_by_role={
            'default': {
                'rationalizable_coordination_in_the_matrix__arena__resp2prev_0',
            },
        },
    ),
    rationalizable_coordination_in_the_matrix__arena_2=ScenarioConfig(
        description=('focals are resident, three visitors join who always ' +
                     'select the same option as their partner in the previous' +
                     'interaction and are greedy in doing so'),
        tags={
            'resident',
            'flexibility',
            'scarcity',
        },
        substrate='rationalizable_coordination_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 5 + (False,) * 3,
        bots_by_role={
            'default': {
                'rationalizable_coordination_in_the_matrix__arena__resp2prev_greedy_0',
            },
        },
    ),
    rationalizable_coordination_in_the_matrix__arena_3=ScenarioConfig(
        description=('visiting a background population where all initially ' +
                     'choose option A (without greed) and then switch '+
                     'to either B or C after some time'),
        tags={
            'visitor',
            'convention_following',
            'flexibility',
        },
        substrate='rationalizable_coordination_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'rationalizable_coordination_in_the_matrix__arena__flip_a2b_0',
                'rationalizable_coordination_in_the_matrix__arena__flip_a2c_0',
            },
        },
    ),
    rationalizable_coordination_in_the_matrix__arena_4=ScenarioConfig(
        description=('visiting a background population where all initially ' +
                     'choose option B (without greed) and then switch '+
                     'to either A or C after some time'),
        tags={
            'visitor',
            'convention_following',
            'flexibility',
        },
        substrate='rationalizable_coordination_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'rationalizable_coordination_in_the_matrix__arena__flip_b2a_0',
                'rationalizable_coordination_in_the_matrix__arena__flip_b2c_0',
            },
        },
    ),
    rationalizable_coordination_in_the_matrix__arena_5=ScenarioConfig(
        description=('visiting a background population where all initially ' +
                     'choose option C (without greed) and then switch '+
                     'to either A or B after some time'),
        tags={
            'visitor',
            'convention_following',
            'flexibility',
        },
        substrate='rationalizable_coordination_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'rationalizable_coordination_in_the_matrix__arena__flip_c2a_0',
                'rationalizable_coordination_in_the_matrix__arena__flip_c2b_0',
            },
        },
    ),
    rationalizable_coordination_in_the_matrix__arena_6=ScenarioConfig(
        description=('visiting a background population where all typically ' +
                     'choose option A and do not collect it too greedily'),
        tags={
            'visitor',
            'convention_following',
        },
        substrate='rationalizable_coordination_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'rationalizable_coordination_in_the_matrix__arena__pure_a_0',
            },
        },
    ),
    rationalizable_coordination_in_the_matrix__arena_7=ScenarioConfig(
        description=('visiting a background population where all typically ' +
                     'choose option B and do not collect it too greedily'),
        tags={
            'visitor',
            'convention_following',
        },
        substrate='rationalizable_coordination_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'rationalizable_coordination_in_the_matrix__arena__pure_b_0',
            },
        },
    ),
    rationalizable_coordination_in_the_matrix__arena_8=ScenarioConfig(
        description=('visiting a background population where all typically ' +
                     'choose option C and do not collect it too greedily'),
        tags={
            'visitor',
            'convention_following',
        },
        substrate='rationalizable_coordination_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'rationalizable_coordination_in_the_matrix__arena__pure_c_0',
            },
        },
    ),
    rationalizable_coordination_in_the_matrix__repeated_0=ScenarioConfig(
        description='meeting any pure strategy player',
        tags={
            'flexibility',
        },
        substrate='rationalizable_coordination_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) * 1 + (False,) * 1,
        bots_by_role={
            'default': {
                'rationalizable_coordination_in_the_matrix__repeated__pure_a_margin_0',
                'rationalizable_coordination_in_the_matrix__repeated__pure_b_margin_0',
                'rationalizable_coordination_in_the_matrix__repeated__pure_c_margin_0',
            },
        },
    ),
    rationalizable_coordination_in_the_matrix__repeated_1=ScenarioConfig(
        description=('meeting an agent who plays the best response to ' +
                     'what the focal agent did in the last round.'),
        tags={
            'flexibility',
        },
        substrate='rationalizable_coordination_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) * 1 + (False,) * 1,
        bots_by_role={
            'default': {
                'rationalizable_coordination_in_the_matrix__repeated__resp2prev_margin_0',
            },
        },
    ),
    rationalizable_coordination_in_the_matrix__repeated_2=ScenarioConfig(
        description=('versus mixture of opponents who often flip to other ' +
                     'strategies after some number of interactions'),
        tags={
            'flexibility',
        },
        substrate='rationalizable_coordination_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) * 1 + (False,) * 1,
        bots_by_role={
            'default': {
                'rationalizable_coordination_in_the_matrix__repeated__pure_a_margin_0',
                'rationalizable_coordination_in_the_matrix__repeated__flip_a2b_0',
                'rationalizable_coordination_in_the_matrix__repeated__flip_a2b_1',
                'rationalizable_coordination_in_the_matrix__repeated__flip_a2c_0',
                'rationalizable_coordination_in_the_matrix__repeated__flip_a2c_1',
                'rationalizable_coordination_in_the_matrix__repeated__pure_b_margin_0',
                'rationalizable_coordination_in_the_matrix__repeated__flip_b2c_0',
                'rationalizable_coordination_in_the_matrix__repeated__flip_b2c_1',
                'rationalizable_coordination_in_the_matrix__repeated__flip_b2a_0',
                'rationalizable_coordination_in_the_matrix__repeated__flip_b2a_1',
                'rationalizable_coordination_in_the_matrix__repeated__pure_c_margin_0',
                'rationalizable_coordination_in_the_matrix__repeated__flip_c2a_0',
                'rationalizable_coordination_in_the_matrix__repeated__flip_c2a_1',
                'rationalizable_coordination_in_the_matrix__repeated__flip_c2b_0',
                'rationalizable_coordination_in_the_matrix__repeated__flip_c2b_1'
            },
        },
    ),
    rationalizable_coordination_in_the_matrix__repeated_3=ScenarioConfig(
        description='meeting an agent who almost always chooses resource a',
        tags={
            'convention_following',
            'acceptance',
        },
        substrate='rationalizable_coordination_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) * 1 + (False,) * 1,
        bots_by_role={
            'default': {
                'rationalizable_coordination_in_the_matrix__repeated__pure_a_margin_0',
            },
        },
    ),
    rationalizable_coordination_in_the_matrix__repeated_4=ScenarioConfig(
        description='meeting an agent who almost always chooses resource b',
        tags={
            'convention_following',
            'acceptance',
        },
        substrate='rationalizable_coordination_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) * 1 + (False,) * 1,
        bots_by_role={
            'default': {
                'rationalizable_coordination_in_the_matrix__repeated__pure_b_margin_0',
            },
        },
    ),
    rationalizable_coordination_in_the_matrix__repeated_5=ScenarioConfig(
        description='meeting an agent who almost always chooses resource c',
        tags={
            'convention_following',
        },
        substrate='rationalizable_coordination_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) * 1 + (False,) * 1,
        bots_by_role={
            'default': {
                'rationalizable_coordination_in_the_matrix__repeated__pure_c_margin_0',
            },
        },
    ),
    running_with_scissors_in_the_matrix__arena_0=ScenarioConfig(
        description=('versus a background population containing bots ' +
                     'implementing all three pure strategies'),
        tags={
            'visitor',
            'flexibility',
        },
        substrate='running_with_scissors_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 1 + (False,) * 7,
        bots_by_role={
            'default': {
                'running_with_scissors_in_the_matrix__arena__rock_margin_0',
                'running_with_scissors_in_the_matrix__arena__rock_margin_1',
                'running_with_scissors_in_the_matrix__arena__paper_margin_0',
                'running_with_scissors_in_the_matrix__arena__paper_margin_1',
                'running_with_scissors_in_the_matrix__arena__scissors_margin_0',
                'running_with_scissors_in_the_matrix__arena__scissors_margin_1',
            },
        }
    ),
    running_with_scissors_in_the_matrix__arena_1=ScenarioConfig(
        description=('versus gullible bots'),
        tags={
            'deception',
            'visitor',
            'versus_gullible',
        },
        substrate='running_with_scissors_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 1 + (False,) * 7,
        bots_by_role={
            'default': {
                'running_with_scissors_in_the_matrix__arena__free_0',
            },
        }
    ),
    running_with_scissors_in_the_matrix__arena_2=ScenarioConfig(
        description=('versus mixture of opponents who play rock and some who ' +
                     'flip to scissors after two interactions'),
        tags={
            'visitor',
            'flexibility',
        },
        substrate='running_with_scissors_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 1 + (False,) * 7,
        bots_by_role={
            'default': {
                'running_with_scissors_in_the_matrix__arena__rock_margin_0',
                'running_with_scissors_in_the_matrix__arena__rock_margin_1',
                'running_with_scissors_in_the_matrix__arena__flip_r2s_0',
            },
        }
    ),
    running_with_scissors_in_the_matrix__arena_3=ScenarioConfig(
        description=('versus mixture of opponents who play paper and some ' +
                     'who flip to rock after two interactions'),
        tags={
            'visitor',
            'flexibility',
        },
        substrate='running_with_scissors_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 1 + (False,) * 7,
        bots_by_role={
            'default': {
                'running_with_scissors_in_the_matrix__arena__paper_margin_0',
                'running_with_scissors_in_the_matrix__arena__paper_margin_1',
                'running_with_scissors_in_the_matrix__arena__flip_p2r_0',
            },
        }
    ),
    running_with_scissors_in_the_matrix__arena_4=ScenarioConfig(
        description=('versus mixture of opponents who play scissors and some ' +
                     'who flip to paper after two interactions'),
        tags={
            'visitor',
            'exploitation',
        },
        substrate='running_with_scissors_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 1 + (False,) * 7,
        bots_by_role={
            'default': {
                'running_with_scissors_in_the_matrix__arena__scissors_margin_0',
                'running_with_scissors_in_the_matrix__arena__scissors_margin_1',
                'running_with_scissors_in_the_matrix__arena__flip_s2p_0',
            },
        }
    ),
    running_with_scissors_in_the_matrix__arena_5=ScenarioConfig(
        description=('visiting a population of pure paper bots'),
        tags={
            'visitor',
            'exploitation',
        },
        substrate='running_with_scissors_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'running_with_scissors_in_the_matrix__arena__paper_margin_0',
                'running_with_scissors_in_the_matrix__arena__paper_margin_1',
            },
        }
    ),
    running_with_scissors_in_the_matrix__arena_6=ScenarioConfig(
        description=('visiting a population of pure rock bots'),
        tags={
            'visitor',
            'exploitation',
        },
        substrate='running_with_scissors_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'running_with_scissors_in_the_matrix__arena__rock_margin_0',
                'running_with_scissors_in_the_matrix__arena__rock_margin_1',
            },
        }
    ),
    running_with_scissors_in_the_matrix__arena_7=ScenarioConfig(
        description=('visiting a population of pure scissors bots'),
        tags={
            'visitor',
            'exploitation',
        },
        substrate='running_with_scissors_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'running_with_scissors_in_the_matrix__arena__scissors_margin_0',
                'running_with_scissors_in_the_matrix__arena__scissors_margin_1',
            },
        }
    ),
    running_with_scissors_in_the_matrix__one_shot_0=ScenarioConfig(
        description='versus mixed strategy opponent',
        tags={
            'flexibility',
        },
        substrate='running_with_scissors_in_the_matrix__one_shot',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'running_with_scissors_in_the_matrix__one_shot__rock_margin_0',
                'running_with_scissors_in_the_matrix__one_shot__paper_margin_0',
                'running_with_scissors_in_the_matrix__one_shot__scissors_margin_0',
            },
        }
    ),
    running_with_scissors_in_the_matrix__one_shot_1=ScenarioConfig(
        description='versus pure rock opponent',
        tags={
            'exploitation',
        },
        substrate='running_with_scissors_in_the_matrix__one_shot',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'running_with_scissors_in_the_matrix__one_shot__rock_margin_0',
            },
        }
    ),
    running_with_scissors_in_the_matrix__one_shot_2=ScenarioConfig(
        description='versus pure paper opponent',
        tags={
            'exploitation',
        },
        substrate='running_with_scissors_in_the_matrix__one_shot',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'running_with_scissors_in_the_matrix__one_shot__paper_margin_0',
            },
        }
    ),
    running_with_scissors_in_the_matrix__one_shot_3=ScenarioConfig(
        description='versus pure scissors opponent',
        tags={
            'exploitation',
        },
        substrate='running_with_scissors_in_the_matrix__one_shot',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'running_with_scissors_in_the_matrix__one_shot__scissors_margin_0',
            },
        }
    ),
    running_with_scissors_in_the_matrix__repeated_0=ScenarioConfig(
        description='versus mixed strategy opponent',
        tags={
            'flexibility',
        },
        substrate='running_with_scissors_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'running_with_scissors_in_the_matrix__repeated__rock_margin_0',
                'running_with_scissors_in_the_matrix__repeated__paper_margin_0',
                'running_with_scissors_in_the_matrix__repeated__scissors_margin_0',
            },
        }
    ),
    running_with_scissors_in_the_matrix__repeated_1=ScenarioConfig(
        description=('versus opponent who plays the best response to ' +
                     'what the focal player did in the last round.'),
        tags={
            'flexibility',
        },
        substrate='running_with_scissors_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'running_with_scissors_in_the_matrix__repeated__resp2prev_margin_0',
            },
        }
    ),
    running_with_scissors_in_the_matrix__repeated_2=ScenarioConfig(
        description=('versus opponent who sometimes plays a pure strategy ' +
                     'but sometimes plays the best response to what the ' +
                     'focal player did in the last round'),
        tags={
            'flexibility',
        },
        substrate='running_with_scissors_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'running_with_scissors_in_the_matrix__repeated__resp2prev_margin_0',
                'running_with_scissors_in_the_matrix__repeated__rock_margin_0',
                'running_with_scissors_in_the_matrix__repeated__paper_margin_0',
                'running_with_scissors_in_the_matrix__repeated__scissors_margin_0',
            },
        }
    ),
    running_with_scissors_in_the_matrix__repeated_3=ScenarioConfig(
        description=('versus mixture of opponents who often flip to other ' +
                     'strategies after two interactions'),
        tags={
            'flexibility',
        },
        substrate='running_with_scissors_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'running_with_scissors_in_the_matrix__repeated__rock_0',
                'running_with_scissors_in_the_matrix__repeated__rock_margin_0',
                'running_with_scissors_in_the_matrix__repeated__flip_r2s_0',
                'running_with_scissors_in_the_matrix__repeated__paper_0',
                'running_with_scissors_in_the_matrix__repeated__paper_margin_0',
                'running_with_scissors_in_the_matrix__repeated__flip_p2r_0',
                'running_with_scissors_in_the_matrix__repeated__scissors_0',
                'running_with_scissors_in_the_matrix__repeated__scissors_margin_0',
                'running_with_scissors_in_the_matrix__repeated__flip_s2p_0',
            },
        }
    ),
    running_with_scissors_in_the_matrix__repeated_4=ScenarioConfig(
        description=('versus mixture of opponents who either flip to another ' +
                     'strategy after one interaction and keep it forever or ' +
                     'continue to change, always best responding to what ' +
                     'the focal player just did'),
        tags={
            'flexibility',
        },
        substrate='running_with_scissors_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'running_with_scissors_in_the_matrix__repeated__flip_r2s_1',
                'running_with_scissors_in_the_matrix__repeated__flip_p2r_1',
                'running_with_scissors_in_the_matrix__repeated__flip_s2p_1',
                'running_with_scissors_in_the_matrix__repeated__resp2prev_margin_0',
            },
        }
    ),
    running_with_scissors_in_the_matrix__repeated_5=ScenarioConfig(
        description='versus gullible opponent',
        tags={
            'deception',
            'versus_gullible',
        },
        substrate='running_with_scissors_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'running_with_scissors_in_the_matrix__repeated__free_0',
            },
        }
    ),
    running_with_scissors_in_the_matrix__repeated_6=ScenarioConfig(
        description='versus pure rock opponent',
        tags={
            'exploitation',
        },
        substrate='running_with_scissors_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'running_with_scissors_in_the_matrix__repeated__rock_margin_0',
            },
        }
    ),
    running_with_scissors_in_the_matrix__repeated_7=ScenarioConfig(
        description='versus pure paper opponent',
        tags={
            'exploitation',
        },
        substrate='running_with_scissors_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'running_with_scissors_in_the_matrix__repeated__paper_margin_0',
            },
        }
    ),
    running_with_scissors_in_the_matrix__repeated_8=ScenarioConfig(
        description='versus pure scissors opponent',
        tags={
            'exploitation',
        },
        substrate='running_with_scissors_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'running_with_scissors_in_the_matrix__repeated__scissors_margin_0',
            },
        }
    ),
    stag_hunt_in_the_matrix__arena_0=ScenarioConfig(
        description='visiting unconditional stag players',
        tags={
            'visitor',
            'convention_following',
        },
        substrate='stag_hunt_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'stag_hunt_in_the_matrix__arena__puppet_stag_0',
                'stag_hunt_in_the_matrix__arena__puppet_stag_margin_0',
            },
        },
    ),
    stag_hunt_in_the_matrix__arena_1=ScenarioConfig(
        description='visiting unconditional hare players',
        tags={
            'visitor',
            'convention_following',
        },
        substrate='stag_hunt_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'stag_hunt_in_the_matrix__arena__puppet_hare_0',
                'stag_hunt_in_the_matrix__arena__puppet_hare_margin_0',
            },
        },
    ),
    stag_hunt_in_the_matrix__arena_2=ScenarioConfig(
        description=('focals are resident and visitors are unconditional ' +
                     'stag players'),
        tags={
            'resident',
            'prosocial_newcomers',
        },
        substrate='stag_hunt_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 5 + (False,) * 3,
        bots_by_role={
            'default': {
                'stag_hunt_in_the_matrix__arena__puppet_stag_0',
                'stag_hunt_in_the_matrix__arena__puppet_stag_margin_0',
            },
        },
    ),
    stag_hunt_in_the_matrix__arena_3=ScenarioConfig(
        description=('focals are resident and visitors are unconditional' +
                     'hare players'),
        tags={
            'resident',
            'antisocial_newcomers',
        },
        substrate='stag_hunt_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 5 + (False,) * 3,
        bots_by_role={
            'default': {
                'stag_hunt_in_the_matrix__arena__puppet_hare_0',
                'stag_hunt_in_the_matrix__arena__puppet_hare_margin_0',
            },
        },
    ),
    stag_hunt_in_the_matrix__arena_4=ScenarioConfig(
        description=('visiting a population of hair-trigger grim ' +
                     'reciprocator bots who initially play stag but, if ' +
                     'any partner plays hare once, they give up on trying to ' +
                     'cooperate and play hare in all future interactions'),
        tags={
            'visitor',
            'reciprocity',
        },
        substrate='stag_hunt_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'stag_hunt_in_the_matrix__arena__puppet_grim_one_strike_0',
                'stag_hunt_in_the_matrix__arena__puppet_grim_one_strike_margin_0',
            },
        },
    ),
    stag_hunt_in_the_matrix__arena_5=ScenarioConfig(
        description=('visiting a population of two-strikes grim ' +
                     'reciprocator bots who initially play stag but, if ' +
                     'their partners play hare twice, they give up on trying ' +
                     'to cooperate and play hare in all future interactions'),
        tags={
            'visitor',
            'reciprocity',
        },
        substrate='stag_hunt_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) + (False,) * 7,
        bots_by_role={
            'default': {
                'stag_hunt_in_the_matrix__arena__puppet_grim_two_strikes_0',
                'stag_hunt_in_the_matrix__arena__puppet_grim_two_strikes_margin_0',
            },
        },
    ),
    stag_hunt_in_the_matrix__arena_6=ScenarioConfig(
        description=(
            'visiting a mixed population of k-strikes grim reciprocator bots ' +
            'with k values from 1 to 3, they initially play stag but, if ' +
            'their partners play hare k times, they then play hare in all ' +
            'future interactions'
        ),
        tags={
            'visitor',
            'reciprocity',
        },
        substrate='stag_hunt_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 3 + (False,) * 5,
        bots_by_role={
            'default': {
                'stag_hunt_in_the_matrix__arena__puppet_grim_one_strike_0',
                'stag_hunt_in_the_matrix__arena__puppet_grim_one_strike_margin_0',
                'stag_hunt_in_the_matrix__arena__puppet_grim_three_strikes_0',
                'stag_hunt_in_the_matrix__arena__puppet_grim_three_strikes_margin_0',
                'stag_hunt_in_the_matrix__arena__puppet_grim_two_strikes_0',
                'stag_hunt_in_the_matrix__arena__puppet_grim_two_strikes_margin_0',
            },
        },
    ),
    stag_hunt_in_the_matrix__arena_7=ScenarioConfig(
        description='visiting a mixture of pure hare and pure stag players',
        tags={
            'visitor',
            'flexibility',
        },
        substrate='stag_hunt_in_the_matrix__arena',
        roles=('default',) * 8,
        is_focal=(True,) * 3 + (False,) * 5,
        bots_by_role={
            'default': {
                'stag_hunt_in_the_matrix__arena__puppet_stag_0',
                'stag_hunt_in_the_matrix__arena__puppet_stag_margin_0',
                'stag_hunt_in_the_matrix__arena__puppet_hare_0',
                'stag_hunt_in_the_matrix__arena__puppet_hare_margin_0',
            },
        },
    ),
    stag_hunt_in_the_matrix__repeated_0=ScenarioConfig(
        description='partner may play either stag or hare',
        tags={
            'flexibility',
        },
        substrate='stag_hunt_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'stag_hunt_in_the_matrix__repeated__puppet_hare_margin_0',
                'stag_hunt_in_the_matrix__repeated__puppet_hare_margin_1',
                'stag_hunt_in_the_matrix__repeated__puppet_stag_margin_0',
                'stag_hunt_in_the_matrix__repeated__puppet_stag_margin_1',
            },
        },
    ),
    stag_hunt_in_the_matrix__repeated_1=ScenarioConfig(
        description='partner typically plays stag',
        tags={
            'partner_cooperates_unconditionally',
        },
        substrate='stag_hunt_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'stag_hunt_in_the_matrix__repeated__puppet_stag_margin_0',
                'stag_hunt_in_the_matrix__repeated__puppet_stag_margin_1',
            },
        },
    ),
    stag_hunt_in_the_matrix__repeated_2=ScenarioConfig(
        description='partner typically plays hare',
        tags={
            'acceptance',
            'partner_defects_unconditionally',
        },
        substrate='stag_hunt_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'stag_hunt_in_the_matrix__repeated__puppet_hare_margin_0',
                'stag_hunt_in_the_matrix__repeated__puppet_hare_margin_1',
            },
        },
    ),
    stag_hunt_in_the_matrix__repeated_3=ScenarioConfig(
        description=('partner is a hair-trigger grim reciprocator, i.e. one ' +
                     'who initially cooperates but, if defected on once, will' +
                     ' retaliate by defecting forever after'),
        tags={
            'reciprocity',
        },
        substrate='stag_hunt_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'stag_hunt_in_the_matrix__repeated__puppet_grim_one_strike_margin_0',
                'stag_hunt_in_the_matrix__repeated__puppet_grim_one_strike_margin_1',
            },
        },
    ),
    stag_hunt_in_the_matrix__repeated_4=ScenarioConfig(
        description=('partner is a two-strikes grim reciprocator, i.e. one ' +
                     'who initially cooperates, but if defected on twice, ' +
                     'will retaliate by defecting forever after'),
        tags={
            'reciprocity',
        },
        substrate='stag_hunt_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'stag_hunt_in_the_matrix__repeated__puppet_grim_two_strikes_margin_0',
                'stag_hunt_in_the_matrix__repeated__puppet_grim_two_strikes_margin_1',
            },
        },
    ),
    stag_hunt_in_the_matrix__repeated_5=ScenarioConfig(
        description='partner is a tit-for-tat conditional cooperator',
        tags={
            'reciprocity',
        },
        substrate='stag_hunt_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'stag_hunt_in_the_matrix__repeated__puppet_tft_margin_0',
                'stag_hunt_in_the_matrix__repeated__puppet_tft_margin_1',
            },
        },
    ),
    stag_hunt_in_the_matrix__repeated_6=ScenarioConfig(
        description=('partner is a tit-for-tat conditional cooperator who ' +
                     'occasionally plays hare instead of stag'),
        tags={
            'reciprocity',
            'forgiveness',
        },
        substrate='stag_hunt_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'stag_hunt_in_the_matrix__repeated__puppet_tft_tremble_margin_0',
                'stag_hunt_in_the_matrix__repeated__puppet_tft_tremble_margin_1',
            },
        },
    ),
    stag_hunt_in_the_matrix__repeated_7=ScenarioConfig(
        description='partner plays stag for a while then switches to hare',
        tags={
            'flexibility',
        },
        substrate='stag_hunt_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'stag_hunt_in_the_matrix__repeated__puppet_flip_0',
            },
        },
    ),
    stag_hunt_in_the_matrix__repeated_8=ScenarioConfig(
        description=('partner initially plays hare, but if punished, partner ' +
                     'then switches to tit-for-tat conditional cooperation'),
        tags={
            'teaching',
            'reciprocity',
            'forgiveness',
            'sanctioning',
        },
        substrate='stag_hunt_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'stag_hunt_in_the_matrix__repeated__puppet_corrigible_0',
            },
        },
    ),
    stag_hunt_in_the_matrix__repeated_9=ScenarioConfig(
        description=('partner initially plays hare, but if punished, partner ' +
                     'then switches to noisy tit-for-tat conditional ' +
                     'cooperation'),
        tags={
            'teaching',
            'reciprocity',
            'forgiveness',
            'sanctioning',
        },
        substrate='stag_hunt_in_the_matrix__repeated',
        roles=('default',) * 2,
        is_focal=(True,) + (False,),
        bots_by_role={
            'default': {
                'stag_hunt_in_the_matrix__repeated__puppet_corrigible_tremble_0',
            },
        },
    ),
    territory__inside_out_0=ScenarioConfig(
        description='focals are resident and visited by an aggressor',
        tags={
            'resident',
            'antisocial_newcomers',
            'defense',
        },
        substrate='territory__inside_out',
        roles=('default',) * 5,
        is_focal=(True,) * 4 + (False,),
        bots_by_role={
            'default': {
                'territory__inside_out__aggressor_0',
                'territory__inside_out__aggressor_1',
                'territory__inside_out__aggressor_2',
                'territory__inside_out__aggressor_3',
                'territory__inside_out__aggressor_with_extra_training_0',
            },
        },
    ),
    territory__inside_out_1=ScenarioConfig(
        description='visiting a population of aggressors',
        tags={
            'visitor',
            'defense',
        },
        substrate='territory__inside_out',
        roles=('default',) * 5,
        is_focal=(True,) + (False,) * 4,
        bots_by_role={
            'default': {
                'territory__inside_out__aggressor_0',
                'territory__inside_out__aggressor_1',
                'territory__inside_out__aggressor_2',
                'territory__inside_out__aggressor_3',
                'territory__inside_out__aggressor_with_extra_training_0',
            },
        },
    ),
    territory__inside_out_2=ScenarioConfig(
        description='focals are resident, visited by a bot that does nothing',
        tags={
            'resident',
            'restraint',
        },
        substrate='territory__inside_out',
        roles=('default',) * 5,
        is_focal=(True,) * 4 + (False,),
        bots_by_role={'default': {'noop_bot'}},
    ),
    territory__inside_out_3=ScenarioConfig(
        description='focals visit a resident population that does nothing.',
        tags={
            'visitor',
            'chance_to_benefit_oneself_without_imposing_social_cost',
        },
        substrate='territory__inside_out',
        roles=('default',) * 5,
        is_focal=(True,) + (False,) * 4,
        bots_by_role={'default': {'noop_bot'}},
    ),
    territory__inside_out_4=ScenarioConfig(
        description=('focals are resident, visited by a bot that claims a ' +
                     'moderate size territory and mostly tolerates its ' +
                     'neighbors'),
        tags={
            'resident',
            'defense',
        },
        substrate='territory__inside_out',
        roles=('default',) * 5,
        is_focal=(True,) * 4 + (False,),
        bots_by_role={
            'default': {
                'territory__inside_out__somewhat_tolerant_bot_0',
                'territory__inside_out__somewhat_tolerant_bot_1',},
        },
    ),
    territory__inside_out_5=ScenarioConfig(
        description=('focals visit a resident population that claims a ' +
                     'moderate size territory and mostly tolerates its ' +
                     'neighbors'),
        tags={
            'visitor',
            'defense',
        },
        substrate='territory__inside_out',
        roles=('default',) * 5,
        is_focal=(True,) + (False,) * 4,
        bots_by_role={
            'default': {
                'territory__inside_out__somewhat_tolerant_bot_0',
                'territory__inside_out__somewhat_tolerant_bot_1',},
        },
    ),
    territory__open_0=ScenarioConfig(
        description='focals are resident and visited by an aggressor',
        tags={
            'resident',
            'antisocial_newcomers',
            'defense',
        },
        substrate='territory__open',
        roles=('default',) * 9,
        is_focal=(True,) * 8 + (False,),
        bots_by_role={
            'default': {
                'territory__open__aggressor_0',
                'territory__open__aggressor_1',
                'territory__open__aggressor_2',
                'territory__open__aggressor_3',
                'territory__open__aggressor_with_extra_training_0',
            },
        },
    ),
    territory__open_1=ScenarioConfig(
        description='visiting a population of aggressors',
        tags={
            'visitor',
            'defense',
        },
        substrate='territory__open',
        roles=('default',) * 9,
        is_focal=(True,) + (False,) * 8,
        bots_by_role={
            'default': {
                'territory__open__aggressor_0',
                'territory__open__aggressor_1',
                'territory__open__aggressor_2',
                'territory__open__aggressor_3',
                'territory__open__aggressor_with_extra_training_0',
            },
        },
    ),
    territory__open_2=ScenarioConfig(
        description='focals are resident, visited by a bot that does nothing',
        tags={
            'resident',
            'restraint',
        },
        substrate='territory__open',
        roles=('default',) * 9,
        is_focal=(True,) * 8 + (False,),
        bots_by_role={'default': {'noop_bot'}},
    ),
    territory__open_3=ScenarioConfig(
        description='focals visit a resident population that does nothing',
        tags={
            'visitor',
            'chance_to_benefit_oneself_without_imposing_social_cost',
        },
        substrate='territory__open',
        roles=('default',) * 9,
        is_focal=(True,) + (False,) * 8,
        bots_by_role={'default': {'noop_bot'}},
    ),
    territory__rooms_0=ScenarioConfig(
        description='focals are resident and visited by an aggressor',
        tags={
            'resident',
            'antisocial_newcomers',
            'defense',
        },
        substrate='territory__rooms',
        roles=('default',) * 9,
        is_focal=(True,) * 8 + (False,),
        bots_by_role={
            'default': {
                'territory__rooms__aggressor_0',
                'territory__rooms__aggressor_1',
                'territory__rooms__aggressor_2',
                'territory__rooms__aggressor_3',
                'territory__rooms__aggressor_with_extra_training_0',
            },
        },
    ),
    territory__rooms_1=ScenarioConfig(
        description='visiting a population of aggressors',
        tags={
            'visitor',
            'defense',
        },
        substrate='territory__rooms',
        roles=('default',) * 9,
        is_focal=(True,) + (False,) * 8,
        bots_by_role={
            'default': {
                'territory__rooms__aggressor_0',
                'territory__rooms__aggressor_1',
                'territory__rooms__aggressor_2',
                'territory__rooms__aggressor_3',
                'territory__rooms__aggressor_with_extra_training_0',
            },
        },
    ),
    territory__rooms_2=ScenarioConfig(
        description='focals are resident, visited by a bot that does nothing',
        tags={
            'resident',
            'restraint',
        },
        substrate='territory__rooms',
        roles=('default',) * 9,
        is_focal=(True,) * 8 + (False,),
        bots_by_role={'default': {'noop_bot'}},
    ),
    territory__rooms_3=ScenarioConfig(
        description='focals visit a resident population that does nothing',
        tags={
            'visitor',
            'chance_to_benefit_oneself_without_imposing_social_cost',
        },
        substrate='territory__rooms',
        roles=('default',) * 9,
        is_focal=(True,) + (False,) * 8,
        bots_by_role={'default': {'noop_bot'}},
    ),
    territory__rooms_4=ScenarioConfig(
        description='visiting a population of considerate claimers',
        tags={
            'visitor',
        },
        substrate='territory__rooms',
        roles=('default',) * 9,
        is_focal=(True,) + (False,) * 8,
        bots_by_role={
            'default': {'territory__rooms__puppet_considerate_claimer_0'}
        },
    ),
    territory__rooms_5=ScenarioConfig(
        description=('focals are resident, visited by a couple considerate ' +
                     'claimers'),
        tags={
            'resident',
        },
        substrate='territory__rooms',
        roles=('default',) * 9,
        is_focal=(True,) * 7 + (False,) * 2,
        bots_by_role={
            'default': {'territory__rooms__puppet_considerate_claimer_0'}
        },
    ),
    territory__rooms_6=ScenarioConfig(
        description=('focals are resident, visited by a couple considerate ' +
                     'claimers'),
        tags={
            'resident',
        },
        substrate='territory__rooms',
        roles=('default',) * 9,
        is_focal=(True,) * 7 + (False,) * 2,
        bots_by_role={
            'default': {'territory__rooms__puppet_considerate_claimer_0'}
        },
    ),
    territory__rooms_7=ScenarioConfig(
        description=('focals are resident, visited by an inconsiderate ' +
                     'claimer'),
        tags={
            'resident',
        },
        substrate='territory__rooms',
        roles=('default',) * 9,
        is_focal=(False,) + (True,) * 8,  # Reversed is_focal order for variety.
        bots_by_role={
            'default': {'territory__rooms__puppet_inconsiderate_claimer_0'}
        },
    ),
    territory__rooms_8=ScenarioConfig(
        description=('focals are resident, visited by an inconsiderate ' +
                     'claimer who becomes a zapper later in the episode'),
        tags={
            'resident',
            'flexibility',
            'defense',
        },
        substrate='territory__rooms',
        roles=('default',) * 9,
        is_focal=(True,) * 8 + (False,) * 1,
        bots_by_role={
            'default': {
                'territory__rooms__puppet_inconsiderate_claimer_to_zapper_0'
            }
        },
    ),
    territory__rooms_9=ScenarioConfig(
        description=('the environment is not as crowded as it could be, ' +
                     'focals are resident, and are visited by ' +
                     'an inconsiderate claimer who becomes a zapper later ' +
                     'in the episode'),
        tags={
            'resident',
            'flexibility',
            'defense',
        },
        substrate='territory__rooms',
        roles=('default',) * 3,
        is_focal=(True,) * 2 + (False,) * 1,
        bots_by_role={
            'default': {
                'territory__rooms__puppet_inconsiderate_claimer_to_zapper_0'
            }
        },
    ),
    territory__rooms_10=ScenarioConfig(
        description=(
            'the environment is not as crowded as it could be, focals' +
            'are resident, and are visited by an inconsiderate ' +
            'claimer who becomes a resource destroyer later in the episode'),
        tags={
            'resident',
            'flexibility',
            'defense',
        },
        substrate='territory__rooms',
        roles=('default',) * 3,
        is_focal=(True,) * 2 + (False,) * 1,
        bots_by_role={
            'default': {
                'territory__rooms__puppet_inconsiderate_claimer_to_destroyer_0'
            }
        },
    ),
    territory__rooms_11=ScenarioConfig(
        description=(
            'the environment is not as crowded as it could be, focals' +
            'visit a background population of considerate claimers'),
        tags={
            'visitor',
            'chance_to_benefit_oneself_without_imposing_social_cost',
        },
        substrate='territory__rooms',
        roles=('default',) * 3,
        is_focal=(True,) + (False,) * 2,
        bots_by_role={
            'default': {
                'territory__rooms__puppet_considerate_claimer_0'
            }
        },
    ),
    territory__rooms_12=ScenarioConfig(
        description=(
            'the environment is not as crowded as it could be, focals' +
            'are resident, and are visited by an aggressive background bot'),
        tags={
            'resident',
            'defense',
        },
        substrate='territory__rooms',
        roles=('default',) * 3,
        is_focal=(False,) + (True,) * 2,  # Reversed is_focal order for variety.
        bots_by_role={
            'default': {
                'territory__rooms__aggressor_0',
                'territory__rooms__aggressor_1',
                'territory__rooms__aggressor_2',
                'territory__rooms__aggressor_3',
                'territory__rooms__aggressor_with_extra_training_0',
            }
        },
    ),
    territory__rooms_13=ScenarioConfig(
        description=(
            'the environment is not as crowded as it could be, focals' +
            'visit an aggressive background population'),
        tags={
            'visitor',
            'defense',
        },
        substrate='territory__rooms',
        roles=('default',) * 3,
        is_focal=(False,) * 2 + (True,),  # Reversed is_focal order for variety.
        bots_by_role={
            'default': {
                'territory__rooms__aggressor_0',
                'territory__rooms__aggressor_1',
                'territory__rooms__aggressor_2',
                'territory__rooms__aggressor_3',
                'territory__rooms__aggressor_with_extra_training_0',
            }
        },
    ),
    territory__rooms_14=ScenarioConfig(
        description=(
            'the environment is not as crowded as it could be, focals' +
            'are resident, and are visited by a couple bots who do nothing'),
        tags={
            'resident',
            'defense',
        },
        substrate='territory__rooms',
        roles=('default',) * 5,
        is_focal=(False,) * 2 + (True,) * 3,  # Reversed is_focal for variety.
        bots_by_role={'default': {'noop_bot'}},
    ),
    # keep-sorted end
)
