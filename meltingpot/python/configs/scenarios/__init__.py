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
"""Test scenario configs."""

import collections
import dataclasses
from typing import AbstractSet, Collection, Mapping, Sequence

import immutabledict


@dataclasses.dataclass(frozen=True)
class Scenario:
  description: str
  tags: AbstractSet[str]
  substrate: str
  is_focal: Sequence[bool]
  bots: AbstractSet[str]


SCENARIOS: Mapping[str, Scenario] = immutabledict.immutabledict(
    # keep-sorted start numeric=yes block=yes
    allelopathic_harvest_0=Scenario(
        description='focals are resident and a visitor prefers green',
        tags=frozenset({
            'resident',
        }),
        substrate='allelopathic_harvest',
        is_focal=(True,) * 15 + (False,) * 1,
        bots=frozenset({
            'ah3gs_bot_finding_berry_two_the_most_tasty_0',
            'ah3gs_bot_finding_berry_two_the_most_tasty_1',
            'ah3gs_bot_finding_berry_two_the_most_tasty_4',
            'ah3gs_bot_finding_berry_two_the_most_tasty_5',
        }),
    ),
    allelopathic_harvest_1=Scenario(
        description='visiting a green preferring population',
        tags=frozenset({
            'convention_following',
            'visitor',
        }),
        substrate='allelopathic_harvest',
        is_focal=(True,) * 4 + (False,) * 12,
        bots=frozenset({
            'ah3gs_bot_finding_berry_two_the_most_tasty_0',
            'ah3gs_bot_finding_berry_two_the_most_tasty_1',
            'ah3gs_bot_finding_berry_two_the_most_tasty_4',
            'ah3gs_bot_finding_berry_two_the_most_tasty_5',
        }),
    ),
    arena_running_with_scissors_in_the_matrix_0=Scenario(
        description='versus gullible bots',
        tags=frozenset({
            'deception',
            'half_and_half',
            'versus_free',
        }),
        substrate='arena_running_with_scissors_in_the_matrix',
        is_focal=(True,) * 4 + (False,) * 4,
        bots=frozenset({
            'arena_rws_free_0',
            'arena_rws_free_1',
            'arena_rws_free_2',
        }),
    ),
    arena_running_with_scissors_in_the_matrix_1=Scenario(
        description='versus mixture of pure bots',
        tags=frozenset({
            'half_and_half',
            'versus_pure_all',
        }),
        substrate='arena_running_with_scissors_in_the_matrix',
        is_focal=(True,) * 4 + (False,) * 4,
        bots=frozenset({
            'arena_rws_pure_paper_0',
            'arena_rws_pure_paper_1',
            'arena_rws_pure_paper_2',
            'arena_rws_pure_paper_3',
            'arena_rws_pure_rock_0',
            'arena_rws_pure_rock_1',
            'arena_rws_pure_rock_2',
            'arena_rws_pure_rock_3',
            'arena_rws_pure_scissors_0',
            'arena_rws_pure_scissors_1',
            'arena_rws_pure_scissors_2',
            'arena_rws_pure_scissors_3',
        }),
    ),
    arena_running_with_scissors_in_the_matrix_2=Scenario(
        description='versus pure rock bots',
        tags=frozenset({
            'half_and_half',
            'versus_pure_rock',
        }),
        substrate='arena_running_with_scissors_in_the_matrix',
        is_focal=(True,) * 4 + (False,) * 4,
        bots=frozenset({
            'arena_rws_pure_rock_0',
            'arena_rws_pure_rock_1',
            'arena_rws_pure_rock_2',
            'arena_rws_pure_rock_3',
        }),
    ),
    arena_running_with_scissors_in_the_matrix_3=Scenario(
        description='versus pure paper bots',
        tags=frozenset({
            'half_and_half',
            'versus_pure_paper',
        }),
        substrate='arena_running_with_scissors_in_the_matrix',
        is_focal=(True,) * 4 + (False,) * 4,
        bots=frozenset({
            'arena_rws_pure_paper_0',
            'arena_rws_pure_paper_1',
            'arena_rws_pure_paper_2',
            'arena_rws_pure_paper_3',
        }),
    ),
    arena_running_with_scissors_in_the_matrix_4=Scenario(
        description='versus pure scissors bots',
        tags=frozenset({
            'half_and_half',
            'versus_pure_scissors',
        }),
        substrate='arena_running_with_scissors_in_the_matrix',
        is_focal=(True,) * 4 + (False,) * 4,
        bots=frozenset({
            'arena_rws_pure_scissors_0',
            'arena_rws_pure_scissors_1',
            'arena_rws_pure_scissors_2',
            'arena_rws_pure_scissors_3',
        }),
    ),
    bach_or_stravinsky_in_the_matrix_0=Scenario(
        description='visiting pure bach fans',
        tags=frozenset({
            'convention_following',
            'versus_pure_bach',
            'visitor',
        }),
        substrate='bach_or_stravinsky_in_the_matrix',
        is_focal=(True,) * 1 + (False,) * 7,
        bots=frozenset({
            'bach_fan_0',
            'bach_fan_1',
            'bach_fan_2',
        }),
    ),
    bach_or_stravinsky_in_the_matrix_1=Scenario(
        description='visiting pure stravinsky fans',
        tags=frozenset({
            'convention_following',
            'versus_pure_stravinsky',
            'visitor',
        }),
        substrate='bach_or_stravinsky_in_the_matrix',
        is_focal=(True,) * 1 + (False,) * 7,
        bots=frozenset({
            'stravinsky_fan_0',
            'stravinsky_fan_1',
            'stravinsky_fan_2',
        }),
    ),
    capture_the_flag_0=Scenario(
        description='focal team versus shaped a3c bot team',
        tags=frozenset({
            'half_and_half',
            'learned_teamwork',
        }),
        substrate='capture_the_flag',
        is_focal=(True, False) * 4,
        bots=frozenset({
            'ctf_pseudorewards_for_main_game_events_a3c_2',
            'ctf_pseudorewards_for_main_game_events_a3c_6',
        }),
    ),
    capture_the_flag_1=Scenario(
        description='focal team versus shaped vmpo bot team',
        tags=frozenset({
            'half_and_half',
            'learned_teamwork',
        }),
        substrate='capture_the_flag',
        is_focal=(True, False,) * 4,
        bots=frozenset({
            'ctf_pseudorewards_for_main_game_events_vmpo_0',
            'ctf_pseudorewards_for_main_game_events_vmpo_3',
            'ctf_pseudorewards_for_main_game_events_vmpo_4',
            'ctf_pseudorewards_for_main_game_events_vmpo_6',
            'ctf_pseudorewards_for_main_game_events_vmpo_7',
        }),
    ),
    capture_the_flag_2=Scenario(
        description='ad hoc teamwork with shaped a3c bots',
        tags=frozenset({
            'ad_hoc_teamwork',
            'visitor',
        }),
        substrate='capture_the_flag',
        is_focal=(True,) * 1 + (False,) * 7,
        bots=frozenset({
            'ctf_pseudorewards_for_main_game_events_a3c_2',
            'ctf_pseudorewards_for_main_game_events_a3c_6',
        }),
    ),
    capture_the_flag_3=Scenario(
        description='ad hoc teamwork with shaped vmpo bots',
        tags=frozenset({
            'ad_hoc_teamwork',
            'visitor',
        }),
        substrate='capture_the_flag',
        is_focal=(True,) * 1 + (False,) * 7,
        bots=frozenset({
            'ctf_pseudorewards_for_main_game_events_vmpo_0',
            'ctf_pseudorewards_for_main_game_events_vmpo_3',
            'ctf_pseudorewards_for_main_game_events_vmpo_4',
            'ctf_pseudorewards_for_main_game_events_vmpo_6',
            'ctf_pseudorewards_for_main_game_events_vmpo_7',
        }),
    ),
    chemistry_branched_chain_reaction_0=Scenario(
        description='focals meet X preferring bots',
        tags=frozenset({
            'half_and_half',
        }),
        substrate='chemistry_branched_chain_reaction',
        is_focal=(True,) * 4 + (False,) * 4,
        bots=frozenset({
            'chemistry_branched_chain_reaction_X_specialist_0',
            'chemistry_branched_chain_reaction_X_specialist_1',
            'chemistry_branched_chain_reaction_X_specialist_2',
        }),
    ),
    chemistry_branched_chain_reaction_1=Scenario(
        description='focals meet Y preferring bots',
        tags=frozenset({
            'half_and_half',
        }),
        substrate='chemistry_branched_chain_reaction',
        is_focal=(True,) * 4 + (False,) * 4,
        bots=frozenset({
            'chemistry_branched_chain_reaction_Y_specialist_0',
            'chemistry_branched_chain_reaction_Y_specialist_1',
            'chemistry_branched_chain_reaction_Y_specialist_2',
        }),
    ),
    chemistry_branched_chain_reaction_2=Scenario(
        description='focals are resident',
        tags=frozenset({
            'resident',
        }),
        substrate='chemistry_branched_chain_reaction',
        is_focal=(True,) * 7 + (False,) * 1,
        bots=frozenset({
            'chemistry_branched_chain_reaction_X_specialist_0',
            'chemistry_branched_chain_reaction_X_specialist_1',
            'chemistry_branched_chain_reaction_X_specialist_2',
            'chemistry_branched_chain_reaction_Y_specialist_0',
            'chemistry_branched_chain_reaction_Y_specialist_1',
            'chemistry_branched_chain_reaction_Y_specialist_2',
        }),
    ),
    chemistry_branched_chain_reaction_3=Scenario(
        description='visiting another population',
        tags=frozenset({
            'convention_following',
            'visitor',
        }),
        substrate='chemistry_branched_chain_reaction',
        is_focal=(True,) * 1 + (False,) * 7,
        bots=frozenset({
            'chemistry_branched_chain_reaction_X_specialist_0',
            'chemistry_branched_chain_reaction_X_specialist_1',
            'chemistry_branched_chain_reaction_X_specialist_2',
            'chemistry_branched_chain_reaction_Y_specialist_0',
            'chemistry_branched_chain_reaction_Y_specialist_1',
            'chemistry_branched_chain_reaction_Y_specialist_2',
        }),
    ),
    chemistry_metabolic_cycles_0=Scenario(
        description='focals meet food1 preferring bots',
        tags=frozenset({
            'half_and_half',
        }),
        substrate='chemistry_metabolic_cycles',
        is_focal=(True,) * 4 + (False,) * 4,
        bots=frozenset({
            'chemistry_metabolic_cycles_food1_specialist_0',
            'chemistry_metabolic_cycles_food1_specialist_1',
        }),
    ),
    chemistry_metabolic_cycles_1=Scenario(
        description='focals meet food2 preferring bots',
        tags=frozenset({
            'half_and_half',
        }),
        substrate='chemistry_metabolic_cycles',
        is_focal=(True,) * 4 + (False,) * 4,
        bots=frozenset({
            'chemistry_metabolic_cycles_food2_specialist_0',
            'chemistry_metabolic_cycles_food2_specialist_1',
        }),
    ),
    chemistry_metabolic_cycles_2=Scenario(
        description='focals are resident',
        tags=frozenset({
            'resident',
        }),
        substrate='chemistry_metabolic_cycles',
        is_focal=(True,) * 7 + (False,) * 1,
        bots=frozenset({
            'chemistry_metabolic_cycles_food1_specialist_0',
            'chemistry_metabolic_cycles_food1_specialist_1',
            'chemistry_metabolic_cycles_food2_specialist_0',
            'chemistry_metabolic_cycles_food2_specialist_1',
        }),
    ),
    chemistry_metabolic_cycles_3=Scenario(
        description='visiting another population',
        tags=frozenset({
            'visitor',
        }),
        substrate='chemistry_metabolic_cycles',
        is_focal=(True,) * 1 + (False,) * 7,
        bots=frozenset({
            'chemistry_metabolic_cycles_food1_specialist_0',
            'chemistry_metabolic_cycles_food1_specialist_1',
            'chemistry_metabolic_cycles_food2_specialist_0',
            'chemistry_metabolic_cycles_food2_specialist_1',
        }),
    ),
    chicken_in_the_matrix_0=Scenario(
        description='meeting a mixture of pure bots',
        tags=frozenset({
            'half_and_half',
            'versus_pure_all',
        }),
        substrate='chicken_in_the_matrix',
        is_focal=(True,) * 4 + (False,) * 4,
        bots=frozenset({
            'chicken_pure_dove_0',
            'chicken_pure_dove_1',
            'chicken_pure_dove_2',
            'chicken_pure_dove_3',
            'chicken_pure_hawk_0',
            'chicken_pure_hawk_1',
            'chicken_pure_hawk_2',
            'chicken_pure_hawk_3',
        }),
    ),
    chicken_in_the_matrix_1=Scenario(
        description='visiting a pure dove population',
        tags=frozenset({
            'versus_pure_dove',
            'visitor',
        }),
        substrate='chicken_in_the_matrix',
        is_focal=(True,) * 1 + (False,) * 7,
        bots=frozenset({
            'chicken_pure_dove_0',
            'chicken_pure_dove_1',
            'chicken_pure_dove_2',
            'chicken_pure_dove_3',
        }),
    ),
    chicken_in_the_matrix_2=Scenario(
        description='focals are resident and visitors are hawks',
        tags=frozenset({
            'resident',
            'versus_pure_hawk',
        }),
        substrate='chicken_in_the_matrix',
        is_focal=(True,) * 5 + (False,) * 3,
        bots=frozenset({
            'chicken_pure_hawk_0',
            'chicken_pure_hawk_1',
            'chicken_pure_hawk_2',
            'chicken_pure_hawk_3',
        }),
    ),
    chicken_in_the_matrix_3=Scenario(
        description='visiting a gullible population',
        tags=frozenset({
            'deception',
            'versus_free',
            'visitor',
        }),
        substrate='chicken_in_the_matrix',
        is_focal=(True,) * 1 + (False,) * 7,
        bots=frozenset({
            'chicken_free_0',
            'chicken_free_1',
            'chicken_free_2',
            'chicken_free_3',
        }),
    ),
    chicken_in_the_matrix_4=Scenario(
        description='visiting grim reciprocators',
        tags=frozenset({
            'reciprocity',
            'versus_puppet',
            'visitor',
        }),
        substrate='chicken_in_the_matrix',
        is_focal=(True,) * 2 + (False,) * 6,
        bots=frozenset({
            'chicken_puppet_grim',
        }),
    ),
    clean_up_0=Scenario(
        description='visiting an altruistic population',
        tags=frozenset({
            'versus_cleaners',
            'visitor',
        }),
        substrate='clean_up',
        is_focal=(True,) * 3 + (False,) * 4,
        bots=frozenset({
            'cleanup_cleaner_1',
            'cleanup_cleaner_2',
        }),
    ),
    clean_up_1=Scenario(
        description='focals are resident and visitors free ride',
        tags=frozenset({
            'resident',
            'versus_consumers',
        }),
        substrate='clean_up',
        is_focal=(True,) * 4 + (False,) * 3,
        bots=frozenset({
            'cleanup_consumer_0',
            'cleanup_consumer_1',
            'cleanup_consumer_2',
        }),
    ),
    clean_up_2=Scenario(
        description='visiting a turn-taking population that cleans first',
        tags=frozenset({
            'versus_puppet',
            'visitor',
        }),
        substrate='clean_up',
        is_focal=(True,) * 3 + (False,) * 4,
        bots=frozenset({
            'cleanup_puppet_alternate_clean_first',
        }),
    ),
    clean_up_3=Scenario(
        description='visiting a turn-taking population that eats first',
        tags=frozenset({
            'versus_puppet',
            'visitor',
        }),
        substrate='clean_up',
        is_focal=(True,) * 3 + (False,) * 4,
        bots=frozenset({
            'cleanup_puppet_alternate_eat_first',
        }),
    ),
    clean_up_4=Scenario(
        description='focals are visited by one reciprocator',
        tags=frozenset({
            'resident',
            'versus_puppet',
        }),
        substrate='clean_up',
        is_focal=(True,) * 6 + (False,) * 1,
        bots=frozenset({
            'cleanup_puppet_reciprocator_threshold_low',
        }),
    ),
    clean_up_5=Scenario(
        description='focals are visited by two suspicious reciprocators',
        tags=frozenset({
            'resident',
            'versus_puppet',
        }),
        substrate='clean_up',
        is_focal=(True,) * 5 + (False,) * 2,
        bots=frozenset({
            'cleanup_puppet_reciprocator_threshold_mid',
        }),
    ),
    clean_up_6=Scenario(
        description='focals are visited by one suspicious reciprocator',
        tags=frozenset({
            'resident',
            'versus_puppet',
        }),
        substrate='clean_up',
        is_focal=(True,) * 6 + (False,) * 1,
        bots=frozenset({
            'cleanup_puppet_reciprocator_threshold_mid',
        }),
    ),
    collaborative_cooking_impassable_0=Scenario(
        description='visiting a vmpo population',
        tags=frozenset({
            'convention_following',
            'visitor',
        }),
        substrate='collaborative_cooking_impassable',
        is_focal=(True,) * 1 + (False,) * 3,
        bots=frozenset({
            'collaborative_cooking_impassable_vmpo_pop_size_ten_0',
            'collaborative_cooking_impassable_vmpo_pop_size_ten_2',
            'collaborative_cooking_impassable_vmpo_pop_size_ten_3',
            'collaborative_cooking_impassable_vmpo_pop_size_ten_4',
            'collaborative_cooking_impassable_vmpo_pop_size_ten_6',
            'collaborative_cooking_impassable_vmpo_pop_size_ten_7',
            'collaborative_cooking_impassable_vmpo_pop_size_ten_9',
        }),
    ),
    collaborative_cooking_impassable_1=Scenario(
        description='focals are resident',
        tags=frozenset({
            'resident',
        }),
        substrate='collaborative_cooking_impassable',
        is_focal=(True,) * 3 + (False,) * 1,
        bots=frozenset({
            'collaborative_cooking_impassable_vmpo_pop_size_ten_0',
            'collaborative_cooking_impassable_vmpo_pop_size_ten_2',
            'collaborative_cooking_impassable_vmpo_pop_size_ten_3',
            'collaborative_cooking_impassable_vmpo_pop_size_ten_4',
            'collaborative_cooking_impassable_vmpo_pop_size_ten_6',
            'collaborative_cooking_impassable_vmpo_pop_size_ten_7',
            'collaborative_cooking_impassable_vmpo_pop_size_ten_9',
        }),
    ),
    collaborative_cooking_passable_0=Scenario(
        description='visiting uncoordinated generalists',
        tags=frozenset({
            'convention_following',
            'versus_uncoordinated_generalist',
            'visitor',
        }),
        substrate='collaborative_cooking_passable',
        is_focal=(True,) * 1 + (False,) * 3,
        bots=frozenset({
            'collaborative_cooking_passable_vmpo_pop_size_ten_5',
        }),
    ),
    collaborative_cooking_passable_1=Scenario(
        description='focals are resident and visited by an uncoordinated generalist',
        tags=frozenset({
            'resident',
            'versus_uncoordinated_generalist',
        }),
        substrate='collaborative_cooking_passable',
        is_focal=(True,) * 3 + (False,) * 1,
        bots=frozenset({
            'collaborative_cooking_passable_vmpo_pop_size_ten_5',
        }),
    ),
    commons_harvest_closed_0=Scenario(
        description='focals are resident and visited by two zappers',
        tags=frozenset({
            'resident',
        }),
        substrate='commons_harvest_closed',
        is_focal=(True,) * 14 + (False,) * 2,
        bots=frozenset({
            'closed_commons_zapper_0',
            'closed_commons_zapper_1',
            'closed_commons_zapper_2',
            'closed_commons_zapper_3',
        }),
    ),
    commons_harvest_closed_1=Scenario(
        description='focals are resident and visited by six zappers',
        tags=frozenset({
            'resident',
        }),
        substrate='commons_harvest_closed',
        is_focal=(True,) * 10 + (False,) * 6,
        bots=frozenset({
            'closed_commons_zapper_0',
            'closed_commons_zapper_1',
            'closed_commons_zapper_2',
            'closed_commons_zapper_3',
        }),
    ),
    commons_harvest_closed_2=Scenario(
        description='visiting a population of zappers',
        tags=frozenset({
            'visitor',
        }),
        substrate='commons_harvest_closed',
        is_focal=(True,) * 4 + (False,) * 12,
        bots=frozenset({
            'closed_commons_zapper_0',
            'closed_commons_zapper_1',
            'closed_commons_zapper_2',
            'closed_commons_zapper_3',
        }),
    ),
    commons_harvest_open_0=Scenario(
        description='focals are resident and visited by two zappers',
        tags=frozenset({
            'resident',
        }),
        substrate='commons_harvest_open',
        is_focal=(True,) * 14 + (False,) * 2,
        bots=frozenset({
            'open_commons_zapper_0',
            'open_commons_zapper_1',
        }),
    ),
    commons_harvest_open_1=Scenario(
        description='focals are resident and visited by six zappers',
        tags=frozenset({
            'resident',
        }),
        substrate='commons_harvest_open',
        is_focal=(True,) * 10 + (False,) * 6,
        bots=frozenset({
            'open_commons_zapper_0',
            'open_commons_zapper_1',
        }),
    ),
    commons_harvest_partnership_0=Scenario(
        description='meeting good partners',
        tags=frozenset({
            'half_and_half',
            'versus_good_partners',
        }),
        substrate='commons_harvest_partnership',
        is_focal=(True,) * 8 + (False,) * 8,
        bots=frozenset({
            'partnership_commons_putative_good_partner_4',
            'partnership_commons_putative_good_partner_5',
            'partnership_commons_putative_good_partner_7',
        }),
    ),
    commons_harvest_partnership_1=Scenario(
        description='focals are resident and visitors are good partners',
        tags=frozenset({
            'resident',
            'versus_good_partners',
        }),
        substrate='commons_harvest_partnership',
        is_focal=(True,) * 12 + (False,) * 4,
        bots=frozenset({
            'partnership_commons_putative_good_partner_4',
            'partnership_commons_putative_good_partner_5',
            'partnership_commons_putative_good_partner_7',
        }),
    ),
    commons_harvest_partnership_2=Scenario(
        description='visiting a population of good partners',
        tags=frozenset({
            'versus_good_partners',
            'visitor',
        }),
        substrate='commons_harvest_partnership',
        is_focal=(True,) * 4 + (False,) * 12,
        bots=frozenset({
            'partnership_commons_putative_good_partner_4',
            'partnership_commons_putative_good_partner_5',
            'partnership_commons_putative_good_partner_7',
        }),
    ),
    commons_harvest_partnership_3=Scenario(
        description='focals are resident and visited by two zappers',
        tags=frozenset({
            'resident',
            'versus_zappers',
        }),
        substrate='commons_harvest_partnership',
        is_focal=(True,) * 14 + (False,) * 2,
        bots=frozenset({
            'partnership_commons_zapper_1',
            'partnership_commons_zapper_2',
        }),
    ),
    commons_harvest_partnership_4=Scenario(
        description='focals are resident and visited by six zappers',
        tags=frozenset({
            'resident',
            'versus_zappers',
        }),
        substrate='commons_harvest_partnership',
        is_focal=(True,) * 10 + (False,) * 6,
        bots=frozenset({
            'partnership_commons_zapper_1',
            'partnership_commons_zapper_2',
        }),
    ),
    commons_harvest_partnership_5=Scenario(
        description='visiting a population of zappers',
        tags=frozenset({
            'versus_zappers',
            'visitor',
        }),
        substrate='commons_harvest_partnership',
        is_focal=(True,) * 4 + (False,) * 12,
        bots=frozenset({
            'partnership_commons_zapper_1',
            'partnership_commons_zapper_2',
        }),
    ),
    king_of_the_hill_0=Scenario(
        description='focal team versus default vmpo bot team',
        tags=frozenset({
            'half_and_half',
            'learned_teamwork',
        }),
        substrate='king_of_the_hill',
        is_focal=(True, False) * 4,
        bots=frozenset({
            'koth_default_vmpo_0',
            'koth_default_vmpo_1',
            'koth_default_vmpo_2',
            'koth_default_vmpo_3',
            'koth_default_vmpo_4',
            'koth_default_vmpo_5',
            'koth_default_vmpo_6',
            'koth_default_vmpo_7',
        }),
    ),
    king_of_the_hill_1=Scenario(
        description='focal team versus shaped a3c bot team',
        tags=frozenset({
            'half_and_half',
            'learned_teamwork',
        }),
        substrate='king_of_the_hill',
        is_focal=(True, False) * 4,
        bots=frozenset({
            'koth_zap_while_in_control_a3c_0',
            'koth_zap_while_in_control_a3c_1',
            'koth_zap_while_in_control_a3c_2',
            'koth_zap_while_in_control_a3c_3',
            'koth_zap_while_in_control_a3c_4',
            'koth_zap_while_in_control_a3c_5',
            'koth_zap_while_in_control_a3c_6',
            'koth_zap_while_in_control_a3c_7',
        }),
    ),
    king_of_the_hill_2=Scenario(
        description='focal team versus shaped vmpo bot team',
        tags=frozenset({
            'half_and_half',
            'learned_teamwork',
        }),
        substrate='king_of_the_hill',
        is_focal=(True, False) * 4,
        bots=frozenset({
            'koth_zap_while_in_control_vmpo_0',
            'koth_zap_while_in_control_vmpo_1',
            'koth_zap_while_in_control_vmpo_2',
            'koth_zap_while_in_control_vmpo_3',
            'koth_zap_while_in_control_vmpo_4',
            'koth_zap_while_in_control_vmpo_5',
            'koth_zap_while_in_control_vmpo_6',
            'koth_zap_while_in_control_vmpo_7',
        }),
    ),
    king_of_the_hill_3=Scenario(
        description='ad hoc teamwork with default vmpo bots',
        tags=frozenset({
            'ad_hoc_teamwork',
            'visitor',
        }),
        substrate='king_of_the_hill',
        is_focal=(True,) * 1 + (False,) * 7,
        bots=frozenset({
            'koth_default_vmpo_0',
            'koth_default_vmpo_1',
            'koth_default_vmpo_2',
            'koth_default_vmpo_3',
            'koth_default_vmpo_4',
            'koth_default_vmpo_5',
            'koth_default_vmpo_6',
            'koth_default_vmpo_7',
        }),
    ),
    king_of_the_hill_4=Scenario(
        description='ad hoc teamwork with shaped a3c bots',
        tags=frozenset({
            'ad_hoc_teamwork',
            'visitor',
        }),
        substrate='king_of_the_hill',
        is_focal=(True,) * 1 + (False,) * 7,
        bots=frozenset({
            'koth_zap_while_in_control_a3c_0',
            'koth_zap_while_in_control_a3c_1',
            'koth_zap_while_in_control_a3c_2',
            'koth_zap_while_in_control_a3c_3',
            'koth_zap_while_in_control_a3c_4',
            'koth_zap_while_in_control_a3c_5',
            'koth_zap_while_in_control_a3c_6',
            'koth_zap_while_in_control_a3c_7',
        }),
    ),
    king_of_the_hill_5=Scenario(
        description='ad hoc teamwork with shaped vmpo bots',
        tags=frozenset({
            'ad_hoc_teamwork',
            'visitor',
        }),
        substrate='king_of_the_hill',
        is_focal=(True,) * 1 + (False,) * 7,
        bots=frozenset({
            'koth_zap_while_in_control_vmpo_0',
            'koth_zap_while_in_control_vmpo_1',
            'koth_zap_while_in_control_vmpo_2',
            'koth_zap_while_in_control_vmpo_3',
            'koth_zap_while_in_control_vmpo_4',
            'koth_zap_while_in_control_vmpo_5',
            'koth_zap_while_in_control_vmpo_6',
            'koth_zap_while_in_control_vmpo_7',
        }),
    ),
    prisoners_dilemma_in_the_matrix_0=Scenario(
        description='visiting unconditional cooperators',
        tags=frozenset({
            'versus_pure_cooperators',
            'visitor',
        }),
        substrate='prisoners_dilemma_in_the_matrix',
        is_focal=(True,) * 1 + (False,) * 7,
        bots=frozenset({
            'prisoners_dilemma_cooperator_2',
            'prisoners_dilemma_cooperator_4',
        }),
    ),
    prisoners_dilemma_in_the_matrix_1=Scenario(
        description='focals are resident and visitors are unconditional cooperators',
        tags=frozenset({
            'resident',
            'versus_pure_cooperators',
        }),
        substrate='prisoners_dilemma_in_the_matrix',
        is_focal=(True,) * 6 + (False,) * 2,
        bots=frozenset({
            'prisoners_dilemma_cooperator_2',
            'prisoners_dilemma_cooperator_4',
        }),
    ),
    prisoners_dilemma_in_the_matrix_2=Scenario(
        description='focals are resident and visitors defect',
        tags=frozenset({
            'resident',
            'versus_pure_defectors',
        }),
        substrate='prisoners_dilemma_in_the_matrix',
        is_focal=(True,) * 6 + (False,) * 2,
        bots=frozenset({
            'prisoners_dilemma_defector_0',
            'prisoners_dilemma_defector_2',
        }),
    ),
    prisoners_dilemma_in_the_matrix_3=Scenario(
        description='meeting gullible bots',
        tags=frozenset({
            'deception',
            'half_and_half',
            'versus_free',
        }),
        substrate='prisoners_dilemma_in_the_matrix',
        is_focal=(True,) * 4 + (False,) * 4,
        bots=frozenset({
            'prisoners_dilemma_free_0',
            'prisoners_dilemma_free_1',
            'prisoners_dilemma_free_2',
        }),
    ),
    prisoners_dilemma_in_the_matrix_4=Scenario(
        description='visiting a population of grim reciprocators',
        tags=frozenset({
            'reciprocity',
            'versus_puppet',
            'visitor',
        }),
        substrate='prisoners_dilemma_in_the_matrix',
        is_focal=(True,) * 1 + (False,) * 7,
        bots=frozenset({
            'prisoners_dilemma_puppet_grim_threshold_high',
        }),
    ),
    prisoners_dilemma_in_the_matrix_5=Scenario(
        description='visiting a population of hair-trigger grim reciprocators',
        tags=frozenset({
            'reciprocity',
            'versus_puppet',
            'visitor',
        }),
        substrate='prisoners_dilemma_in_the_matrix',
        is_focal=(True,) * 1 + (False,) * 7,
        bots=frozenset({
            'prisoners_dilemma_puppet_grim_threshold_low',
        }),
    ),
    pure_coordination_in_the_matrix_0=Scenario(
        description='focals are resident and visitor is mixed',
        tags=frozenset({
            'resident',
            'versus_pure_all',
        }),
        substrate='pure_coordination_in_the_matrix',
        is_focal=(True,) * 7 + (False,) * 1,
        bots=frozenset({
            'pure_coordination_type_1_specialist_0',
            'pure_coordination_type_1_specialist_1',
            'pure_coordination_type_2_specialist_0',
            'pure_coordination_type_2_specialist_1',
            'pure_coordination_type_3_specialist_0',
            'pure_coordination_type_3_specialist_1',
        }),
    ),
    pure_coordination_in_the_matrix_1=Scenario(
        description='visiting resource 1 fans',
        tags=frozenset({
            'versus_pure_type_1',
            'visitor',
        }),
        substrate='pure_coordination_in_the_matrix',
        is_focal=(True,) * 1 + (False,) * 7,
        bots=frozenset({
            'pure_coordination_type_1_specialist_0',
            'pure_coordination_type_1_specialist_1',
        }),
    ),
    pure_coordination_in_the_matrix_2=Scenario(
        description='visiting resource 2 fans',
        tags=frozenset({
            'versus_pure_type_2',
            'visitor',
        }),
        substrate='pure_coordination_in_the_matrix',
        is_focal=(True,) * 1 + (False,) * 7,
        bots=frozenset({
            'pure_coordination_type_2_specialist_0',
            'pure_coordination_type_2_specialist_1',
        }),
    ),
    pure_coordination_in_the_matrix_3=Scenario(
        description='visiting resource 3 fans',
        tags=frozenset({
            'versus_pure_type_3',
            'visitor',
        }),
        substrate='pure_coordination_in_the_matrix',
        is_focal=(True,) * 1 + (False,) * 7,
        bots=frozenset({
            'pure_coordination_type_3_specialist_0',
            'pure_coordination_type_3_specialist_1',
        }),
    ),
    pure_coordination_in_the_matrix_4=Scenario(
        description='meeting uncoordinated strangers',
        tags=frozenset({
            'half_and_half',
            'versus_pure_all',
        }),
        substrate='pure_coordination_in_the_matrix',
        is_focal=(True,) * 4 + (False,) * 4,
        bots=frozenset({
            'pure_coordination_type_1_specialist_0',
            'pure_coordination_type_1_specialist_1',
            'pure_coordination_type_2_specialist_0',
            'pure_coordination_type_2_specialist_1',
            'pure_coordination_type_3_specialist_0',
            'pure_coordination_type_3_specialist_1',
        }),
    ),
    rationalizable_coordination_in_the_matrix_0=Scenario(
        description='focals are resident and visitor is mixed',
        tags=frozenset({
            'resident',
            'versus_pure_all',
        }),
        substrate='rationalizable_coordination_in_the_matrix',
        is_focal=(True,) * 7 + (False,) * 1,
        bots=frozenset({
            'rationalizable_coordination_type_1_specialist_0',
            'rationalizable_coordination_type_1_specialist_1',
            'rationalizable_coordination_type_2_specialist_0',
            'rationalizable_coordination_type_2_specialist_1',
            'rationalizable_coordination_type_3_specialist_0',
            'rationalizable_coordination_type_3_specialist_1',
        }),
    ),
    rationalizable_coordination_in_the_matrix_1=Scenario(
        description='visiting resource 1 fans',
        tags=frozenset({
            'versus_pure_type_1',
            'visitor',
        }),
        substrate='rationalizable_coordination_in_the_matrix',
        is_focal=(True,) * 1 + (False,) * 7,
        bots=frozenset({
            'rationalizable_coordination_type_1_specialist_0',
            'rationalizable_coordination_type_1_specialist_1',
        }),
    ),
    rationalizable_coordination_in_the_matrix_2=Scenario(
        description='visiting resource 2 fans',
        tags=frozenset({
            'versus_pure_type_2',
            'visitor',
        }),
        substrate='rationalizable_coordination_in_the_matrix',
        is_focal=(True,) * 1 + (False,) * 7,
        bots=frozenset({
            'rationalizable_coordination_type_2_specialist_0',
            'rationalizable_coordination_type_2_specialist_1',
        }),
    ),
    rationalizable_coordination_in_the_matrix_3=Scenario(
        description='visiting resource 3 fans',
        tags=frozenset({
            'versus_pure_type_3',
            'visitor',
        }),
        substrate='rationalizable_coordination_in_the_matrix',
        is_focal=(True,) * 1 + (False,) * 7,
        bots=frozenset({
            'rationalizable_coordination_type_3_specialist_0',
            'rationalizable_coordination_type_3_specialist_1',
        }),
    ),
    rationalizable_coordination_in_the_matrix_4=Scenario(
        description='meeting uncoordinated strangers',
        tags=frozenset({
            'half_and_half',
            'versus_pure_all',
        }),
        substrate='rationalizable_coordination_in_the_matrix',
        is_focal=(True,) * 4 + (False,) * 4,
        bots=frozenset({
            'rationalizable_coordination_type_1_specialist_0',
            'rationalizable_coordination_type_1_specialist_1',
            'rationalizable_coordination_type_2_specialist_0',
            'rationalizable_coordination_type_2_specialist_1',
            'rationalizable_coordination_type_3_specialist_0',
            'rationalizable_coordination_type_3_specialist_1',
        }),
    ),
    running_with_scissors_in_the_matrix_0=Scenario(
        description='versus gullible opponent',
        tags=frozenset({
            'deception',
            'half_and_half',
            'versus_free',
        }),
        substrate='running_with_scissors_in_the_matrix',
        is_focal=(True,) * 1 + (False,) * 1,
        bots=frozenset({
            'classic_rws_free_0',
            'classic_rws_free_1',
            'classic_rws_free_2',
        }),
    ),
    running_with_scissors_in_the_matrix_1=Scenario(
        description='versus mixed strategy opponent',
        tags=frozenset({
            'half_and_half',
            'versus_pure_all',
        }),
        substrate='running_with_scissors_in_the_matrix',
        is_focal=(True,) * 1 + (False,) * 1,
        bots=frozenset({
            'classic_rws_pure_paper_0',
            'classic_rws_pure_paper_1',
            'classic_rws_pure_paper_2',
            'classic_rws_pure_paper_3',
            'classic_rws_pure_rock_0',
            'classic_rws_pure_rock_1',
            'classic_rws_pure_rock_2',
            'classic_rws_pure_rock_3',
            'classic_rws_pure_scissors_0',
            'classic_rws_pure_scissors_1',
            'classic_rws_pure_scissors_2',
            'classic_rws_pure_scissors_3',
        }),
    ),
    running_with_scissors_in_the_matrix_2=Scenario(
        description='versus pure rock opponent',
        tags=frozenset({
            'half_and_half',
            'versus_pure_rock',
        }),
        substrate='running_with_scissors_in_the_matrix',
        is_focal=(True,) * 1 + (False,) * 1,
        bots=frozenset({
            'classic_rws_pure_rock_0',
            'classic_rws_pure_rock_1',
            'classic_rws_pure_rock_2',
            'classic_rws_pure_rock_3',
        }),
    ),
    running_with_scissors_in_the_matrix_3=Scenario(
        description='versus pure paper opponent',
        tags=frozenset({
            'half_and_half',
            'versus_pure_paper',
        }),
        substrate='running_with_scissors_in_the_matrix',
        is_focal=(True,) * 1 + (False,) * 1,
        bots=frozenset({
            'classic_rws_pure_paper_0',
            'classic_rws_pure_paper_1',
            'classic_rws_pure_paper_2',
            'classic_rws_pure_paper_3',
        }),
    ),
    running_with_scissors_in_the_matrix_4=Scenario(
        description='versus pure scissors opponent',
        tags=frozenset({
            'half_and_half',
            'versus_pure_scissors',
        }),
        substrate='running_with_scissors_in_the_matrix',
        is_focal=(True,) * 1 + (False,) * 1,
        bots=frozenset({
            'classic_rws_pure_scissors_0',
            'classic_rws_pure_scissors_1',
            'classic_rws_pure_scissors_2',
            'classic_rws_pure_scissors_3',
        }),
    ),
    stag_hunt_in_the_matrix_0=Scenario(
        description='visiting a population of stags',
        tags=frozenset({
            'versus_pure_stag',
            'visitor',
        }),
        substrate='stag_hunt_in_the_matrix',
        is_focal=(True,) * 1 + (False,) * 7,
        bots=frozenset({
            'stag_hunt_stag_specialist_3',
            'stag_hunt_stag_specialist_5',
        }),
    ),
    stag_hunt_in_the_matrix_1=Scenario(
        description='visiting a population of hares',
        tags=frozenset({
            'versus_pure_hare',
            'visitor',
        }),
        substrate='stag_hunt_in_the_matrix',
        is_focal=(True,) * 1 + (False,) * 7,
        bots=frozenset({
            'stag_hunt_hare_specialist_0',
            'stag_hunt_hare_specialist_1',
            'stag_hunt_hare_specialist_2',
        }),
    ),
    stag_hunt_in_the_matrix_2=Scenario(
        description='visiting a population of grim reciprocators',
        tags=frozenset({
            'reciprocity',
            'versus_puppet',
            'visitor',
        }),
        substrate='stag_hunt_in_the_matrix',
        is_focal=(True,) * 2 + (False,) * 6,
        bots=frozenset({
            'stag_hunt_puppet_grim',
        }),
    ),
    territory_open_0=Scenario(
        description='focals are resident and visited by a shaped bot',
        tags=frozenset({
            'resident',
        }),
        substrate='territory_open',
        is_focal=(True,) * 8 + (False,) * 1,
        bots=frozenset({
            'territory_open_painter_0',
            'territory_open_painter_1',
            'territory_open_painter_2',
            'territory_open_painter_3',
        }),
    ),
    territory_open_1=Scenario(
        description='visiting a population of shaped bots',
        tags=frozenset({
            'convention_following',
            'visitor',
        }),
        substrate='territory_open',
        is_focal=(True,) * 1 + (False,) * 8,
        bots=frozenset({
            'territory_open_painter_0',
            'territory_open_painter_1',
            'territory_open_painter_2',
            'territory_open_painter_3',
        }),
    ),
    territory_rooms_0=Scenario(
        description='focals are resident and visited by an aggressor',
        tags=frozenset({
            'resident',
        }),
        substrate='territory_rooms',
        is_focal=(True,) * 8 + (False,) * 1,
        bots=frozenset({
            'territory_closed_reply_to_zapper_0',
            'territory_closed_reply_to_zapper_1',
        }),
    ),
    territory_rooms_1=Scenario(
        description='visiting a population of aggressors',
        tags=frozenset({
            'convention_following',
            'visitor',
        }),
        substrate='territory_rooms',
        is_focal=(True,) * 1 + (False,) * 8,
        bots=frozenset({
            'territory_closed_reply_to_zapper_0',
            'territory_closed_reply_to_zapper_1',
        }),
    ),
    # keep-sorted end
)


def scenarios_by_substrate(
    scenarios: Mapping[str, Scenario]
) -> Mapping[str, Collection[str]]:
  by_substrate = collections.defaultdict(list)
  for scenario_name, scenario in scenarios.items():
    by_substrate[scenario.substrate].append(scenario_name)
  for key, value in by_substrate.items():
    by_substrate[key] = tuple(value)
  return immutabledict.immutabledict(**by_substrate)
