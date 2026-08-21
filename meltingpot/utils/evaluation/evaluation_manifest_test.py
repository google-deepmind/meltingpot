# Copyright 2026 DeepMind Technologies Limited.
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
"""Tests for evaluation result manifests."""

import dataclasses
from unittest import mock

from absl.testing import absltest
from meltingpot.configs import bots as bot_configs
from meltingpot.configs import scenarios as scenario_configs
from meltingpot.utils.evaluation import evaluation_manifest
import numpy as np
import pandas as pd


def _results() -> pd.DataFrame:
  return pd.DataFrame({
      'background_player_names': [
          ('background_a',),
          ('background_b',),
          ('background_c',),
      ],
      'background_player_returns': [
          (1.0,),
          (2.5,),
          (-4.0,),
      ],
      'focal_player_names': [
          ('focal_a', 'focal_b'),
          ('focal_a', 'focal_b'),
          ('focal_a', 'focal_b'),
      ],
      'focal_player_returns': [
          (3.0, -0.0),
          (np.nan, np.inf),
          (-np.inf, 7.25),
      ],
      'focal_per_capita_return': [1.5, np.nan, -np.inf],
      'background_per_capita_return': [1.0, 2.5, -4.0],
      'video_path': [
          '/tmp/run/episode_0.mp4',
          '/tmp/run/episode_1.mp4',
          '/tmp/run/episode_2.mp4',
      ],
  })


class ContentHashTest(absltest.TestCase):

  def test_is_deterministic(self):
    results = _results()
    self.assertEqual(
        evaluation_manifest.content_hash(results),
        evaluation_manifest.content_hash(results.copy(deep=True)),
    )

  def test_ignores_video_paths_and_derived_per_capita_columns(self):
    left = _results()
    right = _results()
    right['video_path'] = ['/other/a', '/other/b', '/other/c']
    right['focal_per_capita_return'] = [999.0, 999.0, 999.0]
    right['background_per_capita_return'] = [-999.0, -999.0, -999.0]

    self.assertEqual(
        evaluation_manifest.content_hash(left),
        evaluation_manifest.content_hash(right),
    )

  def test_changed_return_changes_only_its_episode_leaf(self):
    original = _results()
    changed = _results()
    changed.at[1, 'focal_player_returns'] = (10.0, 11.0)

    original_leaves = evaluation_manifest.episode_hashes(original)
    changed_leaves = evaluation_manifest.episode_hashes(changed)

    self.assertEqual(original_leaves[0], changed_leaves[0])
    self.assertNotEqual(original_leaves[1], changed_leaves[1])
    self.assertEqual(original_leaves[2], changed_leaves[2])
    self.assertNotEqual(
        evaluation_manifest.content_hash(original),
        evaluation_manifest.content_hash(changed),
    )

  def test_changed_player_name_changes_content(self):
    original = _results()
    changed = _results()
    changed.at[0, 'focal_player_names'] = ('different', 'focal_b')

    self.assertNotEqual(
        evaluation_manifest.content_hash(original),
        evaluation_manifest.content_hash(changed),
    )

  def test_episode_order_is_part_of_content(self):
    original = _results()
    reversed_results = original.iloc[::-1].reset_index(drop=True)

    self.assertNotEqual(
        evaluation_manifest.content_hash(original),
        evaluation_manifest.content_hash(reversed_results),
    )

  def test_dataframe_index_is_not_part_of_content(self):
    original = _results()
    reindexed = original.copy()
    reindexed.index = [100, 200, 300]

    self.assertEqual(
        evaluation_manifest.content_hash(original),
        evaluation_manifest.content_hash(reindexed),
    )

  def test_exact_float_representation_distinguishes_signed_zero(self):
    positive = _results().iloc[[0]].copy()
    negative = _results().iloc[[0]].copy()
    positive.at[positive.index[0], 'focal_player_returns'] = (3.0, 0.0)

    self.assertNotEqual(
        evaluation_manifest.content_hash(positive),
        evaluation_manifest.content_hash(negative),
    )

  def test_empty_results_have_deterministic_root(self):
    empty = pd.DataFrame()
    self.assertEqual(
        evaluation_manifest.content_hash(empty),
        evaluation_manifest.content_hash(empty.copy()),
    )
    self.assertEmpty(evaluation_manifest.episode_hashes(empty))

  def test_missing_columns_are_rejected_for_nonempty_results(self):
    with self.assertRaisesRegex(ValueError, 'missing columns'):
      evaluation_manifest.content_hash(pd.DataFrame({'x': [1]}))

  def test_name_return_lengths_must_match(self):
    results = _results().iloc[[0]].copy()
    results.at[results.index[0], 'focal_player_returns'] = (1.0,)

    with self.assertRaisesRegex(ValueError, 'must have equal length'):
      evaluation_manifest.content_hash(results)

  def test_returns_must_be_numeric(self):
    results = _results().iloc[[0]].copy()
    results.at[results.index[0], 'focal_player_returns'] = ('bad', 1.0)

    with self.assertRaisesRegex(ValueError, 'numeric returns'):
      evaluation_manifest.content_hash(results)


class ConfigurationHashTest(absltest.TestCase):

  def test_substrate_configuration_hash_is_deterministic(self):
    first = evaluation_manifest.configuration_hash('clean_up')
    second = evaluation_manifest.configuration_hash('clean_up')
    self.assertEqual(first, second)
    self.assertLen(first, 64)

  def test_scenario_configuration_hash_is_deterministic(self):
    first = evaluation_manifest.configuration_hash('clean_up_20')
    second = evaluation_manifest.configuration_hash('clean_up_20')
    self.assertEqual(first, second)
    self.assertLen(first, 64)

  def test_scenario_bot_pool_changes_configuration_hash(self):
    base = scenario_configs.ScenarioConfig(
        description='same description',
        tags={'same_tag'},
        substrate='clean_up',
        roles=('default', 'default'),
        is_focal=(True, False),
        bots_by_role={'default': {'bot_a'}},
    )
    changed = scenario_configs.ScenarioConfig(
        description='same description',
        tags={'same_tag'},
        substrate='clean_up',
        roles=('default', 'default'),
        is_focal=(True, False),
        bots_by_role={'default': {'bot_b'}},
    )

    with mock.patch.object(
        evaluation_manifest,
        '_substrate_signature',
        return_value={'fixed': True},
    ):
      with mock.patch.object(
          evaluation_manifest.scenario_configs,
          'SCENARIO_CONFIGS',
          {'example': base},
      ):
        first = evaluation_manifest.configuration_hash('example')
      with mock.patch.object(
          evaluation_manifest.scenario_configs,
          'SCENARIO_CONFIGS',
          {'example': changed},
      ):
        second = evaluation_manifest.configuration_hash('example')

    self.assertNotEqual(first, second)

  def test_background_bot_config_changes_configuration_hash(self):
    scenario = scenario_configs.ScenarioConfig(
        description='same',
        tags={'same'},
        substrate='clean_up',
        roles=('default', 'default'),
        is_focal=(True, False),
        bots_by_role={'default': {'bot_a'}},
    )
    base_bot = bot_configs.BotConfig(
        substrate='clean_up',
        roles={'default'},
        model_path='/models/clean_up/model_a',
        puppeteer_builder=None,
    )
    changed_bot = bot_configs.BotConfig(
        substrate='clean_up',
        roles={'default'},
        model_path='/models/clean_up/model_b',
        puppeteer_builder=None,
    )

    with mock.patch.object(
        evaluation_manifest,
        '_substrate_signature',
        return_value={'fixed': True},
    ):
      with mock.patch.object(
          evaluation_manifest.scenario_configs,
          'SCENARIO_CONFIGS',
          {'example': scenario},
      ):
        with mock.patch.object(
            evaluation_manifest.bot_configs,
            'BOT_CONFIGS',
            {'bot_a': base_bot},
        ):
          first = evaluation_manifest.configuration_hash('example')
        with mock.patch.object(
            evaluation_manifest.bot_configs,
            'BOT_CONFIGS',
            {'bot_a': changed_bot},
        ):
          second = evaluation_manifest.configuration_hash('example')

    self.assertNotEqual(first, second)

  def test_descriptive_metadata_does_not_change_configuration_hash(self):
    base = scenario_configs.ScenarioConfig(
        description='description one',
        tags={'tag_one'},
        substrate='clean_up',
        roles=('default', 'default'),
        is_focal=(True, False),
        bots_by_role={'default': {'bot_a'}},
    )
    changed = scenario_configs.ScenarioConfig(
        description='description two',
        tags={'tag_two'},
        substrate='clean_up',
        roles=('default', 'default'),
        is_focal=(True, False),
        bots_by_role={'default': {'bot_a'}},
    )

    with mock.patch.object(
        evaluation_manifest,
        '_substrate_signature',
        return_value={'fixed': True},
    ):
      with mock.patch.object(
          evaluation_manifest.scenario_configs,
          'SCENARIO_CONFIGS',
          {'example': base},
      ):
        first = evaluation_manifest.configuration_hash('example')
      with mock.patch.object(
          evaluation_manifest.scenario_configs,
          'SCENARIO_CONFIGS',
          {'example': changed},
      ):
        second = evaluation_manifest.configuration_hash('example')

    self.assertEqual(first, second)

  def test_unknown_target_is_rejected(self):
    with self.assertRaisesRegex(ValueError, 'Unknown substrate or scenario'):
      evaluation_manifest.configuration_hash('does_not_exist')


class ManifestTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.results = _results()
    self.manifest = evaluation_manifest.create_manifest(
        self.results, 'clean_up_20'
    )

  def test_contains_episode_and_configuration_fingerprints(self):
    self.assertEqual(self.manifest.schema_version, 1)
    self.assertEqual(self.manifest.hash_algorithm, 'sha256')
    self.assertEqual(self.manifest.target, 'clean_up_20')
    self.assertEqual(self.manifest.target_kind, 'scenario')
    self.assertEqual(self.manifest.num_episodes, 3)
    self.assertLen(self.manifest.configuration_sha256, 64)
    self.assertLen(self.manifest.content_sha256, 64)
    self.assertLen(self.manifest.episode_sha256, 3)
    self.assertIn(('python', mock.ANY), self.manifest.runtime_versions)

  def test_json_round_trip(self):
    decoded = evaluation_manifest.EvaluationManifest.from_json(
        self.manifest.to_json()
    )
    self.assertEqual(decoded, self.manifest)

  def test_file_round_trip(self):
    path = self.create_tempfile().full_path
    self.manifest.write(path)
    decoded = evaluation_manifest.EvaluationManifest.read(path)
    self.assertEqual(decoded, self.manifest)

  def test_verify_accepts_matching_results(self):
    evaluation_manifest.verify_manifest(self.manifest, self.results)

  def test_verify_reports_first_changed_episode(self):
    changed = _results()
    changed.at[1, 'background_player_returns'] = (123.0,)

    with self.assertRaisesRegex(
        evaluation_manifest.EvaluationManifestMismatch,
        r'Episode 1 content hash mismatch',
    ):
      evaluation_manifest.verify_manifest(self.manifest, changed)

  def test_verify_reports_episode_count_change(self):
    changed = self.results.iloc[:2]
    with self.assertRaisesRegex(
        evaluation_manifest.EvaluationManifestMismatch,
        'Episode count mismatch',
    ):
      evaluation_manifest.verify_manifest(self.manifest, changed)

  def test_configuration_check_can_be_disabled(self):
    changed_manifest = dataclasses.replace(
        self.manifest, configuration_sha256='0' * 64
    )

    with self.assertRaisesRegex(
        evaluation_manifest.EvaluationManifestMismatch,
        'Configuration hash mismatch',
    ):
      evaluation_manifest.verify_manifest(changed_manifest, self.results)

    evaluation_manifest.verify_manifest(
        changed_manifest, self.results, check_configuration=False
    )

  def test_runtime_check_is_opt_in(self):
    changed_manifest = dataclasses.replace(
        self.manifest, runtime_versions=(('python', '0.0.0'),)
    )

    evaluation_manifest.verify_manifest(changed_manifest, self.results)

    with self.assertRaisesRegex(
        evaluation_manifest.EvaluationManifestMismatch,
        'Runtime versions',
    ):
      evaluation_manifest.verify_manifest(
          changed_manifest, self.results, check_runtime=True
      )

  def test_from_dict_rejects_invalid_digest(self):
    value = self.manifest.to_dict()
    value['content_sha256'] = 'not-a-digest'
    with self.assertRaisesRegex(ValueError, 'SHA-256'):
      evaluation_manifest.EvaluationManifest.from_dict(value)

  def test_from_dict_rejects_unknown_schema(self):
    value = self.manifest.to_dict()
    value['schema_version'] = 999
    with self.assertRaisesRegex(ValueError, 'Unsupported evaluation manifest'):
      evaluation_manifest.EvaluationManifest.from_dict(value)

  def test_from_dict_rejects_wrong_episode_digest_count(self):
    value = self.manifest.to_dict()
    value['episode_sha256'] = value['episode_sha256'][:-1]
    with self.assertRaisesRegex(ValueError, 'does not match num_episodes'):
      evaluation_manifest.EvaluationManifest.from_dict(value)


if __name__ == '__main__':
  absltest.main()
