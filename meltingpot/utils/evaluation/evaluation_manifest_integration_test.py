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
"""Tests for manifest-producing evaluation convenience functions."""

from unittest import mock

from absl.testing import absltest
from meltingpot.utils.evaluation import evaluation
from meltingpot.utils.evaluation import evaluation_manifest
import pandas as pd


_RESULTS = pd.DataFrame({
    'background_player_names': [('background',)],
    'background_player_returns': [(1.0,)],
    'focal_player_names': [('focal',)],
    'focal_player_returns': [(2.0,)],
})


class EvaluationWithManifestTest(absltest.TestCase):

  def test_population_wrapper_delegates_and_fingerprints_result(self):
    manifest = mock.sentinel.manifest
    population = {'focal': mock.sentinel.policy}
    names_by_role = {'default': {'focal'}}

    with mock.patch.object(
        evaluation, 'evaluate_population', return_value=_RESULTS
    ) as evaluate_population:
      with mock.patch.object(
          evaluation_manifest, 'create_manifest', return_value=manifest
      ) as create_manifest:
        actual_results, actual_manifest = (
            evaluation_manifest.evaluate_population_with_manifest(
                population=population,
                names_by_role=names_by_role,
                target='clean_up',
                num_episodes=7,
                video_root='/tmp/videos',
            )
        )

    self.assertIs(actual_results, _RESULTS)
    self.assertIs(actual_manifest, manifest)
    evaluate_population.assert_called_once_with(
        population=population,
        names_by_role=names_by_role,
        scenario='clean_up',
        num_episodes=7,
        video_root='/tmp/videos',
    )
    create_manifest.assert_called_once_with(_RESULTS, 'clean_up')

  def test_saved_model_wrapper_delegates_and_fingerprints_result(self):
    manifest = mock.sentinel.manifest
    saved_models = {'focal': '/models/focal'}
    names_by_role = {'default': {'focal'}}

    with mock.patch.object(
        evaluation, 'evaluate_saved_models', return_value=_RESULTS
    ) as evaluate_saved_models:
      with mock.patch.object(
          evaluation_manifest, 'create_manifest', return_value=manifest
      ) as create_manifest:
        actual_results, actual_manifest = (
            evaluation_manifest.evaluate_saved_models_with_manifest(
                saved_models=saved_models,
                names_by_role=names_by_role,
                target='clean_up_20',
                num_episodes=9,
                video_root=None,
            )
        )

    self.assertIs(actual_results, _RESULTS)
    self.assertIs(actual_manifest, manifest)
    evaluate_saved_models.assert_called_once_with(
        saved_models=saved_models,
        names_by_role=names_by_role,
        scenario='clean_up_20',
        num_episodes=9,
        video_root=None,
    )
    create_manifest.assert_called_once_with(_RESULTS, 'clean_up_20')


if __name__ == '__main__':
  absltest.main()
