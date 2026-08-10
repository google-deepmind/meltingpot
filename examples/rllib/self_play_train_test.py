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
"""Tests for the self_play_train.py."""

# Ray and Gymnasium are optional example dependencies and absent from core CI.
# pylint: disable=import-error
# pytype: disable=import-error

import importlib.util
import unittest

from absl.testing import absltest

def _module_available(module):
  try:
    return importlib.util.find_spec(module) is not None
  except ModuleNotFoundError:
    return False


_RLLIB_AVAILABLE = all(
    _module_available(module) for module in ('gymnasium', 'ray.rllib'))


@unittest.skipUnless(_RLLIB_AVAILABLE, 'requires the optional RLlib extras')
class TrainingTests(absltest.TestCase):
  """Tests for MeltingPotEnv for RLLib."""

  def test_training(self):
    from . import self_play_train  # pylint: disable=g-import-not-at-top

    config = self_play_train.get_config(
        num_rollout_workers=1,
        rollout_fragment_length=10,
        train_batch_size=20,
        sgd_minibatch_size=20,
        fcnet_hiddens=(4,),
        post_fcnet_hiddens=(4,),
        lstm_cell_size=2)
    results = self_play_train.train(config, num_iterations=1)
    self.assertEqual(results.num_errors, 0)
