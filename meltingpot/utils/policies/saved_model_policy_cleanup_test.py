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
"""Regression test for TF1 SavedModel initialization cleanup."""

import contextlib
from unittest import mock

from absl.testing import absltest
from meltingpot.utils.policies import saved_model_policy


class TF1SavedModelPolicyCleanupTest(absltest.TestCase):

  def test_closes_session_when_saved_model_loading_fails(self):
    graph = mock.Mock()
    session = mock.Mock()

    with mock.patch.object(
        saved_model_policy.tf.compat.v1, 'Graph', return_value=graph
    ), mock.patch.object(
        saved_model_policy.tf.compat.v1, 'Session', return_value=session
    ) as session_factory, mock.patch.object(
        saved_model_policy.TF1SavedModelPolicy,
        '_build_context',
        return_value=contextlib.nullcontext(),
    ), mock.patch.object(
        saved_model_policy.tf.compat.v1.saved_model,
        'load_v2',
        side_effect=RuntimeError('load failed'),
    ):
      with self.assertRaisesRegex(RuntimeError, 'load failed'):
        saved_model_policy.TF1SavedModelPolicy('/invalid/model')

    session_factory.assert_called_once_with(graph=graph)
    session.close.assert_called_once_with()


if __name__ == '__main__':
  absltest.main()
