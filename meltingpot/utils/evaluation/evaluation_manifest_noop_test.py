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
"""Coverage for manifesting scenarios that use the built-in noop bot."""

from absl.testing import absltest
from meltingpot import bot
from meltingpot.utils.evaluation import evaluation_manifest


class NoopBotManifestTest(absltest.TestCase):

  def test_noop_bot_has_explicit_signature(self):
    self.assertEqual(
        evaluation_manifest._bot_signature(bot.NOOP_BOT_NAME),
        {
            'name': 'noop_bot',
            'kind': 'fixed_action',
            'action': bot.NOOP_ACTION,
        },
    )

  def test_noop_scenario_can_be_fingerprinted(self):
    digest = evaluation_manifest.configuration_hash(
        'collaborative_cooking__asymmetric_0'
    )
    self.assertLen(digest, 64)


if __name__ == '__main__':
  absltest.main()
