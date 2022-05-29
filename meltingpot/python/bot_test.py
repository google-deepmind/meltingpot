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
"""Tests of bots."""

import collections
import functools
import threading

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
import numpy as np
import tree

from meltingpot.python import bot as bot_factory
from meltingpot.python import substrate as substrate_factory
from meltingpot.python.utils.scenarios import substrate_transforms


_STEP_TYPE_SPEC = dm_env.specs.BoundedArray(
    shape=(),
    dtype=np.int64,
    minimum=min(dm_env.StepType),
    maximum=max(dm_env.StepType),
)


@functools.lru_cache(maxsize=None)
def _calculate_specs(substrate):
  config = substrate_factory.get_config(substrate)
  environment = substrate_factory.build(config)
  environment = substrate_transforms.with_tf1_bot_required_observations(
      environment)
  timestep_spec = dm_env.TimeStep(
      step_type=_STEP_TYPE_SPEC,
      reward=environment.reward_spec()[0],
      discount=environment.discount_spec(),
      observation=environment.observation_spec()[0])
  action_spec = environment.action_spec()[0]
  return timestep_spec, action_spec


_lock = threading.Lock()
_key_locks = collections.defaultdict(threading.Lock)


def _get_specs(substrate):
  with _lock:
    key_lock = _key_locks[substrate]
  with key_lock:
    return _calculate_specs(substrate)


class BotTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (bot, bot) for bot in bot_factory.AVAILABLE_BOTS
  )
  def test_step_without_error(self, bot_name):
    bot_config = bot_factory.get_config(bot_name)
    timestep_spec, action_spec = _get_specs(bot_config.substrate)
    with bot_factory.build(bot_config) as policy:
      self.assert_compatible(policy, timestep_spec, action_spec)

  def assert_compatible(self, policy, timestep_spec, action_spec):
    timestep = tree.map_structure(
        lambda spec: spec.generate_value(), timestep_spec)
    prev_state = policy.initial_state()
    action, _ = policy.step(timestep, prev_state)
    try:
      action_spec.validate(action)
    except ValueError:
      self.fail(f'Returned action {action!r} does not match action_spec '
                f'{action_spec!r}.')


if __name__ == '__main__':
  absltest.main()
