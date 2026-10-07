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
"""Collective rewards must satisfy their advertised float64 scalar spec."""

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from meltingpot.utils.substrates.wrappers import collective_reward_wrapper
import numpy as np


class RewardEnvironment(dm_env.Environment):
  """A deterministic two-player environment with standard restart semantics."""

  def __init__(self):
    self.frame = np.arange(6, dtype=np.uint8).reshape(1, 2, 3)
    self.observations = ({'RGB': self.frame}, {'RGB': self.frame})
    self.closed = False
    self.last_timestep = None
    self.actions = []
    self.reset_args = None

  def reset(self, *args, **kwargs):
    self.reset_args = (args, kwargs)
    self.last_timestep = dm_env.restart(self.observations)
    return self.last_timestep

  def step(self, actions):
    self.actions.append(actions)
    if len(self.actions) == 1:
      self.last_timestep = dm_env.transition([2, -1], self.observations)
    else:
      self.last_timestep = dm_env.termination(
          np.array([0.5, 1.25], dtype=np.float32), self.observations
      )
    return self.last_timestep

  def observation_spec(self):
    return [
        {'RGB': dm_env.specs.Array(self.frame.shape, self.frame.dtype)}
        for _ in range(2)
    ]

  def action_spec(self):
    return [dm_env.specs.DiscreteArray(2) for _ in range(2)]

  def close(self):
    self.closed = True


class CollectiveRewardDtypeTest(parameterized.TestCase):

  @parameterized.product(
      dtype=(np.int8, np.int32, np.int64, np.float16, np.float32, np.float64),
      step_type=(
          dm_env.StepType.FIRST,
          dm_env.StepType.MID,
          dm_env.StepType.LAST,
      ),
  )
  def test_collective_reward_matches_spec_for_numeric_reward_arrays(
      self, dtype, step_type
  ):
    env = RewardEnvironment()
    wrapped = collective_reward_wrapper.CollectiveRewardWrapper(env)
    rewards = np.asarray([3, -2], dtype=dtype)
    before = rewards.copy()
    source = dm_env.TimeStep(step_type, rewards, 0.5, env.observations)
    actual = wrapped._get_timestep(source)
    self.assertIs(actual.reward, rewards)
    self.assertIs(actual.step_type, step_type)
    self.assertEqual(actual.discount, 0.5)
    for observation, spec in zip(
        actual.observation, wrapped.observation_spec()
    ):
      value = observation['COLLECTIVE_REWARD']
      self.assertEqual(value.dtype, np.dtype(np.float64))
      self.assertEqual(value.shape, ())
      self.assertEqual(float(value), 1.0)
      spec['COLLECTIVE_REWARD'].validate(value)
      self.assertIs(observation['RGB'], env.frame)
    np.testing.assert_array_equal(rewards, before)
    self.assertTrue(all(set(obs) == {'RGB'} for obs in source.observation))

  @parameterized.parameters(
      {'rewards': [3, -2]},
      {'rewards': (3, -2)},
      {'rewards': [1.25, -0.5]},
      {'rewards': ()},
  )
  def test_python_reward_sequences_remain_supported(self, rewards):
    env = RewardEnvironment()
    wrapped = collective_reward_wrapper.CollectiveRewardWrapper(env)
    source = dm_env.transition(rewards, env.observations)
    actual = wrapped._get_timestep(source)
    for obs in actual.observation:
      self.assertEqual(obs['COLLECTIVE_REWARD'], sum(rewards))
      dm_env.specs.Array((), np.float64).validate(obs['COLLECTIVE_REWARD'])
    self.assertIs(actual.reward, rewards)

  def test_standard_restart_contributes_zero_without_changing_reward_none(self):
    env = RewardEnvironment()
    wrapped = collective_reward_wrapper.CollectiveRewardWrapper(env)
    actual = wrapped.reset('seed', option=7)
    self.assertEqual(env.reset_args, (('seed',), {'option': 7}))
    self.assertIsNone(actual.reward)
    self.assertIsNone(actual.discount)
    self.assertTrue(actual.first())
    for obs, spec in zip(actual.observation, wrapped.observation_spec()):
      self.assertEqual(obs['COLLECTIVE_REWARD'], 0.0)
      spec['COLLECTIVE_REWARD'].validate(obs['COLLECTIVE_REWARD'])

  @parameterized.parameters(
      (np.array([2**62, 2**62], dtype=np.int64), float(2**63)),
      (
          np.array([-(2**62), -(2**62), -(2**62)], dtype=np.int64),
          float(-3 * 2**62),
      ),
      (np.array([65504.0, 65504.0], dtype=np.float16), 131008.0),
  )
  def test_sum_does_not_overflow_narrow_or_integer_accumulators(
      self, rewards, expected
  ):
    env = RewardEnvironment()
    wrapped = collective_reward_wrapper.CollectiveRewardWrapper(env)
    result = wrapped._get_timestep(dm_env.transition(rewards, env.observations))
    for obs in result.observation:
      self.assertEqual(obs['COLLECTIVE_REWARD'], expected)
      self.assertTrue(np.isfinite(obs['COLLECTIVE_REWARD']))

  def test_wrapper_lifecycle_returns_spec_valid_observations_on_every_step(
      self,
  ):
    env = RewardEnvironment()
    with collective_reward_wrapper.CollectiveRewardWrapper(env) as wrapped:
      timesteps = [wrapped.reset(), wrapped.step([0, 1]), wrapped.step([1, 0])]
      self.assertEqual(env.actions, [[0, 1], [1, 0]])
      self.assertTrue(timesteps[-1].last())
      for timestep, expected in zip(timesteps, (0.0, 1.0, 1.75)):
        for obs, specs in zip(timestep.observation, wrapped.observation_spec()):
          for name, spec in specs.items():
            spec.validate(obs[name])
          self.assertEqual(obs['COLLECTIVE_REWARD'], expected)
      assert env.last_timestep is not None
      self.assertIs(timesteps[-1].reward, env.last_timestep.reward)
      self.assertFalse(env.closed)
    self.assertTrue(env.closed)

  def test_existing_float64_rewards_keep_the_same_sum(self):
    rng = np.random.default_rng(42)
    env = RewardEnvironment()
    wrapped = collective_reward_wrapper.CollectiveRewardWrapper(env)
    for count in (1, 2, 8, 32):
      rewards = rng.normal(size=count)
      timestep = dm_env.transition(rewards, env.observations)
      result = wrapped._get_timestep(timestep)
      for obs in result.observation:
        np.testing.assert_array_equal(obs['COLLECTIVE_REWARD'], np.sum(rewards))


if __name__ == '__main__':
  absltest.main()
