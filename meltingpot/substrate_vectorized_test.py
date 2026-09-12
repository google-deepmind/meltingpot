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
"""Tests for the public vectorized substrate builder."""

from unittest import mock

from absl.testing import absltest
from meltingpot import substrate
from meltingpot.utils.substrates import vector_env


class BuildVectorizedTest(absltest.TestCase):

  def test_delegates_without_building_environment_in_parent(self):
    factory = mock.Mock()
    factory.valid_roles.return_value = frozenset({"default", "special"})
    vectorized = mock.sentinel.vectorized

    with mock.patch.object(substrate, "get_factory", return_value=factory):
      with mock.patch.object(
          vector_env, "build_vectorized", return_value=vectorized
      ) as build_vectorized:
        actual = substrate.build_vectorized(
            "example",
            roles=("default", "special"),
            num_envs=4,
            start_method="spawn",
        )

    self.assertIs(actual, vectorized)
    build_vectorized.assert_called_once_with(
        "example",
        roles=("default", "special"),
        num_envs=4,
        start_method="spawn",
    )

  def test_rejects_nonpositive_num_envs_before_factory_lookup(self):
    with mock.patch.object(substrate, "get_factory") as get_factory:
      with self.assertRaisesRegex(ValueError, "num_envs must be positive"):
        substrate.build_vectorized("example", roles=("default",), num_envs=0)
    get_factory.assert_not_called()

  def test_rejects_unsupported_roles_before_starting_workers(self):
    factory = mock.Mock()
    factory.valid_roles.return_value = frozenset({"default"})
    with mock.patch.object(substrate, "get_factory", return_value=factory):
      with mock.patch.object(vector_env, "build_vectorized") as build_vectorized:
        with self.assertRaisesRegex(ValueError, "Invalid roles"):
          substrate.build_vectorized(
              "example", roles=("default", "unknown"), num_envs=2
          )
    build_vectorized.assert_not_called()

  def test_builds_and_steps_real_substrates_in_spawned_workers(self):
    name = "prisoners_dilemma_in_the_matrix__repeated"
    config = substrate.get_config(name)
    roles = config.default_player_roles

    with substrate.build_vectorized(
        name,
        roles=roles,
        num_envs=2,
        start_method="spawn",
    ) as env:
      reset = env.reset()
      self.assertLen(reset, 2)
      self.assertTrue(all(timestep.first() for timestep in reset))

      actions = tuple((0,) * len(roles) for _ in range(env.num_envs))
      stepped = env.step(actions)
      self.assertLen(stepped, 2)
      self.assertTrue(all(not timestep.first() for timestep in stepped))


if __name__ == "__main__":
  absltest.main()
