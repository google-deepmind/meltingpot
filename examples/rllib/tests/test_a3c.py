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

from absl.testing import absltest
from examples.rllib import utils
from meltingpot.python import substrate
from examples.rllib import a3c


class PlayerPoliciesTests(absltest.TestCase):
  """Tests the player policies setup"""

  def setUp(self):
    super().setUp()
    # Create a new MeltingPotEnv for each test case
    env_config = substrate.get_config('commons_harvest_open')
    self._env = utils.env_creator(env_config)

  def test_agents_created(self):
    num_players = 16
    playerPolicies = a3c.IndependentAgents(
        self._env.observation_space["player_0"],
        self._env.action_space["player_0"], num_players)
    policies = playerPolicies.policies

    # Check there are the correct number of policies
    self.assertLen(policies.keys(), num_players)

  def test_policy_mapping_fn(self):
    # Create the playerPolicies
    num_players = 16
    agents = a3c.IndependentAgents(self._env.observation_space["player_0"],
                                   self._env.action_space["player_0"],
                                   num_players)

    expected_policies = list(agents.policies.keys())

    # For each player, user policy_mapping_fn to get a policy
    resulting_policies = []
    for player_number in range(num_players):
      policy_name = agents.policy_mapping_fn(f'player_{player_number}')
      resulting_policies.append(policy_name)

    # Check that all policies are used
    self.assertItemsEqual(resulting_policies, expected_policies)


if __name__ == '__main__':
  absltest.main()
