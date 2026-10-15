--[[ Copyright 2022 DeepMind Technologies Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
]]

local components = require 'meltingpot.lua.levels.hidden_agenda.components'
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'
local tests = {}

local function role(name)
  return components.Role{role = name, frozenState = 'frozen'}
end

for _, name in ipairs({'crewmate', 'impostor'}) do
  tests['gettersStartAtZero_' .. name] = function()
    local instance = role(name)
    asserts.EQ(instance:gemsCollectedReward(), 0)
    asserts.EQ(instance:gemsDepositedReward(), 0)
  end
  tests['independentSignedAccumulators_' .. name] = function()
    local instance = role(name)
    for _, amount in ipairs({0.5, 1.25, -0.25}) do
      instance:rewardForGemsCollected(amount)
    end
    for _, amount in ipairs({2, -3.5, 0}) do
      instance:rewardForGemsDeposited(amount)
    end
    asserts.EQ(instance:gemsCollectedReward(), 1.5)
    asserts.EQ(instance:gemsDepositedReward(), -1.5)
    local other = role(name)
    asserts.EQ(other:gemsCollectedReward(), 0)
    asserts.EQ(other:gemsDepositedReward(), 0)
    asserts.EQ(instance:gemsCollectedReward(), 1.5)
  end
end

local function settlement(potential, rewards, withShaping)
  local progress = components.Progress{
      num_players = 3, potential_pseudorewards = potential,
      teleport_spawn_group = 'spawn',
  }
  local ended = 0
  progress.gameObject = {simulation = {endEpisode = function()
    ended = ended + 1
  end}}
  local results = {}
  for index, name in ipairs({'crewmate', 'impostor', 'crewmate'}) do
    local instance = role(name)
    local state = {reward = 0, frozen = false}
    local avatar = {
        addReward = function(_, amount) state.reward = state.reward + amount end,
        disallowMovement = function() state.frozen = true end,
    }
    local object = {
        hasComponent = function() return false end,
        getComponent = function(_, key)
          if key == 'Role' then return instance end
          if key == 'Avatar' then return avatar end
          error('Unexpected component: ' .. key)
        end,
    }
    instance.gameObject = object
    local collected = withShaping and index * 0.5 or 0
    local deposited = withShaping and (index == 2 and -0.25 or 1.25) or 0
    instance:rewardForGemsCollected(collected)
    instance:rewardForGemsDeposited(deposited)
    avatar:addReward(collected + deposited)
    progress.avatars[index] = object
    results[index] = {state = state, groupReward = rewards[name],
                      shaping = collected + deposited}
  end
  progress:gameEnd(rewards.impostor, rewards.crewmate)
  asserts.EQ(ended, 1)
  for _, result in ipairs(results) do
    local expected = result.groupReward
    if not potential then expected = expected + result.shaping end
    asserts.EQ(result.state.reward, expected)
    asserts.EQ(result.state.frozen, true)
  end
end

for _, potential in ipairs({false, true}) do
  for _, shaping in ipairs({false, true}) do
    for _, winningSide in ipairs({'crewmate', 'impostor', 'draw'}) do
      local label = tostring(potential) .. '_' .. tostring(shaping) .. '_' ..
          winningSide
      tests['terminalSettlement_' .. label] = function()
        local rewards = {crewmate = 0, impostor = 0}
        if winningSide == 'crewmate' then
          rewards = {crewmate = 4, impostor = -4}
        elseif winningSide == 'impostor' then
          rewards = {crewmate = -4, impostor = 4}
        end
        settlement(potential, rewards, shaping)
      end
    end
  end
end

return test_runner.run(tests)
