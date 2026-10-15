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

local components = require 'meltingpot.lua.levels.gift_refinements.components'
local read_settings = require 'common.read_settings'
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'
local tests = {}

local function fixture(roleRewards, sourceType, role)
  local players = {}
  for index = 1, 2 do
    local inventory = components.Inventory{
        capacityPerType = 10, numTokenTypes = 3,
    }
    local tracker = components.TokenTracker{
        numPlayers = 2, numTokenTypes = 3,
    }
    tracker:reset()
    local beam = components.GiftBeam{
        cooldownTime = 2, beamLength = 3, beamRadius = 0,
        agentRole = role, giftMultiplier = 3, successfulGiftReward = 10,
        roleRewardForGifting = roleRewards,
    }
    local rewards = {}
    local avatar = {
        getIndex = function() return index end,
        addReward = function(_, amount)
          assert(type(amount) == 'number', 'Reward must be numeric')
          rewards[#rewards + 1] = amount
        end,
    }
    local members = {Inventory = inventory, TokenTracker = tracker,
                     GiftBeam = beam, Avatar = avatar}
    local object = {getComponent = function(_, name) return members[name] end}
    for _, part in pairs(members) do part.gameObject = object end
    inventory:start()
    players[index] = {object = object, inventory = inventory, tracker = tracker,
                      beam = beam, rewards = rewards}
  end
  if sourceType > 0 then players[1].inventory:addTokens(sourceType, 1) end
  return players[1], players[2]
end

for _, kind in ipairs({'empty', 'sparse', 'parsed'}) do
  for _, sourceType in ipairs({0, 1, 3}) do
    tests['missingRole_' .. kind .. '_type_' .. sourceType] = function()
      local rewards = {}
      if kind == 'sparse' then rewards.known = 0.2 end
      if kind == 'parsed' then
        rewards = read_settings.any()
        rewards.known = 0.2
      end
      local source, target = fixture(rewards, sourceType, 'unlisted')
      asserts.EQ(target.beam:onHit(source.object, 'gift'), true)
      asserts.EQ(#source.rewards, 0)
      asserts.EQ(#target.rewards, 0)
      asserts.EQ(rawget(rewards, 'unlisted'), nil)
      local expected = sourceType == 0 and 0 or sourceType == 3 and 1 or 3
      asserts.EQ(target.inventory.inventory:sum(), expected)
      asserts.EQ(source.inventory.inventory:sum(), 0)
      asserts.EQ(source.tracker.giftsGivenToAny, expected)
      asserts.EQ(target.tracker.giftsReceivedFromAny, expected)
    end
  end
end

for _, amount in ipairs({0, 0.2, -2}) do
  for _, sourceType in ipairs({0, 1, 3}) do
    tests['configured_' .. amount .. '_type_' .. sourceType] = function()
      local source, target = fixture({known = amount}, sourceType, 'known')
      target.beam:onHit(source.object, 'gift')
      if sourceType == 1 then
        asserts.tablesEQ(source.rewards, {amount, amount * 10})
      else
        asserts.tablesEQ(source.rewards, {amount})
      end
      asserts.EQ(#target.rewards, 0)
    end
  end
end

function tests.unrelatedBeamLeavesInventoriesAndRewardsUnchanged()
  local source, target = fixture({}, 1, 'unlisted')
  asserts.EQ(target.beam:onHit(source.object, 'other'), nil)
  asserts.EQ(source.inventory.inventory:sum(), 1)
  asserts.EQ(target.inventory.inventory:sum(), 0)
  asserts.EQ(#source.rewards, 0)
end

return test_runner.run(tests)
