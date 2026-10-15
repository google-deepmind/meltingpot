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
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'
local tests = {}

local function inventory(capacity)
  return components.Inventory{capacityPerType = capacity, numTokenTypes = 3}
end

function tests.returnValueCountsOnlyNewTokens()
  local inv = inventory(10)
  asserts.EQ(inv:addTokens(1, 3), 3)
  asserts.EQ(inv:addTokens(1, 2), 2)
  asserts.EQ(inv.inventory(1):val(), 5)
end

function tests.saturatedInventoryReportsZeroAdded()
  local inv = inventory(5)
  inv:addTokens(2, 5)
  asserts.EQ(inv:addTokens(2, 2), 0)
  asserts.EQ(inv.inventory(2):val(), 5)
end

function tests.partialCapacityReportsAcceptedPortion()
  local inv = inventory(5)
  inv:addTokens(3, 4)
  asserts.EQ(inv:addTokens(3, 3), 1)
  asserts.EQ(inv.inventory(3):val(), 5)
end

function tests.zeroRequestDoesNotCountExistingTokens()
  local inv = inventory(5)
  inv:addTokens(1, 2)
  asserts.EQ(inv:addTokens(1, 0), 0)
  asserts.EQ(inv.inventory(1):val(), 2)
end

function tests.smallInventoriesMatchConservationOracle()
  for capacity = 1, 5 do
    for initial = 0, capacity do
      for amount = 0, 7 do
        for tokenType = 1, 3 do
          local inv = inventory(capacity)
          inv.inventory(tokenType):val(initial)
          local accepted = inv:addTokens(tokenType, amount)
          local expected = math.min(amount, capacity - initial)
          asserts.EQ(accepted, expected)
          asserts.EQ(inv.inventory(tokenType):val(), initial + expected)
          for other = 1, 3 do
            if other ~= tokenType then
              asserts.EQ(inv.inventory(other):val(), 0)
            end
          end
        end
      end
    end
  end
end

function tests.removeAndResetBehaviorRemainUnchanged()
  local inv = inventory(5)
  inv:addTokens(2, 4)
  asserts.EQ(inv:removeTokens(2, 2), 2)
  asserts.EQ(inv:removeTokens(2, 7), 2)
  asserts.EQ(inv:getHighestTypeAvailable(), 0)
  inv:addTokens(3, 4)
  inv:reset()
  asserts.EQ(inv:getLowestTypeAvailable(), 0)
  asserts.EQ(inv:addTokens(1, 3), 3)
end

local function player(index, capacity)
  local inv = inventory(capacity)
  local tracker = components.TokenTracker{numPlayers = 2, numTokenTypes = 3}
  tracker:reset()
  local beam = components.GiftBeam{
      cooldownTime = 2, beamLength = 3, beamRadius = 0,
      agentRole = 'default', giftMultiplier = 3, successfulGiftReward = 1,
      roleRewardForGifting = {default = 0},
  }
  local reward = 0
  local avatar = {getIndex = function() return index end,
                 addReward = function(_, amount) reward = reward + amount end}
  local members = {Inventory = inv, TokenTracker = tracker,
                   GiftBeam = beam, Avatar = avatar}
  local object = {getComponent = function(_, name) return members[name] end}
  for _, part in pairs(members) do part.gameObject = object end
  inv:start()
  return {object = object, inventory = inv, tracker = tracker, beam = beam,
          reward = function() return reward end}
end

for _, initial in ipairs({0, 2, 4, 5}) do
  tests['giftingTracksActualTransfer_' .. initial] = function()
    local sender, receiver = player(1, 5), player(2, 5)
    sender.inventory:addTokens(1, 2)
    receiver.inventory:addTokens(2, initial)
    local expected = math.min(3, 5 - initial)
    asserts.EQ(receiver.beam:onHit(sender.object, 'gift'), true)
    asserts.EQ(receiver.inventory.inventory(2):val(), initial + expected)
    asserts.EQ(sender.inventory.inventory(1):val(), 1)
    asserts.EQ(receiver.tracker.giftsReceived(1, 1):val(), expected)
    asserts.EQ(receiver.tracker.giftsReceivedFromAny, expected)
    asserts.EQ(sender.tracker.giftsGiven(2, 1):val(), expected)
    asserts.EQ(sender.tracker.giftsGivenToAny, expected)
    asserts.EQ(sender.reward(), 0)
    asserts.EQ(receiver.reward(), 0)
  end
end

function tests.repeatedGiftsDoNotRecountPriorTransfers()
  local sender, receiver = player(1, 5), player(2, 5)
  sender.inventory:addTokens(1, 3)
  receiver.inventory:addTokens(2, 2)
  for _ = 1, 3 do receiver.beam:onHit(sender.object, 'gift') end
  asserts.EQ(receiver.inventory.inventory(2):val(), 5)
  asserts.EQ(receiver.tracker.giftsReceivedFromAny, 3)
  asserts.EQ(sender.tracker.giftsGivenToAny, 3)
  asserts.EQ(sender.inventory:getHighestTypeAvailable(), 0)
end

function tests.mostRefinedGiftStillTransfersOneToken()
  local sender, receiver = player(1, 5), player(2, 5)
  sender.inventory:addTokens(3, 1)
  receiver.inventory:addTokens(3, 3)
  receiver.beam:onHit(sender.object, 'gift')
  asserts.EQ(receiver.inventory.inventory(3):val(), 4)
  asserts.EQ(receiver.tracker.giftsReceived(1, 3):val(), 1)
  asserts.EQ(sender.tracker.giftsGivenToAny, 1)
end

return test_runner.run(tests)
