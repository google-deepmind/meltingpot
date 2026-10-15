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

-- Reward preferences belong to the receiving avatar, not the beam target.
local components = require 'meltingpot.lua.levels.the_matrix.components'
local tensor = require 'system.tensor'
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'
local tests = {}

local function player(preferred, extra, zeroDefault)
  local avatar = {reward = 0, deliveries = 0}
  function avatar:addReward(amount)
    self.reward = self.reward + amount
    self.deliveries = self.deliveries + 1
  end
  local members = {Avatar = avatar}
  if preferred then
    local taste = components.InteractionTaste{
        mostTastyResourceClass = preferred,
        extraReward = extra,
        zeroDefaultInteractionReward = zeroDefault or false,
    }
    taste:reset()
    members.InteractionTaste = taste
  end
  avatar.gameObject = {
      hasComponent = function(_, name) return members[name] ~= nil end,
      getComponent = function(_, name) return assert(members[name]) end,
  }
  return avatar
end

local function deliver(row, column, target, rowReward, columnReward, floor)
  local zapper = components.GameInteractionZapper{
      cooldownTime = 2, beamLength = 3, beamRadius = 1,
      framesTillRespawn = 2, numResources = 2,
      rewardFloor = floor or -1e6,
  }
  zapper.gameObject = target.gameObject
  local rowInventory = tensor.DoubleTensor{4, 1}
  local columnInventory = tensor.DoubleTensor{1, 4}
  zapper:sendRewardsToBothInteractants(
      rowReward, columnReward, rowInventory, columnInventory, row, column)
  asserts.tablesEQ(rowInventory:val(), {4, 1})
  asserts.tablesEQ(columnInventory:val(), {1, 4})
end

function tests.differentPreferencesApplyToTheirOwners()
  for _, rowIsTarget in ipairs({false, true}) do
    local row = player(1, 10)
    local column = player(2, 20)
    deliver(row, column, rowIsTarget and row or column, 3, 5)
    asserts.EQ(row.reward, 13)
    asserts.EQ(column.reward, 25)
    asserts.EQ(row.deliveries, 1)
    asserts.EQ(column.deliveries, 1)
  end
end

function tests.preferenceOnOnlyOnePlayerDoesNotLeakToItsPartner()
  for _, rowIsTarget in ipairs({false, true}) do
    for _, rowHasTaste in ipairs({false, true}) do
      local row = rowHasTaste and player(1, 10) or player()
      local column = rowHasTaste and player() or player(2, 20)
      deliver(row, column, rowIsTarget and row or column, 3, 5)
      asserts.EQ(row.reward, rowHasTaste and 13 or 3)
      asserts.EQ(column.reward, rowHasTaste and 5 or 25)
    end
  end
end

function tests.zeroDefaultOnlyAffectsTheConfiguredRecipient()
  for _, rowIsTarget in ipairs({false, true}) do
    local row = player(1, 10, true)
    local column = player(2, 20, false)
    deliver(row, column, rowIsTarget and row or column, 3, 5)
    asserts.EQ(row.reward, 10)
    asserts.EQ(column.reward, 25)
  end
end

function tests.absentAndDisabledTastesKeepMatrixPayoffs()
  for _, disabled in ipairs({false, true}) do
    local row = disabled and player(-1, 10, true) or player()
    local column = disabled and player(-1, 20, true) or player()
    deliver(row, column, row, 3, 5)
    asserts.EQ(row.reward, 3)
    asserts.EQ(column.reward, 5)
  end
end

function tests.rewardFloorIsStillAppliedBeforeTasteAdjustment()
  for _, rowIsTarget in ipairs({false, true}) do
    local row = player(1, 10)
    local column = player(2, 20)
    deliver(row, column, rowIsTarget and row or column, 0, -1, 0)
    asserts.EQ(row.reward, 0)
    asserts.EQ(column.reward, 0)
    asserts.EQ(row.deliveries, 0)
    asserts.EQ(column.deliveries, 0)
  end
end

function tests.identicalPreferencesRemainCompatible()
  local row = player(1, 10)
  local column = player(1, 10)
  deliver(row, column, column, 3, 5)
  asserts.EQ(row.reward, 13)
  asserts.EQ(column.reward, 5)
end

return test_runner.run(tests)
