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

local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'
local tests = {}

local components = require 'meltingpot.lua.levels.externality_mushrooms.components'

local function regrowth(minimum)
  local part = components.MushroomRegrowth{
    mushroomsToProbabilities = {eaten = {grown = 1}},
    minPotentialMushrooms = minimum or 1,
  }
  local object = {spawns = 0}
  function object:getComponent(name)
    assert(name == 'Transform')
    return {queryPosition = function() return nil end}
  end
  function object:setState(_) self.spawns = self.spawns + 1 end
  part.gameObject = {simulation = {getGameObjectFromPiece = function() return object end}}
  part:reset()
  return part, object
end

local function checkCount(part, expected)
  local actual = 0
  for _ in pairs(part._potentialMushrooms) do actual = actual + 1 end
  asserts.EQ(actual, expected)
  asserts.EQ(part._numPotentialMushrooms, actual)
end

function tests.duplicateRegistrationCountsOneSlot()
  local part = regrowth()
  for _ = 1, 4 do part:registerPotentialMushroom(7) end
  checkCount(part, 1)
end

function tests.absentAndRepeatedRemovalDoNotMakeCountsNegative()
  local part = regrowth()
  part:deregisterPotentialMushroom(9)
  checkCount(part, 0)
  part:registerPotentialMushroom(7)
  part:deregisterPotentialMushroom(7)
  part:deregisterPotentialMushroom(7)
  checkCount(part, 0)
end

function tests.duplicateRegistrationCannotBypassMinimumHabitatSize()
  local part, object = regrowth(2)
  part:registerPotentialMushroom(7)
  part:registerPotentialMushroom(7)
  part:grow('eaten')
  asserts.EQ(object.spawns, 0)
  part:registerPotentialMushroom(8)
  part:grow('eaten')
  asserts.EQ(object.spawns, 1)
  checkCount(part, 2)
end

function tests.unmatchedRemovalCannotPreventGrowthInValidHabitat()
  local part, object = regrowth(2)
  part:registerPotentialMushroom(7)
  part:registerPotentialMushroom(8)
  part:deregisterPotentialMushroom(99)
  part:grow('eaten')
  asserts.EQ(object.spawns, 1)
  checkCount(part, 2)
end

function tests.growableCallbacksKeepCountsForLiveToLiveTransitions()
  local part = regrowth()
  local state = 'wait'
  local growable = components.MushroomGrowable{}
  growable.gameObject = {
    getState = function() return state end,
    getPiece = function() return 7 end,
    simulation = {getSceneObject = function()
      return {getComponent = function() return part end}
    end},
  }
  local update
  growable:registerUpdaters({registerUpdater = function(_, spec) update = spec.updateFn end})
  growable:onStateChange('grown')
  update()
  checkCount(part, 1)
  for _, nextState in ipairs({'firstMushroom', 'secondMushroom', 'thirdMushroom'}) do
    local old = state
    state = nextState
    growable:onStateChange(old)
    update()
    checkCount(part, 0)
  end
  state = 'wait'
  growable:onStateChange('thirdMushroom')
  update()
  checkCount(part, 1)
end

function tests.independentSlotsAndResetRetainMembership()
  local part = regrowth()
  for _, id in ipairs({2, 5, 8}) do part:registerPotentialMushroom(id) end
  checkCount(part, 3)
  part:deregisterPotentialMushroom(5)
  checkCount(part, 2)
  part:registerPotentialMushroom(5)
  checkCount(part, 3)
  part:reset()
  checkCount(part, 0)
  part:registerPotentialMushroom(8)
  checkCount(part, 1)
end

return test_runner.run(tests)
