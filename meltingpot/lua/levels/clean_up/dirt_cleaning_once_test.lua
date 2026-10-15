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

local components = require 'meltingpot.lua.levels.clean_up.components'

local function dirt()
  local cleaning = components.DirtCleaning{}
  local object = {state = 'dirt', changes = 0}
  function object:getState() return self.state end
  function object:setState(state)
    self.pending = state
    self.changes = self.changes + 1
  end
  function object:flush()
    local old = self.state
    self.state = self.pending
    self.pending = nil
    if cleaning.onStateChange then cleaning:onStateChange(old) end
  end
  cleaning.gameObject = object
  if cleaning.reset then cleaning:reset() end
  return cleaning, object
end

local function cleaner(index, withTaste, withCleaner)
  local record = {rewardCalls = 0, cumulants = 0}
  local avatar = {getIndex = function() return index end}
  local taste = {cleaned = function() record.rewardCalls = record.rewardCalls + 1 end}
  local beam = {setCumulant = function() record.cumulants = record.cumulants + 1 end}
  local parts = {Avatar = avatar}
  if withTaste then parts.Taste = taste end
  if withCleaner then parts.Cleaner = beam end
  return {
    getComponent = function(_, name) return parts[name] end,
    hasComponent = function(_, name) return parts[name] ~= nil end,
  }, record
end

for _, order in ipairs({{1, 2}, {2, 1}}) do
  tests['oneCleanerPerDirt' .. order[1]] = function()
    local component, object = dirt()
    local a, ar = cleaner(1, true, true)
    local b, br = cleaner(2, true, true)
    local players, records = {a, b}, {ar, br}
    asserts.EQ(component:onHit(players[order[1]], 'cleanHit'), true)
    asserts.EQ(object:getState(), 'dirt')  -- Engine changes are deferred.
    asserts.EQ(not component:onHit(players[order[2]], 'cleanHit'), true)
    asserts.EQ(records[order[1]].rewardCalls, 1)
    asserts.EQ(records[order[1]].cumulants, 1)
    asserts.EQ(records[order[2]].rewardCalls, 0)
    asserts.EQ(records[order[2]].cumulants, 0)
    asserts.EQ(object.changes, 1)
  end
end

function tests.repeatedHitsByOneCleanerDoNotMultiplyRewards()
  local component, object = dirt()
  local player, record = cleaner(1, true, true)
  for _ = 1, 4 do component:onHit(player, 'cleanHit') end
  asserts.EQ(record.rewardCalls, 1)
  asserts.EQ(record.cumulants, 1)
  asserts.EQ(object.changes, 1)
end

function tests.regrowthAndEpisodeResetAllowAnotherCleaning()
  local component, object = dirt()
  local player, record = cleaner(1, true, true)
  for cycle = 1, 3 do
    asserts.EQ(component:onHit(player, 'cleanHit'), true)
    object:flush()
    asserts.EQ(not component:onHit(player, 'cleanHit'), true)
    asserts.EQ(record.rewardCalls, cycle)
    object:setState('dirt')
    object:flush()
  end
  if component.reset then component:reset() end
  component:onHit(player, 'cleanHit')
  asserts.EQ(record.rewardCalls, 4)
end

function tests.otherBeamsAndCleanTilesDoNotConsumeTheCleaning()
  local component, object = dirt()
  local player, record = cleaner(1, true, true)
  asserts.EQ(not component:onHit(player, 'zapHit'), true)
  asserts.EQ(object.changes, 0)
  asserts.EQ(component:onHit(player, 'cleanHit'), true)
  object:flush()
  component:onHit(player, 'cleanHit')
  asserts.EQ(record.rewardCalls, 1)
end

for _, flags in ipairs({{false, false}, {true, false}, {false, true}}) do
  tests['optionalComponents' .. tostring(flags[1]) .. tostring(flags[2])] = function()
    local component, object = dirt()
    local player, record = cleaner(1, flags[1], flags[2])
    component:onHit(player, 'cleanHit')
    component:onHit(player, 'cleanHit')
    asserts.EQ(record.rewardCalls, flags[1] and 1 or 0)
    asserts.EQ(record.cumulants, flags[2] and 1 or 0)
    asserts.EQ(object.changes, 1)
  end
end

function tests.independentDirtCellsCanBothBeCleaned()
  local a = dirt()
  local b = dirt()
  local player, record = cleaner(1, true, true)
  asserts.EQ(a:onHit(player, 'cleanHit'), true)
  asserts.EQ(b:onHit(player, 'cleanHit'), true)
  asserts.EQ(record.rewardCalls, 2)
end

return test_runner.run(tests)
