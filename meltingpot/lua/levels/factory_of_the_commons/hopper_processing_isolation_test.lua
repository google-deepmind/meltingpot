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

local components = require(
    'meltingpot.lua.levels.factory_of_the_commons.components')
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'
local tests = {}

local function fixture()
  local hopper = components.HopperMouth{
      framesToProcess = 17,
      closed = 'closed', opening = 'opening', open = 'open',
  }
  local needed = false
  local receiver = {
      hasNeededObjects = function() return needed end,
      setHasNeededObjects = function(_, value) needed = value end,
  }
  local current = 'open'
  local transform = {queryDisc = function() return {{}} end}
  hopper.gameObject = {
      setState = function(_, state)
        assert(state == 'closed' or state == 'opening' or state == 'open')
        current = state
      end,
      getComponent = function(_, name)
        if name == 'Receiver' then return receiver end
        if name == 'Transform' then return transform end
        error('Unexpected component ' .. name)
      end,
  }
  hopper:reset()
  hopper:update()
  return {hopper = hopper, receiver = receiver,
          state = function() return current end}
end

local function stateAt(step)
  if step >= 4 and step <= 15 then return 'closed' end
  if step == 3 or step == 16 then return 'opening' end
  return 'open'
end

function tests.singleMachineKeepsItsExistingCycle()
  local item = fixture()
  item.receiver:setHasNeededObjects(true)
  item.hopper:update()
  asserts.EQ(item.hopper._counter, 17)
  for step = 1, 17 do
    item.hopper:update()
    asserts.EQ(item.state(), stateAt(step))
  end
  asserts.EQ(item.hopper:isOpen(), true)
  asserts.EQ(item.receiver:hasNeededObjects(), false)
end

function tests.resettingAnIdleMachineDoesNotCancelItsNeighbor()
  local a, b = fixture(), fixture()
  a.hopper:processing()
  b.hopper:reset()
  for step = 1, 4 do a.hopper:update() end
  asserts.EQ(a.state(), 'closed')
  asserts.EQ(a.hopper._counter, 13)
end

function tests.startingOneMachineDoesNotConsumeAnotherInput()
  local a, b = fixture(), fixture()
  a.hopper:processing()
  b.receiver:setHasNeededObjects(true)
  b.hopper:update()
  asserts.EQ(b.hopper._counter, 17)
end

for _, reverse in ipairs({false, true}) do
  for _, lag in ipairs({0, 2, 8}) do
    tests['independentCycles_' .. tostring(reverse) .. '_' .. lag] = function()
      local a, b = fixture(), fixture()
      for step = 0, 17 + lag do
        if step == 0 then a.hopper:processing() end
        if step == lag then b.hopper:processing() end
        local function tickA() if step > 0 then a.hopper:update() end end
        local function tickB() if step > lag then b.hopper:update() end end
        if reverse then tickB(); tickA() else tickA(); tickB() end
        asserts.EQ(a.state(), stateAt(step))
        asserts.EQ(b.state(), stateAt(math.max(step - lag, 0)))
      end
      asserts.EQ(a.hopper:isOpen(), true)
      asserts.EQ(b.hopper:isOpen(), true)
    end
  end
end

function tests.idleMachinesRemainOpen()
  local a, b = fixture(), fixture()
  for _ = 1, 20 do a.hopper:update(); b.hopper:update() end
  asserts.EQ(a.state(), 'open')
  asserts.EQ(b.state(), 'open')
end

return test_runner.run(tests)
