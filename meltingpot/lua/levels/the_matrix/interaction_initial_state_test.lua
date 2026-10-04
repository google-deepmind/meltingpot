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

local components = require 'meltingpot.lua.levels.the_matrix.components'
local tensor = require 'system.tensor'
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'
local tests = {}

local function fixture(count, index)
  local zapper = components.GameInteractionZapper{
      cooldownTime = 2, beamLength = 3, beamRadius = 0,
      framesTillRespawn = 2, numResources = count,
  }
  zapper.gameObject = {getComponent = function()
    return {getIndex = function() return index end}
  end}
  zapper:start()
  return zapper
end

for _, count in ipairs({2, 3}) do
  tests['initialSentinel_' .. count] = function()
    local zapper = fixture(count, 1)
    local expected = tensor.DoubleTensor(2, count):fill(-1)
    asserts.tablesEQ(zapper.latest_interaction_inventories:val(), expected:val())
    asserts.EQ(zapper.interacted_this_step, 0)
  end
  tests['restartClearsLastInteraction_' .. count] = function()
    local zapper = fixture(count, 1)
    zapper.latest_interaction_inventories:fill(5)
    zapper.interacted_this_step = 1
    zapper:start()
    local expected = tensor.DoubleTensor(2, count):fill(-1)
    asserts.tablesEQ(zapper.latest_interaction_inventories:val(), expected:val())
    asserts.EQ(zapper.interacted_this_step, 0)
  end
  tests['reportAndUpdateRetainExistingSemantics_' .. count] = function()
    local row = fixture(count, 1)
    local column = fixture(count, 2)
    local rowInventory = tensor.DoubleTensor(count):fill(2)
    local columnInventory = tensor.DoubleTensor(count):fill(3)
    row:reportInteraction(1, 2, 0, 0, rowInventory, columnInventory)
    column:reportInteraction(1, 2, 0, 0, rowInventory, columnInventory)
    asserts.tablesEQ(row.latest_interaction_inventories:val(),
                     {rowInventory:val(), columnInventory:val()})
    asserts.tablesEQ(column.latest_interaction_inventories:val(),
                     {columnInventory:val(), rowInventory:val()})
    row:update()
    column:update()
    local expected = tensor.DoubleTensor(2, count):fill(-1)
    asserts.tablesEQ(row.latest_interaction_inventories:val(), expected:val())
    asserts.tablesEQ(column.latest_interaction_inventories:val(), expected:val())
    asserts.tablesEQ(rowInventory:val(), tensor.DoubleTensor(count):fill(2):val())
  end
end

return test_runner.run(tests)
