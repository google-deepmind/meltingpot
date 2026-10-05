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
local core = require 'meltingpot.lua.modules.component_library'
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'
local tests = {}

local function fixture()
  local receiver = components.Receiver{}
  local stateManager = core.StateManager{
      initialState = 'hopper_mouth_open',
      stateConfigs = {{state = 'hopper_mouth_open'},
                      {state = 'hopper_mouth_closing'},
                      {state = 'hopper_mouth_closed'}},
  }
  local states = {}
  local grid = {setState = function(_, _, state)
    states[#states + 1] = state
  end}
  local object = {
      _id = 1,
      started = function() return false end,
      getPiece = function() return 1 end,
      setState = function(_, state) stateManager:setState(grid, state) end,
      hasComponent = function(_, name) return name == 'Receiver' end,
  }
  local hopper = components.HopperMouth{
      closed = 'hopper_mouth_closed', opening = 'hopper_mouth_closing',
      open = 'hopper_mouth_open',
  }
  hopper:reset()
  hopper:setIsOpen(true)
  object.getComponent = function(_, name)
    if name == 'Receiver' then return receiver end
    if name == 'HopperMouth' then return hopper end
    if name == 'StateManager' then return stateManager end
    if name == 'Transform' then
      return {queryDisc = function() return {{}} end}
    end
    error('Unexpected component: ' .. name)
  end
  receiver.gameObject = object
  stateManager.gameObject = object
  hopper.gameObject = object
  stateManager:awake()
  return receiver, object, hopper, states
end

local function deliver(receiverObject, tokenType, indicatorType)
  local token = components.Token{type = tokenType}
  local receivable = components.Receivable{
      waitState = 'wait', liveState = 'live',
  }
  local indicator = {
      hasComponent = function(_, name) return name == 'ReceiverIndicator' end,
      getComponent = function()
        return {getType = function() return indicatorType end}
      end,
  }
  local transform = {queryDisc = function(_, layer)
    if layer == 'lowestPhysical' then return {receiverObject} end
    return {indicator}
  end}
  receivable.gameObject = {getComponent = function(_, name)
    if name == 'Token' then return token end
    if name == 'Transform' then return transform end
    error('Unexpected component: ' .. name)
  end}
  receivable:_setReceiver()
end

function tests.completeInputChangesOnlyItsInventoryFlag()
  local receiver, _, _, states = fixture()
  asserts.EQ(receiver:setHasNeededObjects(true), true)
  asserts.EQ(receiver:hasNeededObjects(), true)
  asserts.EQ(#states, 0)
  asserts.EQ(receiver:setHasNeededObjects(false), false)
end

function tests.firstCubeChangesOnlyItsInventoryFlag()
  local receiver, _, _, states = fixture()
  asserts.EQ(receiver:setHasOneOfTwoCubes(true), true)
  asserts.EQ(receiver:hasOneOfTwoCubes(), true)
  asserts.EQ(#states, 0)
  asserts.EQ(receiver:setHasOneOfTwoCubes(false), false)
end

function tests.twoCubeRecipeAcceptsBothInputsWithoutAnInvalidState()
  local receiver, object, _, states = fixture()
  deliver(object, 'BlueCube', 'TwoBlocks')
  asserts.EQ(receiver:hasOneOfTwoCubes(), true)
  assert(not receiver:hasNeededObjects())
  deliver(object, 'BlueCube', 'TwoBlocks')
  asserts.EQ(receiver:hasOneOfTwoCubes(), false)
  asserts.EQ(receiver:hasNeededObjects(), true)
  asserts.EQ(#states, 0)
end

function tests.matchingSingleInputStartsTheHopperAnimation()
  local receiver, object, hopper, states = fixture()
  deliver(object, 'Apple', 'Apple')
  asserts.EQ(receiver:hasNeededObjects(), true)
  asserts.EQ(#states, 0)
  hopper:update()
  asserts.EQ(hopper._counter, 17)
  for _ = 1, 4 do hopper:update() end
  asserts.EQ(states[#states], 'PTID_1_hopper_mouth_closed')
  asserts.EQ(receiver:hasNeededObjects(), false)
end

function tests.nonMatchingRecipeLeavesFlagsUnchanged()
  local receiver, object, _, states = fixture()
  deliver(object, 'Apple', 'TwoBlocks')
  assert(not receiver:hasNeededObjects())
  assert(not receiver:hasOneOfTwoCubes())
  asserts.EQ(#states, 0)
end

function tests.clearingFlagsRemainsAStateFreeOperation()
  local receiver, _, _, states = fixture()
  receiver:setHasNeededObjects(false)
  receiver:setHasOneOfTwoCubes(false)
  asserts.EQ(#states, 0)
end

return test_runner.run(tests)
