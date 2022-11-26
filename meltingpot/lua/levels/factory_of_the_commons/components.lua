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

local args = require 'common.args'
local class = require 'common.class'
local helpers = require 'common.helpers'
local set = require 'common.set'
local log = require 'common.log'
local events = require 'system.events'
local random = require 'system.random'
local tensor = require 'system.tensor'

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')


-- The 'Receivable' component enables an object to be placed on a receiver.
-- Sets Receiver as having needed objects.
local Receivable = class.Class(component.Component)

function Receivable:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Receivable')},
      {'waitState', args.stringType},
      {'liveState', args.stringType},
  })
  Receivable.Base.__init__(self, kwargs)
  self._config.waitState = kwargs.waitState
  self._config.liveState = kwargs.liveState
  self._isDropped = false
end

function Receivable:_checkIfOverReceiver()
  local underlyingObject = self.gameObject:getComponent('Transform'):queryDisc(
      'lowestPhysical', 0)

  local isOverReceiver = false
  for _, object in ipairs(underlyingObject) do
    if object:hasComponent('Receiver') then
      if object:getComponent('HopperMouth'):isOpen() == true then
        isOverReceiver = true
      end
    end
  end
  return isOverReceiver
end

function Receivable:_setReceiver()
  local underlyingObject = self.gameObject:getComponent('Transform'):queryDisc(
      'lowestPhysical', 0)
  local hopper = self.gameObject:getComponent('Transform'):queryDisc(
      'upperPhysical', 1)
  local isOverReceiver = false
  for _, receiver in ipairs(underlyingObject) do
    if receiver:hasComponent('Receiver') then
      if receiver:getComponent('HopperMouth'):isOpen() == true then
        for _, hopperIndicator in ipairs(hopper) do
          if hopperIndicator:hasComponent('ReceiverIndicator') then
            local oneCube = receiver:getComponent('Receiver'):hasOneOfTwoCubes()
            local tokenType = self.gameObject:getComponent('Token'):getType()
            local indicatorType = hopperIndicator:getComponent(
                'ReceiverIndicator'):getType()
            if tokenType == 'BlueCube' and
                indicatorType == 'TwoBlocks' and
                oneCube ~= true then
              receiver:getComponent('Receiver'):setHasOneOfTwoCubes(true)
            elseif tokenType == 'BlueCube' and
                indicatorType == 'TwoBlocks' and
                oneCube == true or
                tokenType == indicatorType then
              receiver:getComponent('Receiver'):setHasOneOfTwoCubes(false)
              receiver:getComponent('Receiver'):setHasNeededObjects(true)
              local test = receiver:getComponent('Receiver'):hasNeededObjects()
            end
          end
        end
      end
    end
  end
  return isOverReceiver
end

function Receivable:_isGrasped()
  return self.gameObject:getComponent('Graspable'):isGrasped()
end

function Receivable:_beReceived()
  self.gameObject:getComponent('ReceiverDropAnimation'):startFall()
end

function Receivable:isDropping()
  return self._config.isDropping
end

function Receivable:setIsDropping(x)
  if self._config.isDropping and not x then
    local avatarIndex = self.gameObject:getComponent(
        'Graspable'):getWhoIsGrasping()
    if avatarIndex then
      local avatarGrasp = self.gameObject.simulation:getAvatarFromIndex(
          avatarIndex):getComponent('AvatarGrasp')
      avatarGrasp:drop()
    end
    self.gameObject:setState(self._config.waitState)
    self._isDropped = true
  end
  self._config.isDropping = x
end

function Receivable:registerUpdaters(updaterRegistry)
  local grasp = function()
    if not self._isDropped then
      if not self._config.isDropping then
        if self:_checkIfOverReceiver() then
          if not self:_isGrasped() then
            self:_setReceiver()
            self:_beReceived()
          end
        end
      end
    end
    if self._config.liveState == self.gameObject:getState() then
      self._isDropped = false
    end
  end
  updaterRegistry:registerUpdater{
      updateFn = grasp,
      priority = self.gameObject:getComponent('Graspable'):priority() + 1,
  }
end


--[[ A receiver component which accepts items from the avatar and gives the
avatar a reward (or all avatars if globalReward=True).
]]
local Receiver = class.Class(component.Component)

function Receiver:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Receiver')},
  })
  Receiver.Base.__init__(self, kwargs)
  self._kwargs = kwargs
end

function Receiver:hasNeededObjects()
  return self._config.hasNeededObjects
end

function Receiver:setHasNeededObjects(x)
  self._config.hasNeededObjects = x
  if self._config.hasNeededObjects == true then
    self.gameObject:setState(self._config.waitState)
  end
  return self._config.hasNeededObjects
end

function Receiver:hasOneOfTwoCubes()
  return self._config.hasOneOfTwoCubes
end

function Receiver:setHasOneOfTwoCubes(x)
  self._config.hasOneOfTwoCubes = x
  if self._config.hasOneOfTwoCubes == true then
    self.gameObject:setState(self._config.waitState)
  end
  return self._config.hasOneOfTwoCubes
end

local Token = class.Class(component.Component)

function Token:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Token')},
      {'type', args.stringType},
  })
  Token.Base.__init__(self, kwargs)
  self._config.type = kwargs.type
end

function Token:getType()
  return self._config.type
end


-- Animation for dropping object into receiver hopper.
local ReceiverDropAnimation = class.Class(component.Component)

function ReceiverDropAnimation:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('ReceiverDropAnimation')},
      {'framesToDrop', args.default(2), args.positive},
      {'dropOne', args.stringType},
      {'dropTwo', args.stringType},
  })
  ReceiverDropAnimation.Base.__init__(self, kwargs)
  self._config.framesToDrop = kwargs.framesToDrop
  self._config.dropOne = kwargs.dropOne
  self._config.dropTwo = kwargs.dropTwo
end

function ReceiverDropAnimation:reset()
  self._counter = 0
end

function ReceiverDropAnimation:startFall()
  self._counter = self._config.framesToDrop
  self.gameObject:getComponent('Receivable'):setIsDropping(true)
end

function ReceiverDropAnimation:update()
  if self._counter == 2 then
    self.gameObject:setState(self._config.dropOne)
  end
  if self._counter == 1 then
    self.gameObject:setState(self._config.dropTwo)
  end
  if self._counter == 0 then
    self.gameObject:getComponent('Receivable'):setIsDropping(false)
  end
  self._counter = self._counter - 1
end


local ConveyerBeltOnAnimation = class.Class(component.Component)

function ConveyerBeltOnAnimation:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('ConveyerBeltOnAnimation')},
      {'framesToDispenseObject', args.default(6), args.positive},
      {'waitState', args.stringType},
      {'stateOne', args.stringType},
      {'stateTwo', args.stringType},
      {'stateThree', args.stringType},
  })
  ConveyerBeltOnAnimation.Base.__init__(self, kwargs)
  self._config.framesToDispenseObject = kwargs.framesToDispenseObject
  self._config.waitState = kwargs.waitState
  self._config.stateOne = kwargs.stateOne
  self._config.stateTwo = kwargs.stateTwo
  self._config.stateThree = kwargs.stateThree
end

function ConveyerBeltOnAnimation:reset()
  self._counter = 0
end

function ConveyerBeltOnAnimation:update()
  local frame = self._counter % 6
  if frame == 5 then
    self.gameObject:setState(self._config.stateThree)
  end
  if frame == 3 then
    self.gameObject:setState(self._config.stateTwo)
  end
  if frame == 1 then
    self.gameObject:setState(self._config.stateOne)
  end
  self._counter = self._counter + 1
end


local ReceiverIndicator = class.Class(component.Component)

function ReceiverIndicator:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('ReceiverIndicator')},
      {'framesToTurnOffIndicator', args.default(8), args.positive},
      {'waitState', args.stringType},
      {'liveState', args.stringType},
      {'secondLiveState', args.stringType},
      {'count', args.stringType},
      {'type', args.stringType},
  })
  ReceiverIndicator.Base.__init__(self, kwargs)
  self._config.framesToTurnOffIndicator = kwargs.framesToTurnOffIndicator
  self._config.waitState = kwargs.waitState
  self._config.liveState = kwargs.liveState
  self._config.secondLiveState = kwargs.secondLiveState
  self._config.count = kwargs.count
  self._config.type = kwargs.type
end

function ReceiverIndicator:getCount()
  return self._config.count
end

function ReceiverIndicator:getType()
  return self._config.type
end

function ReceiverIndicator:processingObjects()
 self._counter = self._config.framesToTurnOffIndicator
end

function ReceiverIndicator:reset()
  self._counter = 0
end

function ReceiverIndicator:update()
  local hopperInventory = self.gameObject:getComponent('Transform'):queryDisc(
      'lowestPhysical', 1)
  if self._config.liveState then
    for _, hopperMouth in ipairs(hopperInventory) do
      if hopperMouth:hasComponent('Receiver') then
        if self:getType() == 'TwoBlocks' and
            hopperMouth:getComponent('Receiver'):hasOneOfTwoCubes() then
          self.gameObject:setState(self._config.secondLiveState)
        elseif hopperMouth:getComponent('Receiver'):hasNeededObjects() then
          self.gameObject:setState(self._config.waitState)
          self:processingObjects()
        else
          self.gameObject:setState(self._config.liveState)
        end
      end
    end
  end
  if self._counter > 0 then
    self.gameObject:setState(self._config.waitState)
    self._counter = self._counter -1
  end
end


local HopperMouth = class.Class(component.Component)

function HopperMouth:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('HopperMouth')},
      {'framesToProcess', args.default(17), args.positive},
      {'closed', args.stringType},
      {'opening', args.stringType},
      {'open', args.stringType},
  })
  HopperMouth.Base.__init__(self, kwargs)
  self._config.framesToProcess = kwargs.framesToProcess
  self._config.closed = kwargs.closed
  self._config.opening = kwargs.opening
  self._config.open = kwargs.open
end

function HopperMouth:processing()
  self._counter = self._config.framesToProcess
  processing = true
  return
end

function HopperMouth:reset()
  processing = false
  self._counter = 0
end

function HopperMouth:isOpen()
  return self._config.isOpen
end

function HopperMouth:resetInventory()
  self.gameObject:getComponent('Receiver'):setHasNeededObjects(false)
  return
end

function HopperMouth:setIsOpen(x)
  self._config.isOpen = x
  if self._config.isOpen then
  end
  return self._config.isOpen
end

function HopperMouth:update()
  if not processing == true then
    self.gameObject:setState(self._config.open)
    self:setIsOpen(true)
  elseif processing == true then
    if self._counter > 0 then
      if self._counter == 15 then
        self.gameObject:setState(self._config.opening)
        self:resetInventory()
      end
      if self._counter == 14 then
        self.gameObject:setState(self._config.closed)
        self:setIsOpen(false)
      end
      if self._counter == 2 then
        self.gameObject:setState(self._config.opening)
      end
      if self._counter == 1 then
        self.gameObject:setState(self._config.open)
        self:setIsOpen(true)
        self:reset()
      end
      self._counter = self._counter - 1
    end
  else
    self.gameObject:setState(self._config.closed)
    self:setIsOpen(false)
  end
  local hopper = self.gameObject:getComponent('Transform'):queryDisc(
      'upperPhysical', 1)
  for _, neededObjects in ipairs(hopper) do
    if self.gameObject:getComponent('Receiver'):hasNeededObjects() then
      if not processing == true then
        self:processing()
      end
    end
  end
end


local AppleComponent = class.Class(component.Component)

function AppleComponent:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AppleComponent')},
      {'liveState', args.stringType},
      {'waitState', args.stringType},
      {'rewardForEating', args.numberType},
  })
  AppleComponent.Base.__init__(self, kwargs)
  self._config.liveState = kwargs.liveState
  self._config.waitState = kwargs.waitState
  self._config.rewardForEating = kwargs.rewardForEating
end

function AppleComponent:reset()
  self._waitState = self._config.waitState
  self._liveState = self._config.liveState
end

function AppleComponent:onEnter(enteringGameObject, contactName)
  if contactName == 'avatar' then
    if self.gameObject:getState() == self._liveState then
      -- Reward the player who ate the edible.
      local avatarComponent = enteringGameObject:getComponent('Avatar')
      avatarComponent:addReward(self._config.rewardForEating)
      -- Change the edible to its wait (disabled) state.
      self.gameObject:setState(self._waitState)
    end
  end
end


-- Animation for apple ejecting from converter output.
local ObjectJumpAnimation = class.Class(component.Component)

function ObjectJumpAnimation:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('ObjectJumpAnimation')},
      {'framesToJump', args.default(8), args.positive},
      {'jump', args.stringType},
      {'drop', args.stringType},
      {'waitState', args.stringType},
  })
  ObjectJumpAnimation.Base.__init__(self, kwargs)
  self._config.framesToJump = kwargs.framesToJump
  self._config.jump = kwargs.jump
  self._config.drop = kwargs.drop
  self._config.waitState = kwargs.waitState
  self._counter = 0
end

function ObjectJumpAnimation:startJump()
  self._counter = self._config.framesToJump
  self.gameObject:setState(self._config.waitState)
end

function ObjectJumpAnimation:update()
  if self._counter == 2 then
    self.gameObject:setState(self._config.jump)
  end
  if self._counter == 1 then
    self.gameObject:setState(self._config.drop)
  end
  self._counter = self._counter - 1
end


-- Animation for apple ejecting from converter output.
local SecondObjectJumpAnimation = class.Class(component.Component)

function SecondObjectJumpAnimation:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('SecondObjectJumpAnimation')},
      {'framesToJump', args.default(11), args.positive},
      {'jump', args.stringType},
      {'drop', args.stringType},
      {'waitState', args.stringType},
  })
  SecondObjectJumpAnimation.Base.__init__(self, kwargs)
  self._config.framesToJump = kwargs.framesToJump
  self._config.jump = kwargs.jump
  self._config.drop = kwargs.drop
  self._config.waitState = kwargs.waitState
end

function SecondObjectJumpAnimation:reset()
  self._counter = 0
end

function SecondObjectJumpAnimation:startJump()
  self._counter = self._config.framesToJump
  self.gameObject:setState(self._config.waitState)
end

function SecondObjectJumpAnimation:update()
  if self._counter == 2 then
    self.gameObject:setState(self._config.jump)
  end
  if self._counter == 1 then
    self.gameObject:setState(self._config.drop)
  end
  self._counter = self._counter - 1
  if self._counter == 0 then
    self:reset()
  end
end


-- Animation for apple ejecting from converter output.
local ObjectDispensingAnimation = class.Class(component.Component)

function ObjectDispensingAnimation:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('ObjectDispensingAnimation')},
      {'framesToJump', args.default(6), args.positive},
      {'frameOne', args.stringType},
      {'frameTwo', args.stringType},
      {'frameThree', args.stringType},
      {'waitState', args.stringType},
  })
  ObjectDispensingAnimation.Base.__init__(self, kwargs)
  self._config.framesToJump = kwargs.framesToJump
  self._config.frameOne = kwargs.frameOne
  self._config.frameTwo = kwargs.frameTwo
  self._config.frameThree = kwargs.frameThree
  self._config.waitState = kwargs.waitState
end

function ObjectDispensingAnimation:reset()
  self._counter = 0
end

function ObjectDispensingAnimation:startJump()
  self._counter = self._config.framesToJump
end

function ObjectDispensingAnimation:update()
  local state = self.gameObject:getState()
  if self._counter == 3 then
    self.gameObject:setState(self._config.frameOne)
  end
  if self._counter == 2 then
    self.gameObject:setState(self._config.frameTwo)
  end
  if self._counter == 1 then
    self.gameObject:setState(self._config.frameThree)
  end
  if self._counter == 0 then
    self.gameObject:setState(self._config.waitState)
  end
  self._counter = self._counter - 1
  if self._counter == 0 then
    self:reset()
  end
end


-- Animation for apple ejecting from converter output.
local DoubleObjectDispensingAnimation = class.Class(component.Component)

function DoubleObjectDispensingAnimation:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('DoubleObjectDispensingAnimation')},
      {'framesToJump', args.default(9), args.positive},
      {'frameOne', args.stringType},
      {'frameTwo', args.stringType},
      {'frameThree', args.stringType},
      {'frameFour', args.stringType},
      {'frameFive', args.stringType},
      {'frameSix', args.stringType},
      {'waitState', args.stringType},
  })
  DoubleObjectDispensingAnimation.Base.__init__(self, kwargs)
  self._config.framesToJump = kwargs.framesToJump
  self._config.frameOne = kwargs.frameOne
  self._config.frameTwo = kwargs.frameTwo
  self._config.frameThree = kwargs.frameThree
  self._config.frameFour = kwargs.frameFour
  self._config.frameFive = kwargs.frameFive
  self._config.frameSix = kwargs.frameSix
  self._config.waitState = kwargs.waitState
end

function DoubleObjectDispensingAnimation:reset()
  self._counter = 0
end

function DoubleObjectDispensingAnimation:startJump()
  self._counter = self._config.framesToJump
end

function DoubleObjectDispensingAnimation:update()
  local state = self.gameObject:getState()
  if self._counter == 7 then
    self.gameObject:setState(self._config.frameOne)
  end
  if self._counter == 6 then
    self.gameObject:setState(self._config.frameTwo)
  end
  if self._counter == 5 then
    self.gameObject:setState(self._config.frameThree)
  end
  if self._counter == 3 then
    self.gameObject:setState(self._config.frameFour)
  end
  if self._counter == 2 then
    self.gameObject:setState(self._config.frameFive)
  end
  if self._counter == 1 then
    self.gameObject:setState(self._config.frameSix)
  end
  if self._counter == 0 then
    self.gameObject:setState(self._config.waitState)
  end
  self._counter = self._counter - 1
  if self._counter == 0 then
    self:reset()
  end
end


local DispenserIndicator = class.Class(component.Component)

function DispenserIndicator:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('DispenserIndicator')},
      {'objectOne', args.stringType},
      {'objectTwo', args.stringType},
  })
  DispenserIndicator.Base.__init__(self, kwargs)
  self._config.objectOne = kwargs.objectOne
  self._config.objectTwo = kwargs.objectTwo
end

function DispenserIndicator:getObjectOne()
  return self._config.objectOne
end

function DispenserIndicator:getObjectTwo()
  return self._config.objectTwo
end

function DispenserIndicator:reset()
  self._counter = 0
end

function DispenserIndicator:registerUpdaters(updaterRegistry)
  local indicator = function()
    local dispensingAnimation = self.gameObject:getComponent(
        'Transform'):queryDisc('overlay', 2)
    local hopper = self.gameObject:getComponent('Transform'):queryDisc(
        'lowestPhysical', 2)
    -- Looks for hopper. Must be within a radius of 2 cells of connected game
    -- object.
    local dispenser = self.gameObject:getComponent('Transform'):queryDisc(
        'midPhysical', 1)
    -- Looks for dispenser. Must be on a cell adjacent to connected game object.
    local offsetPosition = tensor.Tensor(
        self.gameObject:getPosition()):cadd(tensor.Tensor({0, 2})):val()
    -- Locates position 2 cells from connected game object.
    local secondOffsetPosition = tensor.Tensor(
        self.gameObject:getPosition()):cadd(tensor.Tensor({0, 3})):val()
    -- Locates position 3 cells from connected game object.
    local allTokens = self.gameObject.simulation:getAllGameObjectsWithComponent(
        'Token')
    -- Find all game objects with Token component.
    for _, receiverInventory in ipairs(hopper) do
      if receiverInventory:hasComponent('Receiver') then
        local dispenserObjectOne = self:getObjectOne()
        local dispenserObjectTwo = self:getObjectTwo()
        local hopperInventory = receiverInventory:getComponent('Receiver')
        if hopperInventory:hasNeededObjects() then
          -- If the machine only dispenses one object, then find and
          -- teleport that object.
          if self:getObjectTwo() == "NoneNeeded" then
            if self:getObjectOne() == "Apple" then
              local appleToMove
              for i, apple in ipairs(allTokens) do
                local tokenType = apple:getComponent('Token'):getType()
                if apple:getState() == 'waitState' and tokenType == 'Apple' then
                appleToMove = apple
                  -- Queue an apple in waitState to teleport onto dispensing
                  -- area when triggered.
                end
              end
              appleToMove:getComponent('ObjectJumpAnimation'):startJump()
              appleToMove:teleport(offsetPosition, 'S')
              for i, dispensingApple in ipairs(dispensingAnimation) do
              -- Run any dispensing animation in waitState in vicinity.
                dispensingApple:getComponent(
                    'ObjectDispensingAnimation'):startJump()
              end
            end
            if self:getObjectOne() == "PinkCube" then
              local pinkCubeToMove
              for i, token in ipairs(allTokens) do
                local tokenType = token:getComponent('Token'):getType()
                if token:getState() == 'waitState' and
                    tokenType == 'PinkCube' then
                  pinkCubeToMove = token
                  break
                end
              end
              pinkCubeToMove:getComponent('ObjectJumpAnimation'):startJump()
              pinkCubeToMove:teleport(offsetPosition, 'S')
              for i, dispensingPinkCube in ipairs(dispensingAnimation) do
                dispensingPinkCube:getComponent(
                    'ObjectDispensingAnimation'):startJump()
              end
            end
          elseif self:getObjectTwo() ~= "NoneNeeded" then
            -- Check if hopper has required objects.
            if self:getObjectOne() == "Apple" and
                self:getObjectTwo() == "Apple" then
                -- Find all invisible apples.
              appleList = {}
              for i, apple in ipairs(allTokens) do
                local tokenType = apple:getComponent('Token'):getType()
                if apple:getState() == 'waitState' and
                    tokenType == 'Apple' then
                  table.insert(appleList, apple)
                end
              end
              appleList[1]:getComponent('ObjectJumpAnimation'):startJump()
              appleList[1]:teleport(offsetPosition, 'S')
              appleList[2]:getComponent('ObjectJumpAnimation'):startJump()
              appleList[2]:teleport(secondOffsetPosition, 'S')
              for i, dispensingObjects in ipairs(dispensingAnimation) do
                dispensingObjects:getComponent(
                    'ObjectDispensingAnimation'):startJump()
              end
            end
            if dispenserObjectOne == "BlueCube" and
                dispenserObjectTwo == "Banana" then
              local bananaToMove
              local blueCubeToMove
              for i, token in ipairs(allTokens) do
                local tokenType = token:getComponent('Token'):getType()
                if token:getState() == 'waitState' and
                    tokenType == 'Banana' then
                  bananaToMove = token
                end
                if token:getState() == 'waitState' and
                    tokenType == 'BlueCube' then
                  blueCubeToMove = token
                end
              end
              blueCubeToMove:getComponent('ObjectJumpAnimation'):startJump()
              blueCubeToMove:teleport(offsetPosition, 'S')
              bananaToMove:getComponent('SecondObjectJumpAnimation'):startJump()
              bananaToMove:teleport(secondOffsetPosition, 'S')
              for i, dispensingObjects in ipairs(dispensingAnimation) do
                dispensingObjects:getComponent(
                    'DoubleObjectDispensingAnimation'):startJump()
              end
            end
            if dispenserObjectOne == "Apple" and
                dispenserObjectTwo == "BlueCube" then
              local appleToMove
              local blueCubeToMove
              for i, token in ipairs(allTokens) do
                local tokenType = token:getComponent('Token'):getType()
                if token:getState() == 'waitState' and
                    tokenType == 'Apple' then
                  appleToMove = token
                end
                if token:getState() == 'waitState' and
                    tokenType == 'BlueCube' then
                  blueCubeToMove = token
                  break
                end
              end
              blueCubeToMove:getComponent('ObjectJumpAnimation'):startJump()
              blueCubeToMove:teleport(offsetPosition, 'S')
              appleToMove:getComponent('SecondObjectJumpAnimation'):startJump()
              appleToMove:teleport(secondOffsetPosition, 'S')
              for i, dispensingObjects in ipairs(dispensingAnimation) do
                dispensingObjects:getComponent(
                    'DoubleObjectDispensingAnimation'):startJump()
              end
            end
          end
        end
      end
    end
  end
  updaterRegistry:registerUpdater{
    updateFn = indicator,
    priority = 200,
}
end


local allComponents = {
  -- Apple components.
  AppleComponent = AppleComponent,
  ObjectJumpAnimation = ObjectJumpAnimation,
  SecondObjectJumpAnimation = SecondObjectJumpAnimation,
  ObjectDispensingAnimation = ObjectDispensingAnimation,
  DoubleObjectDispensingAnimation = DoubleObjectDispensingAnimation,
  -- Object components.
  Receivable = Receivable,
  Token = Token,
  -- Machinery components.
  Receiver = Receiver,
  ReceiverIndicator = ReceiverIndicator,
  DispenserIndicator = DispenserIndicator,
  ReceiverDropAnimation = ReceiverDropAnimation,
  ConveyerBeltOnAnimation = ConveyerBeltOnAnimation,
  HopperMouth = HopperMouth,
  ConveyorMovement = ConveyorMovement,
}

component_registry.registerAllComponents(allComponents)

return allComponents
