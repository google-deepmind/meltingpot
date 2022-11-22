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

local helpers = require 'common.helpers'
local log = require 'common.log'
local args = require 'common.args'
local class = require 'common.class'
local set = require 'common.set'
local tensor = require 'system.tensor'

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')

local _GRASP_PRIORITY = 140
local _MOVE_PRIORITY = 150

local _DIRECTION = {
    N = tensor.Tensor({0, -1}),
    E = tensor.Tensor({1, 0}),
    S = tensor.Tensor({0, 1}),
    W = tensor.Tensor({-1, 0}),
}

-- The `Graspable` component enables an object to be grasped by AvatarGrasp.
local Graspable = class.Class(component.Component)

function Graspable:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Graspable')},
      {'graspableStates', args.tableType},
      -- If set to a disconnect state then automatically disconnect from the
      -- avatar currently holding this (if there is one).
      {'disconnectStates', args.tableType},
  })
  Graspable.Base.__init__(self, kwargs)
  self._config.graspableStates = kwargs.graspableStates
  self._config.disconnectStates = kwargs.disconnectStates

  self._graspableStatesSet = set.Set(self._config.graspableStates)
  self._disconnectStatesSet = set.Set(self._config.disconnectStates)
end

function Graspable:pickUp(hitterObject)
  if self._graspableStatesSet[self.gameObject:getState()] then
    hitterObject:getComponent('AvatarGrasp'):grasp(self.gameObject)
    self._isGrasped = true
  end
end

function Graspable:setGrasped(status)
  self._isGrasped = status
end

function Graspable:isGrasped()
  return self._isGrasped
end

function Graspable:getWhoIsGrasping()
  return self._whoIsGrasping
end

function Graspable:setWhoIsGrasping(avatarIndex)
  self._whoIsGrasping = avatarIndex
end

function Graspable:reset()
  self._isGrasped = false
end

function Graspable:registerUpdaters(updaterRegistry)
  local grasp = function()
    self._isGrasped = false
  end

  updaterRegistry:registerUpdater{
      updateFn = grasp,
      priority = _GRASP_PRIORITY - 1,
  }
end

function Graspable:priority()
  return _GRASP_PRIORITY
end

function Graspable:getAvatarHoldingThis()
  if self._whoIsGrasping then
    local avatarObject = self.gameObject.simulation:getAvatarFromIndex(
        self._whoIsGrasping)
    if avatarObject and avatarObject:hasComponent('AvatarGrasp') then
      return avatarObject
    end
  end
  return nil
end

function Graspable:onStateChange(oldState)
  local newState = self.gameObject:getState()
  if self._disconnectStatesSet[newState] then
    local avatarObject = self:getAvatarHoldingThis()
    if avatarObject then
      local avatarGrasp = avatarObject:getComponent('AvatarGrasp')
      avatarGrasp:drop()
    end
  end
end


-- The `AvatarGrasp` component endows an avatar with the ability to grasp an
-- object in the direction they are facing.

local AvatarGrasp = class.Class(component.Component)

function AvatarGrasp:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AvatarGrasp')},
      {'shape', args.stringType},
      {'palette', args.tableType},
      {'graspAction', args.stringType},
      -- If multiple objects are at the same position then grasp them
      -- according to their layer in order `precedenceOrder`.
      {'precedenceOrder', args.tableType},
  })
  AvatarGrasp.Base.__init__(self, kwargs)
  self._config.shape = kwargs.shape
  self._config.palette = kwargs.palette
  self._config.graspAction = kwargs.graspAction
  self._config.precedenceOrder = kwargs.precedenceOrder
end

function AvatarGrasp:awake()
  self._graspBeam = 'grasp_' .. self:avatarIndex()
  self._handBeam = 'hand_' .. self:avatarIndex()
end

function AvatarGrasp:reset()
  self._avatar = self.gameObject:getComponent('Avatar')
  self._graspedObject = nil
end

function insertAtIfNotPresent(tbl, insert_at, element)
  for _, value in pairs(tbl) do
    if value == element then
      return
    end
  end
  table.insert(tbl, insert_at, element)
end

function AvatarGrasp:addHits(worldConfig)
  -- Add the grasp beam underneath the overlay layer.
  local insert_at = 0
  for index, layer in pairs(worldConfig.renderOrder) do
    if layer == 'overlay' then
      insert_at = index
      break
    end
  end
  insertAtIfNotPresent(worldConfig.renderOrder, insert_at, self._graspBeam)
  worldConfig.hits[self._graspBeam] = {
      layer = self._graspBeam,
      sprite = self._graspBeam,
  }
  insertAtIfNotPresent(worldConfig.renderOrder, insert_at, self._handBeam)
  worldConfig.hits[self._handBeam] = {
      layer = self._handBeam,
      sprite = self._handBeam,
  }
end

function AvatarGrasp:addSprites(tileSet)
  tileSet:addShape(self._handBeam, {
      palette = self._config.palette,
      text = self._config.shape})
end

function AvatarGrasp:drop()
  local graspable = self._graspedObject:getComponent('Graspable')
  graspable:setGrasped(false)
  graspable:setWhoIsGrasping(nil)

  self._avatar:disconnect(self._graspedObject)
  self._graspedObject = nil
end

function AvatarGrasp:getGraspableObjectInFront()
  local transform = self.gameObject:getComponent('Transform')
  for _, layer in ipairs(self._config.precedenceOrder) do
    local hit, foundObject, _ = transform:rayCastDirection(layer, 1)
    if hit and foundObject:hasComponent('Graspable') then
      return foundObject
    end
  end
  return nil
end

function AvatarGrasp:getGraspableObjectByMeInFront()
  local graspableObject = self:getGraspableObjectInFront()
  if not graspableObject then
    -- Return `nil` if there is no object in front or if it does not have the
    -- `Graspable` component.
    return nil
  end
  local graspable = graspableObject:getComponent('Graspable')
  local whoIsGrasping = graspable:getWhoIsGrasping()
  if not whoIsGrasping then
    -- Return the object if no one is grasping the object.
    return graspableObject
  end
  if self:avatarIndex() == whoIsGrasping then
    -- Return the object if I am grasping the object.
    return graspableObject
  end
  -- Return `nil` if anyone else is grasping the object.
  return nil
end

function AvatarGrasp:registerUpdaters(updaterRegistry)
  local grasp = function()
    local isHoldingObject = self._graspedObject ~= nil
    if isHoldingObject then
      local objectInFront = self:getGraspableObjectByMeInFront()
      if objectInFront then
        self.gameObject:hitBeam(self._handBeam, 1, 0)
      end
    end

    local playerVolatileVariables = self._avatar:getVolatileData()
    local graspAction =
        playerVolatileVariables.actions[self._config.graspAction] == 1
    if graspAction and not isHoldingObject then
      -- pick up object
      local objectInFront = self:getGraspableObjectByMeInFront()
      if objectInFront then
        objectInFront:getComponent('Graspable'):pickUp(self.gameObject)
      end
    end
    if graspAction and isHoldingObject then
      -- drop object
      self:drop()
    end
  end

  local rotate_grasped = function()
    -- Only rotate the grasped object if there is one connected.
    if not self._graspedObject then
      return
    end

    -- Only rotate the grasped object if the player rotates.
    local playerVolatileVariables = self._avatar:getVolatileData()
    local turn_action = playerVolatileVariables.actions['turn']
    if turn_action == 0 then
      return
    end

    -- Determine the direction the player will be facing on the next turn.
    local _COMPASS = {N = 1, E = 2, S = 3, W = 4}
    local _COMPASSKEYS = {'N', 'E', 'S', 'W'}
    local function rotate(facing, turn)
      return _COMPASSKEYS[(_COMPASS[facing] - 1 + turn + 4) % 4 + 1]
    end

    local playerDir = rotate(self.gameObject:getOrientation(), turn_action)
    local objectDir = rotate(self._graspedObject:getOrientation(), turn_action)

    -- Teleport the object to the location the player will be facing.
    local offsetPosition = tensor.Tensor(self.gameObject:getPosition()):cadd(
        _DIRECTION[playerDir]):val()
    self._avatar:disconnect(self._graspedObject)
    self._graspedObject:teleport(offsetPosition, objectDir)
    self._avatar:connect(self._graspedObject)
  end

  updaterRegistry:registerUpdater{
      updateFn = grasp,
      priority = _GRASP_PRIORITY,
  }
  updaterRegistry:registerUpdater{
      updateFn = rotate_grasped,
      priority = _MOVE_PRIORITY - 1,
  }
end

function AvatarGrasp:avatarIndex()
  return self.gameObject:getComponent('Avatar'):getIndex()
end

function AvatarGrasp:grasp(gameObject)
  if not self._graspedObject then
    self._graspedObject = gameObject
    self._avatar:connect(self._graspedObject)
    self._graspedObject:getComponent('Graspable'):setWhoIsGrasping(
        self:avatarIndex())
  end
end

local allComponents = {
  AvatarGrasp = AvatarGrasp,
  Graspable = Graspable,
}

-- Register all components from this module in the component registry.
component_registry.registerAllComponents(allComponents)

