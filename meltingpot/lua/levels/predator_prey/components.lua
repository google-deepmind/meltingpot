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


local Role = class.Class(component.Component)

function Role:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Role')},
      {'isPredator', args.booleanType},
  })
  Role.Base.__init__(self, kwargs)
  self._config.isPredator = kwargs.isPredator
end

function Role:isPredator()
  return self._config.isPredator
end

function Role:isPrey()
  return not self._config.isPredator
end


--[[ The `PredatorInteractBeam` component endows a predator with the ability
to fire a 1 length, 0 radius beam to interact with other objects that they are
facing. ]]
local PredatorInteractBeam = class.Class(component.Component)

function PredatorInteractBeam:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('PredatorInteractBeam')},
      {'cooldownTime', args.numberType},
      {'shapes', args.tableType},
      {'palettes', args.tableType},
  })
  PredatorInteractBeam.Base.__init__(self, kwargs)

  self._config.cooldownTime = kwargs.cooldownTime
  self._config.shape = kwargs.shapes[1]
  self._config.palette = kwargs.palettes[1]
end

function PredatorInteractBeam:awake()
  self.hitAndSpriteName = 'predator'
  self._showForDuration = 0
end

function PredatorInteractBeam:getHitName()
  return self.hitAndSpriteName
end

function PredatorInteractBeam:addHits(worldConfig)
  worldConfig.hits[self.hitAndSpriteName] = {
      layer = self.hitAndSpriteName,
      sprite = self.hitAndSpriteName,
  }
  component.insertIfNotPresent(worldConfig.renderOrder, self.hitAndSpriteName)
end

function PredatorInteractBeam:addSprites(tileSet)
  tileSet:addShape(self.hitAndSpriteName,
                   {palette = self._config.palette,
                    text = self._config.shape,
                    noRotate = true})
end

function PredatorInteractBeam:registerUpdaters(updaterRegistry)
  local interact = function()
    local playerVolatileVariables = (
        self.gameObject:getComponent('Avatar'):getVolatileData())
    local actions = playerVolatileVariables.actions
    -- Execute the beam if applicable.
    if self._showForDuration > 0 then
      self.gameObject:hitBeam(self.hitAndSpriteName, 1, 0)
      self._showForDuration = self._showForDuration - 1
    else
      if actions['interact'] == 1 then
        self._coolingTimer = self._config.cooldownTime
        self.gameObject:hitBeam(self.hitAndSpriteName, 1, 0)
      end
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = interact,
      priority = 140,
  }
end

function PredatorInteractBeam:showForDuration(duration)
  self._showForDuration = duration - 1
end

--[[ Reset gets called just before start regardless of whether we have a new
environment instance or one that was reset by calling reset() in python.]]
function PredatorInteractBeam:reset()
  local kwargs = self._kwargs
  self._playerIndex = self.gameObject:getComponent('Avatar'):getIndex()
  -- Set the beam cooldown timer to its `ready` state (i.e. coolingTimer = 0).
  self._coolingTimer = 0
end


--[[ The `InteractEatAcorn` component endows a prey avatar with the ability to
eat an acorn if in their inventory for a reward. ]]
local InteractEatAcorn = class.Class(component.Component)

function InteractEatAcorn:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('InteractEatAcorn')},
      {'cooldownTime', args.numberType},
      {'shapes', args.tableType},
      {'palettes', args.tableType},
      {'isEating', args.booleanType},
      -- `defaultState` is the default player state with arms down.
      {'defaultState', args.stringType},
  })
  InteractEatAcorn.Base.__init__(self, kwargs)

  self._config.cooldownTime = kwargs.cooldownTime
  self._config.shape = kwargs.shapes[1]
  self._config.palette = kwargs.palettes[1]
  self._config.isEating = kwargs.isEating
  self._config.defaultState = kwargs.defaultState

  self._inventoryObject = nil
end

function InteractEatAcorn:awake()
  self.hitAndSpriteName = 'interact_' .. self.gameObject:getUniqueState()
  self._showForDuration = 0
end

function InteractEatAcorn:getHitName()
  return self.hitAndSpriteName
end

function InteractEatAcorn:isEating()
  return self._config.isEating
end

function InteractEatAcorn:setIsEating(x)
  self._config.isEating = x
  return self._config.isEating
end

function InteractEatAcorn:setInventoryObject(obj)
  self._inventoryObject = obj
end

function InteractEatAcorn:getAvatarsInventory()
  return self._inventoryObject
end

function InteractEatAcorn:addHits(worldConfig)
  worldConfig.hits[self.hitAndSpriteName] = {
      layer = self.hitAndSpriteName,
      sprite = self.hitAndSpriteName,
  }
  component.insertIfNotPresent(worldConfig.renderOrder, self.hitAndSpriteName)
end

function InteractEatAcorn:addSprites(tileSet)
  tileSet:addShape(self.hitAndSpriteName,
                   {palette = self._config.palette,
                    text = self._config.shape,
                    noRotate = true})
end

function InteractEatAcorn:registerUpdaters(updaterRegistry)
  local interact = function()
    local playerVolatileVariables = (
        self.gameObject:getComponent('Avatar'):getVolatileData())
    local actions = playerVolatileVariables.actions
    -- Execute eat action if applicable.
    if self._config.cooldownTime >= 0 then
      if self._coolingTimer > 0 then
        self._coolingTimer = self._coolingTimer - 1
      else
        if actions['interact'] == 1 then
          self._coolingTimer = self._config.cooldownTime
          local inventoryObject = self:getAvatarsInventory()
          if inventoryObject:getHeldItem() ~= 'empty' then
            -- Only allow eating an acorn if stamina bar is invisible and
            -- in default "arms down" state.
            local stamina = self.gameObject:getComponent('Stamina')
            local state = self.gameObject:getState()
            if stamina:getBand() == 'invisible' and
                state == self._config.defaultState then
              inventoryObject:setHeldItem('empty')
              self.gameObject:getComponent('AvatarEatingAnimation'):sitDown()
            end
          end
        end
      end
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = interact,
      priority = 140,
  }
end

function InteractEatAcorn:showForDuration(duration)
  self._showForDuration = duration - 1
end

--[[ Reset gets called just before start regardless of whether we have a new
environment instance or one that was reset by calling reset() in python.]]
function InteractEatAcorn:reset()
  local kwargs = self._kwargs
  self._playerIndex = self.gameObject:getComponent('Avatar'):getIndex()
  -- Set the eat action cooldown timer to its `ready` state
  -- (i.e. coolingTimer = 0).
  self._coolingTimer = 0
end


--[[ An inventory component which holds an item and visualises it. It will
be placed on a separate object, one associated with each avatar, and connected
to it.
]]
local Inventory = class.Class(component.Component)

function Inventory:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Inventory')},
      -- `playerIndex` (int): index of the player whose inventory this is.
      {'playerIndex', args.numberType},
  })
  Inventory.Base.__init__(self, kwargs)
  self._playerIndex = kwargs.playerIndex
end

function Inventory:postStart()
  local sim = self.gameObject.simulation
  -- Store a reference to the connected avatar object.
  self._avatarObject = sim:getAvatarFromIndex(self._playerIndex)
  self._avatarObject:getComponent('InteractEatAcorn'):setInventoryObject(self)
end

function Inventory:getHeldItem()
  local state = self.gameObject:getState()
  return state
end

function Inventory:setHeldItem(item)
  self.gameObject:setState(item)
end


local AvatarEdible = class.Class(component.Component)

function AvatarEdible:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AvatarEdible')},
      {'groupRadius', args.default(2), args.positive},
      {'framesToDisplayBeingEaten', args.default(5), args.positive},
      {'predatorRewardForEating', args.default(1), args.numberType},
  })
  AvatarEdible.Base.__init__(self, kwargs)
  self._groupRadius = kwargs.groupRadius
  self._framesToDisplayBeingEaten = kwargs.framesToDisplayBeingEaten
  self._config.predatorRewardForEating = kwargs.predatorRewardForEating
end

function AvatarEdible:reset()
  self.dead = false
end

function AvatarEdible:_countGroupSize(targetRole)
  local function isTargetRole(object)
    if targetRole == 'prey' then
      return object:getComponent('Role'):isPrey()
    elseif targetRole == 'predator' then
      return object:getComponent('Role'):isPredator()
    end
  end

  local layer = self.gameObject:getLayer()
  local transform = self.gameObject:getComponent('Transform')
  local objectsNearby = transform:queryDisc(layer, self._groupRadius)

  local groupSize = 0
  local objectsWithTargetRoleNearby = {}
  for _, object in pairs(objectsNearby) do
    if object:hasComponent('Role') and isTargetRole(object) then
      if object:getComponent('Stamina'):getBand() ~= 'red' then
        -- Never count avatars with stamina currently in the 'red' band.
        if object:hasComponent('InteractEatAcorn') then
          -- Prey have the 'InteractEatAcorn' component.
          if not object:getComponent('InteractEatAcorn'):isEating() then
            -- Prey only count if they are not currently eating an acorn.
            groupSize = groupSize + 1
            table.insert(objectsWithTargetRoleNearby, object)
          end
        else
          -- Predators do not have the 'InteractEatAcorn' component.
          groupSize = groupSize + 1
          table.insert(objectsWithTargetRoleNearby, object)
        end
      end
    end
  end
  return groupSize, objectsWithTargetRoleNearby
end


function AvatarEdible:_beEaten()
  local deadState = self.gameObject:getComponent('Avatar'):getWaitState()
  self.gameObject:setState(deadState)
  self.dead = true
end

function AvatarEdible:onHit(hitterObject, hitName)
  local selfRole = self.gameObject:getComponent('Role')
  if hitName == 'predator' and selfRole:isPrey() then
    local preyGroupSize, preyNearby = self:_countGroupSize('prey')
    local predatorGroupSize, _ = self:_countGroupSize('predator')
    if preyGroupSize <= predatorGroupSize then
      -- Case where the group is too small so the prey will be eaten.
      self:_beEaten()
      hitterObject:getComponent('PredatorInteractBeam'):showForDuration(
          self._framesToDisplayBeingEaten)
      hitterObject:getComponent('Avatar'):disallowMovementUntil(
          self._framesToDisplayBeingEaten)
      local hitterAvatar = hitterObject:getComponent('Avatar')
      hitterAvatar:addReward(self._config.predatorRewardForEating)
      events:add(
        'prey_consumed', 'dict',
        'predator_player_index', hitterAvatar:getIndex(),
        'prey_player_index', self.gameObject:getComponent('Avatar'):getIndex()
      )  -- (int, int)
    else
      -- Case where the group is big enough to avoid being eaten.
      for _, preyObject in ipairs(preyNearby) do
        if preyObject:getComponent('AvatarEdible'):alive() then
          preyObject:getComponent('AvatarAnimation'):armsUp()
        end
      end
    end
  elseif hitName == 'predator' and selfRole:isPredator() then
    -- Case where a predator eats another predator for zero reward (since
    -- predators are not tasty). Competition is the main reason one might want
    -- to do this.
    self:_beEaten()
    -- It takes a lot of energy to incapacitate a predator so stamina is
    -- correspondingly decreased substantially. This sometimes gives prey a
    -- chance to escape.
    local hitterStamina = hitterObject:getComponent('Stamina'):addValue(-4)

    events:add(
        'predator_consumed', 'dict',
        'eater_player_index', hitterObject:getComponent('Avatar'):getIndex(),
        'eaten_player_index', self.gameObject:getComponent('Avatar'):getIndex()
      )  -- (int, int)
  end
end

function AvatarEdible:alive()
  return not self.dead
end


local AvatarRespawn = class.Class(component.Component)

function AvatarRespawn:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AvatarRespawn')},
      {'framesTillRespawn', args.positive},
  })
  AvatarRespawn.Base.__init__(self, kwargs)
  self._config.framesTillRespawn = kwargs.framesTillRespawn
end

function AvatarRespawn:registerUpdaters(updaterRegistry)
  local avatar = self.gameObject:getComponent('Avatar')
  local aliveState = avatar:getAliveState()
  local waitState = avatar:getWaitState()
  local respawn = function()
    local spawnGroup = avatar:getSpawnGroup()
    self.gameObject:teleportToGroup(spawnGroup, aliveState)
    self.playerRespawnedThisStep = true
    if self.gameObject:hasComponent('AvatarEdible') then
      local avatarEdible = self.gameObject:getComponent('AvatarEdible')
      avatarEdible.dead = false
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = respawn,
      priority = 135,
      state = waitState,
      startFrame = self._config.framesTillRespawn
  }
end


local AvatarAnimation = class.Class(component.Component)

function AvatarAnimation:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AvatarAnimation')},
      {'framesToRaiseArms', args.default(5), args.positive},
      {'upState', args.stringType},
      {'downState', args.stringType},
  })
  AvatarAnimation.Base.__init__(self, kwargs)
  self._config.framesToRaiseArms = kwargs.framesToRaiseArms
  self._config.upState = kwargs.upState
  self._config.downState = kwargs.downState
end

function AvatarAnimation:reset()
  self._counter = 0
end

function AvatarAnimation:armsUp()
  self._counter = self._config.framesToRaiseArms
  self.gameObject:setState(self._config.upState)
end

function AvatarAnimation:registerUpdaters(updaterRegistry)
  local updateAnimation = function()
    if self._counter == 1 then
      if self.gameObject:getComponent('AvatarEdible'):alive() then
        self.gameObject:setState(self._config.downState)
      end
    end
    self._counter = self._counter - 1
  end

  updaterRegistry:registerUpdater{
      updateFn = updateAnimation,
      priority = 295,
  }
end


local AvatarEatingAnimation = class.Class(component.Component)

function AvatarEatingAnimation:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AvatarEatingAnimation')},
      {'framesToEatAcorn', args.default(26), args.positive},
      {'sit', args.stringType},
      {'prepToEat', args.stringType},
      {'firstBite', args.stringType},
      {'secondBite', args.stringType},
      {'lastBite', args.stringType},
      {'downState', args.stringType},
      {'acornReward', args.default(3.0), args.positive},
  })
  AvatarEatingAnimation.Base.__init__(self, kwargs)
  self._config.framesToEatAcorn = kwargs.framesToEatAcorn
  self._config.sit = kwargs.sit
  self._config.prepToEat = kwargs.prepToEat
  self._config.firstBite = kwargs.firstBite
  self._config.secondBite = kwargs.secondBite
  self._config.lastBite = kwargs.lastBite
  self._config.downState = kwargs.downState
  self._config.acornReward = kwargs.acornReward
  self._config.oneThirdAcornReward = self._config.acornReward / 3.0
end

function AvatarEatingAnimation:_resetAnimation()
  self._counter = 0
  self.gameObject:getComponent('InteractEatAcorn'):setIsEating(false)
end

function AvatarEatingAnimation:reset()
  self:_resetAnimation()
end

function AvatarEatingAnimation:registerUpdaters(updaterRegistry)
  local avatar = self.gameObject:getComponent('Avatar')
  local transform = self.gameObject:getComponent('Transform')
  local deadState = avatar:getWaitState()

  local updateAnimation = function()
    if self.gameObject:getState() ~= deadState then
      if self._counter == self._config.framesToEatAcorn then
        -- Give the avatar a pseudoreward if applicable on the first eat frame.
        if self.gameObject:hasComponent('AcornTaste') then
          local pseudoreward = self.gameObject:getComponent(
              'AcornTaste'):getAcornConsumptionReward()
          avatar:addReward(pseudoreward)
        end
      end
      if self._counter > 0 then
        avatar:disallowMovementUntil(0)
        self.gameObject:getComponent('InteractEatAcorn'):setIsEating(true)
      end
      if self._counter == 21 then
        if self.gameObject:getComponent('AvatarEdible'):alive() then
          self.gameObject:setState(self._config.prepToEat)
        end
      end
      if self._counter == 16 then
        if self.gameObject:getComponent('AvatarEdible'):alive() then
          self.gameObject:setState(self._config.firstBite)
          avatar:addReward(self._config.oneThirdAcornReward)
        end
      end
      if self._counter == 11 then
        if self.gameObject:getComponent('AvatarEdible'):alive() then
          self.gameObject:setState(self._config.secondBite)
          avatar:addReward(self._config.oneThirdAcornReward)
        end
      end
      if self._counter == 6 then
        if self.gameObject:getComponent('AvatarEdible'):alive() then
          self.gameObject:setState(self._config.lastBite)
          avatar:addReward(self._config.oneThirdAcornReward)
        end
      end
      if self._counter == 1 then
        -- Done eating the acorn.
        self.gameObject:getComponent('InteractEatAcorn'):setIsEating(false)
        if self.gameObject:getComponent('AvatarEdible'):alive() then
          self.gameObject:setState(self._config.downState)
        end
        events:add('acorn_consumed', 'dict',
                   'player_index', avatar:getIndex())  -- int
        -- Check if acorn was consumed while standing on safety grass.
        local objectOrNilBelow = transform:queryPosition('midPhysical')
        if objectOrNilBelow and objectOrNilBelow:getState() == 'safe_grass' then
          if self.gameObject:hasComponent('AcornTaste') then
            local pseudoreward = self.gameObject:getComponent(
                'AcornTaste'):getSafeAcornConsumptionReward()
            avatar:addReward(pseudoreward)
          end
          events:add('acorn_consumed_safely', 'dict',
                     'player_index', avatar:getIndex())  -- int
        end
      end
    else
      self:_resetAnimation()
    end
    self._counter = self._counter - 1
  end

  updaterRegistry:registerUpdater{
      updateFn = updateAnimation,
      priority = 300,
  }
end

function AvatarEatingAnimation:sitDown()
  if self.gameObject:getComponent('AvatarEdible'):alive() then
    self._counter = self._config.framesToEatAcorn
    self.gameObject:setState(self._config.sit)
  end
end


--[[ Reward players based on their distance to other nearby avatars having a
specific role.]]
local ProxemicTaste = class.Class(component.Component)

function ProxemicTaste:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('ProxemicTaste')},
      -- `distanceToReward` maps distances to rewards to add up per step per
      -- avatar object found nearby.
      {'distanceToReward', args.default({}), args.tableType},
      -- `layer` indicates which layer to count objects. It will typically be
      -- the layer where Avatars are.
      {'layer', args.default('upperPhysical'), args.stringType},
      -- `roleToCount` indicates whether to count prey, predators, or both.
      {'roleToCount', args.default('avatar'),
       args.oneOf('prey', 'predator', 'avatar')},
  })
  ProxemicTaste.Base.__init__(self, kwargs)
  self._distanceToReward = kwargs.distanceToReward
  self._layer = kwargs.layer
  self._roleToCount = kwargs.roleToCount
end

function ProxemicTaste:reset()
  local role = self.gameObject:getComponent('Role')
  self._subtractFromCount = 0
  if role:isPrey() and self._roleToCount == 'prey' then
    self._subtractFromCount = 1
  end
  if role:isPredator() and self._roleToCount == 'predator' then
    self._subtractFromCount = 1
  end
  if self._roleToCount == 'avatar' then
    self._subtractFromCount = 1
  end
end

function ProxemicTaste:_countAvatarsWithRole(neighbors)
  local counts = {
      avatar = 0,
      prey = 0,
      predator = 0,
  }
  for _, neighbor in ipairs(neighbors) do
    if neighbor:hasComponent('Avatar') and
        neighbor:getComponent('Avatar'):isAlive() then
      counts['avatar'] = counts['avatar'] + 1
      if neighbor:getComponent('Role'):isPrey() then
        counts['prey'] = counts['prey'] + 1
      elseif neighbor:getComponent('Role'):isPredator() then
        counts['predator'] = counts['predator'] + 1
      end
    end
  end
  return counts[self._roleToCount]
end

function ProxemicTaste:registerUpdaters(updaterRegistry)
  local avatar = self.gameObject:getComponent('Avatar')
  local transform = self.gameObject:getComponent('Transform')

  local proximityTasteUpdate = function()
    if avatar:isAlive() then
      local rewardToDeliver = 0
      for distance, rewardAtThisDistance in pairs(self._distanceToReward) do
        local neighbors = transform:queryDisc(self._layer, distance)
        local numCounted = self:_countAvatarsWithRole(neighbors)
        -- If needed, subtract 1 from numCounted so self does not count.
        local extraReward = (
            rewardAtThisDistance * (numCounted - self._subtractFromCount))
        rewardToDeliver = rewardToDeliver + extraReward
      end
      avatar:addReward(rewardToDeliver)
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = proximityTasteUpdate,
      priority = 2,
  }
end


--[[ Optionally provide extra rewards for collecting and eating acorns.

This is off by default. Acorns should not provide any reward till they have
been consumed in the main version of the substrate. This component is typically
only used when training certain background populations.
]]
local AcornTaste = class.Class(component.Component)

function AcornTaste:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AcornTaste')},
      -- No effect when `collectReward` is 0.0.
      {'collectReward', args.default(0.0), args.numberType},
      -- No effect when `eatReward` is 0.0.
      {'eatReward', args.default(0.0), args.numberType},
      -- No effect when `safeAcornConsumptionReward` is 0.0.
      {'safeAcornConsumptionReward', args.default(0.0), args.numberType},
  })
  AcornTaste.Base.__init__(self, kwargs)
  self._collectReward = kwargs.collectReward
  self._extraEatReward = kwargs.eatReward
  self._safeAcornConsumptionReward = kwargs.safeAcornConsumptionReward
end

function AcornTaste:getAcornCollectionReward()
  return self._collectReward
end

function AcornTaste:getAcornConsumptionReward()
  return self._extraEatReward
end

function AcornTaste:getSafeAcornConsumptionReward()
  return self._safeAcornConsumptionReward
end


-- Make the apples edible.
local AppleEdible = class.Class(component.Component)

function AppleEdible:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AppleEdible')},
      {'liveState', args.stringType},
      {'waitState', args.stringType},
      {'rewardForEating', args.numberType},
  })
  AppleEdible.Base.__init__(self, kwargs)

  self._config.liveState = kwargs.liveState
  self._config.waitState = kwargs.waitState
  self._config.rewardForEating = kwargs.rewardForEating
end

function AppleEdible:reset()
  self._waitState = self._config.waitState
  self._liveState = self._config.liveState
end

function AppleEdible:setWaitState(newWaitState)
  self._waitState = newWaitState
end

function AppleEdible:getWaitState()
  return self._waitState
end

function AppleEdible:setLiveState(newLiveState)
  self._liveState = newLiveState
end

function AppleEdible:getLiveState()
  return self._liveState
end

function AppleEdible:onEnter(enteringGameObject, contactName)
  if contactName == 'avatar' and enteringGameObject and
      enteringGameObject:hasComponent('Role') then
    local role = enteringGameObject:getComponent('Role')
    if self.gameObject:getState() == self._liveState and role:isPrey() then
      -- Reward the player who ate the edible.
      local avatarComponent = enteringGameObject:getComponent('Avatar')
      avatarComponent:addReward(self._config.rewardForEating)
      events:add('apple_consumed', 'dict',
                 'player_index', avatarComponent:getIndex())  -- int
      -- Change the edible to its wait (disabled) state.
      self.gameObject:setState(self._waitState)
    end
  end
end


-- Make acorns pickuppable.
local AcornPickUppable = class.Class(component.Component)

function AcornPickUppable:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AcornPickUppable')},
      {'liveState', args.stringType},
      {'waitState', args.stringType},
  })
  AcornPickUppable.Base.__init__(self, kwargs)

  self._config.liveState = kwargs.liveState
  self._config.waitState = kwargs.waitState
end

function AcornPickUppable:reset()
  self._waitState = self._config.waitState
  self._liveState = self._config.liveState
end

function AcornPickUppable:setWaitState(newWaitState)
  self._waitState = newWaitState
end

function AcornPickUppable:getWaitState()
  return self._waitState
end

function AcornPickUppable:setLiveState(newLiveState)
  self._liveState = newLiveState
end

function AcornPickUppable:getLiveState()
  return self._liveState
end

function AcornPickUppable:onEnter(enteringGameObject, contactName)
  if contactName == 'avatar' and enteringGameObject and
      enteringGameObject:hasComponent('Role') then
    local role = enteringGameObject:getComponent('Role')
    if self.gameObject:getState() == self._liveState and role:isPrey() then
      local avatarsInventory = enteringGameObject:getComponent(
          'InteractEatAcorn'):getAvatarsInventory()
      -- If prey avatar's inventory is empty, add acorn to inventory.
      if avatarsInventory:getHeldItem() == 'empty' then
        avatarsInventory:setHeldItem('acorn')
        local avatarComponent = enteringGameObject:getComponent('Avatar')
        events:add('acorn_collected', 'dict',
                   'player_index', avatarComponent:getIndex())  -- int
        -- Change the edible to its wait (disabled) state.
        self.gameObject:setState(self._waitState)
        -- Give the avatar a pseudoreward if applicable.
        if enteringGameObject:hasComponent('AcornTaste') then
          local pseudoreward = enteringGameObject:getComponent(
              'AcornTaste'):getAcornCollectionReward()
          avatarComponent:addReward(pseudoreward)
        end
      end
    end
  end
end


local allComponents = {
  -- Avatar components.
  Role = Role,
  Inventory = Inventory,
  PredatorInteractBeam = PredatorInteractBeam,
  InteractEatAcorn = InteractEatAcorn,
  AvatarEdible = AvatarEdible,
  AvatarRespawn = AvatarRespawn,
  AvatarAnimation = AvatarAnimation,
  AvatarEatingAnimation = AvatarEatingAnimation,
  -- Avatar components used to define pseudorewards for background bot training.
  ProxemicTaste = ProxemicTaste,
  AcornTaste = AcornTaste,
  -- Apple components.
  AppleEdible = AppleEdible,
  AcornPickUppable = AcornPickUppable,
}

component_registry.registerAllComponents(allComponents)

return allComponents
