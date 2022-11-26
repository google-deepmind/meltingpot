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
local log = require 'common.log'
local random = require 'system.random'
local tensor = require 'system.tensor'
local set = require 'common.set'
local events = require 'system.events'

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')


local MushroomEating = class.Class(component.Component)

function MushroomEating:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('MushroomEating')},
      {'liveStates', args.tableType},
      {'waitState', args.default('wait'), args.stringType},
      {'totalReward', args.tableType},
      {'numSporesReleasedWhenEaten', args.tableType},
      {'digestionTimes', args.tableType},
      {'destroyOnEating', args.tableType},
  })
  MushroomEating.Base.__init__(self, kwargs)
  self._totalReward = kwargs.totalReward
  self._numSporesReleasedWhenEaten = kwargs.numSporesReleasedWhenEaten
  self._waitState = kwargs.waitState
  self._liveStates = kwargs.liveStates
  self._liveStatesSet = set.Set(self._liveStates)
  self._waitState = kwargs.waitState
  self._digestionTimes = kwargs.digestionTimes
  self._destroyOnEating = kwargs.destroyOnEating
end

function MushroomEating:_getOthers(eatingPlayerIndex)
  local numPlayers = self.gameObject.simulation:getNumPlayers()
  local others = {}
  for idx = 1, numPlayers do
    if idx ~= eatingPlayerIndex then
      local avatarObject = self.gameObject.simulation:getAvatarFromIndex(idx)
      table.insert(others, avatarObject)
    end
  end
  return others
end

function MushroomEating:_rewardEveryone(playerIndex)
  local avatarWhoAteThis = self.gameObject.simulation:getAvatarFromIndex(
      playerIndex):getComponent('Avatar')
  local otherAvatars = self:_getOthers(playerIndex)
  local numPlayers = self.gameObject.simulation:getNumPlayers()

  local mushroomType = self.gameObject:getState()
  events:add('eating_mushroom', 'dict',
          'player_index', playerIndex,
          'mushroom_type', mushroomType)
  if mushroomType == 'fullInternalityZeroExternality' then
    -- Only reward self.
    avatarWhoAteThis:addReward(self._totalReward[mushroomType])
  elseif mushroomType == 'halfInternalityHalfExternality' then
    local partialReward = self._totalReward[mushroomType] / numPlayers
    -- Reward self.
    avatarWhoAteThis:addReward(partialReward)
    -- Reward others.
    for _, avatarObject in ipairs(otherAvatars) do
      local avatarComponent = avatarObject:getComponent('Avatar')
      avatarComponent:addReward(partialReward)
    end
  elseif mushroomType == 'zeroInternalityFullExternality' then
    -- Only reward others.
    local partialReward = self._totalReward[mushroomType] / (numPlayers - 1)
    for _, avatarObject in ipairs(otherAvatars) do
      local avatarComponent = avatarObject:getComponent('Avatar')
      avatarComponent:addReward(partialReward)
    end
  elseif mushroomType == 'negativeInternalityNegativeExternality' then
    -- Reward (or punish) everyone.
    local partialReward = self._totalReward[mushroomType] / numPlayers
    avatarWhoAteThis:addReward(partialReward)
    for _, avatarObject in ipairs(otherAvatars) do
      local avatarComponent = avatarObject:getComponent('Avatar')
      avatarComponent:addReward(partialReward)
    end
  else
    assert(false, 'Unrecognized mushroom type: ' .. mushroomType)
  end
end

function MushroomEating:onEnter(enteringGameObject, contactName)
  if contactName == 'avatar' then
    if self._liveStatesSet[self.gameObject:getState()] then
      local avatarComponent = enteringGameObject:getComponent('Avatar')
      local playerIndex = avatarComponent:getIndex()
      self:_rewardEveryone(playerIndex)
      -- Another mushroom grows.
      local mushroomRegrowth = self.gameObject.simulation:getSceneObject(
          ):getComponent('MushroomRegrowth')
      local mushroomType = self.gameObject:getState()
      for n = 1, self._numSporesReleasedWhenEaten[mushroomType] do
        mushroomRegrowth:grow(self.gameObject:getState())
      end
      -- Some mushrooms destroy other mushrooms when eaten.
      if rawget(self._destroyOnEating, mushroomType) then
        local mushroomTypeToDestroy = self._destroyOnEating[
            mushroomType].typeToDestroy
        local percentToDestroy = self._destroyOnEating[
            mushroomType].percentToDestroy
        mushroomRegrowth:destroyRandomMushrooms(mushroomTypeToDestroy,
                                                percentToDestroy)
      end
      -- Freeze the avatar who ate the mushroom while it digests.
      local timeToDigest = self._digestionTimes[self.gameObject:getState()]
      avatarComponent:disallowMovementUntil(timeToDigest)
      -- Change this object to its wait (disabled) state.
      self.gameObject:setState(self._waitState)
      -- Set the cumulant tracking that this mushroom was eaten.
      self:setCumulants(enteringGameObject)
    end
  end
end

function MushroomEating:setCumulants(enteringGameObject)
  local cumulantsComponent = enteringGameObject:getComponent('Cumulants')
  local mushroomType = self.gameObject:getState()
  if mushroomType == 'fullInternalityZeroExternality' then
    cumulantsComponent.ate_mushroom_fize = 1
  elseif mushroomType == 'halfInternalityHalfExternality' then
    cumulantsComponent.ate_mushroom_hihe = 1
  elseif mushroomType == 'zeroInternalityFullExternality' then
    cumulantsComponent.ate_mushroom_zife = 1
  elseif mushroomType == 'negativeInternalityNegativeExternality' then
    cumulantsComponent.ate_mushroom_nine = 1
  end
end


local MushroomGrowable = class.Class(component.Component)

function MushroomGrowable:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('MushroomGrowable')},
  })
  MushroomGrowable.Base.__init__(self, kwargs)
end

function MushroomGrowable:registerUpdaters(updaterRegistry)
  local regrowth = self.gameObject.simulation:getSceneObject():getComponent(
      'MushroomRegrowth')

  local registration = function()
    if self._mustRegister then
      regrowth:registerPotentialMushroom(self.gameObject:getPiece())
    elseif self._mustDeregister then
      regrowth:deregisterPotentialMushroom(self.gameObject:getPiece())
    end
    self._mustRegister = false
    self._mustDeregister = false
  end

  updaterRegistry:registerUpdater{
      updateFn = registration,
      priority = 500,
  }
end

function MushroomGrowable:onStateChange(oldState)
  local regrowth = self.gameObject.simulation:getSceneObject():getComponent(
      'MushroomRegrowth')
  local newState = self.gameObject:getState()
  if newState == 'wait' then
    self._mustRegister = true
  else
    self._mustDeregister = true
  end
end


local MushroomRegrowth = class.Class(component.Component)

function MushroomRegrowth:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('MushroomRegrowth')},
      {'mushroomsToProbabilities', args.tableType},
      {'minPotentialMushrooms', args.default(10), args.numberType},
  })
  MushroomRegrowth.Base.__init__(self, kwargs)
  self.mushroomsToProbabilities = kwargs.mushroomsToProbabilities
  self._minPotentialMushrooms = kwargs.minPotentialMushrooms

  self._potentialMushrooms = {}
end

function MushroomRegrowth:reset()
  self._potentialMushrooms = set.Set{}
  self._numPotentialMushrooms = 0
end

function MushroomRegrowth:grow(eatenMushroom)
  local growthProbabilities = self.mushroomsToProbabilities[eatenMushroom]
  for mushroom, probability in pairs(growthProbabilities) do
    if self._numPotentialMushrooms >= self._minPotentialMushrooms then
      if random:uniformReal(0, 1) < probability then
        local piece = random:choice(set.toSortedList(self._potentialMushrooms))
        if piece then
          local object = self.gameObject.simulation:getGameObjectFromPiece(
              piece)
          local avatarOrNil = object:getComponent('Transform'):queryPosition(
              'upperPhysical')
          if not avatarOrNil then
            -- Do not spawn mushrooms where avatars are currently standing.
            object:setState(mushroom)
          end
        end
      end
    end
  end
end

function MushroomRegrowth:destroyRandomMushrooms(mushroomType, percentToDestroy)
  local simulation = self.gameObject.simulation
  local shroomObjects = simulation:getGroupShuffledWithProbability(
    mushroomType, percentToDestroy)
  for _, object in ipairs(shroomObjects) do
    object:setState('wait')
  end
end

function MushroomRegrowth:registerPotentialMushroom(mushroomPiece)
  self._potentialMushrooms[mushroomPiece] = true
  self._numPotentialMushrooms = self._numPotentialMushrooms + 1
end

function MushroomRegrowth:deregisterPotentialMushroom(mushroomPiece)
  self._potentialMushrooms[mushroomPiece] = nil
  self._numPotentialMushrooms = self._numPotentialMushrooms - 1
end


local Destroyable = class.Class(component.Component)

function Destroyable:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Destroyable')},
      {'initialHealth', args.positive},
      {'waitState', args.stringType},
  })
  Destroyable.Base.__init__(self, kwargs)
  self._config.initialHealth = kwargs.initialHealth
  self._config.waitState = kwargs.waitState
end

function Destroyable:reset()
  self._variables = {}
  self._variables.health = self._config.initialHealth
end

function Destroyable:onHit(hitterGameObject, hitName)
  if hitName == 'zapHit' then
    self._variables.health = self._variables.health - 1
    if self._variables.health == 0 then
      -- Reset the health state variable.
      self._variables.health = self._config.initialHealth
      -- Remove the resource from the map.
      self.gameObject:setState(self._config.waitState)
      -- Set the cumulant tracking that this mushroom was eaten.
      self:setCumulants(hitterGameObject)
      -- Beams pass through a destroyed destroyable.
      return false
    end
    -- Beams do not pass through after hitting an undestroyed destroyable.
    return true
  end
end

function Destroyable:setCumulants(hitterGameObject)
  local cumulantsComponent = hitterGameObject:getComponent('Cumulants')
  local mushroomType = self.gameObject:getState()
  if mushroomType == 'fullInternalityZeroExternality' then
    cumulantsComponent.destroyed_mushroom_fize = 1
  elseif mushroomType == 'halfInternalityHalfExternality' then
    cumulantsComponent.destroyed_mushroom_hihe = 1
  elseif mushroomType == 'zeroInternalityFullExternality' then
    cumulantsComponent.destroyed_mushroom_zife = 1
  elseif mushroomType == 'negativeInternalityNegativeExternality' then
    cumulantsComponent.destroyed_mushroom_nine = 1
  end
end


local Perishable = class.Class(component.Component)

function Perishable:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Perishable')},
      {'waitState', args.stringType},
      {'delayPerState', args.tableType},
  })
  Perishable.Base.__init__(self, kwargs)
  self._config.waitState = kwargs.waitState
  self._delayPerState = kwargs.delayPerState
end

function Perishable:registerUpdaters(updaterRegistry)
  local perish = function()
    self.gameObject:setState(self._config.waitState)
  end

  for state, delay in pairs(self._delayPerState) do
    updaterRegistry:registerUpdater{
        updateFn = perish,
        priority = 3,
        startFrame = delay,
        state = state,
    }
  end
end


local Cumulants = class.Class(component.Component)

function Cumulants:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Cumulants')},
  })
  Cumulants.Base.__init__(self, kwargs)
end

function Cumulants:reset()
  self:_resetBinaryCumulants()
end

function Cumulants:_resetBinaryCumulants()
  self.ate_mushroom_fize = 0
  self.ate_mushroom_hihe = 0
  self.ate_mushroom_zife = 0
  self.ate_mushroom_nine = 0
  self.destroyed_mushroom_fize = 0
  self.destroyed_mushroom_hihe = 0
  self.destroyed_mushroom_zife = 0
  self.destroyed_mushroom_nine = 0
end

function Cumulants:registerUpdaters(updaterRegistry)
  local resetCumuluants = function()
    self:_resetBinaryCumulants()
  end

  updaterRegistry:registerUpdater{
      updateFn = resetCumuluants,
      priority = 900,
  }
end


local allComponents = {
  -- Game object components.
  MushroomEating = MushroomEating,
  MushroomGrowable = MushroomGrowable,
  Destroyable = Destroyable,
  Perishable = Perishable,
  -- Avatar components.
  Cumulants = Cumulants,
  -- Scene components.
  MushroomRegrowth = MushroomRegrowth,
}

component_registry.registerAllComponents(allComponents)

return allComponents
