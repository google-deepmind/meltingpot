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
local events = require 'system.events'
local random = require 'system.random'
local tensor = require 'system.tensor'

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')

local _COMPASS = {'N', 'E', 'S', 'W'}

local _DIRECTION = {
    N = tensor.Tensor({0, -1}),
    E = tensor.Tensor({1, 0}),
    S = tensor.Tensor({0, 1}),
    W = tensor.Tensor({-1, 0}),
}

local AllBeamBlocker = class.Class(component.Component)

function AllBeamBlocker:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AllBeamBlocker')},
  })
  AllBeamBlocker.Base.__init__(self, kwargs)
end

function AllBeamBlocker:onHit(hittingGameObject, hitName)
  -- no beams pass through.
  return true
end

local Resource = class.Class(component.Component)

function Resource:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Resource')},
      {'initialHealth', args.positive},
      {'destroyedState', args.stringType},
      {'reward', args.numberType},
      {'rewardRate', args.numberType},
      {'rewardDelay', args.numberType},
      {'delayTillSelfRepair', args.default(15), args.ge(0)},  -- frames
      {'selfRepairProbability', args.default(0.1), args.ge(0.0), args.le(1.0)},
  })
  Resource.Base.__init__(self, kwargs)

  self._config.initialHealth = kwargs.initialHealth
  self._config.destroyedState = kwargs.destroyedState
  self._config.reward = kwargs.reward
  self._config.rewardRate = kwargs.rewardRate
  self._config.rewardDelay = kwargs.rewardDelay
  self._config.delayTillSelfRepair = kwargs.delayTillSelfRepair
  self._config.selfRepairProbability = kwargs.selfRepairProbability
end

function Resource:reset()
  self._health = self._config.initialHealth
  self._rewardingStatus = 'inactive'
  self._claimedByAvatarComponent = nil
  self._neverYetClaimed = true
  self._destroyed = false
  self._framesSinceZapped = nil
end

function Resource:registerUpdaters(updaterRegistry)
  local provideRewards = function()
    if self.gameObject:getState() ~= self._config.destroyedState then
      if self._claimedByAvatarComponent.gameObject:hasComponent('Taste') then
        local avatarObject = self._claimedByAvatarComponent.gameObject
        local tasteComponent = avatarObject:getComponent('Taste')
        tasteComponent:addDefaultReward(self._config.reward)
      else
        self._claimedByAvatarComponent:addReward(self._config.reward)
      end
      self._rewardingStatus = 'active'
    end
  end
  updaterRegistry:registerUpdater{
      updateFn = provideRewards,
      group = 'claimedResources',
      probability = self._config.rewardRate,
      startFrame = self._config.rewardDelay,
  }
  local function releaseClaimOfDeadAgent()
    if self._claimedByAvatarComponent:isWait() and not self._destroyed then
      local stateManager = self.gameObject:getComponent('StateManager')
      self.gameObject:setState(stateManager:getInitialState())
      self._rewardingStatus = 'inactive'
      self._claimedByAvatarComponent = nil
    end
  end
  updaterRegistry:registerUpdater{
      updateFn = releaseClaimOfDeadAgent,
      group = 'claimedResources',
      priority = 2,
      startFrame = 5,
  }
end

function Resource:_claim(hittingGameObject)
  self._claimedByAvatarComponent = hittingGameObject:getComponent('Avatar')
  local claimedByIndex = self._claimedByAvatarComponent:getIndex()
  local claimedName = 'claimed_by_' .. tostring(claimedByIndex)
  if self.gameObject:getState() ~= claimedName and not self._destroyed then
    self.gameObject:setState(claimedName)
    self._rewardingStatus = 'inactive'
    -- If player has a role that gets rewarded for claiming, apply that reward.
    if hittingGameObject:hasComponent('Taste') then
      hittingGameObject:getComponent('Taste'):addRewardIfApplicable(
        self._neverYetClaimed)
    end
    self._neverYetClaimed = false
    -- Report the claiming event.
    events:add('claimed_resource', 'dict',
               'player_index', claimedByIndex)  -- int
  end
end

function Resource:onHit(hittingGameObject, hitName)
  if string.sub(hitName, 1, string.len('directionHit')) == 'directionHit' then
    self:_claim(hittingGameObject)
  end

  for i = 1, self._numPlayers do
    local beamName = 'claimBeam_' .. tostring(i)
    if hitName == beamName then
      self:_claim(hittingGameObject)
      -- Claims pass through resources.
      return false
    end
  end

  if hitName == 'zapHit' then
    self._health = self._health - 1
    self._framesSinceZapped = 0
    if self._health == 0 then
      -- Reset the health state variable.
      self._health = self._config.initialHealth
      -- Remove the resource from the map.
      self.gameObject:setState(self._config.destroyedState)
      -- Tell the reward indicator the resource was destroyed.
      self._rewardingStatus = 'inactive'
      -- Destroy the resource's associated texture objects.
      self._texture_object:setState('destroyed')
      -- Tell the resource's associated damage indicator.
      self._associatedDamageIndicator:setState('inactive')
      -- Record the destruction event.
      local playerIndex = hittingGameObject:getComponent('Avatar'):getIndex()
      events:add('destroyed_resource', 'dict',
                 'player_index', playerIndex)  -- int
      self._destroyed = true
      -- Zaps pass through a destroyed resource.
      return false
    end
    -- Zaps do not pass through after hitting an undestroyed resource.
    return true
  end

  -- Other beams (if any exist) pass through.
  return false
end

function Resource:start()
  self._numPlayers = self.gameObject.simulation:getNumPlayers()
end

function Resource:postStart()
  self._texture_object = self.gameObject:getComponent(
      'Transform'):queryPosition('lowerPhysical')
  self._associatedDamageIndicator = self.gameObject:getComponent(
      'Transform'):queryPosition('superDirectionIndicatorLayer')
end

function Resource:update()
  if self._health < self._config.initialHealth then
    self._associatedDamageIndicator:setState('damaged')
    if self._framesSinceZapped >= self._config.delayTillSelfRepair then
      if random:uniformReal(0, 1) < self._config.selfRepairProbability then
        self._health = self._health + 1
        if self._health == self._config.initialHealth then
          self._associatedDamageIndicator:setState('inactive')
        end
      end
    end
    self._framesSinceZapped = self._framesSinceZapped + 1
  end
end

function Resource:getRewardingStatus()
  return self._rewardingStatus
end



local ResourceClaimer = class.Class(component.Component)

function ResourceClaimer:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('ResourceClaimer')},
      {'playerIndex', args.numberType},
      {'beamLength', args.numberType},
      {'beamRadius', args.numberType},
      {'beamWait', args.numberType},
      {'color', args.tableType},
  })
  ResourceClaimer.Base.__init__(self, kwargs)

  self._kwargs = kwargs
  self._config.beamLength = kwargs.beamLength
  self._config.beamRadius = kwargs.beamRadius
  self._config.beamWait = kwargs.beamWait
  self._config.beamColor = kwargs.color

  -- Each player makes claims for their own dedicated resource.
  self._playerIndex = kwargs.playerIndex
  self._claimBeamName = 'claimBeam_' .. tostring(self._playerIndex)
  self._beamSpriteName = 'claimBeamSprite_' .. tostring(self._playerIndex)
end

function ResourceClaimer:reset()
  self._cooldown = 0
end

function ResourceClaimer:addSprites(tileSet)
  tileSet:addColor(self._beamSpriteName, self._config.beamColor)
end

function ResourceClaimer:addHits(worldConfig)
  worldConfig.hits[self._claimBeamName] = {
      layer = 'superDirectionIndicatorLayer',
      sprite = self._beamSpriteName,
  }
end

function ResourceClaimer:registerUpdaters(updaterRegistry)
  local claim = function()
    local playerVolatileVariables = (
        self.gameObject:getComponent('Avatar'):getVolatileData())
    local actions = playerVolatileVariables.actions
    if self._config.beamWait >= 0 then
      if self._cooldown > 0 then
        self._cooldown = self._cooldown - 1
      else
        if actions['fireClaim'] == 1 then
          self._cooldown = self._config.beamWait
          self.gameObject:hitBeam(self._claimBeamName,
                                  self._config.beamLength,
                                  self._config.beamRadius)
        end
      end
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = claim,
  }
end


local RewardIndicator = class.Class(component.Component)

function RewardIndicator:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('RewardIndicator')},
  })
  RewardIndicator.Base.__init__(self, kwargs)
end

function RewardIndicator:reset()
  self._pairedResource = nil
end

function RewardIndicator:postStart()
  local transformComponent = self.gameObject:getComponent('Transform')
  local objectToPair = transformComponent:queryPosition('upperPhysical')

  assert(objectToPair, 'Nothing to pair at position ' ..
         helpers.tostring(self.gameObject:getPosition()))
  assert(objectToPair:hasComponent('Resource'),
         'Object to pair is not a resource.')

  self._pairedResource = objectToPair
end

function RewardIndicator:update()
  local resourceComponent = self._pairedResource:getComponent('Resource')
  local newStatus = resourceComponent:getRewardingStatus()
  local resourceState = resourceComponent.gameObject:getState()
  if newStatus == "active" then
    self.gameObject:setState("dry_" .. resourceState)
  else
    self.gameObject:setState("inactive")
  end
end


local Taste = class.Class(component.Component)

function Taste:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Taste')},
      -- Choose `role` = `none` to select the default game behavior.
      {'role', args.default('none'), args.oneOf('none',
                                                'rewarded_per_claim',
                                                'rewarded_per_claim_only')},
      -- Amount of reward to deliver to applicable roles for claiming resources.
      {'rewardAmount', args.default(0), args.numberType},
      -- Multiply the claim reward for applicable roles by this value when
      -- claiming a resource that has not yet been claimed by anyone.
      {'firstClaimRewardMultiplier', args.default(1.0), args.numberType},
  })
  Taste.Base.__init__(self, kwargs)
  self._role = kwargs.role
  self._rewardAmount = kwargs.rewardAmount
  self._firstClaimRewardMultiplier = kwargs.firstClaimRewardMultiplier
end

function Taste:addRewardIfApplicable(neverYetClaimed)
  if self._role == 'rewarded_per_claim' or
      self._role == 'rewarded_per_claim_only' then
    local rewardAmount = self._rewardAmount
    if neverYetClaimed then
      rewardAmount = rewardAmount * self._firstClaimRewardMultiplier
    end
    self.gameObject:getComponent('Avatar'):addReward(rewardAmount)
  end
end

function Taste:addDefaultReward(defaultReward)
  if self._role == 'rewarded_per_claim_only' then
    -- Agents with this role do not get the game's usual reward.
    self.gameObject:getComponent('Avatar'):addReward(0.0)
  else
    -- Default condition when no special role was applied.
    self.gameObject:getComponent('Avatar'):addReward(defaultReward)
  end
end


-- The `Paintbrush` component endows an avatar with the ability to grasp an
-- object in the direction they are facing.

local Paintbrush = class.Class(component.Component)

function Paintbrush:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Paintbrush')},
      {'shape', args.tableType},
      {'palette', args.tableType},
      {'playerIndex', args.numberType},
  })
  Paintbrush.Base.__init__(self, kwargs)
  self._config.shape = kwargs.shape
  self._config.palette = kwargs.palette
  self._config.playerIndex = kwargs.playerIndex
end

function Paintbrush:addSprites(tileSet)
  for j=1, 4 do
    local spriteData = {
      palette = self._config.palette,
      text = self._config.shape[j],
      noRotate = true
    }
    tileSet:addShape(
      'brush' .. self._config.playerIndex .. '.' .. _COMPASS[j], spriteData)
  end
end

function Paintbrush:addHits(worldConfig)
  local playerIndex = self._config.playerIndex
  for j=1, 4 do
    local hitName = 'directionHit' .. playerIndex
    worldConfig.hits[hitName] = {
        layer = 'directionIndicatorLayer',
        sprite = 'brush' .. self._config.playerIndex,
  }
  end
end


function Paintbrush:registerUpdaters(updaterRegistry)
  local playerIndex = self._config.playerIndex
  self._avatar = self.gameObject:getComponent('Avatar')
  local drawBrush = function()
    local beam = 'directionHit' .. playerIndex
    self.gameObject:hitBeam(beam, 1, 0)
  end
  updaterRegistry:registerUpdater{
      updateFn = drawBrush,
      priority = 130,
  }
end


local Destructable = class.Class(component.Component)

function Destructable:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Destructable')},
  })
  Destructable.Base.__init__(self, kwargs)
end

function Destructable:onHit(hittingGameObject, hitName)
  if hitName == 'zapHit' then
    self.gameObject:setState('destroyed')
  end
  return false
end

local allComponents = {
    -- Non-avatar components.
    AllBeamBlocker = AllBeamBlocker,
    Resource = Resource,
    RewardIndicator = RewardIndicator,
    Paintbrush = Paintbrush,
    Destructable = Destructable,

    -- Avatar components.
    ResourceClaimer = ResourceClaimer,
    Taste = Taste,
}

component_registry.registerAllComponents(allComponents)

return allComponents
