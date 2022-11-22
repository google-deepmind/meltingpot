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
local tensor = require 'system.tensor'

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')

local _HOLD_PRIORITY = 140
local _SHOVE_PRIORITY = 135

local _COMPASS = {'N', 'E', 'S', 'W'}
local _OPPOSITECOMPASS = {N = 'S', E = 'W', S = 'N', W = 'E'}

local Grappling = class.Class(component.Component)

function Grappling:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Grappling')},
      {'shape', args.tableType},
      {'palette', args.tableType},
      {'liveState', args.stringType},
      {'grappledState', args.stringType},
      {'grapplingState', args.stringType},
  })
  Grappling.Base.__init__(self, kwargs)
  self._config.shape = kwargs.shape
  self._config.palette = kwargs.palette
  self._config.liveState = kwargs.liveState
  self._config.grappledState = kwargs.grappledState
  self._config.grapplingState = kwargs.grapplingState
end

function Grappling:reset()
  self._chanceToHold = true
  self._chanceToShove = true
  self._heldBy = nil
  self._freezeCounter = 0
  self._avatar = self.gameObject:getComponent('Avatar')
end

function Grappling:addHits(worldConfig)
  -- Add the hold beam layer.
  component.insertIfNotPresent(worldConfig.renderOrder, 'holdBeamLayer')
  worldConfig.hits['hold'] = {
      layer = 'holdBeamLayer',
      sprite = 'holdBeam',
  }
  -- Add the shove beam layer on top.
  component.insertIfNotPresent(worldConfig.renderOrder, 'shoveBeamLayer')
  worldConfig.hits['shove'] = {
      layer = 'shoveBeamLayer',
      sprite = 'ShoveBeam',
  }
  worldConfig.hits['pull'] = {
      layer = 'shoveBeamLayer',
      sprite = 'ShoveBeam',
  }
end

function Grappling:addSprites(tileSet)
  for j=1, 4 do
    local spriteData = {
      palette = self._config.palette,
      text = self._config.shape[j],
      noRotate = true
    }
    tileSet:addShape(
      'holdBeam' .. '.' .. _COMPASS[j], spriteData)
  end
end

function Grappling:registerUpdaters(updaterRegistry)
  local function hold()
    self._heldBy = nil
    local playerVolatileVariables = self._avatar:getVolatileData()
    local holdAction = playerVolatileVariables.actions['hold'] == 1
    if holdAction and self._chanceToHold then
      self.gameObject:hitBeam('hold', 2, 0)
      self.gameObject:setState(self._config.grapplingState)
    end
  end
  updaterRegistry:registerUpdater{
      updateFn = hold,
      priority = _HOLD_PRIORITY ,
  }

  local function shove()
    local playerVolatileVariables = self._avatar:getVolatileData()
    local shoveAction = playerVolatileVariables.actions['shove'] == 1
    if shoveAction and self._chanceToShove then
      self.gameObject:hitBeam('shove', 2, 0)
      self.gameObject:setState(self._config.grapplingState)
    end
    local shoveAction = playerVolatileVariables.actions['shove'] == -1
    if shoveAction and self._chanceToShove then
      self.gameObject:hitBeam('pull', 2, 0)
      self.gameObject:setState(self._config.grapplingState)
    end
  end
  updaterRegistry:registerUpdater{
      updateFn = shove,
      priority = _SHOVE_PRIORITY,
  }
end

function Grappling:onHit(hittingGameObject, hitName)
  if hitName == 'hold' then
    local attackingAvatar = hittingGameObject:getComponent('Avatar')
    -- Both grapplers are stuck in one place till done grappling.
    self._avatar:disallowMovementUntil(2)
    self:disallowGrapplingUntil(2)
    self.gameObject:setState(self._config.grappledState)
    attackingAvatar:disallowMovementUntil(2)
    self._heldBy = attackingAvatar:getIndex()
    -- return `true` to prevent the beam from passing through a hit player.
    return true
  end

  -- Must already be held by an avatar in order to be shoved by them.
  if hitName == 'shove' and self._heldBy then
    local attacker = self.gameObject.simulation:getAvatarFromIndex(self._heldBy)
    local attackerOrientation = attacker:getOrientation()
    self.gameObject:moveAbs(attackerOrientation)
    -- return `true` to prevent the beam from passing through a hit player.
    return true
  end
  if hitName == 'pull' and self._heldBy then
    local attacker = self.gameObject.simulation:getAvatarFromIndex(self._heldBy)
    local attackerOrientation = attacker:getOrientation()
    local moveDirection = _OPPOSITECOMPASS[attackerOrientation]
    self.gameObject:moveAbs(moveDirection)
    -- return `true` to prevent the beam from passing through a hit player.
    return true
  end
end

--[[ Prevent an avatar from grappling (using hold and push actions).]]
function Grappling:disallowGrappling()
  self._chanceToHold = false
  self._chanceToShove = false
end

--[[ No need to call `allowGrappling` unless after calling `disallowGrappling`.
]]
function Grappling:allowGrappling()
  self._chanceToHold = true
  self._chanceToShove = true
end

--[[ Prevent grappling actions for `numFrames` steps, then allow them again.]]
function Grappling:disallowGrapplingUntil(numFrames)
  self:disallowGrappling()
  self._freezeCounter = numFrames
end

function Grappling:_handleTimedFreeze()
  local oldFreezeCounter = self._freezeCounter
  if oldFreezeCounter == 1 then
    self.gameObject:setState(self._config.liveState)
    self:allowGrappling()
  end
  self._freezeCounter = math.max(self._freezeCounter - 1, 0)
end

function Grappling:update()
  self:_handleTimedFreeze()
  if self.gameObject:getState() == self._config.grapplingState then
    self.gameObject:setState(self._config.liveState)
  end
end

function Grappling:readyToShoot()
  if self._chanceToHold then
    return 1.0
  else
    return 0.0
  end
end


local allComponents = {
  Grappling = Grappling,
}

-- Register all components from this module in the component registry.
component_registry.registerAllComponents(allComponents)

