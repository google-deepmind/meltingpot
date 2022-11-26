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
local set = require 'common.set'
local events = require 'system.events'
local random = require 'system.random'
local tensor = require 'system.tensor'

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')


local _DIRECTION = {
    N = tensor.Tensor({0, -1}),
    E = tensor.Tensor({1, 0}),
    S = tensor.Tensor({0, 1}),
    W = tensor.Tensor({-1, 0}),
}


local Stamina = class.Class(component.Component)

function Stamina:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Stamina')},
      {'maxStamina', args.positive},
      {'classConfig', args.tableType},
      {'amountInvisible', args.ge(0)},
      {'amountRed', args.ge(0)},
      {'amountYellow', args.ge(0)},
      {'amountGreen', args.ge(0)},
      -- `costlyActions` table of actions affecting stamina eg {'move', 'turn'}.
      {'costlyActions', args.tableType},
  })
  Stamina.Base.__init__(self, kwargs)

  self._config.maxStamina = kwargs.maxStamina
  self._config.classConfig = self:parseClass(kwargs.classConfig)
  self._config.amountInvisible = kwargs.amountInvisible
  self._config.amountRed = kwargs.amountRed
  self._config.amountYellow = kwargs.amountYellow
  self._config.amountGreen = kwargs.amountGreen
  local sum = (self._config.amountInvisible +
               self._config.amountRed +
               self._config.amountYellow +
               self._config.amountGreen - 1)
  assert(sum == self._config.maxStamina,
         "Color amounts must sum to max stamina but " .. sum .. " != "
                                                    .. self._config.maxStamina)

  self._config.costlyActions = kwargs.costlyActions
end

function Stamina:parseClass(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.stringType},
      {'greenFreezeTime', args.ge(0)},
      {'yellowFreezeTime', args.ge(0)},
      {'redFreezeTime', args.ge(0)},
      {'decrementRate', args.default(1), args.gt(0), args.le(1)},
  })
  classConfig = {}
  classConfig.greenFreezeTime = kwargs.greenFreezeTime
  classConfig.yellowFreezeTime = kwargs.yellowFreezeTime
  classConfig.redFreezeTime = kwargs.redFreezeTime
  classConfig.decrementRate = kwargs.decrementRate

  classConfig.decrementInterval = 1.0 / classConfig.decrementRate
  return classConfig
end

function Stamina:reset()
  self._value = self._config.maxStamina
  self._lastAction = nil
  self._frozenFramesRemaining = 0
  self._allowRecovery = true
  self._costlyFramesSinceLastDelta = 0
end

--[[ Return the current stamina band. This function will always return one of
{'green', 'yellow', 'red', 'invisible'}.

`invisible` indicates the highest stamina values.
`green` indicates high stamina.
`yellow` indicates a stamina level in between green and red.
`red` indicates low stamina.
]]
function Stamina:getBand()
  local x = self._config.amountInvisible
  local r = self._config.amountRed
  local y = self._config.amountYellow
  local g = self._config.amountGreen

  if self._value >= 0 and self._value < r then
    return 'red'
  elseif self._value >= r and self._value < r + y then
    return 'yellow'
  elseif self._value >= r + y and self._value < r + y + g then
    return 'green'
  elseif self._value >= r + y + g then
    return 'invisible'
  end
end

function Stamina:registerUpdaters(updaterRegistry)
  local avatar = self.gameObject:getComponent('Avatar')

  local function updateStamina()
    if self._frozenFramesRemaining > 0 then
      return
    end
    -- If not frozen then do the following.
    -- First, check if a costly action was taken and decrement stamina if so.
    local playerVolatileVariables = (
        self.gameObject:getComponent('Avatar'):getVolatileData())
    local actions = playerVolatileVariables.actions
    self._lastAction = nil
    for _, costly_action_name in pairs(self._config.costlyActions) do
      if actions[costly_action_name] ~= 0 then
        self._costlyFramesSinceLastDelta  = self._costlyFramesSinceLastDelta + 1
        local decrementInterval = self._config.classConfig.decrementInterval
        if self._costlyFramesSinceLastDelta == decrementInterval then
          local newStamina = self._value - 1
          self._value = (newStamina > 0) and newStamina or 0
          self._costlyFramesSinceLastDelta = 0
        end
        self._lastAction = 'costly'
      end
    end
    -- Next, check if agent didn't do anything on the last step. If so, then
    -- increase stamina (it regenerates overtime while the avatar rests).
    if self._lastAction == nil and self._allowRecovery then
      local newStamina = self._value + 1
      self._value = math.min(newStamina, self._config.maxStamina)
    end
    self._lastAction = self._lastAction or 'not_costly'
  end

  local function applyStamina()
    if self._frozenFramesRemaining > 0 then
      self._frozenFramesRemaining = self._frozenFramesRemaining - 1
      return
    end
    -- If not already frozen then do the following.
    local greenFreezeTime = self._config.classConfig.greenFreezeTime
    local yellowFreezeTime = self._config.classConfig.yellowFreezeTime
    local redFreezeTime = self._config.classConfig.redFreezeTime
    local band = self:getBand()
    if band == 'invisible' or band == 'green' then
      if self._lastAction == 'costly' and greenFreezeTime > 0 then
        avatar:disallowMovementUntil(greenFreezeTime)
        self._frozenFramesRemaining = greenFreezeTime
      end
    elseif band == 'yellow' then
      if self._lastAction == 'costly' and yellowFreezeTime > 0 then
        avatar:disallowMovementUntil(yellowFreezeTime)
        self._frozenFramesRemaining = yellowFreezeTime
      end
    elseif band == 'red' then
      if self._lastAction == 'costly' and redFreezeTime > 0 then
        avatar:disallowMovementUntil(redFreezeTime)
        self._frozenFramesRemaining = redFreezeTime
      end
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = updateStamina,
      priority = 4,
  }
  updaterRegistry:registerUpdater{
      updateFn = applyStamina,
      priority = 200,
  }
end

function Stamina:getValue()
  return self._value
end

function Stamina:addValue(value)
  if value < 0 then
    self._value = math.max(self._value + value, 0)
  else
    self._value = math.min(self._value + value, self._config.maxStamina)
  end
end

function Stamina:startPreventingRecovery()
  self._allowRecovery = false
end

function Stamina:stopPreventingRecovery()
  self._allowRecovery = true
end

function Stamina:onStateChange(oldState)
  local avatar = self.gameObject:getComponent('Avatar')
  local waitState = avatar:getWaitState()
  if oldState == waitState and avatar:isAlive() then
    -- Respawning.
    self:reset()
  end
end

--[[ Return the current stamina amount as a number between 0 and 1.

Returns 1 when value = `maxStamina` and returns 0 when value == 0.
]]
function Stamina:getNormalizedValue()
  if self.gameObject:getComponent('Avatar'):isAlive() then
    return self._value / self._config.maxStamina
  else
    -- Also show zero stamina while dead.
    return 0
  end
end


local StaminaModulatedByNeed = class.Class(component.Component)

function StaminaModulatedByNeed:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('StaminaModulatedByNeed')},
      {'needComponent', args.default('PeriodicNeed'), args.stringType},
      -- Decrease stamina by `lossPerStepBeyondThreshold` on every step that the
      -- threshold defined in `needComponent` has been exceeded.
      {'lossPerStepBeyondThreshold', args.default(1), args.ge(0)},
  })
  StaminaModulatedByNeed.Base.__init__(self, kwargs)
  self._config.needComponent = kwargs.needComponent
  self._config.lossPerStepBeyondThreshold = kwargs.lossPerStepBeyondThreshold
end

function StaminaModulatedByNeed:registerUpdaters(updaterRegistry)
  local needComponent = self.gameObject:getComponent(self._config.needComponent)
  local staminaComponent = self.gameObject:getComponent('Stamina')

  local function updateStaminaAccordingToNeedLevel()
    local needValue = needComponent:getNeed()
    -- Reduce stamina by 1 on every timestep that need is at or above threshold.
    if needValue >= 1 then
      staminaComponent:addValue(-self._config.lossPerStepBeyondThreshold)
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = updateStaminaAccordingToNeedLevel,
      priority = 5,
  }
end


--[[ The `StaminaObservation` component adds an observation that is 1 when
the avatar can fire (from the Zapper component) and <1 if in cooldown time.

The resulting observation key will be `playerIndex`.READY_TO_SHOOT.
]]
local StaminaObservation = class.Class(component.Component)

function StaminaObservation:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('StaminaObservation')},
      {'staminaComponent', args.default('Stamina'), args.stringType},
  })
  StaminaObservation.Base.__init__(self, kwargs)
  self._config.staminaComponent = kwargs.staminaComponent
end

function StaminaObservation:addObservations(tileSet, world, observations)
  local playerIndex = self.gameObject:getComponent('Avatar'):getIndex()
  local stamina = self.gameObject:getComponent(self._config.staminaComponent)

  observations[#observations + 1] = {
      name = tostring(playerIndex) .. '.STAMINA',
      type = 'Doubles',
      shape = {},
      func = function(grid)
        return stamina:getNormalizedValue()
      end
  }
end


--[[ Optionally send a shaping reward when stamina falls into specified bands.

The `red` band  indicates stamina is lowest.
The `yellow` band indicates stamina is just above the lowest.
The `green` band indicates stamina is relativelt high.
the `invisible` band indicates stamina is at its highest.

This is off by default. It can be used to train background populations that
rapidly learn to avoid letting their stamina drop too low.
]]
local RewardForStaminaLevel = class.Class(component.Component)

function RewardForStaminaLevel:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('RewardForStaminaLevel')},
      -- Selecting `rewardValue` of 0.0 makes this component do nothing.
      {'rewardValue', args.default(0.0), args.numberType},
      -- Which stamina bands to reward.
      {'bands', args.default({'red'}), args.tableType},
  })
  RewardForStaminaLevel.Base.__init__(self, kwargs)
  self._config.rewardValue = kwargs.rewardValue
  self._config.setOfBandsToReward = set.Set(kwargs.bands)
end

function RewardForStaminaLevel:registerUpdaters(updaterRegistry)
  local stamina = self.gameObject:getComponent('Stamina')
  local avatar = self.gameObject:getComponent('Avatar')

  local function sendRewardIfStaminaInRedBand()
    if self._config.setOfBandsToReward[stamina:getBand()] then
      avatar:addReward(self._config.rewardValue)
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = sendRewardIfStaminaInRedBand,
      priority = 2,
  }
end


local StaminaBar = class.Class(component.Component)

function StaminaBar:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('StaminaBar')},
      {'playerIndex', args.ge(0)},
      {'waitState', args.stringType},
      {'layer', args.stringType},
      {'direction', args.oneOf('N', 'E', 'S', 'W')}
  })
  StaminaBar.Base.__init__(self, kwargs)

  self._playerIndex = kwargs.playerIndex
  self._waitState = kwargs.waitState
  self._layer = kwargs.layer
  self._direction = kwargs.direction
end

function StaminaBar:addHits(worldConfig)
  -- Each avatar has a unique layer for its stamina bar. Ensure it is added to
  -- the back of the render order.
  component.insertIfNotPresent(worldConfig.renderOrder, self._layer)
end

function StaminaBar:_setLevel(level)
  self.gameObject:setState('level_' .. tostring(level))
end

function StaminaBar:registerUpdaters(updaterRegistry)
  local function setBarState()
    if self.gameObject:getState() ~= self._waitState then
      local stamina = self._staminaComponent:getValue()
      self:_setLevel(stamina)
    end
  end

  local function rotate()
    self.gameObject:setOrientation(self._direction)
  end

  updaterRegistry:registerUpdater{
      updateFn = setBarState,
      -- The stamina bar should be the very last thing to update, after all
      -- effects that could potentially change stamina levels have resolved.
      priority = 2,
  }
  updaterRegistry:registerUpdater{
      updateFn = rotate,
      priority = 100,
  }
end

function StaminaBar:postStart()
  local sim = self.gameObject.simulation
  self._avatarObject = sim:getAvatarFromIndex(self._playerIndex)
  local avatarComponent = self._avatarObject:getComponent('Avatar')
  self._staminaComponent = self._avatarObject:getComponent('Stamina')

  -- Note that it is essential to set the state before teleporting it.
  -- This is because pieces with no assigned layer have no position, and thus
  -- cannot be teleported.
  self:_setLevel(self._staminaComponent:getValue())
  local avatarPosition = tensor.Tensor(self._avatarObject:getPosition())
  local offsetPosition = avatarPosition:cadd(_DIRECTION[self._direction]):val()
  self.gameObject:teleport(offsetPosition, self._direction)

  -- Connect this object to the avatar game object.
  avatarComponent:connect(self.gameObject)end

function StaminaBar:avatarStateChange(behavior)
  local avatarComponent = self._avatarObject:getComponent('Avatar')
  -- If the avatar's state has changed, then also update the state of
  -- the avatar connector.
  if behavior == 'respawn' then
    avatarComponent:disconnect(self.gameObject)
    -- Set the respawning player's stamina level.
    self:_setLevel(self._staminaComponent:getValue())
    -- When coming to life, also teleport to the right location.
    local avatarPosition = tensor.Tensor(self._avatarObject:getPosition())
    local offsetPosition = avatarPosition:cadd(_DIRECTION[self._direction]):val()
    self.gameObject:teleport(offsetPosition, self._direction)
    avatarComponent:connect(self.gameObject)
  elseif behavior == 'die' then
    self.gameObject:setState(self._waitState)
  end
end


local allComponents = {
    -- Avatar components.
    Stamina = Stamina,
    StaminaModulatedByNeed = StaminaModulatedByNeed,
    StaminaObservation = StaminaObservation,
    RewardForStaminaLevel = RewardForStaminaLevel,
    -- Overlay object components.
    StaminaBar = StaminaBar,
}

component_registry.registerAllComponents(allComponents)

return allComponents
