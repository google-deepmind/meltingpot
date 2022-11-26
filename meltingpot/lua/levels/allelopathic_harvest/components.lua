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

-- For Lua 5.2 compatibility.
local unpack = unpack or table.unpack


local function vectorMax(vector)
  return math.max(unpack(vector:val()))
end


local Berry = class.Class(component.Component)

function Berry:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Berry')},
      {'colorId', args.numberType},
      {'unripePrefix', args.default('unripe'), args.stringType},
      {'ripePrefix', args.default('unripe'), args.stringType},
  })
  Berry.Base.__init__(self, kwargs)

  self._kwargs = kwargs
end

function Berry:reset()
  self._variables = {}
  -- `unripe` and `ripe` states are state variables since agents can color
  -- berries dynamically. Thus they must be set back to their initial value from
  -- their kwargs on calls to `initVariables` (which is itself called by
  -- `reset`).
  self._variables.unripeState = (
      self._kwargs.unripePrefix .. '_' .. tostring(self._kwargs.colorId))
  self._variables.ripeState = (
      self._kwargs.ripePrefix .. '_' .. tostring(self._kwargs.colorId))
  self._variables.colorId = self._kwargs.colorId
end

function Berry:isRipe()
  return self.gameObject:getState() == self._variables.ripeState
end

function Berry:getUnripeState()
  return self._variables.unripeState
end

function Berry:getRipeState()
  return self._variables.ripeState
end

function Berry:setBerryType(colorId)
  local previousBerryId = self._variables.colorId
  self._variables.colorId = colorId
  self._variables.unripeState = (
      self._kwargs.unripePrefix .. '_' .. tostring(colorId))
  self._variables.ripeState = (
      self._kwargs.ripePrefix .. '_' .. tostring(colorId))
  -- Update global ripe/unripe berry counters.
  local ripeBerriesPerType, unripeBerriesPerType = self:_getBerriesPerType()
  if self:isRipe() then
    self:_incrementRipeBerries(ripeBerriesPerType)
    self:_decrementRipeBerries(ripeBerriesPerType, previousBerryId)
  else
    self:_incrementUnripeBerries(unripeBerriesPerType)
    self:_decrementUnripeBerries(unripeBerriesPerType, previousBerryId)
  end
end

function Berry:ripen()
  assert(self.gameObject:getState() == self._variables.unripeState,
        'Error: cannot ripen because current state is not unripe.')
  self.gameObject:setState(self._variables.ripeState)
  -- Update global ripe/unripe state.
  local ripeBerriesPerType, unripeBerriesPerType = self:_getBerriesPerType()
  self:_incrementRipeBerries(ripeBerriesPerType)
  self:_decrementUnripeBerries(unripeBerriesPerType)
end

function Berry:unripen()
  assert(self.gameObject:getState() == self._variables.ripeState,
        'Error: cannot unripen because current state is not ripe.')
  self.gameObject:setState(self._variables.unripeState)
  -- Update global ripe/unripe state.
  local ripeBerriesPerType, unripeBerriesPerType = self:_getBerriesPerType()
  self:_decrementRipeBerries(ripeBerriesPerType)
  self:_incrementUnripeBerries(unripeBerriesPerType)
end

function Berry:start()
  local ripeBerriesPerType, unripeBerriesPerType = self:_getBerriesPerType()
  -- All berries start out unripe.
  self:_incrementUnripeBerries(unripeBerriesPerType)
end

function Berry:getBerryColorId()
  return self._variables.colorId
end

function Berry:_getBerriesPerType()
  local sceneObject = self.gameObject.simulation:getSceneObject()
  local globalBerryTracker = sceneObject:getComponent('GlobalBerryTracker')
  local ripeBerriesPerType = globalBerryTracker:getRipeBerriesPerType()
  local unripeBerriesPerType = globalBerryTracker:getUnripeBerriesPerType()
  return ripeBerriesPerType, unripeBerriesPerType
end

function Berry:_incrementRipeBerries(ripeBerriesPerType, optionalBerryId)
  local colorId = optionalBerryId or self:getBerryColorId()
  local ripeBerriesThisType = ripeBerriesPerType:val()[colorId]
  ripeBerriesPerType(colorId):fill(ripeBerriesThisType + 1)
end

function Berry:_decrementRipeBerries(ripeBerriesPerType, optionalBerryId)
  local colorId = optionalBerryId or self:getBerryColorId()
  local ripeBerriesThisType = ripeBerriesPerType:val()[colorId]
  ripeBerriesPerType(colorId):fill(ripeBerriesThisType - 1)
end

function Berry:_incrementUnripeBerries(unripeBerriesPerType, optionalBerryId)
  local colorId = optionalBerryId or self:getBerryColorId()
  local unripeBerriesThisType = unripeBerriesPerType:val()[colorId]
  unripeBerriesPerType(colorId):fill(unripeBerriesThisType + 1)
end

function Berry:_decrementUnripeBerries(unripeBerriesPerType, optionalBerryId)
 local colorId = optionalBerryId or self:getBerryColorId()
  local unripeBerriesThisType = unripeBerriesPerType:val()[colorId]
  unripeBerriesPerType(colorId):fill(unripeBerriesThisType - 1)
end


local Edible = class.Class(component.Component)

function Edible:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Edible')},
      {'eatingSetsColorToNewborn', args.default(false), args.booleanType},
  })
  Edible.Base.__init__(self, kwargs)
  self._config.eatingSetsColorToNewborn = kwargs.eatingSetsColorToNewborn
end

function Edible:reset()
  self._variables = {}
end

function Edible:onEnter(avatarObject, contactName)
  local berry = self.gameObject:getComponent('Berry')
  local isRipe = berry:isRipe()
  if contactName == 'avatar' and isRipe and not self._variables.triggered then
    self._variables.triggered = true
    -- Reward the player who ate the berry.
    local taste = avatarObject:getComponent('Taste'):getMostTasty()
    local rewardToDeliver = avatarObject:getComponent('Taste'):getReward(
        self.gameObject:getState())
    avatarObject:getComponent('Avatar'):addReward(rewardToDeliver)
    -- Replace the berry with an unripe berry.
    berry:unripen()
    -- If applicable, set the avatar's color back to the newborn color.
    if self._config.eatingSetsColorToNewborn then
      -- A certain number of berries may be consumed "cyprically", i.e. without
      -- revealing by avatar color that they were eaten. Once that number has
      -- been passed then eating a cryptic berry sets avatar color to newborn.
      -- The number of cryptic berries allowed can only be restored by planting.
      avatarObject:getComponent('ColorZapper'):eatCrypticBerry()
    end
    -- Record the eating event.
    local sceneObject = self.gameObject.simulation:getSceneObject()
    local globalBerryTracker = (
          sceneObject:getComponent('GlobalBerryTracker'))
    local playerIndex = avatarObject:getComponent('Avatar'):getIndex()
    local colorId = berry:getBerryColorId()
    local eatingTypesByPlayerMatrix = (
        globalBerryTracker.eatingTypesByPlayerMatrix)
    local pc = eatingTypesByPlayerMatrix(colorId, playerIndex):val()
    eatingTypesByPlayerMatrix(colorId, playerIndex):fill(pc + 1)
    events:add('eating', 'dict',
               'player_index', playerIndex,  -- int
               'berry_color', colorId)  -- int
  end
end

function Edible:onExit(avatarObject, contactName)
  self._variables.triggered = false
end

local Regrowth = class.Class(component.Component)

function Regrowth:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Regrowth')},
      {'minimumTimeToRipen', args.default(5), args.numberType},
      {'baseRate', args.default(0.0000025), args.numberType},
      {'cubicRate', args.default(0.000009), args.numberType},
      {'linearGrowth', args.default(false), args.booleanType},
  })
  Regrowth.Base.__init__(self, kwargs)

  self._config.minimumTimeToRipen = kwargs.minimumTimeToRipen
  self._config.baseRate = kwargs.baseRate
  self._config.cubicRate = kwargs.cubicRate
  self._config.linearGrowth = kwargs.linearGrowth
  self:resetCountdown()
end

function Regrowth:onStateChange(previousState)
  self._countdown = self._config.minimumTimeToRipen
end

function Regrowth:resetCountdown()
  self._countdown = self._config.minimumTimeToRipen
end

function Regrowth:update()
  self._countdown = self._countdown - 1
  if self._countdown > 0 then
    return
  end
  local berry = self.gameObject:getComponent('Berry')
  local probability = self:getProbability()
  if random:uniformReal(0, 1) < probability and not berry:isRipe() then
    berry:ripen()
  end
end

function Regrowth:_getCubicProbability(numBerries)
  local baseRate = self._config.baseRate
  local cubicRate = self._config.cubicRate
  local prob_linear = numBerries * baseRate
  local prob_cubic = numBerries * numBerries * numBerries * baseRate * cubicRate
  local probability = prob_linear + prob_cubic
  return probability
end

function Regrowth:_getLinearProbability(numBerries)
  local probability = numBerries * self._config.baseRate
  return probability
end

function Regrowth:getProbability()
  -- Get global variable references.
  local sceneObject = self.gameObject.simulation:getSceneObject()
  local globalBerryTracker = sceneObject:getComponent('GlobalBerryTracker')
  local ripeBerriesPerType = globalBerryTracker:getRipeBerriesPerType()
  local unripeBerriesPerType = globalBerryTracker:getUnripeBerriesPerType()

  -- Get the id of the current berry type.
  local id = self.gameObject:getComponent('Berry'):getBerryColorId()

  -- Calculate the respawn probability.
  local numRipe = ripeBerriesPerType(id):val()
  local numUnripe = unripeBerriesPerType(id):val()
  local numBerries = numRipe + numUnripe

  if self._config.linearGrowth then
    return self:_getLinearProbability(numBerries)
  end
  -- Default case is cubic growth.
  return self:_getCubicProbability(numBerries)
end


local Coloring = class.Class(component.Component)

function Coloring:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Coloring')},
      {'numColors', args.positive},
      {'unripePrefix', args.default('unripe')},
      {'ripePrefix', args.default('ripe')},
      {'hitPrefix', args.default('fire')},
      {'coloredPlayerPrefix', args.default('coloredPlayer')},
      {'anyUnripeColoringColorsAvatar', args.default(false), args.booleanType},
  })
  Coloring.Base.__init__(self, kwargs)

  self._config.numColors = kwargs.numColors
  self._config.newbornId = self._config.numColors + 1
  self._config.anyUnripeColoringColorsAvatar =
      kwargs.anyUnripeColoringColorsAvatar

  self._config.states = {}
  self._config.hitsToUse = {}
  self._config.hitToIndex = {}
  self._config.coloredPlayerStates = {}
  for idx = 1, self._config.numColors do
    local hit = kwargs.hitPrefix .. '_' .. tostring(idx)
    self._config.hitsToUse[hit] = true
    self._config.states[hit] = {
        unripe = kwargs.unripePrefix .. '_' .. tostring(idx),
        ripe = kwargs.ripePrefix .. '_' .. tostring(idx),
    }
    self._config.hitToIndex[hit] = idx
    self._config.coloredPlayerStates[hit] = (
        kwargs.coloredPlayerPrefix .. '_' .. tostring(idx))
  end
end

function Coloring:reset()
  self._alreadyHit = false
end

function Coloring:getNewStatesFromHit(hitName)
  local result = self._config.states[hitName]
  local beamId = self._config.hitToIndex[hitName]
  return result.unripe, result.ripe, beamId
end

function Coloring:onHit(hitterObject, hit)
  if self._config.hitsToUse[hit] then
    if self._alreadyHit then
      return true
    end
    self._alreadyHit = true
    local berry = self.gameObject:getComponent('Berry')
    local isRipe = berry:isRipe()
    local sourceColorId = berry:getBerryColorId()
    if isRipe == false then
      local unripeState, ripeState, targetColorId = self:getNewStatesFromHit(
          hit)
      -- Only change color, if color is different.
      if targetColorId ~= sourceColorId then
        -- Assume beam colors are in the same order as berry colors.
        self.gameObject:getComponent('Berry'):setBerryType(targetColorId)
        self.gameObject:setState(unripeState)
        self.gameObject:getComponent('Regrowth'):resetCountdown()
        -- Provide pseudoreward if applicable,
        if hitterObject:hasComponent('RewardForColoring') then
          local rewarder = hitterObject:getComponent('RewardForColoring')
          rewarder:rewardIfApplicable(targetColorId, self.gameObject:getPiece())
        end
        local playerIndex = hitterObject:getComponent('Avatar'):getIndex()
        events:add('replanting', 'dict',
                   'player_index', playerIndex,  -- int
                   'source_berry', sourceColorId,  -- int
                   'target_berry', targetColorId)  -- int
        -- Update all planting metrics.
        self:_updateMetrics(berry, hitterObject)
      end
      local colorZapper = hitterObject:getComponent('ColorZapper')
      if self._config.anyUnripeColoringColorsAvatar then
        -- Set the color of the zapper to the color in which they replanted or
        -- pretended to replant if it was already colored that way.
        colorZapper:setColor(
            targetColorId, self._config.coloredPlayerStates[hit])
        colorZapper:resetCrypticBerryConsumption()
      else
        if targetColorId ~= sourceColorId then
          colorZapper:setColor(
              targetColorId, self._config.coloredPlayerStates[hit])
          colorZapper:resetCrypticBerryConsumption()
        end
      end
    end
    -- Do not pass through hit berries.
    return true
  end
  -- Non-coloring beams (like zappers) can pass through berries.
  return false
end

function Coloring:update()
  self._alreadyHit = false
end

function Coloring:_updateMetrics(berry, hitterObject)
  -- Record the coloring event.
  local sceneObject = self.gameObject.simulation:getSceneObject()
  local globalBerryTracker = (
    sceneObject:getComponent('GlobalBerryTracker'))
  local berryColorId = berry:getBerryColorId()
  -- Accumulate berry by player.
  local playerIndex = hitterObject:getComponent('Avatar'):getIndex()
  local coloringByPlayerMatrix = (
      globalBerryTracker.coloringByPlayerMatrix)
  local pc = coloringByPlayerMatrix(berryColorId, playerIndex):val()
  coloringByPlayerMatrix(berryColorId, playerIndex):fill(pc + 1)
  -- Accumulate berry by taste of colorer data.
  local hitterTasteId = (
    hitterObject:getComponent('Taste').mostTastyBerryId)
  local berryTypesByTasteOfColorer = (
      globalBerryTracker.berryTypesByTasteOfColorer)
  local tc = berryTypesByTasteOfColorer(berryColorId, hitterTasteId):val()
  berryTypesByTasteOfColorer(berryColorId, hitterTasteId):fill(tc + 1)
  -- Accumulate berry by color or colorer data.
  local hitterColorId = hitterObject:getComponent('ColorZapper').colorId
  local berryTypesByColorOfColorer = (
        globalBerryTracker.berryTypesByColorOfColorer)
  if hitterColorId > 0 then  -- Newborn avatars have no color.
    local bc = berryTypesByColorOfColorer(berryColorId,
                                          hitterColorId):val()
    berryTypesByColorOfColorer(berryColorId, hitterColorId):fill(bc + 1)
  elseif hitterColorId == 0 then
    -- Handle case of a newborn avatar.
    local bc = berryTypesByColorOfColorer(berryColorId,
                                          self._config.newbornId):val()
    berryTypesByColorOfColorer(berryColorId,
                               self._config.newbornId):fill(bc + 1)
  end
end


local Taste = class.Class(component.Component)

function Taste:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Taste')},
      {'mostTastyBerryId', args.numberType},
      {'ripePrefix', args.default('ripe'), args.stringType},
      -- Reward provided by the most tasty berry.
      {'rewardMostTasty', args.default(2), args.numberType},
      -- Reward provided by all other berries.
      {'rewardDefault', args.default(1), args.numberType},
  })
  Taste.Base.__init__(self, kwargs)
  self._kwargs = kwargs
end

function Taste:reset()
  self.mostTastyBerryId = self._kwargs.mostTastyBerryId
  self._mostTasty = (
      self._kwargs.ripePrefix .. '_' .. self._kwargs.mostTastyBerryId)
  self._rewardMostTasty = self._kwargs.rewardMostTasty
  self._rewardDefault = self._kwargs.rewardDefault
end

function Taste:start()
  local sceneObject = self.gameObject.simulation:getSceneObject()
  self._newbornId = (
      sceneObject:getComponent('GlobalBerryTracker'):getNumBerryTypes() + 1)
end

function Taste:preUpdate()
  local zapperComponent = self.gameObject:getComponent('Zapper')
  local sceneObject = self.gameObject.simulation:getSceneObject()
  local globalZapTracker = sceneObject:getComponent('GlobalZapTracker')

  local zapperIndex = zapperComponent['zapperIndex']
  if zapperIndex then
    local zapperObject = self.gameObject.simulation:getAvatarFromIndex(
        zapperIndex)
    -- Get the zapper's tasteId.
    local zapperTasteId = zapperObject:getComponent('Taste').mostTastyBerryId
    -- Accumulate `tasteByTasteZapCounts` using the new information.
    local tasteByTasteZapCounts = globalZapTracker.tasteByTasteZapCounts
    local c = tasteByTasteZapCounts(self.mostTastyBerryId, zapperTasteId):val()
    tasteByTasteZapCounts(self.mostTastyBerryId, zapperTasteId):fill(c + 1)
    -- Get the zapper's colorId.
    local zapperColorId = zapperObject:getComponent('ColorZapper').colorId
    local tasteByColorZapCounts = globalZapTracker.tasteByColorZapCounts
    if zapperColorId > 0 then
      local tc = tasteByColorZapCounts(self.mostTastyBerryId,
                                       zapperColorId):val()
      tasteByColorZapCounts(self.mostTastyBerryId, zapperColorId):fill(tc + 1)
    elseif zapperColorId == 0 then
      local tc = tasteByColorZapCounts(self.mostTastyBerryId,
                                       self._newbornId):val()
      tasteByColorZapCounts(self.mostTastyBerryId, self._newbornId):fill(tc + 1)
    end
  end
end

function Taste:getMostTasty()
  return self._mostTasty
end

function Taste:getReward(berryColor)
  if berryColor == self._mostTasty then
    return self._rewardMostTasty
  else
    return self._rewardDefault
  end
end


local ColorZapper = class.Class(component.Component)

function ColorZapper:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('ColorZapper')},
      {'cooldownTime', args.numberType},
      {'beamLength', args.numberType},
      {'beamRadius', args.numberType},
      {'numColorZappers', args.positive},
      {'beamColors', args.tableType},
      {'crypticConsumptionThreshold', args.default(1), args.positive},
      {'stochasticallyCryptic', args.default(false), args.booleanType},
  })
  ColorZapper.Base.__init__(self, kwargs)

  self._config.cooldownTime = kwargs.cooldownTime
  self._config.beamLength = kwargs.beamLength
  self._config.beamRadius = kwargs.beamRadius

  self._config.numColorZappers = kwargs.numColorZappers
  self._config.newbornId = self._config.numColorZappers + 1
  self._config.beamColors = kwargs.beamColors

  self._config.crypticConsumptionThreshold = kwargs.crypticConsumptionThreshold
  self._config.stochasticallyCryptic = kwargs.stochasticallyCryptic

  self._config.hitNames = {}
  self._config.layerNames = {}
  self._config.spriteNames = {}
  for idx = 1, self._config.numColorZappers do
    local hitName = 'fire_' .. tostring(idx)
    local layerName = 'beam_' .. hitName
    local spriteName = 'Beam_' .. hitName
    table.insert(self._config.hitNames, hitName)
    table.insert(self._config.layerNames, layerName)
    table.insert(self._config.spriteNames, spriteName)
  end
end

function ColorZapper:reset()
  self.colorId = 0 -- a colorId of 0 indicates the newborn type.
  -- Set the beam cooldown timer to its `ready` state (i.e. coolingTimer = 0).
  self._coolingTimer = 0
  -- The number of cryptic berries recently consumed is zero at the start.
  self._crypticBerriesEaten = 0
  self._disclosureProbability = nil
end

function ColorZapper:addHits(worldConfig)
  for idx = 1, self._config.numColorZappers do
    worldConfig.hits[self._config.hitNames[idx]] = {
        layer = self._config.layerNames[idx],
        sprite = self._config.spriteNames[idx],
    }
    component.insertIfNotPresent(worldConfig.renderOrder, self._config.layerNames[idx])
  end
end

function ColorZapper:addSprites(tileSet)
  for idx = 1, self._config.numColorZappers do
    tileSet:addColor(self._config.spriteNames[idx],
                     self._config.beamColors[idx])
  end
end

function ColorZapper:registerUpdaters(updaterRegistry)
  for beamId = 1, self._config.numColorZappers do
    -- Note: hitName is the same as update name and action name in this case.
    local hitName = self._config.hitNames[beamId]
    local fireColoringBeam = function()
      local state = self.gameObject:getComponent('Avatar'):getVolatileData()
      local actions = state.actions
      -- Execute the beam if applicable.
      if self._config.cooldownTime >= 0 then
        if self._coolingTimer > 0 then
          -- Ensure timer is only decremented once per frame.
          if not self._alreadyDecrementedTimerThisFrame then
            self._coolingTimer = self._coolingTimer - 1
            self._alreadyDecrementedTimerThisFrame = true
          end
        else
          if actions[hitName] == 1 then
            self._coolingTimer = self._config.cooldownTime
            self.gameObject:hitBeam(
                hitName,
                self._config.beamLength,
                self._config.beamRadius
            )
          end
        end
      end
    end
    updaterRegistry:registerUpdater{
      updateFn = fireColoringBeam,
      priority = 140,
    }
  end
end

function ColorZapper:onHit(hittingGameObject, hitName)
  -- Coloring beams do not pass through avatars.
  return true
end

function ColorZapper:preUpdate()
  local zapperComponent = self.gameObject:getComponent('Zapper')
  local sceneObject = self.gameObject.simulation:getSceneObject()
  local globalZapTracker = sceneObject:getComponent('GlobalZapTracker')

  local playerRespawnedThisStep = zapperComponent['playerRespawnedThisStep']
  if playerRespawnedThisStep then
    self.colorId = 0  -- a colorId of 0 indicates the newborn type.
  end

  -- Remap the color ids of the newborn avatars.
  local remappedColorId = self.colorId
  if self.colorId == 0 then
    remappedColorId = self._config.newbornId
  end

  local zapperIndex = zapperComponent['zapperIndex']
  if zapperIndex then
    -- Get the zapper's colorId.
    local zapperObject = self.gameObject.simulation:getAvatarFromIndex(
        zapperIndex)
    local zapperColorId = zapperObject:getComponent('ColorZapper').colorId
    local colorByColorZapCounts = globalZapTracker.colorByColorZapCounts
    -- Newborn avatars do not have a color.
    if zapperColorId > 0 then
      -- Accumulate `colorByColorZapCounts` using the new information.
      local c = colorByColorZapCounts(remappedColorId, zapperColorId):val()
      colorByColorZapCounts(remappedColorId, zapperColorId):fill(c + 1)
    elseif zapperColorId == 0 then
      local c = colorByColorZapCounts(remappedColorId,
                                      self._config.newbornId):val()
      colorByColorZapCounts(remappedColorId, self._config.newbornId):fill(c + 1)
    end
    -- Get the zapper's taste.
    local zapperTasteId = zapperObject:getComponent('Taste').mostTastyBerryId
    local colorByTasteZapCounts = globalZapTracker.colorByTasteZapCounts
    -- Accumulate `colorByTasteZapCounts` using the new information.
    local ct = colorByTasteZapCounts(remappedColorId, zapperTasteId):val()
    colorByTasteZapCounts(remappedColorId, zapperTasteId):fill(ct + 1)
  end

  -- Ensure timers can only be decremented once per frame despite having several
  -- beam zap updates, one per color.
  self._alreadyDecrementedTimerThisFrame = false
end

function ColorZapper:_updateDisclosureProbability()
  local sceneObject = self.gameObject.simulation:getSceneObject()
  local berryTracker = sceneObject:getComponent('GlobalBerryTracker')
  local monocultureFraction = berryTracker:getMonocultureFraction()
  self._disclosureProbability = 1.0 - monocultureFraction
end

function ColorZapper:update()
  if self._config.stochasticallyCryptic then
    self:_updateDisclosureProbability()
  end
end

--[[ Use this function to track avatar color changing when they use the beam.]]
function ColorZapper:setAvatarColorId(colorId)
  self.colorId = colorId
end

function ColorZapper:setColor(idx, coloredPlayerState)
  -- Assume just one connected object with an `AvatarConnector` component.
  local overlayObject = self.gameObject:getComponent(
      'Avatar'):getAllConnectedObjectsWithNamedComponent('AvatarConnector')[1]
  -- Change color of the player that zapped the berry to match the berry.
  overlayObject:setState(coloredPlayerState)
  self:setAvatarColorId(idx)
end

--[[ A certain number of berries may be consumed "cryptically", i.e. without
revealing by avatar color that they were eaten. Once that number has been
passed, then this function sets the avatar to newborn color. When
`stochasticallyCryptic` is true, then this function randomly changes the
avatar's color (revealting it to be a free rider) with probability
`self._disclosureProbability`.
]]
function ColorZapper:eatCrypticBerry()
  if self._config.stochasticallyCryptic then
    if random:uniformReal(0.0, 1.0) < self._disclosureProbability then
      self:setColor(0, 'avatarOverlay')
    end
  else
    self._crypticBerriesEaten = self._crypticBerriesEaten + 1
    local threshold = self._config.crypticConsumptionThreshold
    if self._crypticBerriesEaten >= threshold then
      self:setColor(0, 'avatarOverlay')
    end
  end
end

function ColorZapper:resetCrypticBerryConsumption()
  self._crypticBerriesEaten = 0
end


local RewardForColoring = class.Class(component.Component)

function RewardForColoring:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('RewardForColoring')},
      {'berryIds', args.default({}), args.tableType},
      {'amount', args.default(0.0), args.numberType},
      {'rewardCooldown', args.default(128), args.numberType},
  })
  RewardForColoring.Base.__init__(self, kwargs)
  self._berryIds = set.Set(kwargs.berryIds)
  self._amount = kwargs.amount
  self._rewardCooldown = kwargs.rewardCooldown

  self._cooldownRemainingPerBerry = {}
end

function RewardForColoring:_addReward(piece)
  self.gameObject:getComponent('Avatar'):addReward(self._amount)
  self._cooldownRemainingPerBerry[piece] = self._rewardCooldown
end

function RewardForColoring:rewardIfApplicable(berryId, piece)
  if self._berryIds[berryId] then
    -- berry type is rewarded
    if not self._cooldownRemainingPerBerry[piece] then
      -- specific berry has not been planted before by this agent.
      self:_addReward(piece)
    end
    if self._cooldownRemainingPerBerry[piece] <= 0 then
      -- specific berry was last planted by this agent long enough ago that it
      -- can now provide another reward from planting it again.
      self:_addReward(piece)
    end
  end
end

function RewardForColoring:update()
  for piece, framesSinceRewarded in pairs(self._cooldownRemainingPerBerry) do
    self._cooldownRemainingPerBerry[piece] = framesSinceRewarded - 1
  end
end


local RewardForZapping = class.Class(component.Component)

function RewardForZapping:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('RewardForZapping')},
      {'targetColors', args.default({}), args.tableType},
      {'amounts', args.default({}), args.tableType},
  })
  RewardForZapping.Base.__init__(self, kwargs)
  self._targetColors = set.Set(kwargs.targetColors)
  self._amounts = kwargs.amounts
end

function RewardForZapping:onHit(hitterObject, hit)
  if hit == 'zapHit' then
    local zappedColorId = self.gameObject:getComponent('ColorZapper').colorId
    local hitterTargetColors = hitterObject:getComponent(
      'RewardForZapping'):getTargetColors()
    if hitterTargetColors[zappedColorId] then
      hitterObject:getComponent('Avatar'):addReward(
        self._amounts[zappedColorId])
    end
  end
end

function RewardForZapping:getTargetColors()
  return self._targetColors
end


local GlobalBerryTracker = class.Class(component.Component)

function GlobalBerryTracker:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('GlobalBerryTracker')},
      {'numBerryTypes', args.numberType},
      {'numPlayers', args.numberType},
  })
  GlobalBerryTracker.Base.__init__(self, kwargs)

  self._config.numPlayers = kwargs.numPlayers
  self._config.numBerryTypes = kwargs.numBerryTypes
end

function GlobalBerryTracker:reset()
  self.ripeBerriesPerType = tensor.Int32Tensor(
      self._config.numBerryTypes):fill(0)
  self.unripeBerriesPerType = tensor.Int32Tensor(
      self._config.numBerryTypes):fill(0)
  self.berriesPerType = tensor.Int32Tensor(
      self._config.numBerryTypes):fill(0)

  self.coloringByPlayerMatrix = tensor.Int32Tensor(
      self._config.numBerryTypes, self._config.numPlayers):fill(0)
  self.eatingTypesByPlayerMatrix = tensor.Int32Tensor(
      self._config.numBerryTypes, self._config.numPlayers):fill(0)
  -- Add 1 to accomodate the newborn avatar color.
  self.berryTypesByColorOfColorer = tensor.Int32Tensor(
      self._config.numBerryTypes, self._config.numBerryTypes + 1):fill(0)
  self.berryTypesByTasteOfColorer = tensor.Int32Tensor(
      self._config.numBerryTypes, self._config.numBerryTypes):fill(0)
end

function GlobalBerryTracker:getRipeBerriesPerType()
  return self.ripeBerriesPerType
end

function GlobalBerryTracker:getUnripeBerriesPerType()
  return self.unripeBerriesPerType
end

function GlobalBerryTracker:getNumBerryTypes()
  return self._config.numBerryTypes
end

function GlobalBerryTracker:getMonocultureFraction()
  local numBerries = self.berriesPerType:sum()
  local monocultureFraction = vectorMax(self.berriesPerType) / numBerries
  return monocultureFraction
end

function GlobalBerryTracker:update()
  self.berriesPerType = self.ripeBerriesPerType:clone()
  self.berriesPerType:cadd(self.unripeBerriesPerType)
end


local GlobalZapTracker = class.Class(component.Component)

function GlobalZapTracker:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('GlobalZapTracker')},
      {'numBerryTypes', args.numberType},
      {'numPlayers', args.numberType},
  })
  GlobalZapTracker.Base.__init__(self, kwargs)

  self._config.numBerryTypes = kwargs.numBerryTypes
  self._config.numPlayers = kwargs.numPlayers
end

function GlobalZapTracker:reset()
  self.fullZapCountMatrix = tensor.Int32Tensor(
      self._config.numPlayers, self._config.numPlayers):fill(0)
  -- Add one to accomodate the newborn color.
  self.colorByColorZapCounts = tensor.Int32Tensor(
      self._config.numBerryTypes + 1, self._config.numBerryTypes + 1):fill(0)
  self.colorByTasteZapCounts = tensor.Int32Tensor(
      self._config.numBerryTypes + 1, self._config.numBerryTypes):fill(0)
  self.tasteByTasteZapCounts = tensor.Int32Tensor(
    self._config.numBerryTypes, self._config.numBerryTypes):fill(0)
  self.tasteByColorZapCounts = tensor.Int32Tensor(
    self._config.numBerryTypes, self._config.numBerryTypes + 1):fill(0)
end

function GlobalZapTracker:preUpdate()
  for idx = 1, self._config.numPlayers do
    local avatarObject = self.gameObject.simulation:getAvatarFromIndex(idx)
    local playerZapMatrix = avatarObject:getComponent(
        'Zapper')['playerZapMatrix']
    -- Accumulate the zap count matrix (zappedIndex X zapperIndex).
    self.fullZapCountMatrix:cadd(playerZapMatrix)
  end
end


local allComponents = {
    -- Berry components.
    Berry = Berry,
    Edible = Edible,
    Regrowth = Regrowth,
    Coloring = Coloring,

    -- Avatar components.
    Taste = Taste,
    ColorZapper = ColorZapper,
    RewardForColoring = RewardForColoring,
    RewardForZapping = RewardForZapping,

    -- Scene componenets.
    GlobalBerryTracker = GlobalBerryTracker,
    GlobalZapTracker = GlobalZapTracker,
}

component_registry.registerAllComponents(allComponents)

return allComponents
