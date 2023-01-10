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
local events = require 'system.events'

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')

-- For Lua 5.2 compatibility.
local unpack = unpack or table.unpack

local Resource = class.Class(component.Component)

function Resource:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Resource')},
      {'resourceClass', args.numberType},
      {'visibleType', args.stringType},
      {'waitState', args.stringType},
      {'regenerationRate', args.ge(0.0), args.le(1.0)},
      {'regenerationDelay', args.positive},
  })
  Resource.Base.__init__(self, kwargs)
  self._config.resourceClass = kwargs.resourceClass
  self._config.visibleType = kwargs.visibleType
  self._config.waitState = kwargs.waitState
  self._config.regenerationRate = kwargs.regenerationRate
  self._config.regenerationDelay = kwargs.regenerationDelay
end

function Resource:reset()
  self._variables = {}
end

function Resource:onEnter(enteringGameObject, contactName)
  if self.gameObject:getState() == self._config.visibleType
      and contactName == 'avatar' then
    local sceneObject = self.gameObject.simulation:getSceneObject()
    local theMatrix = sceneObject:getComponent('TheMatrix')
    -- Increment inventory.
    local playerIndex = enteringGameObject:getComponent('Avatar'):getIndex()
    local inventory = theMatrix:getPlayerInventory(playerIndex)
    local amount = inventory(self._config.resourceClass):val()
    inventory(self._config.resourceClass):fill(amount + 1)
    -- Record that at least one resource was collected by this player.
    theMatrix.playerCollectedAtLeastOneResource[playerIndex] = true
    -- Set the 'ready' indicator state to show at least one resource collected.
    if theMatrix.indicators[playerIndex] == 'notReady' then
      theMatrix.indicators[playerIndex] = 'ready'
    end
    -- Remove the resource from the map.
    self.gameObject:setState(self._config.waitState)
    -- Consult the `Taste` component to determine if rewards should be provided.
    if enteringGameObject:hasComponent('Taste') then
      local taste = enteringGameObject:getComponent('Taste')
      local reward = taste:getRewardForGathering(self._config.resourceClass)
      enteringGameObject:getComponent('Avatar'):addReward(reward)
    end
    -- Report the resource collection event.
    self:_reportCollectionEvent(playerIndex, self._config.resourceClass)
    -- Update the avatar cumulants tracking resource collection.
    self:_updateCumulants(enteringGameObject, self._config.resourceClass)
  end
end

function Resource:registerUpdaters(updaterRegistry)
  local transform = self.gameObject:getComponent('Transform')
  local function maybeRespawn()
    if random:uniformReal(0, 1) < self._config.regenerationRate then
      local maybeAvatar = transform:queryPosition('upperPhysical')
      if not maybeAvatar then
        -- Only spawn a resource if an avatar is not currently standing here.
        self.gameObject:setState(self._config.visibleType)
      end
    end
  end
  updaterRegistry:registerUpdater{
      updateFn = maybeRespawn,
      priority = 100,
      state = self._config.waitState,
      startFrame = self._config.regenerationDelay,
  }
end

function Resource:getResourceClass()
  return self._config.resourceClass
end

function Resource:getVisibleState()
  return self._config.visibleType
end

function Resource:regenerate()
  self.gameObject:setState(self._config.visibleType)
end

function Resource:_reportCollectionEvent(playerIndex, resourceClass)
  events:add('collected_resource', 'dict',
             'player_index', playerIndex,
             'class', resourceClass)
end

function Resource:_updateCumulants(avatarObject, collectedResourceClass)
  local gameInteractionZapper = avatarObject:getComponent(
      'GameInteractionZapper')
  gameInteractionZapper:setResourceCollectionCumulant(collectedResourceClass)
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
  if hitName == 'gameInteraction' then
    self._variables.health = self._variables.health - 1
    if self._variables.health == 0 then
      -- Reset the health state variable.
      self._variables.health = self._config.initialHealth
      -- Remove the resource from the map.
      self.gameObject:setState(self._config.waitState)
      -- Report the destruction event.
      self:_reportDestructionEvent(hitterGameObject)
      -- Beams pass through a destroyed destroyable.
      return false
    end
    -- Beams do not pass through after hitting an undestroyed destroyable.
    return true
  end
end

function Destroyable:_reportDestructionEvent(hitterGameObject)
  local playerIndex = hitterGameObject:getComponent('Avatar'):getIndex()
  local resourceClass = self.gameObject:getComponent(
      'Resource'):getResourceClass()
  events:add('destroyed_resource', 'dict',
             'player_index', playerIndex,
             'class', resourceClass)
  local gameInteractionZapper = hitterGameObject:getComponent(
      'GameInteractionZapper')
  gameInteractionZapper:setResourceDestructionCumulant(resourceClass)
end


local TheMatrix = class.Class(component.Component)

--[[ The matrix lives on the static scene game object. It holds the global state
(how much of each resource was collected by each player), needed to resolve
interactions.
]]
function TheMatrix:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('TheMatrix')},
      {'matrix', args.tableType},
      -- If `columnPlayerMatrix` is not specified then automatically use the
      -- transpose of `matrix`.
      {'columnPlayerMatrix', args.default(nil), args.tableType},
      -- By default, all players start out with 1 of each resource type. This
      -- avoids a singularity at 0 in running with scissors, but it is not
      -- desirable for all games. In such cases, pass `true` to initialize at 0.
      {'zeroInitialInventory', args.default(false), args.booleanType},
      -- By default, row players win whenever rewards are tied.
      {'randomTieBreaking', args.default(false), args.booleanType},
      -- By default, players can still interact despite not having picked up
      -- any resources yet.
      {'disallowUnreadyInteractions', args.default(false), args.booleanType},
      -- Set intervals of reward to use to determine result indicator color.
      {'resultIndicatorColorIntervals', args.tableType},
  })
  TheMatrix.Base.__init__(self, kwargs)
  -- Matrices are always square since all players have the same available
  -- strategies as one another.
  self._config.numResources = #kwargs.matrix
  self._config.zeroInitialInventory = kwargs.zeroInitialInventory
  self._rowPlayerMatrix = tensor.DoubleTensor(self:_cleanMatrix(kwargs.matrix))

  self.randomTieBreaking = kwargs.randomTieBreaking
  self.disallowUnreadyInteractions = kwargs.disallowUnreadyInteractions

  if kwargs.columnPlayerMatrix then
    self._columnPlayerMatrix = tensor.DoubleTensor(
        self:_cleanMatrix(kwargs.columnPlayerMatrix))
  else
    self._columnPlayerMatrix = self._rowPlayerMatrix:clone():transpose(1, 2)
  end

  self.resultIndicatorColorIntervals = kwargs.resultIndicatorColorIntervals
end

function TheMatrix:reset()
  self.playerResources = tensor.DoubleTensor(
      self.gameObject.simulation:getNumPlayers(),
      self._config.numResources)
  if self._config.zeroInitialInventory then
    self.playerResources:fill(0)
  else
    self.playerResources:fill(1)
  end
  self.playerCollectedAtLeastOneResource = {}
  for i = 1, self.gameObject.simulation:getNumPlayers() do
    table.insert(self.playerCollectedAtLeastOneResource, false)
  end
  self.indicators = {}
  for i = 1, self.gameObject.simulation:getNumPlayers() do
    table.insert(self.indicators, 'notReady')
  end
end

--[[ Coming in from the command line adds some extra fields to the matrix table
that reflect default values (which we do not use). This function removes them.

This must be done before the matrix can be converted into a tensor.
]]
function TheMatrix:_cleanMatrix(matrixTable)
  local matrixClean = {}
  for _, row in ipairs(matrixTable) do
    table.insert(matrixClean, row)
  end
  return matrixClean
end

function TheMatrix:getNumResources()
  return self._config.numResources
end

function TheMatrix:getPlayerInventory(playerIndex)
  return self.playerResources(playerIndex)
end

function TheMatrix:getRowPlayerMatrix()
  return self._rowPlayerMatrix
end

function TheMatrix:getColumnPlayerMatrix()
  return self._columnPlayerMatrix
end

function TheMatrix:resetInventory(inventory, playerIndex)
  if self._config.zeroInitialInventory then
    inventory:fill(0)
  else
    inventory:fill(1)
  end
  if playerIndex then
    self.playerCollectedAtLeastOneResource[playerIndex] = false
  end
end

function TheMatrix:getColorInterval(reward)
  for idx, interval in ipairs(self.resultIndicatorColorIntervals) do
    if interval[1] <= reward and reward < interval[2] then
      return idx
    end
  end
  assert(false,
          'reward: ' .. tostring(reward) .. ' not found in color intervals.')
end


local SpawnResourcesWhenAllPlayersZapped = class.Class(component.Component)

function SpawnResourcesWhenAllPlayersZapped:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('SpawnResourcesWhenAllPlayersZapped')},
  })
  SpawnResourcesWhenAllPlayersZapped.Base.__init__(self, kwargs)
end

function SpawnResourcesWhenAllPlayersZapped:registerUpdaters(updaterRegistry)
  local simulation = self.gameObject.simulation

  local function step()
    local numLiveAvatars = simulation:getGroupCount('players')
    if numLiveAvatars == 0 then
      local resourceObjects = simulation:getGroupShuffledWithProbability(
          'resourceWaits', 1.0)
      for _, resourceObject in ipairs(resourceObjects) do
        local resource = resourceObject:getComponent('Resource')
        local visibleState = resource:getVisibleState()
        resourceObject:setState(visibleState)
      end
    end
  end
  updaterRegistry:registerUpdater{
      updateFn = step,
      priority = 7,
  }
end


--[[ The `GameInteractionZapper` component endows an avatar with the ability to
fire a beam and be hit by the beams of other avatars. The effect of the beam
depends on the resources in the inventories of both players.
]]
local GameInteractionZapper = class.Class(component.Component)

function GameInteractionZapper:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('GameInteractionZapper')},
      {'cooldownTime', args.numberType},
      {'beamLength', args.numberType},
      {'beamRadius', args.numberType},
      {'framesTillRespawn', args.numberType},
      {'numResources', args.numberType},
      -- If `endEpisodeOnFirstInteraction` set to true, then end the episode
      -- the first time any pair of agents interact.
      {'endEpisodeOnFirstInteraction', args.default(false), args.booleanType},
      -- By default, only the loser of each interaction's inventory gets reset,
      -- if `reset_winner_inventory` set to true then instead reset both the
      -- inventory of both players after each interaction.
      {'reset_winner_inventory', args.default(false), args.booleanType},
      {'reset_loser_inventory', args.default(true), args.booleanType},
      {'losingPlayerDies', args.default(true), args.booleanType},
      {'winningPlayerDies', args.default(false), args.booleanType},
      -- Only deliver rewards larger than `rewardFloor`.
      {'rewardFloor', args.default(-1e6), args.numberType},
      -- Multiply the game's main reward signal by the following value. This is
      -- useful for games like "running with scissors" where the actual reward
      -- values tend to be quite a lot smaller than in other games.
      {'rewardMultiplier', args.default(1.0), args.numberType},
      -- When the value of `rewardFromZappingUnreadyPlayer` is not 0 it will
      -- typically be negative since that would be a situation where we aim to#
      -- deter players from zapping their partners before they are ready (i.e.
      -- before they have collected any resources).
      {'rewardFromZappingUnreadyPlayer', args.default(0), args.numberType},
      -- Both interactants freeze after interacting for `freezeOnInteraction`
      -- steps (zero by default, and if used normally would be just a few steps.
      {'freezeOnInteraction', args.default(0), args.ge(0)},
      -- This color is yellow.
      {'beamColor', args.default({252, 252, 106}), args.tableType},
  })
  GameInteractionZapper.Base.__init__(self, kwargs)

  self._config.cooldownTime = kwargs.cooldownTime
  self._config.beamLength = kwargs.beamLength
  self._config.beamRadius = kwargs.beamRadius
  self._config.beamColor = kwargs.beamColor
  self._config.framesTillRespawn = kwargs.framesTillRespawn

  self._config.numResources = kwargs.numResources

  self._config.endEpisodeOnFirstInteraction = (
      kwargs.endEpisodeOnFirstInteraction)
  self._config.reset_winner_inventory = kwargs.reset_winner_inventory
  self._config.reset_loser_inventory = kwargs.reset_loser_inventory

  self._config.losingPlayerDies = kwargs.losingPlayerDies
  self._config.winningPlayerDies = kwargs.winningPlayerDies
  self._config.rewardFloor = kwargs.rewardFloor
  self._config.rewardMultiplier = kwargs.rewardMultiplier

  self._config.rewardFromZappingUnreadyPlayer = (
      kwargs.rewardFromZappingUnreadyPlayer)

  self._config.freezeOnInteraction = kwargs.freezeOnInteraction
end

function GameInteractionZapper:addHits(worldConfig)
  worldConfig.hits['gameInteraction'] = {
      layer = 'beamInteraction',
      sprite = 'BeamInteraction',
  }
  component.insertIfNotPresent(worldConfig.renderOrder, 'beamInteraction')
end

function GameInteractionZapper:addSprites(tileSet)
  tileSet:addColor('BeamInteraction', self._config.beamColor)
end

function GameInteractionZapper:registerUpdaters(updaterRegistry)
  local aliveState = self:getAliveState()
  local waitState = self:getWaitState()

  local zap = function()
    local avatar = self.gameObject:getComponent('Avatar')
    if avatar:isMovementAllowed() then
      local playerVolatileVariables = avatar:getVolatileData()
      local actions = playerVolatileVariables.actions
      -- Execute the beam if applicable.
      if self.gameObject:getState() == aliveState then
        if self._config.cooldownTime >= 0 then
          if self._coolingTimer > 0 then
            self._coolingTimer = self._coolingTimer - 1
          else
            if actions['interact'] == 1 and self._canZap then
              self._coolingTimer = self._config.cooldownTime
              self.gameObject:hitBeam('gameInteraction',
                                      self._config.beamLength,
                                      self._config.beamRadius)
            end
          end
        end
      end
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = zap,
      priority = 140,
  }

  local respawn = function()
    local spawnGroup = self.gameObject:getComponent('Avatar'):getSpawnGroup()
    self.gameObject:teleportToGroup(spawnGroup, aliveState)
    self.playerRespawnedThisStep = true
  end

  updaterRegistry:registerUpdater{
      updateFn = respawn,
      priority = 135,
      state = waitState,
      startFrame = self._config.framesTillRespawn
  }

  local theMatrix = self.gameObject.simulation:getSceneObject():getComponent(
      'TheMatrix')
  local avatar = self.gameObject:getComponent('Avatar')

  local applyScheduledEffects = function()
    local playerIndex = avatar:getIndex()
    if self._framesTillScheduledEffects == 0 then
      -- Implement scheduled effects.
      for _, effect in ipairs(self._scheduledEffects) do
        effect.func(unpack(effect.arguments))
      end
      self._scheduledEffects = {}
      -- Remove the result indicator on the same frame effects are implemented.
      theMatrix.indicators[playerIndex] = 'notReady'
      self._framesTillScheduledEffects = -1
      -- End the episode after the first interaction if applicable. This is used
      -- in the one-shot substrate variants.
      if self._config.endEpisodeOnFirstInteraction then
        self._endEpisodeOnNextFrame = true
      end
    elseif self._framesTillScheduledEffects > 0 then
      self._framesTillScheduledEffects = self._framesTillScheduledEffects - 1
      local colorIdx = theMatrix:getColorInterval(self._rewardToDetermineColor)
      theMatrix.indicators[playerIndex] = (
          'resultIndicatorColor' .. tostring(colorIdx))
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = applyScheduledEffects,
      priority = 4,
      state = aliveState,
  }

  local endEpisodeIfApplicable = function()
    -- End the episode when the flag is set. This is used in the one-shot
    -- substrate variants.
    if self._endEpisodeOnNextFrame then
      self.gameObject.simulation:endEpisode()
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = endEpisodeIfApplicable,
      priority = 900,
  }

  local function resetSimultaneousInteractionBlocker()
    self.interactedThisStep = false
  end

  updaterRegistry:registerUpdater{
      updateFn = resetSimultaneousInteractionBlocker,
      priority = 890,
  }
end

--[[ Compute payoffs based on row and column player strategy profiles.

`rowProfile` is (1 X numResources) and `colProfile` is (numResources X 1).
]]
function GameInteractionZapper:_computeInteractionRewards(rowProfile,
                                                          colProfile)
  local rowMatrix = self._matrixComponent:getRowPlayerMatrix():clone()
  local colMatrix = self._matrixComponent:getColumnPlayerMatrix():clone()
  local rowReward = rowProfile:mmul(rowMatrix):mmul(colProfile):reshape{}:val()
  local colReward = rowProfile:mmul(colMatrix):mmul(colProfile):reshape{}:val()
  return rowReward, colReward
end

function GameInteractionZapper:_avatarDies(avatarComponent)
  local avatarObject = avatarComponent.gameObject
  local waitType = (
    avatarObject:getComponent('GameInteractionZapper'):getWaitState())
  avatarObject:setState(waitType)
end

function GameInteractionZapper:sendRewardsToBothInteractants(
    rowReward, columnReward,
    rowResources, columnResources,
    rowAvatar, columnAvatar)
  if self.gameObject:hasComponent('InteractionTaste') then
    local interactionTaste = self.gameObject:getComponent('InteractionTaste')
    if rowReward > self._config.rewardFloor then
      local rowRewardToDeliver = interactionTaste:getExtraRewardForInteraction(
          rowReward, rowResources)
      rowAvatar:addReward(rowRewardToDeliver)
    end
    if columnReward > self._config.rewardFloor then
      local colRewardToDeliver = interactionTaste:getExtraRewardForInteraction(
          columnReward, columnResources)
      columnAvatar:addReward(colRewardToDeliver)
    end
  else
    if rowReward > self._config.rewardFloor then
      rowAvatar:addReward(rowReward)
    end
    if columnReward > self._config.rewardFloor then
      columnAvatar:addReward(columnReward)
    end
  end
end

function GameInteractionZapper:setFramesTillScheduledEffects(frames)
  self._framesTillScheduledEffects = frames
end

function GameInteractionZapper:_resolve(
    rowPlayerIndex, columnPlayerIndex, hitterGameObject)
  -- Row player setup
  local rowAvatar = self.gameObject.simulation:getAvatarFromIndex(
    rowPlayerIndex):getComponent('Avatar')
  local rowResources = self._matrixComponent:getPlayerInventory(
    rowPlayerIndex)
  local rowPlayerNumItemsCollected = rowResources:sum()
  local rowProfile = rowResources:reshape{
      1,
      self._config.numResources
  }:clone()
  if rowPlayerNumItemsCollected > 0 then
    rowProfile = rowProfile:div(rowPlayerNumItemsCollected)
  end

  -- Column player setup
  local columnAvatar = self.gameObject.simulation:getAvatarFromIndex(
    columnPlayerIndex):getComponent('Avatar')
  local columnResources = self._matrixComponent:getPlayerInventory(
    columnPlayerIndex)
  local columnPlayerNumItemsCollected = columnResources:sum()
  local columnProfile = columnResources:reshape{
      self._config.numResources,
      1
  }:clone()
  if columnPlayerNumItemsCollected > 0 then
    columnProfile = columnProfile:div(columnPlayerNumItemsCollected)
  end

  -- Calculate payoffs of the interaction for both players.
  local rowReward, columnReward = self:_computeInteractionRewards(rowProfile,
                                                                  columnProfile)

  -- Applly the reward multiplier (it is 1.0 by default).
  rowReward = self._config.rewardMultiplier * rowReward
  columnReward = self._config.rewardMultiplier * columnReward

  self:reportInteraction(rowPlayerIndex, columnPlayerIndex,
                         rowReward, columnReward,
                         rowResources, columnResources,
                         rowAvatar, columnAvatar)
  local hitterZapper = hitterGameObject:getComponent('GameInteractionZapper')
  hitterZapper:reportInteraction(rowPlayerIndex, columnPlayerIndex,
                                 rowReward, columnReward,
                                 rowResources, columnResources,
                                 rowAvatar, columnAvatar)

  self:reportEventAndCumulants(rowPlayerIndex, columnPlayerIndex,
                               rowReward, columnReward,
                               rowResources, columnResources,
                               rowAvatar, columnAvatar)

  -- The player who scored lower dies by default.
  local rowPlayerWon, columnPlayerWon
  if rowReward > columnReward then
    rowPlayerWon = true
  elseif rowReward == columnReward then
    local sceneObject = self.gameObject.simulation:getSceneObject()
    local randomTieBreaking = sceneObject:getComponent(
        'TheMatrix').randomTieBreaking
    if randomTieBreaking then
      -- Break ties randomly if enabled.
      if random:uniformReal(0, 1) <= 0.5 then
        rowPlayerWon = true
      else
        columnPlayerWon = true
      end
    else
      -- By default, the row player wins when rewards are tied.
      rowPlayerWon = true
    end
  else
    columnPlayerWon = true
  end

  -- Set num frames till scheduled effects occur.
  local rowZapper = rowAvatar.gameObject:getComponent('GameInteractionZapper')
  local columnZapper = columnAvatar.gameObject:getComponent(
      'GameInteractionZapper')
  rowZapper:setFramesTillScheduledEffects(self._config.freezeOnInteraction)
  columnZapper:setFramesTillScheduledEffects(self._config.freezeOnInteraction)
  -- Send actual rewards to avatars, taking into account tastes.
  local rewardEffect = {
      func = self.sendRewardsToBothInteractants,
      arguments = {self, rowReward, columnReward,
                         rowResources, columnResources,
                         rowAvatar, columnAvatar},
  }
  table.insert(self._scheduledEffects, rewardEffect)
  -- Populate table of instructions to implement either now or on a later frame.
  if rowPlayerWon then
    -- The row player won so the column player dies.
    if self._config.reset_loser_inventory then
      local effect = {
        func = self._matrixComponent.resetInventory,
        arguments = {self._matrixComponent, columnResources, columnPlayerIndex}
      }
      table.insert(self._scheduledEffects, effect)
    end
    if self._config.reset_winner_inventory then
      self._matrixComponent:resetInventory(rowResources, rowPlayerIndex)
      local effect = {
        func = self._matrixComponent.resetInventory,
        arguments = {self._matrixComponent, rowResources, rowPlayerIndex}
      }
      table.insert(self._scheduledEffects, effect)
    end
    if self._config.losingPlayerDies then
      local effect = {func = self._avatarDies, arguments = {self, columnAvatar}}
      table.insert(self._scheduledEffects, effect)
    end
    if self._config.winningPlayerDies then
      local effect = {func = self._avatarDies, arguments = {self, rowAvatar}}
      table.insert(self._scheduledEffects, effect)
    end
  elseif columnPlayerWon then
    -- The column player won so the row player dies.
    if self._config.reset_loser_inventory then
      local effect = {
        func = self._matrixComponent.resetInventory,
        arguments = {self._matrixComponent, rowResources, rowPlayerIndex}
      }
      table.insert(self._scheduledEffects, effect)
    end
    if self._config.reset_winner_inventory then
      local effect = {
        func = self._matrixComponent.resetInventory,
        arguments = {self._matrixComponent, columnResources, columnPlayerIndex}
      }
      table.insert(self._scheduledEffects, effect)
    end
    if self._config.losingPlayerDies then
      local effect = {func = self._avatarDies, arguments = {self, rowAvatar}}
      table.insert(self._scheduledEffects, effect)
    end
    if self._config.winningPlayerDies then
      local effect = {func = self._avatarDies, arguments = {self, columnAvatar}}
      table.insert(self._scheduledEffects, effect)
    end
  end
  -- If there are any scheduled effects then disallow movement until they occur.
  if #self._scheduledEffects > 0 then
    -- Adding 2 to `freezeOnInteraction` ensures it is impossible to move on the
    -- last frame before the scheduled effects occur.
    local numFramesToFreeze = self._config.freezeOnInteraction + 2
    rowAvatar:disallowMovementUntil(numFramesToFreeze)
    columnAvatar:disallowMovementUntil(numFramesToFreeze)

    rowZapper._rewardToDetermineColor = rowReward
    columnZapper._rewardToDetermineColor = columnReward
  end
end

function GameInteractionZapper:_preventExtraSimultaneousInteraction(
      hitterGameObject)
  -- Prevent interaction of more than two players at a time, in one step.
  if self.interactedThisStep then
    return true
  end
  self.interactedThisStep = true
  local hitterZapper = hitterGameObject:getComponent('GameInteractionZapper')
  if hitterZapper.interactedThisStep then
    return true
  end
  hitterZapper.interactedThisStep = true
  return false
end

function GameInteractionZapper:onHit(hitterGameObject, hitName)
  if hitName == 'gameInteraction' then
    if self:_preventExtraSimultaneousInteraction(hitterGameObject) then
      -- Beams do not pass through.
      return true
    end

    if self._framesTillScheduledEffects >= 0 then
      -- It is not possible to interact with a player who is already interacting
      -- with another player, i.e. players cannot be zapped while frozen.
      -- Return true so beams do not pass through the hit player.
      return true
    end

    local theMatrix = self.gameObject.simulation:getSceneObject():getComponent(
        'TheMatrix')

    local zapperAvatar = hitterGameObject:getComponent('Avatar')
    local zapperIdx = zapperAvatar:getIndex()

    local zappedAvatar = self.gameObject:getComponent('Avatar')
    local zappedIdx = zappedAvatar:getIndex()

    -- Deliver a reward to any player who zaps a player who is not yet ready (ie
    -- who has not yet collected any resources). Typically this reward will be
    -- be zero in which case there is no penalty from zapping an unready player.
    -- In some cases we may make this reward negative in order to discourage
    -- players from zapping others who have not yet collected resources.
    if not theMatrix.playerCollectedAtLeastOneResource[zappedIdx] then
      zapperAvatar:addReward(self._config.rewardFromZappingUnreadyPlayer)
    end

    -- If disallowUnreadyInteractions = true then don't bother computing the
    -- effect of the interaction unless both players are ready.
    if theMatrix.disallowUnreadyInteractions then
      local bothReady = (
          theMatrix.playerCollectedAtLeastOneResource[zapperIdx] and
          theMatrix.playerCollectedAtLeastOneResource[zappedIdx])
      if not bothReady then
        return true  -- block the beam from passing through the hit player.
      end
    end

    -- At this point the interaction is definitely going to be resolved. So we
    -- can safely set the cumulant for both zapper and zapped players.
    self:_setInteractionCumulant(hitterGameObject, self.gameObject)

    if self.gameObject:hasComponent(
        'DyadicRole') and hitterGameObject:hasComponent('DyadicRole') then
      -- If the role component is present then assign row versus column player
      -- according to role.
      local zapperRole = hitterGameObject:getComponent('DyadicRole')
      local zappedRole = self.gameObject:getComponent('DyadicRole')
      -- Only resolve interactions when roles are discordant. That is, a row
      -- player can only interact with a column player and vice versa. A row
      -- player cannot interact with another row player. A column player cannot
      -- interact with another column player.
      if zapperRole:isRowPlayer() and not zappedRole:isRowPlayer() then
        self:_resolve(zapperIdx, zappedIdx, hitterGameObject)
      elseif not zapperRole:isRowPlayer() and zappedRole:isRowPlayer() then
        self:_resolve(zappedIdx, zapperIdx, hitterGameObject)
      end
    else
      -- By default the zapper avatar is the row player and the zapped avatar is
      -- the column player.
      self:_resolve(zapperIdx, zappedIdx, hitterGameObject)
    end

    -- Return `true` to prevent beams from passing through hit players.
    return true
  end
end

function GameInteractionZapper:reportInteraction(rowPlayerIdx, colPlayerIdx,
                                                 rowReward, colReward,
                                                 rowInventory, colInventory,
                                                 rowAvatar, columnAvatar)
  -- Am I being called on the row or column player?
  local selfIndex = self.gameObject:getComponent('Avatar'):getIndex()
  local selfIsRowPlayer = selfIndex == rowPlayerIdx
  if not selfIsRowPlayer then
    assert(selfIndex == colPlayerIdx, 'Self was neither row nor column player.')
  end
  -- Update variables which can be read by a metric reporter.
  if selfIsRowPlayer then
    -- List self first. In this case, self was the row player.
    self.latest_interaction_inventories(1):val(rowInventory:val())
    self.latest_interaction_inventories(2):val(colInventory:val())
  else
    -- List self first. In this case, self was the column player.
    self.latest_interaction_inventories(1):val(colInventory:val())
    self.latest_interaction_inventories(2):val(rowInventory:val())
  end
end

function GameInteractionZapper:reportEventAndCumulants(
    rowPlayerIdx, colPlayerIdx, rowReward, colReward, rowInventory,
    colInventory, rowAvatar, columnAvatar)

  -- Event: interaction: event_type, key1, X, key2, Y, ...
  events:add('interaction', 'dict',
             'row_player_idx', rowPlayerIdx,
             'col_player_idx', colPlayerIdx,
             'row_reward', rowReward,
             'col_reward', colReward,
             'row_inventory', rowInventory,
             'col_inventory', colInventory)

  -- Update cumulants so they can be read by a metric reporter.
  rowAvatar.gameObject:getComponent(
      'GameInteractionZapper'):setArgMaxCumulants(rowInventory)
  columnAvatar.gameObject:getComponent(
      'GameInteractionZapper'):setArgMaxCumulants(colInventory)
end

function GameInteractionZapper:setArgMaxCumulants(inventory)
  if inventory:max(1):val() > 0 then
    local indexOfMaximalItemType = inventory:argMax(1):val()
    local cumulantName = 'argmax_interaction_inventory_was_' .. tostring(
        indexOfMaximalItemType)
    self[cumulantName] = 1
  end
end

function GameInteractionZapper:_resetBinaryCumulants()
  -- Note: cumulant names are in Python style to suggest that they will mainly
  -- be used from there. Also, as observations they will be in all capitals
  -- so the camel case style won't be possible in that form.
  self.interacted_this_step = 0
  self.argmax_interaction_inventory_was_1 = 0
  self.argmax_interaction_inventory_was_2 = 0
  self.argmax_interaction_inventory_was_3 = 0
  self.collected_resource_1 = 0
  self.collected_resource_2 = 0
  self.collected_resource_3 = 0
  self.destroyed_resource_1 = 0
  self.destroyed_resource_2 = 0
  self.destroyed_resource_3 = 0
end

function GameInteractionZapper:_setInteractionCumulant(zapSourceObject,
                                                      zapTargetObject)
  local zapSourceGameInteractionComponent = zapSourceObject:getComponent(
      'GameInteractionZapper')
  local zapTargetGameInteractionComponent = zapTargetObject:getComponent(
      'GameInteractionZapper')
  -- Set interaction cumulant value to 1 for both zapper and zapped players as
  -- both of them were part of an interaction on this frame.
  zapSourceGameInteractionComponent.interacted_this_step = 1
  zapTargetGameInteractionComponent.interacted_this_step = 1
end

function GameInteractionZapper:start()
  -- Set the beam cooldown timer to its `ready` state (i.e. coolingTimer = 0).
  self._coolingTimer = 0
  self.latest_interaction_inventories = tensor.DoubleTensor(
    2, self._config.numResources)

  -- Create variables to hold cumulant data (scalars that can be used as
  -- instantaneous reward signals for generalized value function learning.
  self:_resetBinaryCumulants()

  self._scheduledEffects = {}
  self._framesTillScheduledEffects = -1

  self._endEpisodeOnNextFrame = false

  self._canZap = true
end

function GameInteractionZapper:postStart()
  local sceneObject = self.gameObject.simulation:getSceneObject()
  self._matrixComponent = sceneObject:getComponent('TheMatrix')
end

function GameInteractionZapper:update()
  -- The last interaction variable will be reported as an observation. When no
  -- interaction occured on the previous frame then it will always be set to
  -- the impossible inventory values of {-1, -1}
  self.latest_interaction_inventories:fill(-1)
  -- Reset cumulants.
  self:_resetBinaryCumulants()
end

function GameInteractionZapper:getAliveState()
  return self.gameObject:getComponent('Avatar'):getAliveState()
end

function GameInteractionZapper:getWaitState()
  return self.gameObject:getComponent('Avatar'):getWaitState()
end

function GameInteractionZapper:readyToShoot()
  local normalizedTimeTillReady = self._coolingTimer / self._config.cooldownTime
  return 1 - normalizedTimeTillReady
end

function GameInteractionZapper:setResourceCollectionCumulant(resourceClass)
  local cumulantName = 'collected_resource_' .. tostring(resourceClass)
  self[cumulantName] = 1
end

function GameInteractionZapper:setResourceDestructionCumulant(resourceClass)
  local cumulantName = 'destroyed_resource_' .. tostring(resourceClass)
  self[cumulantName] = 1
end

function GameInteractionZapper:disallowZapping()
  self._canZap = false
end

function GameInteractionZapper:allowZapping()
  self._canZap = true
end


local InventoryObserver = class.Class(component.Component)

function InventoryObserver:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('InventoryObserver')},
  })
  InventoryObserver.Base.__init__(self, kwargs)
end

function InventoryObserver:addObservations(tileSet,
                                           world,
                                           observations,
                                           avatarCount)
  local playerIdx = self.gameObject:getComponent('Avatar'):getIndex()
  local stringPlayerIdx = tostring(playerIdx)

  local sceneObject = self.gameObject.simulation:getSceneObject()
  local matrixComponent = sceneObject:getComponent('TheMatrix')
  local numResources = matrixComponent:getNumResources()
  observations[#observations + 1] = {
      name = stringPlayerIdx .. '.INVENTORY',
      type = 'tensor.DoubleTensor',
      shape = {numResources},
      func = function(grid)
        return matrixComponent:getPlayerInventory(playerIdx)
      end
  }
end


local Taste = class.Class(component.Component)

function Taste:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Taste')},
      {'mostTastyResourceClass', args.numberType},
      {'mostTastyReward', args.default(1), args.numberType},
      {'defaultTastinessReward', args.default(0), args.numberType},
  })
  Taste.Base.__init__(self, kwargs)
  self._kwargs = kwargs
end

function Taste:reset()
  self.mostTastyResourceClass = self._kwargs.mostTastyResourceClass
  self.mostTastyReward = self._kwargs.mostTastyReward
  self.defaultTastinessReward = self._kwargs.defaultTastinessReward
end

function Taste:getRewardForGathering(resourceClass)
  if resourceClass == self.mostTastyResourceClass then
    return self.mostTastyReward
  end
  return self.defaultTastinessReward
end


local InteractionTaste = class.Class(component.Component)

function InteractionTaste:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('InteractionTaste')},
      -- Setting `mostTastyResource` to -1 disables it.
      {'mostTastyResourceClass', args.default(-1), args.numberType},
      -- Add `extra reward` to the reward that would otherwise be obtained in
      -- the case of an interaction playing strategy `mostTastyResourceClass`.
      {'extraReward', args.default(0), args.numberType},
      -- Zero the reward that would otherwise be obtained so all interaction
      -- rewards to this player come from `extraReward`.
      {'zeroDefaultInteractionReward', args.default(false), args.booleanType},
  })
  InteractionTaste.Base.__init__(self, kwargs)
  self._kwargs = kwargs
end

function InteractionTaste:reset()
  self.extraReward = self._kwargs.extraReward
  self.mostTastyResourceClass = self._kwargs.mostTastyResourceClass
  self.zeroDefaultInteractionReward = self._kwargs.zeroDefaultInteractionReward
end

function InteractionTaste:getExtraRewardForInteraction(reward, inventory)
  if self.mostTastyResourceClass > 0 then
    if self.zeroDefaultInteractionReward then
      reward = 0.0
    end
    local amountMostTastyResource = inventory(self.mostTastyResourceClass):val()
    local mostTastyIsMaximal = true
    for idx = 1, inventory:size() do
      if idx ~= self.mostTastyResourceClass then
        mostTastyIsMaximal = amountMostTastyResource > inventory(idx):val()
      end
    end
    if mostTastyIsMaximal then
      return reward + self.extraReward
    end
  end
  -- Case when mostTastyResourceClass is < 0, so this component does nothing.
  return reward
end


local DyadicRole = class.Class(component.Component)

function DyadicRole:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('DyadicRole')},
      {'rowPlayer', args.booleanType},
  })
  DyadicRole.Base.__init__(self, kwargs)
  self._rowPlayer = kwargs.rowPlayer
end

function DyadicRole:isRowPlayer()
  return self._rowPlayer
end


--[[ `ReadyToInteractMarker` adds an extra visual element on top of the players
who have collected at least one resource since their inventory was last reset
according to TheMatrix.
]]
local ReadyToInteractMarker = class.Class(component.Component)

function ReadyToInteractMarker:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('ReadyToInteractMarker')},
      -- `playerIndex` of the avatar to which this object is attached.
      {'playerIndex', args.positive},
  })
  ReadyToInteractMarker.Base.__init__(self, kwargs)
  self._config.playerIndex = kwargs.playerIndex
end

function ReadyToInteractMarker:registerUpdaters(updaterRegistry)
  local simulation = self.gameObject.simulation
  local sceneObject = simulation:getSceneObject()
  local theMatrix = sceneObject:getComponent('TheMatrix')

  local displayReadiness = function()
    local avatarObject = simulation:getAvatarFromIndex(self._config.playerIndex)
    local indicatorState = theMatrix.indicators[self._config.playerIndex]
    if avatarObject:getComponent('Avatar'):isAlive() then
      self.gameObject:setState(indicatorState)
    elseif avatarObject:getComponent('Avatar'):isWait() then
      self.gameObject:setState('avatarMarkingWait')
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = displayReadiness,
      priority = 2,
  }
end


local DisallowMovement = class.Class(component.Component)

function DisallowMovement:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('DisallowMovement')},
  })
  DisallowMovement.Base.__init__(self, kwargs)
end

function DisallowMovement:postStart()
  local avatar = self.gameObject:getComponent('Avatar')
  local gameInteractionZapper = self.gameObject:getComponent(
      'GameInteractionZapper')
  avatar:disallowMovement()
  gameInteractionZapper:disallowZapping()
end


local InitializeAsReadyToInteract = class.Class(component.Component)

function InitializeAsReadyToInteract:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('InitializeAsReadyToInteract')},
      {'playerIndex', args.numberType},
  })
  InitializeAsReadyToInteract.Base.__init__(self, kwargs)
  self._config.playerIndex = kwargs.playerIndex
end

function InitializeAsReadyToInteract:registerUpdaters(updaterRegistry)
  local endEpisodeIfApplicable = function()
    -- End the episode when the flag is set.
    if self._endEpisodeOnNextFrame then
      self.gameObject.simulation:endEpisode()
    end
  end
  updaterRegistry:registerUpdater{
      updateFn = endEpisodeIfApplicable,
      priority = 900,
  }
end

function InitializeAsReadyToInteract:onHit(hitterGameObject, hitName)
  if hitName == 'gameInteraction' then
    self._endEpisodeOnNextFrame = true
    return true
  end
end

function InitializeAsReadyToInteract:postStart()
  local sceneObject = self.gameObject.simulation:getSceneObject()
  local theMatrix = sceneObject:getComponent('TheMatrix')

  self._endEpisodeOnNextFrame = false

  theMatrix.indicators[self._config.playerIndex] = 'ready'
  theMatrix.playerCollectedAtLeastOneResource[self._config.playerIndex] = true
end


local allComponents = {
    -- Object components
    Resource = Resource,
    Destroyable = Destroyable,

    -- Avatar components
    GameInteractionZapper = GameInteractionZapper,
    InventoryObserver = InventoryObserver,
    Taste = Taste,
    InteractionTaste = InteractionTaste,
    DyadicRole = DyadicRole,
    ReadyToInteractMarker = ReadyToInteractMarker,

    -- Avatar debug components
    DisallowMovement = DisallowMovement,
    InitializeAsReadyToInteract = InitializeAsReadyToInteract,

    -- Scene components
    TheMatrix = TheMatrix,
    SpawnResourcesWhenAllPlayersZapped = SpawnResourcesWhenAllPlayersZapped,
}

component_registry.registerAllComponents(allComponents)

return allComponents
