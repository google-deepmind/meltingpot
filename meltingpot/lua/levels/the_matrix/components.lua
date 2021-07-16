--[[ Copyright 2020 DeepMind Technologies Limited.

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


local Resource = class.Class(component.Component)

function Resource:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Resource')},
      {'resourceClass', args.numberType},
      {'visibleType', args.stringType},
      {'waitState', args.stringType},
      {'groupToRespawn', args.stringType},
      {'regenerationRate', args.ge(0.0), args.le(1.0)},
      {'regenerationDelay', args.positive},
  })
  Resource.Base.__init__(self, kwargs)
  self._config.resourceClass = kwargs.resourceClass
  self._config.visibleType = kwargs.visibleType
  self._config.waitState = kwargs.waitState
  self._config.groupToRespawn = kwargs.groupToRespawn
  self._config.regenerationRate = kwargs.regenerationRate
  self._config.regenerationDelay = kwargs.regenerationDelay
end

function Resource:reset()
  self._variables = {}
  self._variables._regenTimer = self._config.regenerationDelay
end

function Resource:onStateChange()
  self._variables._regenTimer = self._config.regenerationDelay
end

function Resource:onEnter(enteringGameObject, contactName)
  if self.gameObject:getState() == self._config.visibleType
      and contactName == 'avatar' then
    local sceneObject = self.gameObject.simulation:getSceneObject()
    -- Increment inventory.
    local playerIndex = enteringGameObject:getComponent('Avatar'):getIndex()
    local inventory = (
        sceneObject:getComponent('TheMatrix'):getPlayerInventory(playerIndex))
    local amount = inventory(self._config.resourceClass):val()
    inventory(self._config.resourceClass):fill(amount + 1)
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
  end
end

function Resource:update()
  if self._variables._regenTimer <= 0 then
    if random:uniformReal(0, 1) < self._config.regenerationRate then
      self.gameObject:setState(self._config.visibleType)
    end
  end
  self._variables._regenTimer = self._variables._regenTimer - 1
end

function Resource:getResourceClass()
  return self._config.resourceClass
end

function Resource:_reportCollectionEvent(playerIndex, resourceClass)
  events:add('collected_resource', 'dict',
             'player_index', playerIndex,
             'class', resourceClass)
end


local Destroyable = class.Class(component.Component)

function Destroyable:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Destroyable')},
      {'initialHealth', args.positive},
      {'visibleType', args.stringType},
      {'waitState', args.stringType},
  })
  Destroyable.Base.__init__(self, kwargs)
  self._config.initialHealth = kwargs.initialHealth
  self._config.visibleType = kwargs.visibleType
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
      {'zero_initial_inventory', args.default(false), args.booleanType},
      -- By default, row players win whenever rewards are tied.
      {'randomTieBreaking', args.default(false), args.booleanType},
  })
  TheMatrix.Base.__init__(self, kwargs)
  -- Matrices are always square since all players have the same available
  -- strategies as one another.
  self._config.numResources = #kwargs.matrix
  self._config.zero_initial_inventory = kwargs.zero_initial_inventory
  self._rowPlayerMatrix = tensor.DoubleTensor(self:_cleanMatrix(kwargs.matrix))

  self.randomTieBreaking = kwargs.randomTieBreaking

  if kwargs.columnPlayerMatrix then
    self._columnPlayerMatrix = tensor.DoubleTensor(
        self:_cleanMatrix(kwargs.columnPlayerMatrix))
  else
    self._columnPlayerMatrix = self._rowPlayerMatrix:clone():transpose(1, 2)
  end
end

function TheMatrix:reset()
  self.playerResources = tensor.DoubleTensor(
      self.gameObject.simulation:getNumPlayers(),
      self._config.numResources)
  if self._config.zero_initial_inventory then
    self.playerResources:fill(0)
  else
    self.playerResources:fill(1)
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

function TheMatrix:resetInventory(inventory)
  if self._config.zero_initial_inventory then
    inventory:fill(0)
  else
    inventory:fill(1)
  end
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
  })
  GameInteractionZapper.Base.__init__(self, kwargs)

  self._config.cooldownTime = kwargs.cooldownTime
  self._config.beamLength = kwargs.beamLength
  self._config.beamRadius = kwargs.beamRadius
  self._config.framesTillRespawn = kwargs.framesTillRespawn

  self._config.numResources = kwargs.numResources

  self._config.endEpisodeOnFirstInteraction =
      kwargs.endEpisodeOnFirstInteraction
  self._config.reset_winner_inventory = kwargs.reset_winner_inventory
  self._config.reset_loser_inventory = kwargs.reset_loser_inventory

  self._config.losingPlayerDies = kwargs.losingPlayerDies
  self._config.winningPlayerDies = kwargs.winningPlayerDies
  self._config.rewardFloor = kwargs.rewardFloor
end

function GameInteractionZapper:addHits(worldConfig)
  worldConfig.hits['gameInteraction'] = {
      layer = 'beamInteraction',
      sprite = 'BeamInteraction',
  }
  table.insert(worldConfig.renderOrder, 'beamInteraction')
end

function GameInteractionZapper:addSprites(tileSet)
  -- This color is yellow.
  tileSet:addColor('BeamInteraction', {252, 252, 106})
end

function GameInteractionZapper:registerUpdaters(updaterRegistry)
  local aliveState = self:getAliveState()
  local waitState = self:getWaitState()

  local zap = function()
    local playerVolatileVariables = (
        self.gameObject:getComponent('Avatar'):getVolatileData())
    local actions = playerVolatileVariables.actions
    -- Execute the beam if applicable.
    if self.gameObject:getState() == aliveState then
      if self._config.cooldownTime >= 0 then
        if self._coolingTimer > 0 then
          self._coolingTimer = self._coolingTimer - 1
        else
          if actions['interact'] == 1 then
            self._coolingTimer = self._config.cooldownTime
            self.gameObject:hitBeam('gameInteraction',
                                    self._config.beamLength,
                                    self._config.beamRadius)
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

function GameInteractionZapper:_resolve(rowPlayerIndex, columnPlayerIndex)
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

  -- Send actual rewards, taking into account tastes.
  if self.gameObject:hasComponent('InteractionTaste') then
    local interactionTaste = self.gameObject:getComponent('InteractionTaste')
    if rowReward > self._config.rewardFloor then
      rowAvatar:addReward(interactionTaste:getExtraRewardForInteraction(
        rowReward, rowResources))
    end
    if columnReward > self._config.rewardFloor then
      columnAvatar:addReward(interactionTaste:getExtraRewardForInteraction(
        columnReward, columnResources))
    end
  else
    if rowReward > self._config.rewardFloor then
      rowAvatar:addReward(rowReward)
    end
    if columnReward > self._config.rewardFloor then
      columnAvatar:addReward(columnReward)
    end
  end

  self:_reportInteraction(rowPlayerIndex, columnPlayerIndex,
                          rowReward, columnReward,
                          rowResources, columnResources)

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

  if rowPlayerWon then
    -- The row player won so the column player dies.
    if self._config.reset_loser_inventory then
      self._matrixComponent:resetInventory(columnResources)
    end
    if self._config.reset_winner_inventory then
      self._matrixComponent:resetInventory(rowResources)
    end
    if self._config.losingPlayerDies then
      self:_avatarDies(columnAvatar)
    end
    if self._config.winningPlayerDies then
      self:_avatarDies(rowAvatar)
    end
  elseif columnPlayerWon then
    -- The column player won so the row player dies.
    if self._config.reset_loser_inventory then
      self._matrixComponent:resetInventory(rowResources)
    end
    if self._config.reset_winner_inventory then
      self._matrixComponent:resetInventory(columnResources)
    end
    if self._config.losingPlayerDies then
      self:_avatarDies(rowAvatar)
    end
    if self._config.winningPlayerDies then
      self:_avatarDies(columnAvatar)
    end
  end
end

function GameInteractionZapper:onHit(hitterGameObject, hitName)
  if hitName == 'gameInteraction' then
    local zapperAvatar = hitterGameObject:getComponent('Avatar')
    local zapperIdx = zapperAvatar:getIndex()

    local zappedAvatar = self.gameObject:getComponent('Avatar')
    local zappedIdx = zappedAvatar:getIndex()

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
        self:_resolve(zapperIdx, zappedIdx)
      elseif not zapperRole:isRowPlayer() and zappedRole:isRowPlayer() then
        self:_resolve(zappedIdx, zapperIdx)
      end
    else
      -- By default the zapper avatar is the row player and the zapped avatar is
      -- the column player.
      self:_resolve(zapperIdx, zappedIdx)
    end

    if self._config.endEpisodeOnFirstInteraction then
      self.gameObject.simulation:endEpisode()
    end

    -- Return `true` to prevent beams from passing through hit players.
    return true
  end
end

function GameInteractionZapper:_reportInteraction(rowPlayerIdx, colPlayerIdx,
                                                  rowReward, colReward,
                                                  rowInventory, colInventory)
  -- Event: interaction: event_type, key1, X, key2, Y, ...
  events:add('interaction', 'dict',
             'row_player_idx', rowPlayerIdx,
             'col_player_idx', colPlayerIdx,
             'row_reward', rowReward,
             'col_reward', colReward,
             'row_inventory', rowInventory,
             'col_inventory', colInventory)
  -- Also update variables which can be read by a metric reporter.
  self.latest_interaction_inventories(1):val(rowInventory:val())
  self.latest_interaction_inventories(2):val(colInventory:val())
end

function GameInteractionZapper:start()
  -- Set the beam cooldown timer to its `ready` state (i.e. coolingTimer = 0).
  self._coolingTimer = 0
  self.latest_interaction_inventories = tensor.DoubleTensor(
    2, self._config.numResources)
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

    -- Scene components
    TheMatrix = TheMatrix,
}

component_registry.registerAllComponents(allComponents)

return allComponents
