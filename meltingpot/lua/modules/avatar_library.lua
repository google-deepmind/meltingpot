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

--[[ A library of common components that may be added to avatar game objects.
]]

local helpers = require 'common.helpers'
local log = require 'common.log'
local class = require 'common.class'
local args = require 'common.args'
local set = require 'common.set'
local tables = require 'common.tables'
local random = require 'system.random'
local tensor = require 'system.tensor'
local tile = require 'system.tile'
local events = require 'system.events'

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')

local _COMPASS = {'N', 'E', 'S', 'W'}


--[[ The `Avatar` component sets up a GameObject to be controlled by an agent.
]]
local Avatar = class.Class(component.Component)

function Avatar:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Avatar')},
      -- `index` (int): player index for the game object bearing this component.
      {'index', args.numberType},
      -- `spawnGroup` (str): group of objects where this avatar may spawn.
      {'spawnGroup', args.stringType},
      -- `postInitialSpawnGroup` (str): group of objects where this avatar may
      --   spawn after its initial spawn (optional, if not provided then it will
      --   default to taking the same value as `spawnGroup`.
      {'postInitialSpawnGroup', args.default('_DEFAULT'), args.stringType},
      -- `aliveState` (str): the state of this object when it is live, i.e.,
      --     when agents and humans can control it.
      {'aliveState', args.stringType},
      -- `additionalLiveStates` (table[str]): additional states when agents and
      --     humans can control the avatar beyond the one named in `aliveState`.
      --     The main reason one would want to name `additionalLiveStates` is to
      --     correctly handle connection and disconnection logic when an avatar
      --     could change state to wait state from any number of live states.
      {'additionalLiveStates', args.default({}), args.tableType},
      -- `waitState` (str): an inactive state, used when it is not able to move
      --     and receiving black frames as observations, e.g. when it is "dead".
      {'waitState', args.stringType},
      -- `speed` (float >=0 and <= 1.0): If 1.0 then all movement actions are
      --   executed, otherwise prevent movements with probability 1 - speed.
      {'speed', args.default(1.0), args.ge(0.0), args.le(1.0)},
      {'actionOrder', args.default({'move', 'turn'}), args.tableType},
      {'actionSpec', args.default(
        {
            move = {default = 0, min = 0, max = #_COMPASS},
            turn = {default = 0, min = -1, max = 1},
        }), args.tableType},
      -- `view`: A configuration table each containing the following entries.
      --   {
      --     left (int),
      --     right (int),
      --     forward (int),
      --     backward (int),
      --     centered (boolean)
      --   }
      {'view', args.tableType},
      -- If `useAbsoluteCoordinates` is true then the avatar moves by selecting
      -- an absolute direction {North, East, South, West} rather than moving
      -- forward/backward/turning along its facing direction. That is, it
      -- replaces the underlying DMLab2D calls to `moveRel` and `turn` by
      -- `moveAbs` and `setOrientation` respectively
      {'useAbsoluteCoordinates', args.default(false), args.booleanType},
      -- `spriteMap` (table {sourceSprite1=targetSprite1,
      --                     sourceSprite2=targetSprite2, ...}) remap source
      -- sprites to target sprites in the view from this avatar's perspective.
      {'spriteMap', args.default({}), args.tableType},
      -- `skipWaitStateRewards` (bool) default True. When true, do not reward
      -- avatars when they are in wait state.
      {'skipWaitStateRewards', args.default(true), args.booleanType},
      -- `randomizeInitialOrientation` (bool) default True. Avatar orientations
      -- are assigned randomly at the start of each episode. If you instead
      -- set this to false then initial Avatar orientation is always North.
      {'randomizeInitialOrientation', args.default(true), args.booleanType},
  })
  Avatar.Base.__init__(self, kwargs)
  self._config.kwargs = kwargs

  self._config.actionOrder = kwargs.actionOrder
  self._config.actionSpec = kwargs.actionSpec
  self._config.view = kwargs.view

  self._config._index = kwargs.index
  self._config.initialSpawnGroup = kwargs.spawnGroup
  self._config.spriteMap = kwargs.spriteMap

  self._config.aliveState = kwargs.aliveState
  self._config.waitState = kwargs.waitState

  self._config.liveStates = kwargs.additionalLiveStates
  table.insert(self._config.liveStates, self._config.aliveState)
  self._config.liveStatesSet = set.Set(self._config.liveStates)

  self._config.skipWaitStateRewards = kwargs.skipWaitStateRewards
  self._config.randomizeInitialOrientation = kwargs.randomizeInitialOrientation

  if kwargs.postInitialSpawnGroup ~= '_DEFAULT' then
    self._config.postInitialSpawnGroup = kwargs.postInitialSpawnGroup
  end
  self._spawnGroup = self._config.initialSpawnGroup
end

-- Call initializeVolatileVariables during `awake` and `reset`.
function Avatar:_initializeVolatileVariables()
  local kwargs = self._config.kwargs
  self.useAbsoluteCoordinates = kwargs.useAbsoluteCoordinates
  self._speed = kwargs.speed
  self._movementAllowed = true
  self._playerVolatileVariables = {}
  self._connectedPiecesSet = {}
  self._freezeCounter = 0
  self._removalCounter = 0
end

function Avatar:awake()
  self:_initializeVolatileVariables()
end

--[[ Map sprites to other sprites from the view perspective of this avatar.

Note: this function should be called during start-up, i.e. before the call
to "addObservations". A good place to call it is in "awake".

Args:
  spriteMap: {originalSprite1 = newSprite1, originalSprite2 = newSprite2, ...}
]]
function Avatar:setSpriteMap(spriteMap)
  self._config.spriteMap = spriteMap
end

function Avatar:registerUpdaters(updaterRegistry)
  local move = function()
    if self._movementAllowed then
      local actions = self._playerVolatileVariables.actions
      if actions['turn'] and actions['turn'] ~= 0 then
        -- Turn the avatar game object.
        self.gameObject:turn(actions['turn'])
        -- Additionally, turn all connected game objects.
        for _, connectedObject in ipairs(self:getConnectedObjects()) do
          connectedObject:turn(actions['turn'])
        end
      end
      if actions['move'] and actions['move'] ~= 0 then
        self.gameObject:moveRel(_COMPASS[actions['move']])
      end
    end
  end

  local moveAbsolute = function()
    if self._movementAllowed then
      local actions = self._playerVolatileVariables.actions
      if actions['turn'] and actions['turn'] ~= 0 then
        -- Turn the avatar game object.
        self.gameObject:setOrientation(_COMPASS[actions['turn']])
        -- Additionally, turn all connected game objects.
        for _, connectedObject in ipairs(self:getConnectedObjects()) do
          connectedObject:setOrientation(_COMPASS[actions['turn']])
        end
      end
      if actions['move'] and actions['move'] ~= 0 then
        self.gameObject:moveAbs(_COMPASS[actions['move']])
      end
    end
  end

  if self.useAbsoluteCoordinates then
    updaterRegistry:registerUpdater{
        updateFn = moveAbsolute,
        priority = 150,
        probability = self._speed,
    }
  else
    updaterRegistry:registerUpdater{
        updateFn = move,
        priority = 150,
        probability = self._speed,
    }
  end
end

function Avatar:discreteActionSpec(actSpec)
  self._config._actionSpecStartOffset = #actSpec
  local id = tostring(self._config._index)
  for a, actionName in ipairs(self._config.actionOrder) do
    local action = self._config.actionSpec[actionName]
    table.insert(actSpec, {
        name = id .. '.' .. actionName,
        min = action.min,
        max = action.max,
    })
  end
end

function Avatar:discreteActions(actions)
  local psActions = self._playerVolatileVariables.actions
  for a, actionName in ipairs(self._config.actionOrder) do
    psActions[actionName] = actions[a + self._config._actionSpecStartOffset]
  end
end

function Avatar:addObservations(tileSet, world, observations)
  local id = self._config._index
  local stringId = tostring(id)
  local playerViewConfig = {
      left = self._config.view.left,
      right = self._config.view.right,
      forward = self._config.view.forward,
      backward = self._config.view.backward,
      centered = self._config.view.centered,
      set = tileSet,
      spriteMap = self._config.spriteMap,
  }

  observations[#observations + 1] = {
      name = stringId .. '.REWARD',
      type = 'Doubles',
      shape = {},
      func = function(grid)
        return self._playerVolatileVariables.reward
      end
  }

  local playerLayerView = world:createView(playerViewConfig)
  local playerLayerViewSpec =
      playerLayerView:observationSpec(stringId .. '.LAYER')
  playerLayerViewSpec.func = function(grid)
    return playerLayerView:observation{
        grid = grid,
        piece = self.gameObject:getPiece(),
        orientation = 'N'
    }
  end
  observations[#observations + 1] = playerLayerViewSpec

  local playerView = tile.Scene{
      shape = playerLayerView:gridSize(),
      set = tileSet
  }

  local spec = {
      name = stringId .. '.RGB',
      type = 'tensor.ByteTensor',
      shape = playerView:shape(),
      func = function(grid)
        local layer_observation = playerLayerView:observation{
            grid = grid,
            piece = self.gameObject:getPiece(),
        }
        return playerView:render(layer_observation)
      end
  }
  observations[#observations + 1] = spec
end

function Avatar:reset()
  self:_initializeVolatileVariables()
end

function Avatar:_startFrame()
  self._playerVolatileVariables.reward = 0
end

--[[ `locator` must be a valid piece id.]]
function Avatar:start(locator)
  self:_startFrame()
  local actions = {}
  for a, actionName in ipairs(self._config.actionOrder) do
    local action = self._config.actionSpec[actionName]
    actions[actionName] = action.default
  end
  self._playerVolatileVariables = {
      reward = 0,
      actions = actions
  }
  local targetTransform = self.gameObject._grid:transform(locator)
  if self._config.randomizeInitialOrientation then
    targetTransform.orientation = random:choice(_COMPASS)
  else
    targetTransform.orientation = 'N'
  end

  local uniqueState = self.gameObject:getUniqueState(
    self.gameObject:getState())
  local piece = self.gameObject._grid:createPiece(
      uniqueState, targetTransform)
  -- Prevent silent failures to create the avatar piece.
  assert(piece,
         'Failed to create avatar piece of type: ' .. uniqueState ..
         ' at: ' .. helpers.tostringOneLine(targetTransform) .. '.')
  self.gameObject:getComponent('Transform'):setPieceAfterDeferredCreation(piece)

  -- Record the avatar startup event.
  events:add('AvatarStarted', 'str', 'success')

  return piece
end

function Avatar:postStart()
  if self._config.postInitialSpawnGroup then
    -- The first spawn was at the externally provided `spawnGroup`. Subsequent
    -- spawns will be at the provided `postInitialSpawnGroup` if applicable.
    self._spawnGroup = self._config.postInitialSpawnGroup
  end
end

function Avatar:preUpdate()
  self:_startFrame()
end

function Avatar:_handleTimedFreeze()
  local oldFreezeCounter = self._freezeCounter
  if oldFreezeCounter == 1 then
    self:allowMovement()
  end
  self._freezeCounter = math.max(self._freezeCounter - 1, 0)
end

function Avatar:_handleScheduledRemoval()
  local oldRemovalCounter = self._removalCounter
  if oldRemovalCounter == 1 then
    local waitState = self:getWaitState()
    self.gameObject:setState(waitState)
  end
  self._removalCounter = math.max(self._removalCounter - 1, 0)
end

function Avatar:update()
  self:_handleTimedFreeze()
  self:_handleScheduledRemoval()
end

function Avatar:getIndex()
  return self._config._index
end

function Avatar:getSpawnGroup()
  return self._spawnGroup
end

function Avatar:addReward(amount)
  local function _addReward(amount)
    self._playerVolatileVariables.reward = (
          self._playerVolatileVariables.reward + amount)
  end
  if self._config.skipWaitStateRewards then
    -- Only add rewards if avatar is not in a wait state.
    if self.gameObject:getState() ~= self._config.waitState then
      _addReward(amount)
    end
  else
    -- Add rewards regardless of whether avatar is in wait state.
    _addReward(amount)
  end
end

function Avatar:getReward()
  return self._playerVolatileVariables.reward
end

-- Return a table with avatar's actions and rewards on the current frame.
function Avatar:getVolatileData()
  return self._playerVolatileVariables
end

--[[ Connects two avatar game objects together in such a way that if one object
is pushed or turned, then they are all pushed or turned.
]]
function Avatar:connect(objectToConnect)
  local piece = objectToConnect:getPiece()
  if not self._connectedPiecesSet[piece] then
    self.gameObject:connect(objectToConnect)
    self._connectedPiecesSet[piece] = true
  end
end

function Avatar:disconnect(objectToDisconnect)
  local piece = objectToDisconnect:getPiece()
  if self._connectedPiecesSet[piece] then
    self._connectedPiecesSet[piece] = nil
    objectToDisconnect:disconnect()
  end
end

--[[ Return an array of connected objects in arbitrary order.]]
function Avatar:getConnectedObjects()
  local sim = self.gameObject.simulation
  local result = {}
  for piece in pairs(self._connectedPiecesSet) do
    table.insert(result, sim:getGameObjectFromPiece(piece))
  end
  return result
end

function Avatar:getAllConnectedObjectsWithNamedComponent(componentName)
  local connectedObjects = self:getConnectedObjects()
  -- Select the first connected object with component named `componentName`.
  local resultObjects = {}
  for _, connectedObject in ipairs(connectedObjects) do
    if connectedObject:hasComponent(componentName) then
      table.insert(resultObjects, connectedObject)
    end
  end
  return resultObjects
end

function Avatar:onStateChange(oldState)
  local newState = self.gameObject:getState()
  local waitState = self:getWaitState()
  local behavior
  if oldState == waitState and self._config.liveStatesSet[newState] then
    behavior = 'respawn'
    self._freezeCounter = 0
    self._removalCounter = 0
  elseif self._config.liveStatesSet[oldState] and newState == waitState then
    behavior = 'die'
  end
  if behavior then
    for i, connectedObject in ipairs(self:getConnectedObjects()) do
      -- Check if there are components that have the function
      -- `avatarStateChange` and call them if so.
      local components = connectedObject:getComponents()
      for _, component in ipairs(components) do
        if component.avatarStateChange then
          component:avatarStateChange(behavior)
        end
      end
    end
  end
end

function Avatar:getAliveState()
  return self._config.aliveState
end

function Avatar:getWaitState()
  return self._config.waitState
end

--[[ Prevent an avatar from moving. Other actions not handled by the `Avatar`
component such as zapping are unaffected.]]
function Avatar:disallowMovement()
  self._movementAllowed = false
end

--[[ No need to call `allowMovement` unless after calling `disallowMovement`.]]
function Avatar:allowMovement()
  self._movementAllowed = true
end

--[[ Prevent movement for `numFrames` steps, then allow it again.]]
function Avatar:disallowMovementUntil(numFrames)
  if numFrames > 0 then
    self:disallowMovement()
    self._freezeCounter = numFrames
  end
end

--[[ Return true if movement is allowed and false otherwise.]]
function Avatar:isMovementAllowed()
  return self._movementAllowed
end

--[[ Remove the avatar (set it to wait state) after `delay` elapsed timesteps.]]
function Avatar:removeAfterDelay(delay)
  self._removalCounter = delay
end

--[[ Return true if avatar is in alive state.]]
function Avatar:isAlive()
  return self._config.liveStatesSet[self.gameObject:getState()]
end

--[[ Return true if avatar is in wait state.]]
function Avatar:isWait()
  return self.gameObject:getState() == self._config.waitState
end

--[[ Return all objects on the specified `layer` in the focal avatar's currently
viewable partial observation window.

Warning: may have undefined results if used with non partial observation
viewing modes (e.g. global views).
]]
function Avatar:queryPartialObservationWindow(layer)
  local transform = self.gameObject:getComponent('Transform')

  local l = self._config.view.left
  local r = self._config.view.right
  local f = self._config.view.forward
  local b = self._config.view.backward

  local absVector1 = transform:getAbsoluteDirectionFromRelative({-l, b})
  local absVector2 = transform:getAbsoluteDirectionFromRelative({r, -f})

  local position = self.gameObject:getPosition()
  local foundObjects = transform:queryRectangle(
      layer,
      {position[1] + absVector1[1], position[2] + absVector1[2]},
      {position[1] + absVector2[1], position[2] + absVector2[2]}
  )
  return foundObjects
end


--[[ The `AvatarDirectionIndicator` component draws the direction indicator in
front of the avatar.
]]
local AvatarDirectionIndicator = class.Class(component.Component)

function AvatarDirectionIndicator:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AvatarDirectionIndicator')},
      {'color', args.default({100, 100, 100, 200}), args.tableType},
  })
  AvatarDirectionIndicator.Base.__init__(self, kwargs)
  self._config.color = kwargs.color
end

function AvatarDirectionIndicator:addSprites(tileSet)
  tileSet:addColor('DirectionSprite', self._config.color)
end

function AvatarDirectionIndicator:addHits(worldConfig)
  worldConfig.hits['directionHit'] = {
      layer = 'directionIndicatorLayer',
      sprite = 'DirectionSprite',
  }
end

function AvatarDirectionIndicator:registerUpdaters(updaterRegistry)
  local drawNose = function()
    -- The "nose" has length 1 and radius 0.
    self.gameObject:hitBeam("directionHit", 1, 0)
  end

  updaterRegistry:registerUpdater{
      updateFn = drawNose,
      priority = 130,
  }
end


--[[ The `Zapper` component endows an avatar with the ability to fire a beam and
be hit by the beams of other avatars.
]]
local Zapper = class.Class(component.Component)

function Zapper:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Zapper')},
      {'cooldownTime', args.numberType},
      {'beamLength', args.numberType},
      {'beamRadius', args.numberType},
      -- The default beam color is yellow.
      {'beamColor', args.default({252, 252, 106}), args.tableType},
      {'framesTillRespawn', args.numberType},
      {'penaltyForBeingZapped', args.numberType},
      {'rewardForZapping', args.numberType},
      {'removeHitPlayer', args.default(true), args.booleanType},
  })
  Zapper.Base.__init__(self, kwargs)

  self._config.cooldownTime = kwargs.cooldownTime
  self._config.beamLength = kwargs.beamLength
  self._config.beamRadius = kwargs.beamRadius
  self._config.beamColor = kwargs.beamColor
  self._config.framesTillRespawn = kwargs.framesTillRespawn
  self._config.penaltyForBeingZapped = kwargs.penaltyForBeingZapped
  self._config.rewardForZapping = kwargs.rewardForZapping
  self._config.removeHitPlayer = kwargs.removeHitPlayer
end

function Zapper:addHits(worldConfig)
  worldConfig.hits['zapHit'] = {
      layer = 'beamZap',
      sprite = 'BeamZap',
  }
  component.insertIfNotPresent(worldConfig.renderOrder, 'beamZap')
end

function Zapper:addSprites(tileSet)
  tileSet:addColor('BeamZap', self._config.beamColor)
end

function Zapper:registerUpdaters(updaterRegistry)
  local aliveState = self:getAliveState()
  local waitState = self:getWaitState()

  local zap = function()
    local playerVolatileVariables = (
        self.gameObject:getComponent('Avatar'):getVolatileData())
    local actions = playerVolatileVariables.actions
    -- Execute the beam if applicable.
    if self.gameObject:getComponent('Avatar'):isAlive() then
      if self._config.cooldownTime >= 0 then
        if self._coolingTimer > 0 then
          self._coolingTimer = self._coolingTimer - 1
        else
          if actions['fireZap'] == 1 then
            self._coolingTimer = self._config.cooldownTime
            self.gameObject:hitBeam(
                'zapHit', self._config.beamLength, self._config.beamRadius)
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

function Zapper:onHit(hittingGameObject, hitName)
  if hitName == 'zapHit' then
    local zappedAvatar = self.gameObject:getComponent('Avatar')
    local zappedIndex = zappedAvatar:getIndex()
    local zapperAvatar = hittingGameObject:getComponent('Avatar')
    local zapperIndex = zapperAvatar:getIndex()
    if self.playerZapMatrix then
      self.playerZapMatrix(zappedIndex, zapperIndex):add(1)
    end
    events:add('zap', 'dict',
               'source', zapperAvatar:getIndex(),  -- int
               'target', zappedAvatar:getIndex())  -- int
    zappedAvatar:addReward(self._config.penaltyForBeingZapped)
    zapperAvatar:addReward(self._config.rewardForZapping)
    if self._config.removeHitPlayer then
      self.gameObject:setState(self:getWaitState())
    end
    -- Temporarily store the index of the zapper avatar in state so it can
    -- be observed elsewhere.
    self.zapperIndex = zapperIndex
    -- Temporarily record that the zapper hit another player on this frame.
    if hittingGameObject:hasComponent('Zapper') then
      local hittingZapper = hittingGameObject:getComponent('Zapper')
      hittingZapper.num_others_player_zapped_this_step = (
          hittingZapper.num_others_player_zapped_this_step + 1)
    end
    -- return `true` to prevent the beam from passing through a hit player.
    return true
  end
end

function Zapper:onStateChange()
  self._respawnTimer = self._config.framesTillRespawn
end

function Zapper:reset()
  self.playerRespawnedThisStep = false
  self._disallowZapping = false
  self._noZappingCounter = 0
  self.num_others_player_zapped_this_step = 0
end

function Zapper:start()
  local scene = self.gameObject.simulation:getSceneObject()
  self.playerZapMatrix = nil
  if scene:hasComponent("GlobalMetricHolder") then
    self.playerZapMatrix = scene:getComponent(
        "GlobalMetricHolder").playerZapMatrix
  end
  -- Set the beam cooldown timer to its `ready` state (i.e. coolingTimer = 0).
  self._coolingTimer = 0
end

function Zapper:_handleTimedZappingPrevention()
  local oldFreezeCounter = self._noZappingCounter
  self._noZappingCounter = math.max(self._noZappingCounter - 1, 0)
  if oldFreezeCounter == 1 then
    self:allowZapping()
  end
end

function Zapper:update()
  -- Metrics must be read from preUpdate since they will get reset in update.
  self.playerRespawnedThisStep = false
  self.zapperIndex = nil
  self.num_others_player_zapped_this_step = 0
  -- Note: After zapping is allowed again after having been disallowed, players
  -- still need to wait another `cooldownTime` frames before they can zap again.
  if self._disallowZapping then
    self._coolingTimer = self._config.cooldownTime + 1
  end
  self:_handleTimedZappingPrevention()
end

function Zapper:getAliveState()
  return self.gameObject:getComponent('Avatar'):getAliveState()
end

function Zapper:getWaitState()
  return self.gameObject:getComponent('Avatar'):getWaitState()
end

-- Integrate all signals that affect whether it is possible to zap into a single
-- float between 0 and 1. It is possible to use the zapping action when 1 is
-- returned. Zapping will be restored sooner the closer to 1 the signal becomes.
function Zapper:readyToShoot()
  local normalizedTimeTillReady = self._coolingTimer / self._config.cooldownTime
  if self.gameObject:getComponent('Avatar'):isAlive() then
    return math.max(1 - normalizedTimeTillReady, 0)
  else
    return 0
  end
end

function Zapper:disallowZapping()
  self._disallowZapping = true
end

function Zapper:allowZapping()
  self._disallowZapping = false
end

--[[ Prevent zapping for `numFrames` steps, then allow it again.]]
function Zapper:disallowZappingUntil(numFrames)
  self:disallowZapping()
  self._noZappingCounter = numFrames
end

-- Return `true` if hit players are removed and `false` otherwise.
function Zapper:getRemoveHitPlayer()
  return self._config.removeHitPlayer
end

--[[ Get all players in reach of being zapped by the focal player in one step.

Return an array of avatars who are in reach and can be zapped by the focal
player on this timestep (if called before actions are applied) or the next
timestep (if called after actions are applied).

Note: This function assumes avatars cannot move and shoot on the same frame.

Args:
  `layer` (string): the layer on which to search for avatar objects. By default,
      the layer will be set to the current layer of the focal game object.

Returns an unsorted array of avatar objects (game objects bearing the avatar
component).
]]
function Zapper:getWhoZappable(layer)
  local layer = layer or self.gameObject:getLayer()
  local transform = self.gameObject:getComponent('Transform')
  local zappableAvatars = {}

  -- Shift left and right by the correct amounts and create forward rays.
  local rays = {}
  for x = -self._config.beamRadius, self._config.beamRadius do
    local ray = {
        positionOffset = {x, 0},
        length = self._config.beamLength - math.abs(x),
        direction = nil
    }
    table.insert(rays, ray)
  end
  -- Send rays to the left and right.
  local leftVector = {-self._config.beamRadius, 0}
  local leftRay = {
      positionOffset = {0, 0},
      length = self._config.beamRadius,
      direction = transform:getAbsoluteDirectionFromRelative(leftVector),
  }
  local rightVector = {self._config.beamRadius, 0}
  local rightRay = {
      positionOffset = {0, 0},
      length = self._config.beamRadius,
      direction = transform:getAbsoluteDirectionFromRelative(rightVector),
  }
  table.insert(rays, leftRay)
  table.insert(rays, rightRay)

  -- Use one ray per horizontal offset (relative coordinate frame).
  for _, ray in ipairs(rays) do
    local success, foundObject, offset = transform:rayCastDirection(
      layer, ray.length, ray.direction, ray.positionOffset)
    -- TODO(b/192927376): Debug why we get ni foundObject when zapping in the
    -- upper-right corner.
    if (success and foundObject ~= nil and
        foundObject:hasComponent('Avatar')) then
      table.insert(zappableAvatars, foundObject)
    end
  end

  return zappableAvatars
end

--[[ Get indices of each player in reach of being zapped by the focal
player in one step.

Return an array of player indices who are in reach and can be zapped by the
focal player on this timestep (if called before actions are applied) or the next
timestep (if called after actions are applied).

Note: This function assumes avatars cannot move and shoot on the same frame.

Args:
  `layer` (string): the layer on which to search for avatar objects. By default,
      the layer will be set to the current layer of the focal game object.

Returns an unsorted array of player indices (note: we sometimes also call player
indices by the name "slot ids").
]]
function Zapper:getZappablePlayerIndices(layer)
  local zappableAvatars = self:getWhoZappable(layer)
  local zappablePlayerIndices = {}
  for i, avatarObject in ipairs(zappableAvatars) do
    table.insert(zappablePlayerIndices,
                 avatarObject:getComponent('Avatar'):getIndex())
  end
  return zappablePlayerIndices
end


--[[ The `ReadyToShootObservation` component adds an observation that is 1 when
the avatar can fire (from the Zapper component) and <1 if in cooldown time.

The resulting observation key will be `playerIndex`.READY_TO_SHOOT.
]]
local ReadyToShootObservation = class.Class(component.Component)

function ReadyToShootObservation:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('ReadyToShootObservation')},
      {'zapperComponent', args.default('Zapper'), args.stringType},
  })
  ReadyToShootObservation.Base.__init__(self, kwargs)
  self._config.zapperComponent = kwargs.zapperComponent
end

function ReadyToShootObservation:addObservations(tileSet, world, observations)
  local playerIndex = self.gameObject:getComponent('Avatar'):getIndex()
  local zapper = self.gameObject:getComponent(self._config.zapperComponent)

  observations[#observations + 1] = {
      name = tostring(playerIndex) .. '.READY_TO_SHOOT',
      type = 'Doubles',
      shape = {},
      func = function(grid)
        return zapper:readyToShoot()
      end
  }
end


local AvatarConnector = class.Class(component.Component)

function AvatarConnector:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AvatarConnector')},
      -- `playerIndex` (int): player index for the avatar to connect to.
      {'playerIndex', args.numberType},
      {'aliveState', args.stringType},
      {'waitState', args.stringType},
  })
  AvatarConnector.Base.__init__(self, kwargs)
  self._kwargs = kwargs
end

function AvatarConnector:reset()
  local kwargs = self._kwargs
  self._playerIndex = kwargs.playerIndex
  self._aliveState = kwargs.aliveState
  self._waitState = kwargs.waitState
end

--[[ Note that postStart is called from the avatar manager, after start has been
called on all other game objects, even avatars.]]
function AvatarConnector:postStart()
  local sim = self.gameObject.simulation
  self._avatarObject = sim:getAvatarFromIndex(self._playerIndex)

  -- Note that it is essential to set the state before teleporting it.
  -- This is because pieces with no assigned layer have no position, and thus
  -- cannot be teleported.
  self.gameObject:setState(self._aliveState)
  self.gameObject:teleport(self._avatarObject:getPosition(),
                           self._avatarObject:getOrientation())

  -- Get the avatar component on the avatar game object and connect to it.
  local avatarComponent = self._avatarObject:getComponent('Avatar')
  avatarComponent:connect(self.gameObject)
end

function AvatarConnector:avatarStateChange(behavior)
  local avatarComponent = self._avatarObject:getComponent('Avatar')
  -- If the avatar's state has changed, then also update the state of
  -- the avatar connector.
  if behavior == 'respawn' then
    avatarComponent:disconnect(self.gameObject)
    self.gameObject:setState(self._aliveState)
    -- When coming to life, also teleport to the right location.
    self.gameObject:teleport(self._avatarObject:getPosition(),
                             self._avatarObject:getOrientation())
    avatarComponent:connect(self.gameObject)
  elseif behavior == 'die' then
    self.gameObject:setState(self._waitState)
  end
end

function AvatarConnector:getAliveState()
  return self._aliveState
end

function AvatarConnector:getWaitState()
  return self._waitState
end


local GraduatedSanctionsMarking = class.Class(component.Component)

function GraduatedSanctionsMarking:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('GraduatedSanctionsMarking')},
      -- `playerIndex` is the index of the player to which the marking object
      -- is attached. This component will be on the marking object.
      {'playerIndex', args.numberType},
      -- `waitState` (str) the state to use when the avatar is zagged out.
      {'waitState', args.stringType},
      -- `initialLevel` (int) sets the initial level.
      {'initialLevel', args.default(1), args.nonNegative},
      -- `recoveryTime` (int or false) number of frames to wait before
      -- automatically setting level back to initial level.
      {'recoveryTime', args.default(false)},
      -- `hitName` (str) defines the `hit` to use to change the marking level.
      {'hitName', args.stringType},
      -- `hitLogic` (array with numLevels tables of the form:
      --     {levelIncrement = (int) x, amount to change level.
      --      sourceReward = (number) y, reward of hitting player.
      --      targetReward = (number) z reward of hit player.
      --      remove = (bool) false, whether to set hit player to wait state.
      --      freeze = (number or nil) nil, whether to freeze, and if so,
      --        for how long.}
      {'hitLogic',
        args.default({
            {levelIncrement = 1, sourceReward = 0, targetReward = -2,
              remove = false, freeze = nil},
            {levelIncrement = 1, sourceReward = 2, targetReward = -8,
              remove = false, freeze = nil},
            {levelIncrement = -2, sourceReward = 10, targetReward = -16,
             remove = false, freeze = nil},
        }),
        args.tableType},
  })
  self.Base.__init__(self, kwargs)
  self._kwargs = kwargs
end

function GraduatedSanctionsMarking:reset()
  local kwargs = self._kwargs
  self._playerIndex = kwargs.playerIndex
  self._waitState = kwargs.waitState
  self._hitName = kwargs.hitName
  self._hitLogic = kwargs.hitLogic
  self._maxLevel = #self._hitLogic
  self._recoveryTime = kwargs.recoveryTime
  self._initialLevel = kwargs.initialLevel

  self._level = kwargs.initialLevel
  self._timeSinceNotInitial = 0
end

function GraduatedSanctionsMarking:_assertNecessaryStatesExist()
  local states = set.Set(self.gameObject:getAllStates())
  for i = 1, self._maxLevel do
    local stateName = 'level_' .. tostring(i)
    assert(states[stateName], 'Missing state: ' .. stateName)
  end
end

function GraduatedSanctionsMarking:registerUpdaters(updaterRegistry)
  local resetToInitialLevel = function()
    local thisAvatar = self._avatarObject:getComponent('Avatar')
    if self._level ~= self._initialLevel and thisAvatar:isAlive() then
      self._timeSinceNotInitial = self._timeSinceNotInitial + 1
      if self._timeSinceNotInitial == self._recoveryTime then
        self._level = self._initialLevel
        self:_setLevel(self._initialLevel)
        self._timeSinceNotInitial = 0
      end
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = resetToInitialLevel,
      priority = 3,
  }
end

function GraduatedSanctionsMarking:start()
  -- Check that enough states exist on the game object to match hitLogic.
  self:_assertNecessaryStatesExist()
end

function GraduatedSanctionsMarking:postStart()
  local sim = self.gameObject.simulation
  self._avatarObject = sim:getAvatarFromIndex(self._playerIndex)
  local avatarComponent = self._avatarObject:getComponent('Avatar')
  self._avatarAliveState = avatarComponent:getAliveState()
  self._avatarWaitState = avatarComponent:getWaitState()

  -- Note that it is essential to set the state before teleporting it.
  -- This is because pieces with no assigned layer have no position, and thus
  -- cannot be teleported.
  self:_setLevel(self._level)
  self.gameObject:teleport(self._avatarObject:getPosition(),
                           self._avatarObject:getOrientation())

  -- Connect this object to the avatar game object.
  avatarComponent:connect(self.gameObject)
end

function GraduatedSanctionsMarking:onHit(hittingGameObject, hitName)
  if hitName == self._hitName then
    local logic = self._hitLogic[self._level]
    local hittingAvatar = hittingGameObject:getComponent('Avatar')
    hittingAvatar:addReward(logic.sourceReward)
    local thisAvatar = self._avatarObject:getComponent('Avatar')
    thisAvatar:addReward(logic.targetReward)
    self._level = self._level + logic.levelIncrement
    if rawget(logic, 'remove') then
      -- Remove the hit player if the logic says to do so.
      -- The actual removal happens one frame after the zap. This was done so
      -- that the removed avatar can still be seen by custom observations
      -- computed on the same frame as the zap.
      local delay = 1
      thisAvatar:removeAfterDelay(delay)
      -- Prevent the zapped avatar from retaliating during the delay frame.
      thisAvatar:disallowMovementUntil(delay)
      self._avatarObject:getComponent('Zapper'):disallowZappingUntil(delay)
      -- Record the removal event:
      events:add('removal_due_to_sanctioning', 'dict',
                 'source', hittingAvatar:getIndex(), -- int (the zapping player)
                 'target', thisAvatar:getIndex()) -- int (the removed player)
    else
      -- Set the hit player to their new level according to the logic.
      self:_setLevel(self._level)

      local freeze = rawget(logic, 'freeze')
      if freeze then
        thisAvatar:disallowMovementUntil(freeze)
        self._avatarObject:getComponent('Zapper'):disallowZappingUntil(freeze)
      end
    end

    -- Reset the time since non initial level transition.
    self._timeSinceNotInitial = 0

    -- Record the sanctioning event:
    events:add('sanctioning', 'dict',
               'source', hittingAvatar:getIndex(),  -- int
               'target', thisAvatar:getIndex())  -- int
  end
  -- Do not handle whether beams are blocked or not here, use the underlying
  -- object instead.
  return false
end

function GraduatedSanctionsMarking:avatarStateChange(behavior)
  local avatarComponent = self._avatarObject:getComponent('Avatar')
  -- If the avatar's state has changed, then also update the state of
  -- the avatar connector.
  if behavior == 'respawn' then
    avatarComponent:disconnect(self.gameObject)
    -- Set the respawning player's marking level.
    self:_setLevel(self._level)
    -- When coming to life, also teleport to the right location.
    self.gameObject:teleport(self._avatarObject:getPosition(),
                             self._avatarObject:getOrientation())
    avatarComponent:connect(self.gameObject)
  elseif behavior == 'die' then
    self.gameObject:setState(self._waitState)
  end
end

function GraduatedSanctionsMarking:_setLevel(level)
  self.gameObject:setState('level_' .. tostring(level))
  -- Record the level change event:
  local thisAvatar = self._avatarObject:getComponent('Avatar')
  events:add('set_sanctioning_level', 'dict',
             'player_index', thisAvatar:getIndex(),  -- int
             'level', level)  -- int
end


--[[ Implement a "drive" that the avatar must satisfy periodically to avoid
incurring a (usually negative) reward per step.

The `PeriodicNeed` component waits for `delay` steps and then delivers
a reward of `reward` per step on every subsequent step. Call the function
`resetDriveLevel` to reset the countdown back up to its initial value.]]
local PeriodicNeed = class.Class(component.Component)

function PeriodicNeed:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('PeriodicNeed')},
      -- `delay` defines how many frames before starting to deliver `reward`
      -- on every step (hunger pain). Eating resets the countdown.
      {'delay', args.positive},
      -- `reward` the reward value to deliver on every step after `delay`.
      {'reward', args.numberType},
  })
  PeriodicNeed.Base.__init__(self, kwargs)
  self._config.delay = kwargs.delay
  self._config.reward = kwargs.reward
end

function PeriodicNeed:reset()
  self._hungerLevel = self._config.delay
end

function PeriodicNeed:update()
  self._hungerLevel = self._hungerLevel - 1
  if self._hungerLevel <= 0 then
    self.gameObject:getComponent('Avatar'):addReward(self._config.reward)
  end
end

-- Call this function to reset the countdown, i.e., to "satisfy" the need.
function PeriodicNeed:resetDriveLevel()
  self._hungerLevel = self._config.delay
end

--[[ Return the current need amount as a number between 0 and 1.

Returns 0 when hungerLevel = delay and returns 1 when hungerLevel <= 0.
]]
function PeriodicNeed:getNeed()
  local normalizedTimeTillActive = self._hungerLevel / self._config.delay
  if self.gameObject:getComponent('Avatar'):isAlive() then
    return math.max(1 - normalizedTimeTillActive, 0)
  else
    return 0
  end
end


--[[ `Role` provides a convenient way to store role data for an avatar.
]]
local Role = class.Class(component.Component)

function Role:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Role')},
      -- `role` (string): returned by `getRole()`.
      {'role', args.stringType},
  })
  Role.Base.__init__(self, kwargs)
  self._kwargs = kwargs
  self._role = kwargs.role
end

function Role:reset()
  self._role = self._kwargs.role
end

function Role:getRole()
  return self._role
end

function Role:setRole(newRole)
  self._role = newRole
end


--[[ The `AvatarIdsInViewObservation` component adds an observation of which
avatar ids (player slot ids) are currently within the focal avatar's partial
observation viewing window.

The AVATAR_IDS_IN_VIEW observation is a binary vector of length num_players.
]]
local AvatarIdsInViewObservation = class.Class(component.Component)

function AvatarIdsInViewObservation:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AvatarIdsInViewObservation')},
      -- `layers` can optionally be used to provide custom layers over which to
      -- look for avatars. If you do not provide any layers then the layer the
      -- avatar spawns on in the first frame of the episode will be used.
      {'layers', args.default({}), args.tableType},
  })
  AvatarIdsInViewObservation.Base.__init__(self, kwargs)
  self._layers = kwargs.layers
end

function AvatarIdsInViewObservation:reset()
  if #self._layers == 0 then
    table.insert(self._layers, self.gameObject:getLayer())
  end
end

function AvatarIdsInViewObservation:_getQueryResult(layers)
  local avatarComponent = self.gameObject:getComponent('Avatar')
  -- First get all the avatar ids in a list.
  local resultsList = {}
  for _, layer in ipairs(layers) do
    local objectsOnLayer = avatarComponent:queryPartialObservationWindow(layer)
    for _, object in ipairs(objectsOnLayer) do
      if object:hasComponent('Avatar') then
        local index = object:getComponent('Avatar'):getIndex()
        table.insert(resultsList, index)
      end
    end
  end
  -- Then reformat the avatar ids list as a binary int32 tensor to output.
  local numPlayers = self.gameObject.simulation:getNumPlayers()
  local resultTensor = tensor.Int32Tensor(numPlayers):fill(0)
  for _, avatarId in ipairs(resultsList) do
    resultTensor(avatarId):add(1)
  end
  return resultTensor
end

function AvatarIdsInViewObservation:addObservations(
    tileSet, world, observations)
  local playerIndex = self.gameObject:getComponent('Avatar'):getIndex()

  observations[#observations + 1] = {
      name = tostring(playerIndex) .. '.AVATAR_IDS_IN_VIEW',
      type = 'tensor.Int32Tensor',
      shape = {self.gameObject.simulation:getNumPlayers()},
      func = function(grid)
        return self:_getQueryResult(self._layers)
      end
  }
end


--[[ The `AvatarIdsInRangeToZapObservation` component adds an observation of
which avatar ids (player slot ids) can be reached by a zap on the current frame.

The AVATAR_IDS_IN_RANGE_TO_ZAP observation is a binary vector of length equal to
the number of players in the episode.
]]
local AvatarIdsInRangeToZapObservation = class.Class(component.Component)

function AvatarIdsInRangeToZapObservation:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AvatarIdsInRangeToZapObservation')},
  })
  AvatarIdsInRangeToZapObservation.Base.__init__(self, kwargs)
end

function AvatarIdsInRangeToZapObservation:reset()
  local numPlayers = self.gameObject.simulation:getNumPlayers()
  self._resultTensor = tensor.Int32Tensor(numPlayers):fill(0)
end

function AvatarIdsInRangeToZapObservation:_getQueryResult()
  self._resultTensor:fill(0)
  -- No Ids are ever in range to zap if the focal avatar is dead.
  local layer = self.gameObject:getLayer()
  if layer and #layer > 0 then
    -- First get all the avatar ids in a list.
    local avatarIds = self.gameObject:getComponent(
      'Zapper'):getZappablePlayerIndices()
    -- Then reformat the avatar ids list as a binary int32 tensor to output.
    for _, avatarId in ipairs(avatarIds) do
      self._resultTensor(avatarId):add(1)
    end
  end
  return self._resultTensor
end

function AvatarIdsInRangeToZapObservation:addObservations(
    tileSet, world, observations)
  local playerIndex = self.gameObject:getComponent('Avatar'):getIndex()

  observations[#observations + 1] = {
      name = tostring(playerIndex) .. '.AVATAR_IDS_IN_RANGE_TO_ZAP',
      type = 'tensor.Int32Tensor',
      shape = {self.gameObject.simulation:getNumPlayers()},
      func = function(grid)
        return self:_getQueryResult()
      end
  }
end


local allComponents = {
    -- Components that are typically on the avatar itself.
    Avatar = Avatar,
    AvatarDirectionIndicator = AvatarDirectionIndicator,
    Zapper = Zapper,
    ReadyToShootObservation = ReadyToShootObservation,
    PeriodicNeed = PeriodicNeed,
    Role = Role,
    AvatarIdsInViewObservation = AvatarIdsInViewObservation,
    AvatarIdsInRangeToZapObservation = AvatarIdsInRangeToZapObservation,

    -- Components that are typically part of other game objects that are
    -- connected to the avatar.
    AvatarConnector = AvatarConnector,
    GraduatedSanctionsMarking = GraduatedSanctionsMarking,
}

-- Register all components from this module in the component registry.
component_registry.registerAllComponents(allComponents)

return allComponents
