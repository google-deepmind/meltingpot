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

-- Simulations implement game logic by maintaining game objects and calling
-- their functions at specific times. This file implements the base class from
-- which level-specific simulations inherit.

local class = require 'common.class'
local helpers = require 'common.helpers'
local log = require 'common.log'
local random = require 'system.random'
local read_settings = require 'common.read_settings'
local tile = require 'system.tile'

local meltingpot = 'meltingpot.lua.modules.'
local game_object = require(meltingpot .. 'game_object')
local component_registry = require(meltingpot .. 'component_registry')
local prefab_utils = require(meltingpot .. 'prefab_utils')
local updater_registry = require(meltingpot .. 'updater_registry')

-- Functions to track from components in a game object.
_COMPONENT_FUNCTIONS = {
    'awake', 'reset', 'start', 'postStart', 'preUpdate', 'update', 'onBlocked',
    'onEnter', 'onExit', 'onHit', 'onStateChange', 'registerUpdaters',
    'addHits', 'addSprites', 'addCustomSprites', 'addObservations',
    'addPlayerCallbacks'}

--[[ The base class of all Simulations.  This object is the container of all
GameObjects, and maintains a registry of them, along with a table mapping states
to GameObjects.
]]

--[[ BaseSimulation holds the list of GameObjects, they determine game behavior.
]]
local BaseSimulation = class.Class()

function BaseSimulation.defaultSettings()
  return {
      map = '',
      charPrefabMap = read_settings.any(),
      prefabs = read_settings.any(),
      buildAvatars = false,
      playerPalettes = read_settings.any(),
      gameObjects = read_settings.default(
        {name = '', components = read_settings.any()}),
      -- The scene is a game object that holds global state/logic.
      scene = {name = 'scene', components = read_settings.any()},
      -- `worldSpriteMap` is an optional table that remaps specific sprites to
      -- other sprites in the WORLD.RGB observation. For example,
      -- {sourceSprite1=targetSprite1, sourceSprite2=targetSprite2, ...}.
      worldSpriteMap = read_settings.any(),
  }
end

local function _stringifyKeys(table)
  newTable = {}
  if table then
    for k, v in pairs(table) do
      newTable[tostring(k)] = v
    end
  end
  return newTable
end

function BaseSimulation:__init__(kwargs)
  self._settings = kwargs.settings
  self._settings.numPlayers = kwargs.numPlayers
  self._settings.charPrefabMap = _stringifyKeys(self._settings.charPrefabMap)
  self._settings.worldSpriteMap = self._settings.worldSpriteMap

  self._variables = {}
  self._variables.gameObjects = {}
  self._variables.avatarObjects = {}
  -- This table contains the mapping from a dmlab2d piece to the game object
  -- that owns that piece.
  self._variables.pieceToGameObject = {}
  self._variables.nextGameObjectId = 1

  -- Initialize avatar indexing tables to be populated in base avatar manager.
  self._variables.avatarPieceToIndex = {}
  self._variables.avatarIndexToPiece = {}

  -- Table of game objects indexed by component functions, to keep track of
  -- only game objects which have at least one component with a function.
  self._variables.objectsByFunctionName = {}
  for _, fnName in pairs(_COMPONENT_FUNCTIONS) do
    self._variables.objectsByFunctionName[fnName] = {}
  end

  -- Add the "scene", a static game object that can hold global logic.
  if self._settings.scene ~= nil then
    self._settings.sceneObject = self:buildGameObjectFromSettings(
        self._settings.scene)
  end

  if self._settings.charPrefabMap ~= nil and self._settings.prefabs ~= nil then
    -- Create game objects from prefabs, ASCII map, and chars to prefabs.
    local objects = prefab_utils.buildGameObjectConfigs(
        self._settings.map, self._settings.prefabs,
        self._settings.charPrefabMap)
    for _, gameObjectConfig in ipairs(objects) do
      table.insert(self._settings.gameObjects, gameObjectConfig)
    end
  end

  if self._settings.buildAvatars then
    -- Create avatar game objects from prefab, and palettes.
    local avatars = prefab_utils.buildAvatarConfigs(
        self._settings.numPlayers, self._settings.prefabs,
        self._settings.playerPalettes)
    for _, avatarConfig in ipairs(avatars) do
      table.insert(self._settings.gameObjects, avatarConfig)
    end
  end

  -- Instantiate and add all game objects (including avatars).
  if self._settings.gameObjects then
    for _, gameObjectConfig in ipairs(self._settings.gameObjects) do
      local gameObject = self:buildGameObjectFromSettings(gameObjectConfig)
      log.v(2, "Creating game object with id " .. gameObject._id)
    end
  end

  -- Check that we do have enough promised avatars
  local numAvatars = 0
  for k, v in pairs(self._variables.avatarObjects) do
    numAvatars = numAvatars + 1
  end
  assert(
    numAvatars == self._settings.numPlayers,
    "Created an environment with " .. self._settings.numPlayers .. " players, "
    .. "but provided " .. numAvatars .. " avatar objects instead. Avatars " ..
    "must be created exactly once, either passed in `game_objects` settings, "
    .. "or by passing `prefabs` containing \"avatar\" and setting " ..
    "`buildAvatars` to `true`.")
end

local function _makeComponent(config, isAvatar)
  local component = config.component
  local kwargs = config.kwargs
  -- Avatar pieces must be created by the Avatar component instead of the
  -- transform component, so must set the `deferPieceCreation` flag.
  if isAvatar and component == 'Transform' then
    kwargs.deferPieceCreation = true
  end
  return component_registry.getComponent(component)(kwargs)
end

local function _defaultStateManager()
  return _makeComponent({
      component = "StateManager",
      kwargs = {
          initialState = "scene",
          stateConfigs = {{
              state = "scene",
          }},
      }},
      false)
end

local function _defaultTransform(isAvatar)
  return _makeComponent({
      component = "Transform",
      kwargs = {
          position = {0, 0},
          orientation = "N",
      }},
      isAvatar)
end

--[[ This function builds a game object from the given configuration and
registers it into the simulation's object tables.  This function is aware of
whether the game object requested is an Avatar object or not.  At the end of
this call, the game object will be almost fully initialised, and contain a
special property: `simulation` that refers to this object.
]]
function BaseSimulation:buildGameObjectFromSettings(gameObjectConfig)
  local componentsConfig = gameObjectConfig.components
  local configuredComponents = {}

  -- Is this an Avatar?
  local isAvatar = false
  for _, config in ipairs(componentsConfig) do
    if config.component == 'Avatar' then
      isAvatar = true
    end
  end

  local hasStateManager = false
  local hasTransform = false
  for _, config in ipairs(componentsConfig) do
    if config.component == "StateManager" then
      hasStateManager = true
    end
    if config.component == "Transform" then
      hasTransform = true
    end
    table.insert(configuredComponents, _makeComponent(config, isAvatar))
  end
  local name = ''
  if rawget(gameObjectConfig, "name") ~= nil then
    name = gameObjectConfig.name
  end

  -- Add default components (with error-level logging to discourage).
  if not hasStateManager then
    log.warn(
      "GameObject '" .. name .. "' did not have a StateManager component, " ..
      "but explicitly specifying one is strongly preferred. Using a default.")
    table.insert(configuredComponents, _defaultStateManager())
  end
  if not hasTransform then
    log.warn(
      "GameObject '" .. name .. "' did not have a Transform component, " ..
      "but explicitly specifying one is strongly preferred. Using a default.")
    table.insert(configuredComponents, _defaultTransform(isAvatar))
  end

  local gameObject = game_object.GameObject{
      id = 'OID_' .. name .. '_' .. self._variables.nextGameObjectId,
      name = name,
      components = configuredComponents
  }
  self._variables.nextGameObjectId = self._variables.nextGameObjectId + 1

  gameObject.simulation = self
  table.insert(self._variables.gameObjects, gameObject)
  for _, fnName in pairs(_COMPONENT_FUNCTIONS) do
    if gameObject:hasComponentWithFunction(fnName) then
      table.insert(self._variables.objectsByFunctionName[fnName], gameObject)
    end
  end
  if isAvatar then
    self._variables.avatarObjects[gameObject._id] = gameObject
  end
  return gameObject
end

--[[ The following callbacks are called during initialisation (in this order) ]]

function BaseSimulation:worldConfig()
  -- This config contains data that you expect not to change during (or between)
  -- episodes. GameObjects add data to its fields.
  local config = {
      outOfBoundsSprite = 'OutOfBounds',
      outOfViewSprite = 'OutOfView',
      -- `updateOrder` sets the order updates are to be called on each frame.
      updateOrder = {},
      -- `renderOrder` is the draw order for layers. The alpha channel allows
      -- control of transparency/opacity viewing to lower layers.
      renderOrder = {
          'logic',
          'alternateLogic',
          'background',
          'lowerPhysical',
          'upperPhysical',  -- Avatars are normally on layer `upperPhysical`.
          'overlay',
          'superOverlay',
      },
      -- `customSprites` holds sprites that can be specified in a spriteMap but
      -- are otherwise not attached to any state. The main use-case for
      -- spriteMaps is the self-vs-other player view type.
      customSprites = {},
      -- `hits` determine the names of the callbacks to execute when a piece
      -- collides with a beam.
      hits = {},
      -- `states` is a dictionary of state configuration tables.
      states = {}
  }
  -- By this point, gameObjects already contain the avatar objects
  for _, gameObject in pairs(
      self._variables.objectsByFunctionName.registerUpdaters) do
    gameObject:registerUpdaters()
  end
  for _, gameObject in pairs(self._variables.gameObjects) do
    gameObject:addStates(config.states)
  end
  for _, gameObject in pairs(
      self._variables.objectsByFunctionName.addHits) do
    gameObject:addHits(config)
  end
  for _, gameObject in pairs(
      self._variables.objectsByFunctionName.addCustomSprites) do
    gameObject:addCustomSprites(config.customSprites)
  end
  self._contacts = {}
  self._updaterRegistry = updater_registry.UpdaterRegistry()
  -- Add all contacts from the registered states into a list to be passed later
  -- to all GameObjects so they can register their callbacks (like with hits).
  for _, stateElement in pairs(config.states) do
    if stateElement.contact then
      table.insert(self._contacts, stateElement.contact)
    end
  end
  -- Notify all GameObjects of the hits and contacts table so they can register
  -- callbacks.
  for _, gameObject in pairs(self._variables.gameObjects) do
    gameObject:setHits(config.hits)
    gameObject:setContacts(self._contacts)
    -- Merge all updaters into the simulation registry.
    self._updaterRegistry:mergeWith(gameObject:getUpdaterRegistry())
  end
  self._updaterRegistry:addUpdateOrder(config.updateOrder)

  log.v(1, 'World Config: ' .. helpers.tostring(config))

  return config
end

function BaseSimulation:addSprites(tileSet)
  tileSet:addColor('OutOfBounds', {0, 0, 0})
  tileSet:addColor('OutOfView', {80, 80, 80})
  for _, gameObject in pairs(
      self._variables.objectsByFunctionName.addSprites) do
    gameObject:addSprites(tileSet)
  end
end

function BaseSimulation:discreteActionSpec()
  local act = {}
  for _, avatarObject in pairs(self._variables.avatarObjects) do
    avatarObject:discreteActionSpec(act)
  end
  return act
end

function BaseSimulation:discreteActions(actions)
  for _, avatarObject in pairs(self._variables.avatarObjects) do
    avatarObject:discreteActions(actions)
  end
end

-- This function adds a third-person, global view of the world as well as any
-- observations from game objects (e.g. for the Avatars).
function BaseSimulation:addObservations(tileSet, world, observations)
  local worldLayerView = world:createView{
      layout = self:textMap().layout,
      spriteMap = self._settings.worldSpriteMap,
  }

  local worldView = tile.Scene{shape = worldLayerView:gridSize(), set = tileSet}
  local spec = {
      name = 'WORLD.RGB',
      type = 'tensor.ByteTensor',
      shape = worldView:shape(),
      func = function(grid)
        return worldView:render(worldLayerView:observation{grid = grid})
      end,
  }
  observations[#observations + 1] = spec
  -- Add all observations from GameObjects, including avatars.
  for _, gameObject in pairs(
      self._variables.objectsByFunctionName.addObservations) do
    gameObject:addObservations(tileSet, world, observations)
  end
end

--[[ End of initialisation callbacks ]]
--[[ The following callbacks are called during starting (in this order) ]]

function BaseSimulation:stateCallbacks(callbacks)
  -- By now we have the hits and the contacts lists.
  for _, gameObject in pairs(self._variables.gameObjects) do
    gameObject:addTypeCallbacks(callbacks)
  end
  for _, gameObject in pairs(
      self._variables.objectsByFunctionName.addPlayerCallbacks) do
    gameObject:addPlayerCallbacks(callbacks)
  end
  self._updaterRegistry:registerCallbacks(callbacks)
  log.v(1, 'Callbacks: ' .. helpers.tostring(callbacks))
end

function BaseSimulation:textMap()
  if not self._settings.map then
    return self._settings.mapsModule[self._settings.mapName]
  else
    return {layout = self._settings.map,
            stateMap = {}}
  end
end

--- After start on all game objects, start the avatar game objects.
function BaseSimulation:_avatarStart(grid)
  -- First determine the number of avatars assigned to each spawn group so they
  -- can be sampled without replacement to avoid spawn collisions.
  local avatarsPerSpawnGroup = {}
  -- We need to cache the spawn groups, just in case they are dynamic. The
  -- reason being that we need them both for computing how many will go to each
  -- group, and later to actually place them in the right group.
  local cachedAvatarSpawnGroups = {}
  for key, avatarObject in pairs(self._variables.avatarObjects) do
    local spawnGroup = avatarObject:getComponent('Avatar'):getSpawnGroup()
    cachedAvatarSpawnGroups[key] = spawnGroup
    if avatarsPerSpawnGroup[spawnGroup] then
      avatarsPerSpawnGroup[spawnGroup] = avatarsPerSpawnGroup[spawnGroup] + 1
    else
      avatarsPerSpawnGroup[spawnGroup] = 1
    end
  end

  -- Sample the right number of points at which to spawn avatars in each group.
  local spawnPointsByGroup = {}
  local spawnCountersByGroup = {}
  for spawnGroup, numAvatarsThisGroup in pairs(avatarsPerSpawnGroup) do
    spawnPointsByGroup[spawnGroup] = grid:groupShuffledWithCount(
      random, spawnGroup, numAvatarsThisGroup)
    assert(#spawnPointsByGroup[spawnGroup] == numAvatarsThisGroup,
           "Insufficient spawn points!")
    spawnCountersByGroup[spawnGroup] = 0
  end

  -- Create the avatars.
  for key, avatarObject in pairs(self._variables.avatarObjects) do
    local spawnGroup = cachedAvatarSpawnGroups[key]
    spawnCountersByGroup[spawnGroup] = spawnCountersByGroup[spawnGroup] + 1
    local idxInGroup = spawnCountersByGroup[spawnGroup]
    local point = spawnPointsByGroup[spawnGroup][idxInGroup]

    -- The call to `start` is where the avatar piece is actually created.
    avatarObject:start(grid, point)
    local avatarPiece = avatarObject:getPiece()
    local avatarIndex = avatarObject:getComponent('Avatar'):getIndex()
    self._variables.avatarPieceToIndex[avatarPiece] = avatarIndex
    self._variables.avatarIndexToPiece[avatarIndex] = avatarPiece
    self._variables.pieceToGameObject[avatarPiece] = avatarObject
  end

  for _, gameObject in pairs(
      self._variables.objectsByFunctionName.postStart) do
    gameObject:postStart(grid)
  end
end

-- This function starts all the non-avatar game objects.  The starting of avatar
-- game objects is handled in a special way by the avatar manager.  The avatar
-- manager start function will always be called after this one.
function BaseSimulation:start(grid)
  self._updaterRegistry:registerGrid(grid)
  self._variables.continueEpisodeAfterThisFrame = true
  -- Call `reset` on all game objects before calling `start` on any of them.
  for _, gameObject in pairs(
      self._variables.objectsByFunctionName.reset) do
    gameObject:reset()
  end
  -- Call `start` on all non-avatar game objects before calling `start` on any
  -- avatar game objects.
  for _, gameObject in pairs(self._variables.gameObjects) do
    if not gameObject:hasComponent("Avatar") then
      gameObject:start(grid)
      local piece = gameObject:getPiece()
      self._variables.pieceToGameObject[piece] = gameObject
    end
  end
  self:_avatarStart(grid)
  log.v(1, "grid\n" .. tostring(grid))
  -- Keep a reference to the grid.
  self._grid = grid
end

--[[ End of starting callbacks ]]
--[[ The following callbacks are called during updating / advancing ]]

function BaseSimulation:update(grid)
  -- Call preUpdate on all gameObjects before calling update on any gameObjects.
  for _, gameObject in pairs(
      self._variables.objectsByFunctionName.preUpdate) do
    gameObject:preUpdate()
  end
  for _, gameObject in pairs(
      self._variables.objectsByFunctionName.update) do
    gameObject:update(grid)
  end
end


--[[
Returns whether the simulation (episode) should continue for at least another
step, as controlled by the components and its updaters. Notice that it is
possible for this function to return true, and still being at the end of the
episode if we have reached the `maxEpisodeLengthFrames`, as that is controlled
by api_factory.
--]]
function BaseSimulation:continue()
  return self._variables.continueEpisodeAfterThisFrame
end

--[[ End of updating / advancing callbacks ]]
--[[ The functions below are part of the user API ]]

-- End the episode.
function BaseSimulation:endEpisode()
  self._variables.continueEpisodeAfterThisFrame = false
end

-- Get the GameObject that owns this dmlab2d piece.
function BaseSimulation:getGameObjectFromPiece(piece)
  return self._variables.pieceToGameObject[piece]
end

-- Returns GameObjects that have the given name, as a list.
function BaseSimulation:getGameObjectsByName(name)
  local objects = {}
  for _, gameObject in pairs(self._variables.gameObjects) do
    if name == gameObject.name then
      table.insert(objects, gameObject)
    end
  end
  return objects
end

-- Returns the avatar GameObjects, as a table. Keys are object IDs, values are
-- the actual GameObjects.
function BaseSimulation:getAvatarGameObjects()
  return self._variables.avatarObjects
end

--[[ Return all game objects that have component `componentName`, as a list.]]
function BaseSimulation:getAllGameObjectsWithComponent(componentName)
  local objects = {}
  for _, gameObject in pairs(self._variables.gameObjects) do
    if gameObject:hasComponent(componentName) then
      table.insert(objects, gameObject)
    end
  end
  return objects
end

--[[ Return an unordered table of all GameObjects.]]
function BaseSimulation:getAllGameObjects()
  return self._variables.gameObjects
end

--[[ Access avatar indices by piece id from gameObjects instantiated on
simulation.]]
function BaseSimulation:getAvatarIndexFromPiece(avatarPiece)
  return self._variables.avatarPieceToIndex[avatarPiece]
end

--[[ Access avatar piece ids by player index from gameObjects instantiated on
simulation.]]
function BaseSimulation:getAvatarPieceFromIndex(avatarIndex)
  return self._variables.avatarIndexToPiece[avatarIndex]
end

-- Access avatar GameObject by player index.
function BaseSimulation:getAvatarFromIndex(avatarIndex)
  return self:getGameObjectFromPiece(
      self._variables.avatarIndexToPiece[avatarIndex])
end

-- Return the number of players in this episode.
function BaseSimulation:getNumPlayers()
  return self._settings.numPlayers
end

-- Return a reference to the scene object.
function BaseSimulation:getSceneObject()
  return self._settings.sceneObject
end

-- Returns the number of objects currently in a state belonging to `group`.
function BaseSimulation:getGroupCount(group)
  return self._grid:groupCount(group)
end

-- Returns a random object currently in a state belonging to a given group.
function BaseSimulation:getGroupRandom(group)
  local piece = self._grid:groupRandom(random, group)
  return self:getGameObjectFromPiece(piece)
end

--[[ Returns objects currently assigned to states belonging to `group` in a
random order.]]
function BaseSimulation:getGroupShuffled(group)
  local pieces = self._grid:groupShuffled(random, group)
  local objects = {}
  for _, piece in ipairs(pieces) do
    table.insert(objects, self:getGameObjectFromPiece(piece))
  end
  return objects
end

--[[ Returns `count` random objects currently assigned to states that belong to
the given group in a random order.]]
function BaseSimulation:getGroupShuffledWithCount(group, count)
  local pieces = self._grid:groupShuffledWithCount(random, group, count)
  local objects = {}
  for _, piece in ipairs(pieces) do
    table.insert(objects, self:getGameObjectFromPiece(piece))
  end
  return objects
end

--[[Returns objects currently assigned to states belonging to a certain group in
a random order, where each object has the given probability of being returned.]]
function BaseSimulation:getGroupShuffledWithProbability(group, probability)
  local pieces = self._grid:groupShuffledWithProbability(
      random, group, probability)
  local objects = {}
  for _, piece in ipairs(pieces) do
    table.insert(objects, self:getGameObjectFromPiece(piece))
  end
  return objects
end

function BaseSimulation:getReward()
  -- This is propagated all the way up to the EnvLuaApi in the Advance function.
  -- It is currently unused by Melting Pot.
end

return {BaseSimulation = BaseSimulation}
