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

--[[ A library of common components that one may add to game objects. The two
mandatory components (StateManager, Transform) are also implemented here.
]]

local helpers = require 'common.helpers'
local log = require 'common.log'
local class = require 'common.class'
local args = require 'common.args'
local events = require 'system.events'
local random = require 'system.random'
local tensor = require 'system.tensor'
local tile = require 'system.tile'

-- For Lua 5.2 compatibility.
local unpack = unpack or table.unpack

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')

local _COMPASS = {'N', 'E', 'S', 'W'}
local _ORIENTATION_TO_INT = {N = 0, E = 1, S = 2, W = 3}
local _DIRECTION = {
    N = tensor.Tensor({0, -1}),
    E = tensor.Tensor({1, 0}),
    S = tensor.Tensor({0, 1}),
    W = tensor.Tensor({-1, 0}),
}


--[[ The StateManager component enables a GameObject to manage various states.
The GameObject will only have one state active at any one time, but can switch
between any of the states included in this manager.
]]

local StateManager = class.Class(component.Component)

function StateManager:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('StateManager')},
      {'initialState', args.stringType},  -- The original state.
      -- A piece configuration table each containing the following
      -- entries.
      --   {
      --     state (string),
      --     layer (string),
      --     sprite (string),
      --     groups (table of strings),
      --     contact (string),
      --   }
      -- Aside from a config for `initialState`, all the rest are optional.
      -- `stateConfigs`: a list of piece configuration tables.
      {'stateConfigs', args.default({}), args.tableType},
  })
  StateManager.Base.__init__(self, kwargs)

  -- A GameObject may have any number of inactive states, they can be
  -- dynamically switched out to become the active state while running.
  self._allStateConfigs = kwargs.stateConfigs
  -- The initial configuration state.
  self._config.initialState = kwargs.initialState
  -- Create index so it is possible to reference piece configs by state.
  self._config.stateToConfigIndex = {}
  for i, config in ipairs(self._allStateConfigs) do
    self._config.stateToConfigIndex[config.state] = i
  end
end

--[[ Component:awake() is called at the end of gameObject initialization. It
only gets called once, i.e., not called again on episode resets.]]
function StateManager:awake()
  self._config.uniqueStatePrefix = 'PTID_' .. self.gameObject._id .. '_'
end

function StateManager:addStates(states)
  -- Note that this function modifies the arg table `states`.
  for _, config in ipairs(self._allStateConfigs) do
    local name = self:getUniqueState(config.state)
    states[name] = {
        groups = config.groups,
    }
    -- For some reason, nils in config get interpreted as an empty table,
    -- so only add them if they are a string.
    if type(config.layer) == 'string' then
      states[name].layer = config.layer
    end
    if type(config.sprite) == 'string' then
      states[name].sprite = config.sprite
    end
    if type(config.contact) == 'string' then
      states[name].contact = config.contact
    end
  end
end

--[[ Returns a uniquified state (as a string) from the given state.
This uniquified string can be used to actually refer unequivocally to the state
belonging to a specific GameObject.  This can be used to refer directly to
callbacks or global states (although the user typiclly does not need to worry
about this).

If no state is passed (or nil), we will retrieve the active state,
uniquified.
]]
function StateManager:getUniqueState(state)
  if state ~= nil then
    return self._config.uniqueStatePrefix .. state
  else
    return self._config.uniqueStatePrefix .. self:getState()
  end
end

function StateManager:_deuniquifyState(unique)
  -- Remove the prefix from the uniquified state
  -- TODO(b/192926589): maybe make sure the prefix is there before removing?
  return unique:sub(#self._config.uniqueStatePrefix + 1)
end

function StateManager:getState()
  if self.gameObject:started() and self.gameObject:getPiece() then
    return self:_deuniquifyState(
        self.gameObject._grid:state(self.gameObject:getPiece()))
  end
  return self._config.initialState
end

function StateManager:_getStateConfig()
  return self._allStateConfigs[
      self._config.stateToConfigIndex[self:getState()]]
end

function StateManager:getLayer()
  return self:_getStateConfig().layer
end

function StateManager:getSprite()
  return self:_getStateConfig().sprite
end

function StateManager:getGroups()
  return self:_getStateConfig().groups
end

function StateManager:getGroupsForState(state)
  self:_validateState(state)
  return self._allStateConfigs[
      self._config.stateToConfigIndex[state]].groups
end

function StateManager:getAllStates()
  local result = {}
  for _, stateConfig in ipairs(self._allStateConfigs) do
    table.insert(result, stateConfig.state)
  end
  return result
end

function StateManager:getInitialState()
  return self._config.initialState
end

function StateManager:_validateState(state)
  assert(type(self._config.stateToConfigIndex[state]) ~= nil,
         'The state "' .. state .. '" is not available.')
end

--[[ `deuniquifyState` is only needed by `gameObject`. It should not really be
considered part of the public API.]]
function StateManager:deuniquifyState(unique)
  return self:_deuniquifyState(unique)
end

--[[ Dynamically change the state of a GameObject.

Use setState to change the active state at run time. Note that it is only
possible to change the state to another piece in the list of states that were
preregistered during initialization `allStateConfigs`.
]]
function StateManager:setState(grid, state)
  local uState = self:getUniqueState(state)
  self:_validateState(uState)
  grid:setState(self.gameObject:getPiece(), uState)
end


--[[ This Component represents the position and orientation of the GameObject.

This is one of the two required componenets (the other being StateManager).

For non-avatar objects, this component is also in charge of creating the
underlying engine piece.  This is done during start, so position and orientation
are undefined before that call.  For avatar objects, the piece is created during
spawning, and then passed on to the Transform. Thus, position and orientation
are only available after `setPieceAfterDeferredCreation`.
]]
local Transform = class.Class(component.Component)

function Transform:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Transform')},
      {'position', args.default({0, 0}), args.tableType},
      {'orientation', args.default(_COMPASS[1]), args.oneOf(unpack(_COMPASS))},
      -- In some cases (mainly for avatars) you want to delay piece creation and
      -- handle it in a different component besides Transform.
      {'deferPieceCreation', args.default(false), args.booleanType},
  })
  Transform.Base.__init__(self, kwargs)

  self._kwargs = kwargs
  self._config.deferPieceCreation = kwargs.deferPieceCreation
  self._variables = {}
end

function Transform:reset()
  self._variables = {}
  -- Track whether start has already been called in order to provide better
  -- feedback in error messages.
  self._variables.started = false
end

function Transform:start()
  self._variables.started = true
  -- The Transform needs access to the private grid as it is tightly linked to
  -- the engine position / orientation.
  self._variables.grid = self.gameObject._grid

  if not self._config.deferPieceCreation then
    local uniqueState = self.gameObject:getUniqueState()
    local transformData = {pos = self._kwargs.position,
                           orientation = self._kwargs.orientation}
    -- Create the piece using the position and orientation requested at init.
    self._variables.piece = self._variables.grid:createPiece(
        uniqueState, transformData)
    -- Prevent silent failures to create the piece.
    assert(self._variables.piece,
           'Failed to create piece of type: ' .. uniqueState ..
           ' at: ' .. helpers.tostringOneLine(transformData) .. '.')
  end
end

function Transform:setPieceAfterDeferredCreation(piece)
  self._variables.piece = piece
end

function Transform:getPiece()
  assert(self._variables.started,
         "Error: Must call Transform:start(...) before Transform:getPiece().")
  return self._variables.piece
end

--[[ Return the current position of this GameObject.

Only available after `start` for non-avatar objects, and only after spawning
for avatar ones.
]]
function Transform:getPosition()
  assert(
    self._variables.started,
    "Error: Must call Transform:start(...) before Transform:getPosition()."
  )
  return self._variables.grid:position(self._variables.piece)
end

--[[ Return the current orientation of this GameObject.

Only available after `start` for non-avatar objects, and only after spawning
for avatar ones.
]]
function Transform:getOrientation()
  return self._variables.grid:transform(self._variables.piece).orientation
end

--[[ Return the layer to which this gameObject is currently assigned.

Only available after `start` for non-avatar objects, and only after spawning
for avatar ones.
]]
function Transform:getLayer()
  return self._variables.grid:layer(self._variables.piece)
end

--[[ Push gameObject in given direction (without regard for its orientation).

If there is a piece in the target location at the time of the move then the
piece stays where it is. Note that it takes effect on the next call to `update`.

Triggers `onContact.contactName.leave` callbacks for pieces at current location
and `onContact.contactName.enter` for pieces at the location moved to. Both
callbacks are triggered even if the move is not possible and `piece` leaves and
enters the same cell.
]]
function Transform:moveAbs(orientation)
  self._variables.grid:moveAbs(self:getPiece(), orientation)
end

--[[ Push gameObject in direction relative to its facing orientation.

If there is a piece in the target location at the time of the move then the
piece stays where it is. Note that it takes effect on the next call to `update`.

Triggers `onContact.contactName.leave` and
`onContact.contactName.enter` callbacks in the same way as `grid:moveAbs(piece,
orientation)`.
]]
function Transform:moveRel(orientation)
  self._variables.grid:moveRel(self:getPiece(), orientation)
end

--[[ Teleports a gameObject to the given position and orientation.

Note that the teleport is applied on the next call to `update`.

Triggers `onContact.contactName.leave` and `onContact.contactName.enter` in the
same way as `grid:moveAbs(piece, orientation)`.
]]
function Transform:teleport(position, orientation)
  self._variables.grid:teleport(self:getPiece(), position)
  self._variables.grid:setOrientation(self:getPiece(), orientation)
end

--[[ Sets position of GameObject to any position matching any piece in a group.

Calls the same add/remove callbacks as grid::setState().

For example, one use case for this function is players respawning after being
tagged out. In this case they reappear at any location where there is a piece on
another layer assigned to the group `spawnPoints`.

The `orient` parameter is optional. Can be one of:
*  grid_world.TELEPORT_ORIENTATION.MATCH_TARGET
*  grid_world.TELEPORT_ORIENTATION.KEEP_ORIGINAL
*  grid_world.TELEPORT_ORIENTATION.PICK_RANDOM

The default is PICK_RANDOM.
]]
function Transform:teleportToGroup(groupName, state, orient)
  self._variables.grid:teleportToGroup(
      self:getPiece(), groupName, self.gameObject:getUniqueState(state), orient)
end

--[[ Turns a GameObject by a given value `angle`:

      *   `0` -> No Op.
      *   `1` -> 90 degrees clockwise.
      *   `2` -> 180 degrees.
      *   `3` -> 90 degrees counterclockwise.
]]
function Transform:turn(angle)
  self._variables.grid:turn(self:getPiece(), angle)
end

--[[ Sets a GameObject to face a particular direction in grid space.:

*   `'N'` -> North (decreasing y).
*   `'E'` -> East (increasing x).
*   `'S'` -> South (increasing y.
*   `'W'` -> West (decreasing x).
]]
function Transform:setOrientation(orientation)
  self._variables.grid:setOrientation(self:getPiece(), orientation)
end

-- Return a bool tracking if :start() was yet called on the transform component.
function Transform:started()
  return self._variables.started
end

--[[ Query game objects in an L1 ball (i.e a diamond) around this game object.

Returns a table of all game objects with an L1 distance to this object less than
or equal to `radius`.
]]
function Transform:queryDiamond(layer, radius)
  local simulation = self.gameObject.simulation
  local position = self:getPosition()
  local neighbors = self._variables.grid:queryDiamond(layer, position, radius)
  local neighborObjects = {}
  for neighborPiece, _ in pairs(neighbors) do
    table.insert(neighborObjects,
                 simulation:getGameObjectFromPiece(neighborPiece))
  end
  return neighborObjects
end

--[[ Query game objects in an L2 ball (i.e. a disc) around this game object.

Returns a table of all game objects with an L2 distance to this object's
position less than or equal to `radius`.
]]
function Transform:queryDisc(layer, radius)
  local simulation = self.gameObject.simulation
  local position = self:getPosition()
  local neighbors = self._variables.grid:queryDisc(layer, position, radius)
  local neighborObjects = {}
  for neighborPiece, _ in pairs(neighbors) do
    table.insert(neighborObjects,
                 simulation:getGameObjectFromPiece(neighborPiece))
  end
  return neighborObjects
end

--[[ Get the number of frames that this object has been in its current state.]]
function Transform:framesOld()
  return self._variables.grid:frames(self.gameObject:getPiece())
end


--[[ Query the game object at a specific layer and position.

Args:
  `layer` (string): the layer to query.
  `position` (optional, table): the position to query. It is optional. If you
      do not pass a position then the position of the current game object will
      be used by default.

Returns the game object at a given layer or nil.
]]
function Transform:queryPosition(layer, position)
  local positionToCheck = position or self.gameObject:getPosition()
  local simulation = self.gameObject.simulation
  local piece = self._variables.grid:queryPosition(layer, positionToCheck)
  return simulation:getGameObjectFromPiece(piece)
end

--[[ Query all game objects in a rectangle at the specified layer and recangle
dimensions.

Returns a table containing all game objects within the rectangle.
]]
function Transform:queryRectangle(layer, positionCorner1, positionCorner2)
  local simulation = self.gameObject.simulation
  local foundPieces = self._variables.grid:queryRectangle(layer,
                                                          positionCorner1,
                                                          positionCorner2)
  local foundObjects = {}
  for piece, _ in pairs(foundPieces) do
    table.insert(foundObjects,
                 simulation:getGameObjectFromPiece(piece))
  end
  return foundObjects
end

--[[ Query by casting a ray in a particular direction from a start point.

Returns whether there is a game object on a line between positionStart and
positionStart + direction on a given layer, not including the start position, or
whether the line is out of bounds. If an object is found, the piece and offset
return values are the first piece found and its offset. In torus topology
rayCastDirection does not change direction, but the offset is not normalised.

Args:
  `layer` (string): the layer to query.
  `distance` (integer): the distance to send the ray.
  `direction` (optional, string in {'N', 'E', 'S', 'W'}): the direction to send
      the ray. It is optional. If you do not pass a direction then the
      orientation of the current game object will be used by default.
  `positionOffset` (optional, 2D table position vector): add this position
      to the avatar's position vector. Use the resulting location as the start
      point for the ray case.

Returns:
  `hit` (boolean): true if an object was found, false otherwise.
  `foundObject` (GameObject): the game object that was hit.
  `offset` (table): vector pointing from the start object to the found object.
]]
function Transform:rayCastDirection(layer, distance, direction, positionOffset)
  assert(layer, 'Must provide layer.')
  assert(distance, 'Must provide distance for ray to travel.')
  local direction = direction or self.gameObject:getOrientation()
  local directionVector
  if type(direction) == 'string' then
    directionVector = _DIRECTION[direction]:clone():mul(distance):val()
  else
    directionVector = direction
  end
  local startLocation
  if positionOffset then
    startLocation = self:getAbsolutePositionFromRelative(positionOffset)
  else
    startLocation = self.gameObject:getPosition()
  end
  local hit, piece, offset = self._variables.grid:rayCastDirection(
      layer, startLocation, directionVector)

  local simulation = self.gameObject.simulation
  local foundObject = simulation:getGameObjectFromPiece(piece)

  return hit, foundObject, offset
end

--[[ Converts `relativePosition` in gameObject relative space to a position in
absolute space.
]]
function Transform:getAbsolutePositionFromRelative(relativePosition)
  local piece = self:getPiece()
  return self._variables.grid:toAbsolutePosition(piece, relativePosition)
end

--[[ Converts `absolutePosition` in absolute coordinates to a position in
gameObject relative coordinates.
]]
function Transform:getRelativePositionFromAbsolute(absolutePosition)
  local piece = self:getPiece()
  return self._variables.grid:toRelativePosition(piece, absolutePosition)
end

--[[ Converts a relative directon vector `relativeDirection` to a direction
vector in absolute coordinates.
]]
function Transform:getAbsoluteDirectionFromRelative(relativeDirection)
  local piece = self:getPiece()
  return self._variables.grid:toAbsoluteDirection(piece, relativeDirection)
end

--[[ Converts an absolute directon vector `absoluteDirection` to a direction
vector in relative coordinates.
]]
function Transform:getRelativeDirectionFromAbsolute(absoluteDirection)
  local piece = self:getPiece()
  return self._variables.grid:toRelativeDirection(piece, absoluteDirection)
end


--[[ The Appearance component manages how GameObjects are rendered visually.
]]

local Appearance = class.Class(component.Component)

function Appearance:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Appearance')},
      {'renderMode',
        args.default('colored_square'),
        args.oneOf('invisible', 'colored_square', 'ascii_shape')},
      {'spriteNames', args.default({}), args.tableType},
      {'spriteRGBColors', args.default({}), args.tableType},
      {'spriteShapes', args.default({}), args.tableType},
      {'palettes', args.default({}), args.tableType},
      -- Boolean flags that determine whether or not to rotate each sprite.
      {'noRotates', args.default({}), args.tableType},
  })
  Appearance.Base.__init__(self, kwargs)

  self._config.renderMode = kwargs.renderMode
  self._config.spriteNames = kwargs.spriteNames
  self._config.spriteRGBColors = kwargs.spriteRGBColors
  self._config.spriteShapes = kwargs.spriteShapes
  self._config.palettes = kwargs.palettes
  self._config.noRotates = kwargs.noRotates
end

function _addShapesToTileSet(
      tileSet, spriteNames, renderMode, spriteRGBColors, spriteShapes, palettes,
      noRotates)
  for i, name in ipairs(spriteNames) do
    if renderMode == 'colored_square' then
      tileSet:addColor(name, spriteRGBColors[i])
    elseif renderMode == 'ascii_shape' then
      local shapes = spriteShapes[i]
      if type(shapes) == 'table' and #shapes == 4 then
        assert (noRotates[i],
          'When providing sprites for all orientations, "noRotates" must be ' ..
          '`true`.')
        for j=1,4 do
          local spriteData = {
              palette = palettes[i],
              text = spriteShapes[i][j],
              noRotate = noRotates[i]
          }
          tileSet:addShape(name .. '.' .. _COMPASS[j], spriteData)
        end
      else
        local spriteData = {
            palette = palettes[i],
            text = spriteShapes[i],
            noRotate = noRotates[i]
        }
        tileSet:addShape(name, spriteData)
      end
    end
  end
end

function Appearance:addSprites(tileSet)
  _addShapesToTileSet(
      tileSet, self._config.spriteNames, self._config.renderMode,
      self._config.spriteRGBColors, self._config.spriteShapes,
      self._config.palettes, self._config.noRotates)
end

function Appearance:getSpriteNames()
  return self._config.spriteNames
end

function Appearance:getSpriteRGBColors()
  return self._config.spriteRGBColors
end


--[[ AdditionalSprites adds sprites that are not associated to any state.

The AdditionalSprites component adds sprites to the `CustomSprites` field of
the world config. You can select any sprite named in custom sprites using a
spriteMap in the avatar's addObservations function.
]]
local AdditionalSprites = class.Class(component.Component)

function AdditionalSprites:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AdditionalSprites')},
      -- All sprites added by this component share the same render mode. If you
      -- need to mix and match render modes then you can just add two instances
      -- of this component.
      {'renderMode',
        args.default('colored_square'),
        args.oneOf('invisible', 'colored_square', 'ascii_shape')},
      {'customSpriteNames', args.default({}), args.tableType},
      {'customSpriteRGBColors', args.default({}), args.tableType},
      {'customSpriteShapes', args.default({}), args.tableType},
      {'customPalettes', args.default({}), args.tableType},
      -- Boolean flags that determine whether or not to rotate each sprite.
      {'customNoRotates', args.default({}), args.tableType},
  })
  AdditionalSprites.Base.__init__(self, kwargs)

  self._config.renderMode = kwargs.renderMode
  self._config.customSpriteNames = kwargs.customSpriteNames
  self._config.customSpriteRGBColors = kwargs.customSpriteRGBColors
  self._config.customSpriteShapes = kwargs.customSpriteShapes
  self._config.customPalettes = kwargs.customPalettes
  self._config.customNoRotates = kwargs.customNoRotates
end

function AdditionalSprites:addSprites(tileSet)
  _addShapesToTileSet(
      tileSet, self._config.customSpriteNames, self._config.renderMode,
      self._config.customSpriteRGBColors, self._config.customSpriteShapes,
      self._config.customPalettes, self._config.customNoRotates)
end

function AdditionalSprites:addCustomSprites(customSprites)
  for _, name in ipairs(self._config.customSpriteNames) do
    table.insert(customSprites, name)
  end
end


--[[ BeamBlockers prevents beams of type `beamType` from passing through.

BeamBlockers are quite commonly used, especially for walls.
]]
local BeamBlocker = class.Class(component.Component)

function BeamBlocker:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('BeamBlocker')},
      {'beamType', args.stringType},
  })
  BeamBlocker.Base.__init__(self, kwargs)
  self._config.beamType = kwargs.beamType
end

function BeamBlocker:onHit(hittingGameObject, hitName)
  -- Beams of type `hitName` do not pass through beam blockers.
  if hitName == self._config.beamType then
    return true
  else
    return false
  end
end


--[[ A utility component to hold global tensors of arbitrary shape that can be
used to register debug observations and events. Usually you would add this
component to the Scene object, and have the GlobalMetricReporter refer to a
specific field here.
]]
local GlobalMetricHolder = class.Class(component.Component)

function GlobalMetricHolder:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('GlobalMetricHolder')},
      -- `metrics` is a table where each entry is a table that defines a single
      --  tensor. Each such definition table has the following fields:
      --  * `type` (string) e.g. 'tensor.DoubleTensor' or 'tensor.Int32Tensor'.
      --  * `shape` (table) e.g. {3, 3}.
      --  * `variable` (string) the name of the variable to hold the tensor.
      {'metrics', args.tableType},
  })
  GlobalMetricHolder.Base.__init__(self, kwargs)
  self._config.metrics = kwargs.metrics
end

function GlobalMetricHolder:reset()
  -- Create and initialise the metrics to zero
  for _, metric in pairs(self._config.metrics) do
    self[metric.variable] = tensor[
        string.gsub(metric.type, "tensor.", "")](unpack(metric.shape)):fill(0)
  end
end

function GlobalMetricHolder:update()
  -- Clear out the metrics for the next step
  for _, metric in pairs(self._config.metrics) do
    self[metric.variable]:fill(0)
  end
end


local GlobalMetricReporter = class.Class(component.Component)

function GlobalMetricReporter:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('GlobalMetricReporter')},
      -- `metrics` is a table where each entry is a table that defines a single
      --  observation. Each such definition table has the following fields:
      --   `name` (string) e.g. RGB.
      --   `type` (string) e.g. 'tensor.DoubleTensor' or 'Doubles'.
      --   `shape` (table) e.g. {3, 3}.
      --   `component` (string) which component to reference on the same game
      --     object as the GlobalMetricReporter component.
      --   `variable` (string) the name of the variable to report.
      {'metrics', args.tableType},
  })
  GlobalMetricReporter.Base.__init__(self, kwargs)
  self._config.metrics = kwargs.metrics
end

function GlobalMetricReporter:addObservations(tileSet, world, observations)
  for _, metric in ipairs(self._config.metrics) do
    observations[#observations + 1] = {
        name = 'WORLD.' .. metric.name,
        type = metric.type,
        shape = metric.shape,
        func = function(grid)
          return self.gameObject:getComponent(metric.component)[metric.variable]
        end
    }
  end
end


local AvatarMetricReporter = class.Class(component.Component)

function AvatarMetricReporter:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AvatarMetricReporter')},
      -- `metrics` is a table where each entry is a table that defines a single
      --  observation. Each such definition table has the following fields:
      --   `name` (string) e.g. RGB.
      --   `type` (string) e.g. 'tensor.DoubleTensor' or 'Doubles'.
      --   `shape` (table) e.g. {3, 3}.
      --   `component` (string) which component to reference on the same game
      --     object as the AvatarMetricReporter component.
      --   `variable` (string) the name of the variable to report.
      --   `index` (int) [OPTIONAL] an index on a tensor. Assumes `variable` is
      --     a one-dimensional tensor, and that the metric of interest is at
      --     this index in the tensor.
      {'metrics', args.tableType},
  })
  AvatarMetricReporter.Base.__init__(self, kwargs)
  self._config.metrics = kwargs.metrics
  -- Check if any metrics have an index, and add that field to the metric. We do
  -- this with `rawget` because the arguments to the constructor are a metatable
  -- with its own index that autocreates keys when first accessed.
  for _, metric in ipairs(self._config.metrics) do
    metric.hasIndex = (rawget(metric, 'index') ~= nil)
  end
end

function AvatarMetricReporter:addObservations(tileSet, world, observations)
  local playerId = self.gameObject:getComponent('Avatar'):getIndex()
  for _, metric in ipairs(self._config.metrics) do
    observations[#observations + 1] = {
        name = playerId .. '.' .. metric.name,
        type = metric.type,
        shape = metric.shape,
        func = function(grid)
          local value = self.gameObject:getComponent(
              metric.component)[metric.variable]
          if metric.hasIndex then
            value = value(metric.index):val()
          end
          return value
        end
    }
  end
end


local LocationObserver = class.Class(component.Component)

function LocationObserver:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('LocationObserver')},
      -- `objectIsAvatar` if true then observation name is 'X.POSITION'
      -- or 'X.ORIENTATION', where X is a player index. Otherwise, observations
      -- are of the form 'WORLD.POSITION_<gameObject.id>'.
      {'objectIsAvatar', args.booleanType},
      -- `alsoReportOrientation` if true then report orientation as well as
      -- position.
      {'alsoReportOrientation', args.default(false), args.booleanType},
  })
  LocationObserver.Base.__init__(self, kwargs)
  self._config.alsoReportOrientation = kwargs.alsoReportOrientation
  self._config.objectIsAvatar = kwargs.objectIsAvatar
end

function LocationObserver:addObservations(tileSet, world, observations)
  local positionObservationName
  local orientationObservationName
  if self._config.objectIsAvatar then
    local playerId = tostring(self.gameObject:getComponent('Avatar'):getIndex())
    positionObservationName = playerId .. '.' .. 'POSITION'
    orientationObservationName = playerId .. '.' .. 'ORIENTATION'
  else
    local uniqueId = tostring(self.gameObject:getUniqueState())
    positionObservationName = 'WORLD.POSITION_' .. uniqueId
    orientationObservationName = 'WORLD.ORIENTATION_' .. uniqueId
  end

  observations[#observations + 1] = {
      name = positionObservationName,
      type = 'tensor.Int32Tensor',
      shape = {2},
      func = function(grid)
        return tensor.Int32Tensor(self.gameObject:getPosition())
      end
  }
  if self._config.alsoReportOrientation then
    observations[#observations + 1] = {
        name = orientationObservationName,
        type = 'tensor.Int32Tensor',
        shape = {},
        func = function(grid)
          return _ORIENTATION_TO_INT[self.gameObject:getOrientation()]
        end
    }
  end
end

--[[ The `StochasticEpisodeEnding` component stochastically ends the episode
after a certain number of frames have passed.

This component is normally added to the scene object since its logic is global.
]]
local StochasticEpisodeEnding = class.Class(component.Component)

function StochasticEpisodeEnding:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('StochasticEpisodeEnding')},
      -- After `minimumFramesPerEpisode` frames, episode has a chance of ending
      -- on all subsequent frames.
      {'minimumFramesPerEpisode', args.positive},
      -- The chance of ending on each frame is `probabilityTerminationPerStep`.
      {'probabilityTerminationPerStep', args.gt(0.0), args.le(1.0)},
  })
  StochasticEpisodeEnding.Base.__init__(self, kwargs)
  self._config.minimumFramesPerEpisode = kwargs.minimumFramesPerEpisode
  self._config.probabilityTerminationPerStep =
      kwargs.probabilityTerminationPerStep
end

function StochasticEpisodeEnding:registerUpdaters(updaterRegistry)
  updaterRegistry:registerUpdater{
      updateFn = function () self.gameObject.simulation:endEpisode() end,
      probability = self._config.probabilityTerminationPerStep,
      startFrame = self._config.minimumFramesPerEpisode,
  }
end


--[[ The `StochasticIntervalEpisodeEnding` component stochastically ends the
episode after a random number of fixed length intervals have passed.

In game theory it is common to consider repeated interactions that end with some
probability after each stage. This assumption is needed in order to avoid
trivial backwards induction solutions in many games, including iterated
prisoners dilemma. Many deep RL frameworks however do not handle well the
possibility of episodes that end at random times. They typically pad all
trajectories with zeros up to a fixed size called the "unroll length". It is
common to choose unroll length to evenly divide episode length in order to avoid
the need for padding. However, this solution rules out stochastic episode ending
on a frame by frame basis, which is desirable for game theoretic reasons. The
purpose of this component then is to combine the best of both worlds. It allows
episodes with duration that cannot be known in advance, but also ensures that
all endings will occur at a multiple of intervalLength (which should be set to
equal the unroll length).

This component is normally added to the scene object since its logic is global.
]]
local StochasticIntervalEpisodeEnding = class.Class(component.Component)

function StochasticIntervalEpisodeEnding:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('StochasticIntervalEpisodeEnding')},
      -- After `minimumFramesPerEpisode` frames, episode has a chance of ending
      -- at the end of each subsequent interval.
      {'minimumFramesPerEpisode', args.positive},
      -- Length in frames of each interval.
      {'intervalLength', args.positive},
      -- Chance of ending per interval is `probabilityTerminationPerInterval`.
      {'probabilityTerminationPerInterval', args.gt(0.0), args.le(1.0)},
  })
  StochasticIntervalEpisodeEnding.Base.__init__(self, kwargs)
  self._config.minimumFramesPerEpisode = kwargs.minimumFramesPerEpisode
  self._config.intervalLength = kwargs.intervalLength
  self._config.probabilityTerminationPerInterval =
      kwargs.probabilityTerminationPerInterval
end

function StochasticIntervalEpisodeEnding:registerUpdaters(updaterRegistry)
  local function maybeEndEpisode()
    if self._t % self._config.intervalLength == 0 then
      if random:uniformReal(0, 1) <
          self._config.probabilityTerminationPerInterval then
        self.gameObject.simulation:endEpisode()
      end
    end
  end
  updaterRegistry:registerUpdater{
      updateFn = maybeEndEpisode,
      startFrame = self._config.minimumFramesPerEpisode,
  }
end

function StochasticIntervalEpisodeEnding:reset()
  self._t = 1
end

function StochasticIntervalEpisodeEnding:update()
  self._t = self._t + 1
end


-- An object that is edible switches state when an avatar touches it, and
-- provides a reward. It can be used in combination to the FixedRateRegrow.
local Edible = class.Class(component.Component)

function Edible:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Edible')},
      {'liveState', args.stringType},
      {'waitState', args.stringType},
      {'rewardForEating', args.numberType},
  })
  Edible.Base.__init__(self, kwargs)

  self._config.liveState = kwargs.liveState
  self._config.waitState = kwargs.waitState
  self._config.rewardForEating = kwargs.rewardForEating
end

function Edible:reset()
  self._waitState = self._config.waitState
  self._liveState = self._config.liveState
end

function Edible:setWaitState(newWaitState)
  self._waitState = newWaitState
end

function Edible:getWaitState()
  return self._waitState
end

function Edible:setLiveState(newLiveState)
  self._liveState = newLiveState
end

function Edible:getLiveState()
  return self._liveState
end

function Edible:onEnter(enteringGameObject, contactName)
  if contactName == 'avatar' then
    if self.gameObject:getState() == self._liveState then
      -- Reward the player who ate the edible.
      local avatarComponent = enteringGameObject:getComponent('Avatar')
      avatarComponent:addReward(self._config.rewardForEating)
      events:add('edible_consumed', 'dict',
                 'player_index', avatarComponent:getIndex())  -- int
      -- Change the edible to its wait (disabled) state.
      self.gameObject:setState(self._waitState)
    end
  end
end


--[[ The `FixedRateRegrow` component enables a game object that is in a
particular (traditionally thought of as "dormant") state to change its state
probabilistically (at a fixed rate). Used primarily for respawning objects.
]]
local FixedRateRegrow = class.Class(component.Component)

function FixedRateRegrow:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('FixedRateRegrow')},
      {'liveState', args.stringType},
      {'waitState', args.stringType},
      {'regrowRate', args.ge(0.0), args.le(1.0)},
  })
  FixedRateRegrow.Base.__init__(self, kwargs)

  self._config.liveState = kwargs.liveState
  self._config.waitState = kwargs.waitState
  self._config.regrowRate = kwargs.regrowRate
end

function FixedRateRegrow:registerUpdaters(updaterRegistry)
  -- Registers an update with high priority that only gets called when the
  -- object is in the `waitState` state.
  updaterRegistry:registerUpdater{
    state = self._config.waitState,
    probability = self._config.regrowRate,
    updateFn = function() self.gameObject:setState(self._config.liveState) end,
  }
end


--[[ The `Animation` component is a simple way to add animations to GameObjects.
This component sets updaters that with a specified delay change the state of the
game object, and might optionally loop.

Although it's main use is to show animations, it can be used whenever you need
regular interval transitions of states, possibly looping.
]]
local Animation = class.Class(component.Component)

function Animation:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Animation')},
      -- The states must all be distinct (even if they map to the same sprite)
      {'states', args.tableType},
      {'gameFramesPerAnimationFrame', args.numberType},
      {'loop', args.booleanType},
      {'randomStartFrame', args.booleanType, args.default(false)},
      {'group', args.stringType, args.default(nil)},
  })
  Animation.Base.__init__(self, kwargs)

  self._config.states = kwargs.states
  self._config.gameFramesPerAnimationFrame = kwargs.gameFramesPerAnimationFrame
  self._config.loop = kwargs.loop
  self._config.randomStartFrame = kwargs.randomStartFrame
  self._config.group = kwargs.group
end

function Animation:postStart()
  if self._config.randomStartFrame then
    self.gameObject:setState(random:choice(self._config.states))
  end
end

function Animation:registerUpdaters(updaterRegistry)
  local prevState = nil
  local firstState = nil
  for _, state in pairs(self._config.states) do
    if firstState == nil then
      firstState = state
    else
      updaterRegistry:registerUpdater{
        state = prevState,
        startFrame = self._config.gameFramesPerAnimationFrame,
        updateFn = function() self.gameObject:setState(state) end,
        group = self._config.group,
      }
    end
    prevState = state
  end
  if self._config.loop then
    updaterRegistry:registerUpdater{
      state = prevState,
      startFrame = self._config.gameFramesPerAnimationFrame,
      updateFn = function() self.gameObject:setState(firstState) end,
      group = self._config.group,
    }
  end
end


--[[ Use `RoleBasedRewardTile` to create a tile that provides rewards to avatars
with specific roles when they step on it.
]]
local RoleBasedRewardTile = class.Class(component.Component)

function RoleBasedRewardTile:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('RoleBasedRewardTile')},
      -- `avatarRoleComponent`: check this component for the getRole function.
      {'avatarRoleComponent', args.stringType},
      -- `getRoleFunction`: call this function on the avatar role component.
      {'getRoleFunction', args.default('getRole'), args.stringType},
      -- `rolesToRewards`: table mapping role (str) to reward (number).
      {'rolesToRewards', args.tableType},
  })
  RoleBasedRewardTile.Base.__init__(self, kwargs)

  self._config.avatarRoleComponent = kwargs.avatarRoleComponent
  self._config.getRoleFunction = kwargs.getRoleFunction
  self._config.rolesToRewards = kwargs.rolesToRewards
end

function RoleBasedRewardTile:onEnter(enteringGameObject, contactName)
  if contactName == 'avatar' then
    if enteringGameObject:hasComponent(self._config.avatarRoleComponent) then
      local roleComponent = enteringGameObject:getComponent(
          self._config.avatarRoleComponent)
      local role = roleComponent[self._config.getRoleFunction](roleComponent)
      local reward = rawget(self._config.rolesToRewards, role)
      if reward then
        local avatarComponent = enteringGameObject:getComponent('Avatar')
        avatarComponent:addReward(reward)
        -- Record that the role based reward tile was triggered.
        events:add('triggered_role_based_reward_tile', 'dict',
                   'player_index', avatarComponent:getIndex(),  -- int
                   'role', role)  -- string
      end
    end
  end
end


local allComponents = {
    -- The two mandatory components:
    StateManager = StateManager,
    Transform = Transform,

    -- Other non-mandatory, but commonly used, components:
    Appearance = Appearance,
    AdditionalSprites = AdditionalSprites,
    BeamBlocker = BeamBlocker,
    GlobalMetricHolder = GlobalMetricHolder,
    GlobalMetricReporter = GlobalMetricReporter,
    AvatarMetricReporter = AvatarMetricReporter,
    LocationObserver = LocationObserver,
    StochasticEpisodeEnding = StochasticEpisodeEnding,
    StochasticIntervalEpisodeEnding = StochasticIntervalEpisodeEnding,
    Edible = Edible,
    FixedRateRegrow = FixedRateRegrow,
    Animation = Animation,
    RoleBasedRewardTile = RoleBasedRewardTile,
}

-- Register all components from this module in the component registry.
component_registry.registerAllComponents(allComponents)

return allComponents
