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

--[[ A GameObject is a container for components. Components endow gameObjects
with behaviors. In Melting Pot, DMLab2D games are written by composing
gameObjects.

For now, gameObjects can only be created at startup time. They cannot be
created or destroyed dynamically. We might change this in the future.
]]

local helpers = require 'common.helpers'
local log = require 'common.log'
local class = require 'common.class'

local meltingpot = 'meltingpot.lua.modules.'
local updater_registry = require(meltingpot .. 'updater_registry')

-- For Lua 5.2 compatibility.
local unpack = unpack or table.unpack

-- Component API functions
_COMPONENT_FUNCTIONS = {
    'awake', 'reset', 'start', 'postStart', 'preUpdate', 'update', 'onBlocked',
    'onEnter', 'onExit', 'onHit', 'onStateChange', 'registerUpdaters',
    'addHits', 'addSprites', 'addCustomSprites', 'addObservations',
    'addPlayerCallbacks'}


local GameObject = class.Class()

--[[ A GameObject is a container for components.

GameObjects have a position in the grid (x, y), and a state (its current
state).  A state determines the look-and-feel, the layer and the groups that
this object belongs to.

Game logic (e.g. when to transition from one state to the next, or how to
respond to interactions with other objects) is implemented through Components.

Users should never need to inherit from GameObject. They just use the base class
and add all the necessary components to create whatever functionality you want.

The GameObject also provides access, to components, to the simulation and the
avatar manager so that components can affect other GameObjects and global state.

In paticular, the following fields and functions are part of the public API for
components:

*   self.simulation: A reference to the game Simulation (see
    base_simulation.lua).
*   self.avatarManager: A reference to the AvatarManager (see
    base_avatar_manager.lua).
*   GameObject:hasComponent(name): Returns whether the game object contains at
    least one component with the given name (as a string).
*   GameObject:getComponent(name): Returns the first instance of the component
    with the given name (as a string).  This function throws an error if a
    component with the requested name does not exist in this GameObject.
*   GameObject:getComponents(name): Returns a list with all the components
    with the given name (as a string).  If none are found, it returns an empty
    list.
*   GameObject:getState(): Returns the current state of the game object
    (e.g. its current state, as a string).
]]
function GameObject:__init__(kwargs)
  assert(kwargs.id ~= nil, 'GameObject\'s id cannot be nil')
  self._id = kwargs.id
  self.name = kwargs.name

  -- Components will be a mapping between the component name and a list of
  -- components with that name.
  self._components = {}
  -- Mapping of components by their API functions (e.g. `start`, `update`, etc.)
  self._components_by_function = {}
  for _, component in ipairs(kwargs.components) do
    self:addComponent(component)
  end

  self._updaterRegistry = updater_registry.UpdaterRegistry()
  --[[ After creation, but before any callbacks are performed, the following
  field will be added:
  *   self.simulation: A reference to the game Simulation (see
      base_simulation.lua).
  ]]
end

function _safe_add_to_table(mapping, func_name, component)
  if mapping[func_name] == nil then
    mapping[func_name] = {}
  end
  table.insert(mapping[func_name], component)
end

--[[
Registers the component's implemented API functions to the GameObject's internal
mapping of components_by_functions.
]]
function GameObject:_registerComponentFunctions(component)
  for _, func in pairs(_COMPONENT_FUNCTIONS) do
    if component[func] ~= nil then
      _safe_add_to_table(self._components_by_function, func, component)
    end
  end
end

function GameObject:hasComponentWithFunction(functionName)
  return self._components_by_function[functionName] ~= nil
end

--[[ Adds a component to this GameObject.  A Component must be added to at most
one GameObject, and it is an error to add it to two or more.
There are two special types of components: StateManager and Transform.  When
adding multiple of those, only the last one added is considered active for the
purposes of the GameObject calls relating explicitly to them (e.g.
getState() or getPiece()).

Once a component is added, its awake() method is called (if implemented).

WARNING: Calling addComponent after initialisation is NOT supported. Callbacks
         from such components will not be properly registered.
]]
function GameObject:addComponent(component)
  if component.gameObject ~= nil then
    error('The component has already been added to a GameObject.')
  end
  component.gameObject = self
  if self._components[component.name] == nil then
    self._components[component.name] = {}
  end
  table.insert(self._components[component.name], component)
  self:_registerComponentFunctions(component)
  -- Store a dedicated reference to certain essential (and unique) components.
  if component.name == 'StateManager' then
    self._stateManager = component
  elseif component.name == 'Transform' then
    self._transform = component
  end -- Do nothing for other components.
  -- Call the awake method on the component if it is implemented.
  if component.awake then
    component:awake()
  end
end

function GameObject:_doOnAllComponents(func)
  local returns = {}
  for k, component_list in pairs(self._components) do
    for _, component in ipairs(component_list) do
      local retValue = func(component)
      if retValue then
        table.insert(returns, retValue)
      end
    end
  end
  return returns
end

function GameObject:_doOnAllStates(func)
  for _, state in ipairs(self:getAllStates()) do
    func(self:getUniqueState(state))
  end
end

--[[ Called by the engine once all the hits have been registered so we can
prepare appropritate callbacks for the underlying states.
--]]
function GameObject:setHits(hitsTable)
  self._hitsTable = hitsTable
end

function GameObject:getUpdaterRegistry()
  return self._updaterRegistry
end

--[[ Called by the engine to retrieve all updaters from the components
--]]
function GameObject:registerUpdaters()
  local updaters = {}
  local priorities = {}
  self:_doOnAllComponents(
    function(component)
      if component.registerUpdaters then
        -- Prepare the registry to register updaters for this component.
        self._updaterRegistry:setGroupPrefix(component.name)
        component:registerUpdaters(self._updaterRegistry)
      end
    end)
  self._updaterRegistry:uniquifyStatesAndAddGroups(self)
end

--[[ Called by the engine once all the contacts have been registered so we can
prepare appropritate callbacks for the underlying states.
--]]
function GameObject:setContacts(contactsTable)
  self._contactsTable = contactsTable
end

--[[ Register the GameObject's states to the provided states registry
]]
function GameObject:addStates(states)
  self._stateManager:addStates(states)
end

function GameObject:addHits(worldConfig)
  self:_doOnAllComponents(
    function(component)
      if component.addHits then
        component:addHits(worldConfig)
      end
    end)
end

function GameObject:addCustomSprites(customSprites)
  self:_doOnAllComponents(
    function(component)
      if component.addCustomSprites then
        component:addCustomSprites(customSprites)
      end
    end)
end

function GameObject:addSprites(tileSet)
  self:_doOnAllComponents(
    function(component)
      if component.addSprites then
        component:addSprites(tileSet)
      end
    end)
end

function GameObject:connect(objectToConnect)
  local ourPiece = self:getPiece()
  local pieceToConnect = objectToConnect:getPiece()
  self._grid:connect(ourPiece, pieceToConnect)
end

function GameObject:disconnect()
  local pieceToDisconnect = self:getPiece()
  self._grid:disconnect(pieceToDisconnect)
end

function GameObject:hitBeam(hitName, length, radius)
  assert(self._hitsTable[hitName],
         'You can only call hitBeam with a hitName that has been registered ' ..
         'previously in the hits table.')
  self._grid:hitBeam(self:getPiece(), hitName, length, radius)
end

-- Wrapping functions for type callbacks belonging to this GameObject.

function GameObject:_onAdd(uState)
  self:_doOnAllComponents(
    function(component)
      if component.onStateChange then
        component:onStateChange(self._previousState)
      end
    end)
end

function GameObject:_onRemove(uState)
  local state = self:getComponent('StateManager'):deuniquifyState(uState)
  self._previousState = state
end

function GameObject:_onSomething(eventFnName, ...)
  local sim = self.simulation
  -- Get the GameObject for the initiator, which is always the first argument
  -- and it is required for all events.
  local args = {...}
  args[1] = sim:getGameObjectFromPiece(args[1])
  return self:_doOnAllComponents(
    function(component)
      if component[eventFnName] then
        return component[eventFnName](component, unpack(args))
      end
    end)
end

function GameObject:_onBlocked(blockerState)
  self:_onSomething('onBlocked', blockerState)
end

function GameObject:_onHit(initiator, hitName)
  local returns = self:_onSomething('onHit', initiator, hitName)
  -- Any component wanting to block the beam should cause full blockage.
  for _, r in pairs(returns) do
    if r then
      return true
    end
  end
  return false
end

function GameObject:_onEnter(initiator, contactName)
  self:_onSomething('onEnter', initiator, contactName)
end

function GameObject:_onExit(initiator, contactName)
  self:_onSomething('onExit', initiator, contactName)
end

-- Removed callbacks that no component has.
function GameObject:_prune_callbacks(state_callbacks)
  if self._components_by_function['onStateChange'] == nil then
    state_callbacks.onAdd = nil
    state_callbacks.onRemove = nil
  end
  if self._components_by_function['onBlocked'] == nil then
    state_callbacks.onBlocked = nil
  end
  if self._components_by_function['onHit'] == nil then
    state_callbacks.onHit = nil
  end
  if self._components_by_function['onEnter'] == nil and
      self._components_by_function['onExit'] == nil then
    state_callbacks.onContact = nil
  end
end

-- Note that this function modifies the arg table `callbacks`.
-- This function initializes an empty callbacks table for each state.
-- Then it calls addTypeCallbacks in all components.
function GameObject:addTypeCallbacks(callbacks)
  -- Prepare the inital callbacks structure for all states.
  self:_doOnAllStates(
    function(uState)
      if not callbacks[uState] then
        callbacks[uState] = {onUpdate = {},
                             onHit = {},
                             onContact = {['avatar'] = {}}}
      end
      callbacks[uState] = {
          onAdd = function (grid, piece) self:_onAdd(uState) end,
          onRemove = function (grid, piece) self:_onRemove(uState) end,
          -- onBlocked is executed when a piece tries (and fails) to move to an
          -- occupied location.
          onBlocked = function(grid, selfPiece, blocker)
              self:_onBlocked(blocker) end,
          -- list of function(grid, piece, framesOld)
          onUpdate = {},
          onHit = {},
          onContact = {['avatar'] = {}},
      }
      for hitName, _ in pairs(self._hitsTable) do
        callbacks[uState].onHit[hitName] = function(
            grid, objPiece, hitterPiece)
          return self:_onHit(hitterPiece, hitName)
        end
      end
      for _, contactName in pairs(self._contactsTable) do
        callbacks[uState].onContact[contactName] = {}
        callbacks[uState].onContact[contactName].enter = function(
            grid, objPiece, initiatorPiece)
          self:_onEnter(initiatorPiece, contactName)
        end
        callbacks[uState].onContact[contactName].leave = function(
            grid, objPiece, initiatorPiece)
          self:_onExit(initiatorPiece, contactName)
        end
      end
      self:_prune_callbacks(callbacks[uState])
    end)
end

--[[ `addPlayerCallbacks` is always called after `addTypeCallbacks`. So the
callbacks table will already exist for it to modify.

Arguments:
`callbacks` (table): table of callbacks. To be modified here and used elsewhere.
]]
function GameObject:addPlayerCallbacks(callbacks)
  self:_doOnAllComponents(
    function(component)
      if component.addPlayerCallbacks then
        component:addPlayerCallbacks(callbacks)
      end
    end)
end

function GameObject:addObservations(tileSet, world, observations)
  self:_doOnAllComponents(
    function(component)
      if component.addObservations then
        component:addObservations(tileSet, world, observations)
      end
    end)
end

function GameObject:discreteActionSpec(actSpec)
  self:_doOnAllComponents(
    function(component)
      if component.discreteActionSpec then
        component:discreteActionSpec(actSpec)
      end
    end)
end

function GameObject:discreteActions(actions)
  self:_doOnAllComponents(
    function(component)
      if component.discreteActions then
        component:discreteActions(actions)
      end
    end)
end

function GameObject:start(grid, optionalLocator)
  self._grid = grid
  -- Make sure that the Transform is initialised first.
  self:getComponent('Transform'):start(optionalLocator)
  self:_doOnAllComponents(
    function(component)
      if component.start and component.name ~= 'Transform' then
        component:start(optionalLocator)
      end
    end)
end

function GameObject:postStart(grid)
  self:_doOnAllComponents(
    function(component)
      if component.postStart then
        component:postStart()
      end
    end)
end

--[[ preUpdate is called for all gameObjects before update is called for any.]]
function GameObject:preUpdate()
  if self._components_by_function['preUpdate'] ~= nil then
    for _, component in pairs(self._components_by_function['preUpdate']) do
      component:preUpdate()
    end
  end
end

function GameObject:update(grid)
  if self._components_by_function['update'] ~= nil then
    for _, component in pairs(self._components_by_function['update']) do
      component:update()
    end
  end
end

function GameObject:hasComponent(name)
  assert(
    self._components,
    'Missing self._components. ' ..
    'Did you mistakenly call hasComponent using a `.` instead of a `:`?')
  return self._components[name] ~= nil
end

-- Gets the first component found of the named type. Typically, only one
-- component of each type is available, so this is enough. However, if you want
-- to retrieve all components of a type, you should use getComponents.
function GameObject:getComponent(name)
  assert(
    self._components,
    'Missing self._components. ' ..
    'Did you mistakenly call getComponent using a `.` instead of a `:`?')
  assert(self._components[name], 'Component does not exist: ' .. name)
  return self._components[name][1]
end

--[[ Like getComponent, except it returns all the components of this type in a
list. If no components of the given type exist then it returns nil. If no type
name is passed in then return an array of all components.]]
function GameObject:getComponents(name)
  assert(
    self._components,
    'Missing self._components. ' ..
    'Did you mistakenly call getComponents using a `.` instead of a `:`?')
  if name then
    return self._components[name]
  end
  local result = {}
  for _, componentsWithSameName in pairs(self._components) do
    for _, component in pairs(componentsWithSameName) do
      table.insert(result, component)
    end
  end
  return result
end

--[[ Return the currently active piece.]]
function GameObject:getPiece()
  return self._transform:getPiece()
end

--[[ Return the type of the currently active piece, uniquified to this object.

If no state is passed (or nil), retrieve the active state, uniquified.
]]
function GameObject:getUniqueState(state)
  return self._stateManager:getUniqueState(state)
end

--[[ Return the type of the currently active piece.]]
function GameObject:getState()
  return self._stateManager:getState()
end

--[[ Return the layer to which the currently active piece is assigned.]]
function GameObject:getLayer()
  return self._stateManager:getLayer()
end

--[[ Return the currently active sprite.]]
function GameObject:getSprite()
  return self._stateManager:getSprite()
end

--[[ Return the groups in which the currently active piece is a member.]]
function GameObject:getGroups()
  return self._stateManager:getGroups()
end

--[[ Return the groups for a particular (possibly inactive) type.]]
function GameObject:getGroupsForState(state)
  return self._stateManager:getGroupsForState(state)
end

--[[ Return the states of all available states for this game object.]]
function GameObject:getAllStates()
  return self._stateManager:getAllStates()
end

--[[ Return a list of all the available sprites for use with this game object.]]
function GameObject:getSpriteNames()
  local spriteNames = {}
  self:_doOnAllComponents(
    function(component)
      if component.getSpriteNames then
        local spriteNamesFromThisComponent = component:getSpriteNames()
        for _, spriteName in ipairs(spriteNamesFromThisComponent) do
          table.insert(spriteNames, spriteName)
        end
      end
    end)
  return spriteNames
end

--[[ Dynamically change the state of a GameObject.

Use setState to change the active state at run time. Note that it is only
possible to change the state to another piece in the list of states that were
preregistered during initialization `allStateConfigs`.

States will be uniquified by the manager, so this parameter is the simple state
(i.e. one registered in the config).
]]
function GameObject:setState(newState)
  self._stateManager:setState(self._grid, newState)
end

--[[ Return the current position of this GameObject.]]
function GameObject:getPosition()
  return self._transform:getPosition()
end

--[[ Return the current orientation of this GameObject.]]
function GameObject:getOrientation()
  return self._transform:getOrientation()
end

--[[ Push this GameObject in the given direction (ignoring its orientation).

If there is a piece in the target location at the time of the move then the
piece stays where it is.

Triggers `onContact.contactName.leave` callbacks for pieces at current location
and `onContact.contactName.enter` for pieces at the location moved to. Both
callbacks are triggered even if the move is not possible and `piece` leaves and
enters the same cell.
]]
function GameObject:moveAbs(orientation)
  self._transform:moveAbs(orientation)
end

--[[ Push this GameObject in direction relative to its facing orientation.

If there is a piece in the target location at the time of the move then the
piece stays where it is.

Triggers `onContact.contactName.leave` and
`onContact.contactName.enter` callbacks in the same way as `grid:moveAbs(piece,
orientation)`.
]]
function GameObject:moveRel(orientation)
  self._transform:moveRel(orientation)
end

--[[ Teleports this GameObject to the given position and orientation.

Triggers `onContact.contactName.leave` and `onContact.contactName.enter` in the
same way as `grid:moveAbs(piece, orientation)`.
]]
function GameObject:teleport(position, orientation)
  self._transform:teleport(position, orientation)
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
function GameObject:teleportToGroup(groupName, state, orient)
  self._transform:teleportToGroup(groupName, state, orient)
end

--[[ Turns this GameObject by a given value `angle`:

      *   `0` -> No Op.
      *   `1` -> 90 degrees clockwise.
      *   `2` -> 180 degrees.
      *   `3` -> 90 degrees counterclockwise.
]]
function GameObject:turn(angle)
  self._transform:turn(angle)
end

--[[ Sets this GameObject to face a particular direction in grid space:

*   `'N'` -> North (decreasing y).
*   `'E'` -> East (increasing x).
*   `'S'` -> South (increasing y.
*   `'W'` -> West (decreasing x).
]]
function GameObject:setOrientation(orientation)
  self._transform:setOrientation(orientation)
end

--[[ Returns false if :start() not yet called on the transform component.]]
function GameObject:started()
  return self._transform:started()
end

function GameObject:reset()
  self:_doOnAllComponents(function(component)
    if component.reset then
      -- Call the component's custom reset function if it was implemented.
      component:reset()
    end
  end)
  self._grid = nil
end

return {GameObject = GameObject}
