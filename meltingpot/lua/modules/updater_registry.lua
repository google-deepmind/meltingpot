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

--[[ The UpdaterRegistry manages adding update functions with to be executed
at engine updates.
--]]

local helpers = require 'common.helpers'
local log = require 'common.log'
local class = require 'common.class'


local UpdaterRegistry = class.Class()

function UpdaterRegistry:__init__(kwargs)
  self._allObjectStates = '__all_object_states__'
  -- Maps a priority to a list of update spec. The update spec is a table with
  -- an update function, a priority, the states it applies to, etc.
  self._updateTable = {}
  --[[ The group prefix is used to create a unique group for all states that
  require the same updater. The main use case is to group updaters by Component,
  so that the same call from all components of all game objects are grouped
  together. This is only used when the group is not set explicitly at
  registration time.
  --]]
  self._groupPrefix = '__'
  -- A count for groups sharing the same prefix. That way, we can register
  -- mutiple updaters with the same group prefix, and they will not clobber each
  -- other.
  self._updaterCount = 0
end

--[[ Register an update function to be executed on updates of the engine.

Registered updates have a numerical priority, with default of 100. Higher
priority means the updates occur earlier in the update cycle. Optionally, an
update function can be registered to only be triggered for a particular game
object state, or set of states. In that case, the update function will only be
called when the game object is in that state. The default is to call the update
function regardless of the game object's state.

You also have the ability to set the update to start executing after a certain
number of frames (after any state change, or at the beginning of the episode).
Finally, you can also set a probability that the function will be called, to
control whether the function is called every step, or only on a fraction of
them.

Updates are controlled via groups (game object states are associated with
arbitrary groups) which is particularly relevant for probabilistic updates.
Whenever a component across multiple game objects registers an update function,
the engine can optimise the probabilistic update and efficiently call the
function only on those components that were probabilistically selected. You can
either manually provide a group when registering an udpate, or let the registry
handle this for you. If you specify a group manually, the registered updater
will take all game objects that are in states that belong to the group, and only
apply the update function on them. If the group is not specified, only the
corresponding updates registered on the same component but across different game
objects will share a group.

This function receives the following parameters, all of which are optional
except the update function `updateFn`:

*   updateFn: The update function to register. The function will be called
    wihtout any arguments.
*   priority: The priority of the update. If not provided, defaults to priority
    100.
*   startFrame: The number of frames to wait after any state change to start
    executing this update.
*   probability: The probability that the update gets called per time step. By
    default this is set to 1, so that the update function is called every frame.
*   group: The group to which the registered updater applies. That is, all game
    objects which are on a state that has this group will have the update
    function registered here called on them. If group is not given, it defaults
    to grouping (corresponding) updates from the same component across all game
    objects.
*   state: The game object state for which this function should be called. If
    not provided, the function will be called for all states.
*   states: a list of states for which this update should be called. This takes
    precedence over `state`. If both are provided, only the value of `states` is
    used.

Typically, you add updaters in the `registerUpdaters` function of a component.
For example,

```
function MyComponent:registerUpdaters(updaterRegistry)
  -- Registers an update with default priority and for all states.
  updaterRegistry:registerUpdater{
    updateFn = self:normalUpdate(),
  }
  -- Registers an update with high priority that only gets called when the
  -- object is in the `_aliveState` state.
  updaterRegistry:registerUpdater{
    priority = 200,
    state = self.config._aliveState,
    updateFn = self:highPriorityUpdate(),
  }
end
```

--]]
function UpdaterRegistry:registerUpdater(params)
  -- Set default parameter value
  setmetatable(
      params,
      {__index = {
          priority = 100,
          startFrame = 0,
          probability = 1.0,
          group = nil,
          state = nil,
          states = nil,
          _updaterName = nil,
      }})
  -- Unwrap parameters, optional or not, into local variables
  local updateFn, priority, startFrame, probability, group, state, states,
      updaterName =
    params[1] or params.updateFn,
    params[2] or params.priority,
    params[3] or params.startFrame,
    params[4] or params.probability,
    params[5] or params.group,
    params[6] or params.state,
    params[7] or params.states,
    params[8] or params._updaterName  -- Internal field, do not use.
  if self._updateTable[priority] == nil then
    self._updateTable[priority] = {}
  end
  local addGroup = false
  -- If no group is needed, create one from prefix and counter, but signal that
  -- we need to add this group to the states it affects (by setting _addGroup).
  if group == nil then
    group = ('UPDATER_GRP__' .. self._groupPrefix .. '_Updater#' ..
             self._updaterCount)
    self._updaterCount = self._updaterCount + 1
    addGroup = true
  end
  table.insert(self._updateTable[priority],
               {updateFn = updateFn,
                startFrame = startFrame,
                probability = probability,
                group = group,
                state = state,
                states = states,
                _updaterName = updaterName,
                _addGroup = addGroup})  -- Whether a new group must be created.
end

function UpdaterRegistry:setGroupPrefix(prefix)
  self._groupPrefix = prefix
  self._updaterCount = 0
end

function UpdaterRegistry:getSortedPriorities()
  local tkeys = {}
  -- populate the table that holds the keys
  for k, _ in pairs(self._updateTable) do table.insert(tkeys, k) end
  -- sort the keys
  table.sort(tkeys, function(a, b) return a > b end)
  return tkeys
end

--[[ After this function is called, all states registered will be uniquified to
the given game object. This means that the updates will only affect the game
object's components and states.

In addition, all states that are relevant for updates will have special groups
added to them so they can be called by the engine updaters.
--]]
function UpdaterRegistry:uniquifyStatesAndAddGroups(gameObject)
  self._byStateAndUpdaterName = {}
  for priority, specs in pairs(self._updateTable) do
    for _, updateSpec in pairs(specs) do
      if updateSpec.states == nil then
        if updateSpec.state == nil then
          updateSpec.states = gameObject:getAllStates()
        else
          updateSpec.states = {updateSpec.state}
        end
        -- We don't need the state anymore, this has been transferred to states.
        updateSpec.state = nil
      end
      updateSpec._updaterName = (
          '_priority_' .. priority .. '_' .. updateSpec.group)

      -- At this point, we only care about specs.states. Uniquify all of them.
      for i, state in ipairs(updateSpec.states) do
        if updateSpec._addGroup then
          table.insert(gameObject:getGroupsForState(state), updateSpec.group)
        end
        updateSpec.states[i] = gameObject:getUniqueState(state)
        if self._byStateAndUpdaterName[state] == nil then
          self._byStateAndUpdaterName[state] = {}
        end
        self._byStateAndUpdaterName[state][updateSpec._updaterName] = true
      end
    end
  end
end

-- Merges this UpdaterRegistry with another. Must be done after uniquification.
function UpdaterRegistry:mergeWith(updaterRegistry)
  if self._byStateAndUpdaterName == nil then
    self._byStateAndUpdaterName = {}
  end
  local shouldAdd = true
  local allStates = true
  for priority, specs in pairs(updaterRegistry._updateTable) do
    for _, updateSpec in pairs(specs) do
      shouldAdd = true
      -- Check if all states and updater names are already included in the
      -- updater. If so, then ignore this spec.
      allStates = true
      for _, state in pairs(updateSpec.states) do
        if not self._byStateAndUpdaterName[state] or
           not self._byStateAndUpdaterName[state][updateSpec._updaterName] then
          allStates = false
          break
        end
      end
      if allStates then
        shouldAdd = false
      end

      if shouldAdd then
        self:registerUpdater{
            updateFn = updateSpec.updateFn,
            priority = priority,
            startFrame = updateSpec.startFrame,
            probability = updateSpec.probability,
            group = updateSpec.group,
            states = updateSpec.states,
            _updaterName = updateSpec._updaterName}

        -- Mark all states in this updater name as seen.
        for _, state in pairs(updateSpec.states) do
          if self._byStateAndUpdaterName[state] == nil then
            self._byStateAndUpdaterName[state] = {}
          end
          self._byStateAndUpdaterName[state][updateSpec._updaterName] = true
        end
      end
    end
  end
end

-- Adds all the priorities into the world config update order.
function UpdaterRegistry:addUpdateOrder(updateOrder)
  local priorities = self:getSortedPriorities()
  for _, priority in ipairs(priorities) do
    local updaterNames = {}
    local specs = self._updateTable[priority]
    -- Gather all updater names with the same priority.
    for _, updateSpec in pairs(specs) do
      updaterNames[updateSpec._updaterName] = true
    end
    for name, _ in pairs(updaterNames) do
      table.insert(updateOrder, name)
    end
  end
end

function UpdaterRegistry:registerCallbacks(callbacks)
  for priority, specs in pairs(self._updateTable) do
    for _, updateSpec in pairs(specs) do
      for _, state in ipairs(updateSpec.states) do
        callbacks[state].onUpdate[updateSpec._updaterName] = updateSpec.updateFn
      end
    end
  end
end

function UpdaterRegistry:registerGrid(grid, callbacks)
  for priority, specs in pairs(self._updateTable) do
    local updaterNames = {}
    for _, updateSpec in pairs(specs) do
      -- Only set the updater once, regardless of how many components requested
      -- it. Per priority.
      if updaterNames[updateSpec._updaterName] == nil then
        grid:setUpdater{
            -- update really should be called `updaterName`
            update = updateSpec._updaterName,
            group = updateSpec.group,
            probability = updateSpec.probability,
            startFrame = updateSpec.startFrame,
        }
        updaterNames[updateSpec._updaterName] = true
      end
    end
  end
end

return {UpdaterRegistry = UpdaterRegistry}
