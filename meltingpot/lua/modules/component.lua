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

local helpers = require 'common.helpers'
local log = require 'common.log'
local class = require 'common.class'

local Component = class.Class()

--[[ A Component is a piece of logic attached to a GameObject.

Component instances can only belong to a single GameObject.  The parent
GameObject can be accessed via self.gameObject, once the component has been
added to the GameObject.

Components have the following methods:

*   awake(): Called when the component is added to a GameObject.
*   reset(): Called once all initialisation is complete (just before `start`).
*   start(): Called once all initialisation is complete, after `reset`.
*   postStart(): Called once, after all calls to `start` on all objects have
    finished.
*   preUpdate(): Called every frame, always before update().
*   update(): Called every frame.
*   onBlocked(blockingGameObject): Called when this object attempts and fails
    to enter an occupied (x, y, layer) absolute location.
*   onEnter(enteringGameObject, contactName): Called when another object enters
    the same (x, y) location (but in another layer) as the containing object.
*   onExit(exitingGameObject, contactName): Called when another object occupying
    the same (x, y) location as this object (but in another layer) leaves the
    location.
*   onHit(hitterGameObject, hitName):  Called when the object is hit by a beam
    of the hitName type created by the hitter object.
*   onStateChange(previousState): Called when the GameObject has changed
    its state.

All of these are optional.

Updates to a component on every frame can be handled in two compatible ways:

1.  Via the `update` function. In this case, the update order is not guaranteed
    in any way, not even within components of a GameObject.
2.  Via registering update functions (by returning a table of priority to update
    function in `register_updaters`). In this case, the priority of updates will
    be guaranteed, that is, update functions with a higher priority will be
    executed before updates in a lower priority. Within a priority, no
    guarantees are provided on execution order.

The following might exist in the future:

*   onDestroy()
*   onConnect(): Called when the object is glued to annother object.
*   onDisconnect(): Called when the object stops being glued to annother object.
]]
function Component:__init__(kwargs)
  self.name = kwargs.name or 'DefaultComponentName'
  self._config = {}
  self._variables = {}

  -- After initialisation and before awake(), the field self.gameObject is set
  -- to the containing GameObject.
  -- Note: self.gameObject is created by GameObject:addComponent.
end


--Utility to insert a value in a table if not already present.
function insertIfNotPresent(tbl, element)
  for _, value in pairs(tbl) do
    if value == element then
      return
    end
  end
  table.insert(tbl, element)
end


return {Component = Component,
        insertIfNotPresent = insertIfNotPresent}
