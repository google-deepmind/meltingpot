--[[ Copyright 2026 DeepMind Technologies Limited.
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

-- Focused test of the production BaseSimulation builder without Lab2D engine.
-- Run: lua meltingpot/lua/modules/base_simulation_config_test.lua
package.path = './?.lua;' .. package.path

package.preload['common.class'] = function()
  return {
    Class = function()
      local cls = {}
      cls.__index = cls
      setmetatable(cls, {
          __call = function(class, kwargs)
            local instance = setmetatable({}, class)
            instance:__init__(kwargs)
            return instance
          end,
      })
      return cls
    end,
  }
end

package.preload['common.helpers'] = function() return {} end
package.preload['common.log'] = function()
  return {v = function() end, warn = function() end}
end
package.preload['system.random'] = function() return {} end
package.preload['common.read_settings'] = function() return {} end
package.preload['system.tile'] = function() return {} end
package.preload['meltingpot.lua.modules.updater_registry'] = function()
  return {}
end
package.preload['meltingpot.lua.modules.prefab_utils'] = function()
  return {}
end
package.preload['meltingpot.lua.modules.component_registry'] = function()
  return {
    getComponent = function(name)
      return function(kwargs)
        return {name = name, received = kwargs}
      end
    end,
  }
end
package.preload['meltingpot.lua.modules.game_object'] = function()
  return {
    GameObject = function(config)
      local obj = {_id = config.id, components = config.components}
      function obj:hasComponentWithFunction(_) return false end
      return obj
    end,
  }
end

local base_simulation = require 'meltingpot.lua.modules.base_simulation'
local simulation = base_simulation.BaseSimulation{
    numPlayers = 0, settings = {map = '', gameObjects = {}}}

local sharedTransform = {position = {3, 4}, orientation = 'E'}
local avatar = simulation:buildGameObjectFromSettings({
    name = 'avatar',
    components = {
        {component = 'Transform', kwargs = sharedTransform},
        {component = 'Avatar', kwargs = {index = 1}},
    },
})
assert(avatar.components[1].received.deferPieceCreation == true,
       'Avatar transforms must defer piece creation')
assert(sharedTransform.deferPieceCreation == nil,
       'Building an avatar must not alter the caller prefab configuration')

-- The exact same configuration may be reused for an ordinary GameObject.
local ordinary = simulation:buildGameObjectFromSettings({
    name = 'ordinary',
    components = {{component = 'Transform', kwargs = sharedTransform}},
})
assert(ordinary.components[1].received.deferPieceCreation == nil,
       'A later non-avatar must not inherit the avatar-only flag')
assert(ordinary.components[1].received.position == sharedTransform.position)
assert(sharedTransform.orientation == 'E')

-- Existing caller-specified options are preserved, even when overridden for
-- this particular avatar constructor call.
local configured = {
    position = {1, 2}, orientation = 'W', deferPieceCreation = false}
local anotherAvatar = simulation:buildGameObjectFromSettings({
    name = 'another_avatar',
    components = {
        {component = 'Transform', kwargs = configured},
        {component = 'Avatar', kwargs = {index = 2}},
    },
})
assert(anotherAvatar.components[1].received.deferPieceCreation == true)
assert(configured.deferPieceCreation == false)

print('BaseSimulation avatar transform configuration regressions passed')
