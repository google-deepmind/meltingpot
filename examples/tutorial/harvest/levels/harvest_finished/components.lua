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
local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')


-- DensityRegrow makes the containing GameObject switch from `waitState` to
-- `liveState` at a rate based on the number of surrounding objects in
-- `liveState` and the configured `baseRate`.
local DensityRegrow = class.Class(component.Component)

function DensityRegrow:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('DensityRegrow')},
      -- `baseRate` indicates the base probability per frame of switching from
      -- wait state to live state.
      {'baseRate', args.ge(0.0), args.le(1.0)},
      -- The name of the state representing the active or alive state.
      {'liveState', args.stringType},
      -- The name of the state representing the inactive or dormans state.
      {'waitState', args.stringType},
      -- The radius of the neighborhood
      {'neighborhoodRadius', args.numberType, args.default(1)},
      -- The layer to query for objects in `liveState`
      {'queryLayer', args.stringType, args.default("lowerPhysical")},
  })
  DensityRegrow.Base.__init__(self, kwargs)

  self._config.baseRate = kwargs.baseRate
  self._config.liveState = kwargs.liveState
  self._config.waitState = kwargs.waitState
  self._config.neighborhoodRadius = kwargs.neighborhoodRadius
  self._config.queryLayer = kwargs.queryLayer
end

function DensityRegrow:registerUpdaters(updaterRegistry)
  updaterRegistry:registerUpdater{
      state = self._config.waitState,
      updateFn = function()
          local transform = self.gameObject:getComponent("Transform")
          -- Get neighbors
          local objects = transform:queryDiamond(
              self._config.queryLayer, self._config.neighborhoodRadius)
          -- Count live neighbors
          local liveNeighbors = 0
          for _, object in pairs(objects) do
            if object:getState() == self._config.liveState then
              liveNeighbors = liveNeighbors + 1
            end
          end
          local actualRate = liveNeighbors * self._config.baseRate
          if random:uniformReal(0, 1) < actualRate then
            self.gameObject:setState(self._config.liveState)
          end
        end,
  }
end


local allComponents = {
    DensityRegrow = DensityRegrow,
}

component_registry.registerAllComponents(allComponents)

return allComponents
