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

-- Entry point lua file for Factory2D levels.
local class = require 'common.class'
local helpers = require 'common.helpers'

local meltingpot = 'meltingpot.lua.modules.'
local avatar_grasp = require(meltingpot .. 'avatar_grasp')
local avatar_grappling = require(meltingpot .. 'avatar_grappling')
local simulation = require(meltingpot .. 'base_simulation')

local stamina = 'meltingpot.lua.levels.stamina.'
local shared_components = require(stamina .. 'shared_components')

local api_factory = require(meltingpot .. 'api_factory')
local component_library = require(meltingpot .. 'component_library')
local avatar_library = require(meltingpot .. 'avatar_library')

-- Finally add the local components for this game, overriding any previously
-- loaded having the same name.
local components = require 'components'


local OverrideSimulation = class.Class(simulation.BaseSimulation)

function OverrideSimulation:worldConfig()
  local config = simulation.BaseSimulation.worldConfig(self)
  local index = 0
  -- Add layer 'midPhysical' below 'upperPhysical'.
  for layerIndex, layerName in ipairs(config.renderOrder) do
    if layerName == 'upperPhysical' then
      index = layerIndex
      break
    end
  end
  table.insert(config.renderOrder, index, 'midPhysical')
  -- Add layer 'appleLayer' below 'midPhysical'.
  for layerIndex, layerName in ipairs(config.renderOrder) do
    if layerName == 'midPhysical' then
      index = layerIndex
      break
    end
  end
  table.insert(config.renderOrder, index, 'appleLayer')
  -- Add layer 'lowestPhysical' below 'lowerPhysical'.
  for layerIndex, layerName in ipairs(config.renderOrder) do
    if layerName == 'lowerPhysical' then
      index = layerIndex
      break
    end
  end
  table.insert(config.renderOrder, index, 'lowestPhysical')
  return config
end

return api_factory.apiFactory{
    Simulation = OverrideSimulation,
    settings = {
        -- Scale each sprite to a square of size `spriteSize` X `spriteSize`.
        spriteSize = 8,
        -- Terminate the episode after this many frames.
        maxEpisodeLengthFrames = 1000,
        -- Settings to pass to simulation.lua.
        simulation = {},
    }
}
