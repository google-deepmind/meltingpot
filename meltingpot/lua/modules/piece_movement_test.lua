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

local grid_world = require 'system.grid_world'
local random = require 'system.random'
local tile_set = require 'common.tile_set'
local helpers = require 'common.helpers'
local log = require 'common.log'
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'

local tests = {}

local function simulateUsage()
  local worldConfig = {
    outOfBoundsSprite = 'OutOfBounds',
    outOfViewSprite = 'OutOfView',
    updateOrder = {'fruit'},
    renderOrder = {'logic', 'pieces'},
    customSprites = {},
    hits = {},
    states = {
        apple = {
            layer = 'logic',
            sprite = 'Apple',
        },
    }
  }

  local world = grid_world.World(worldConfig)
  local tileSet = tile_set.TileSet(world, {width = 5, height = 5})

  tileSet:addColor('OutOfBounds', {0, 0, 0})
  tileSet:addColor('OutOfView', {80, 80, 80})
  tileSet:addColor('Apple', {0, 250, 0})

  local stateCallbacks = {}
  stateCallbacks['apple'] = {
      onContact = {
          avatar = {
              enter = function(grid, applePiece, avatarPiece)
                grid:setState(applePiece, 'appleWait')
              end
          }
      }
  }

  local grid, pieces = world:createGrid{
      layout = '   A ',
      stateMap = {A = 'apple'},
      stateCallbacks = stateCallbacks
  }
  asserts.EQ(#pieces, 1)
  return grid, pieces[1]
end

function tests.moveAbs()
  random:seed(1)
  local grid, piece = simulateUsage()
  log.info("grid\n" .. tostring(grid))
  asserts.tablesEQ(grid:position(0), {3, 0})
  grid:moveAbs(piece, 'E')
  grid:update(random)
  log.info("grid\n" .. tostring(grid))
  asserts.tablesEQ(grid:position(0), {4, 0})
end

function tests.teleport()
  random:seed(1)
  local grid, piece = simulateUsage()
  log.info("grid\n" .. tostring(grid))
  asserts.tablesEQ(grid:position(0), {3, 0})
  grid:teleport(piece, {1, 0})
  grid:update(random)
  log.info("grid\n" .. tostring(grid))
  asserts.tablesEQ(grid:position(0), {1, 0})
end

return test_runner.run(tests)

