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

-- Tests for the `base_simulation` module.

local meltingpot = 'meltingpot.lua.modules.'
local base_simulation = require(meltingpot .. 'base_simulation')
local component_library = require(meltingpot .. 'component_library')
local game_object = require(meltingpot .. 'game_object')

local grid_world = require 'system.grid_world'
local random = require 'system.random'
local tile_set = require 'common.tile_set'
local helpers = require 'common.helpers'
local log = require 'common.log'
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'

local tests = {}

local function getTestGameObjectConfig()
  return {
      components = {
        {
          -- `State` is mandatory. It sets up all the states for the
          -- game object.
          component = 'StateManager',
          kwargs = {
              initialState = 'state1',
              stateConfigs = {
                  {
                      state = 'state1',
                      layer = 'upperLayer',
                      sprite = 'Sprite1',
                      groups = {'testables', 'spawnPoints'},
                  },
                  {
                      state = 'state2',
                      layer = 'lowerLayer',
                      sprite = 'Sprite2',
                      groups = {'testables'},
                  },
              },
          },
        }, {
          -- `Transform` is mandatory. This component manages the location of
          -- the game object as well as its current active piece and sprite.
          component = 'Transform',
          kwargs = {
              position = {2, 2},
              orientation = 'W'
          },
        }, {
          -- `Appearance` is optional. It sets up all the sprites to be used
          -- by the game object.
          component = 'Appearance',
          kwargs = {
              spriteNames = {'Sprite1', 'Sprite2'},
              spriteRGBColors = {{255, 0, 0, 255}, {0, 0, 255, 255}}
          },
        },
      },
  }
end

local function makeTestSimulation()
  local baseSimulation = base_simulation.BaseSimulation{
    numPlayers = 0,
    settings = {map = ""}}
  return baseSimulation
end

local function simulateUsage(baseSimulation)
  local worldConfig = baseSimulation:worldConfig()
  worldConfig.renderOrder = {'lowerLayer', 'upperLayer'}
  worldConfig.states.spawnPoint = {layer = 'logic',
                                  groups = {'spawnPoints'}}

  local world = grid_world.World(worldConfig)
  local tileSet = tile_set.TileSet(world, {width = 5, height = 1})

  baseSimulation:addSprites(tileSet)

  local stateCallbacks = {}
  baseSimulation:stateCallbacks(stateCallbacks, nil)

  local grid = world:createGrid{
      layout = '    .',
      stateMap = {['.'] = 'spawnPoint'},
      stateCallbacks = stateCallbacks
  }

  baseSimulation:start(grid)

  return grid
end

function tests.uniqueGameObjectID()
  local baseSimulation = makeTestSimulation()
  local object1 = baseSimulation:buildGameObjectFromSettings(
      getTestGameObjectConfig())
  local object2 = baseSimulation:buildGameObjectFromSettings(
      getTestGameObjectConfig())
  asserts.NE(object1._id, object2._id)
end

function tests.stressTestUniqueGameObjectID()
  local baseSimulation = makeTestSimulation()
  local ids = {}
  for i = 1, 10000 do
    local object = baseSimulation:buildGameObjectFromSettings(
        getTestGameObjectConfig())
    asserts.EQ(ids[object._id], nil)
    ids[object._id] = 1
  end
end

function tests.getGameObjectFromPiece()
  local baseSimulation = makeTestSimulation()
  local gameObjectConf = getTestGameObjectConfig()
  local gameObject = baseSimulation:buildGameObjectFromSettings(gameObjectConf)
  local grid = simulateUsage(baseSimulation)

  local piece = gameObject:getPiece()
  local returnedGameObject = baseSimulation:getGameObjectFromPiece(piece)

  asserts.tablesEQ(returnedGameObject:getPosition(), gameObject:getPosition())
  asserts.EQ(returnedGameObject:getOrientation(), gameObject:getOrientation())

  asserts.tablesEQ(returnedGameObject:getPosition(), {2, 2})
  asserts.EQ(returnedGameObject:getOrientation(), 'W')
end

return test_runner.run(tests)
