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

-- Tests for the `game_object` module and its mandatory components.

local class = require 'common.class'

local meltingpot = 'meltingpot.lua.modules.'
local component_library = require(meltingpot .. 'component_library')
local game_object = require(meltingpot .. 'game_object')
local component = require(meltingpot .. 'component')

local grid_world = require 'system.grid_world'
local random = require 'system.random'
local tile_set = require 'common.tile_set'
local helpers = require 'common.helpers'
local log = require 'common.log'
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'

local tests = {}

local function makeTestGameObject()
  local gameObject = game_object.GameObject{
      id = "OID_1",
      components = {
          -- `StateManager` is mandatory. It sets up all the states for
          -- the game object.
          component_library.StateManager{
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
                      groups = {'testables', 'inactives'},
                  },
              }
          },
          -- `Transform` is mandatory. This component manages the location of
          -- the game object as well as its current active piece and sprite.
          component_library.Transform{
              position = {2, 0},
              orientation = 'S'
          },
          -- `Appearance` sets up all the sprites to be used by the game object.
          component_library.Appearance{
              spriteNames = {'Sprite1', 'Sprite2'},
              spriteRGBColors = {{255, 0, 0, 255}, {0, 0, 255, 255}}
          },
          -- The location observer component reports its gameObject's position
          -- on each step.
          component_library.LocationObserver{
              objectIsAvatar = false,
              alsoReportOrientation = true,
          },
      }
  }
  return gameObject
end

--[[ Call all mandatory GameObject functions in the same order as apiFactory.]]
local function simulateUsage(gameObject)
  local worldConfig = {
    updateOrder = {},
    renderOrder = {'lowerLayer', 'upperLayer'},
    customSprites = {},
    hits = {},
    states = {
        spawnPoint = {
              layer = 'logic',
              groups = {'spawnPoints'}
          },
    }
  }
  gameObject:addStates(worldConfig.states)
  gameObject:addHits(worldConfig)
  gameObject:setHits(worldConfig.hits)
  gameObject:addCustomSprites(worldConfig.customSprites)
  local contacts = {}
  -- Add all contacts from the registered states into a list to be passed later
  -- to all GameObjects so they can register their callbacks (like with hits).
  for _, typeElement in pairs(worldConfig.states) do
    if typeElement.contact then
      table.insert(contacts, typeElement.contact)
    end
  end
  gameObject:setContacts(contacts)

  local world = grid_world.World(worldConfig)
  local tileSet = tile_set.TileSet(world, {width = 5, height = 1})

  gameObject:addSprites(tileSet)
  local typeCallbacks = {}
  gameObject:addTypeCallbacks(typeCallbacks)

  local grid = world:createGrid{
      layout = '    .',
      stateMap = {['.'] = 'spawnPoint'},
      typeCallbacks = typeCallbacks
  }

  gameObject:start(grid)

  return grid
end

local function updateGrid(grid)
  random:seed(1)
  grid:update(random)
end

function tests.addComponentManyGameObjectsFail()
  local gameObject1 = makeTestGameObject()
  local gameObject2 = makeTestGameObject()
  local blocker = component_library.BeamBlocker{
      beamType = 'type',
  }
  gameObject1:addComponent(blocker)
  asserts.shouldFail(function() gameObject2:addComponent(blocker) end)
end

function tests.addComponentCallsAwake()
  local gameObject = makeTestGameObject()
  local blocker = component_library.BeamBlocker{
      beamType = 'type',
  }
  local awoken = nil
  blocker.awake = function(_self) awoken = 1 end
  gameObject:addComponent(blocker)
  asserts.EQ(awoken, 1)
end

function tests.getComponent()
  local gameObject = makeTestGameObject()
  local blocker1 = component_library.BeamBlocker{
      beamType = 'type1',
  }
  local blocker2 = component_library.BeamBlocker{
      beamType = 'type2',
  }
  gameObject:addComponent(blocker1)
  asserts.EQ(gameObject:getComponent('BeamBlocker')._config.beamType, 'type1')

  gameObject:addComponent(blocker2)
  asserts.EQ(gameObject:getComponent('BeamBlocker')._config.beamType, 'type1')

  local blockers = gameObject:getComponents('BeamBlocker')
  asserts.EQ(#blockers, 2)
  asserts.EQ(blockers[1]._config.beamType, 'type1')
  asserts.EQ(blockers[2]._config.beamType, 'type2')
end

function tests.getState()
  local gameObject = makeTestGameObject()
  asserts.EQ(gameObject:getState(), 'state1')
end

function tests.getUniqueState()
  local gameObject = makeTestGameObject()
  asserts.hasSubstr(gameObject:getUniqueState(), 'state1')
end

function tests.setUniqueStateWithoutGridUpdate()
  local gameObject = makeTestGameObject()
  local grid = simulateUsage(gameObject)
  gameObject:setState('state2')
  -- No update to the engine (e.g. grid:update) means no change in state.
  asserts.hasSubstr(gameObject:getUniqueState(), 'state1')
end

function tests.getUniqueStateAfterSet()
  local gameObject = makeTestGameObject()
  local grid = simulateUsage(gameObject)
  gameObject:setState('state2')
  -- Update the engine so that the change in state is processed
  updateGrid(grid)
  asserts.hasSubstr(gameObject:getUniqueState(), 'state2')
end

function tests.getUniqueStateAfterSetState()
  local gameObject = makeTestGameObject()
  local grid = simulateUsage(gameObject)
  local uptBefore = gameObject:getUniqueState()
  gameObject:setState('state2')
  -- Update the engine so that the change in state is processed
  updateGrid(grid)
  local uptAfter = gameObject:getUniqueState()
  asserts.NE(uptBefore, uptAfter)
  asserts.hasSubstr(uptBefore, gameObject._id)
  asserts.hasSubstr(uptAfter, gameObject._id)
  asserts.hasSubstr(uptAfter, 'state2')
end

function tests.getLayer()
  local gameObject = makeTestGameObject()
  asserts.EQ(gameObject:getLayer(), 'upperLayer')
end

function tests.getSprite()
  local gameObject = makeTestGameObject()
  asserts.EQ(gameObject:getSprite(), 'Sprite1')
end

function tests.getGroups()
  local gameObject = makeTestGameObject()
  asserts.tablesEQ(gameObject:getGroups(), {'testables', 'spawnPoints'})
end

function tests.getGroupsForState()
  local gameObject = makeTestGameObject()
  asserts.tablesEQ(gameObject:getGroupsForState('state2'),
                   {'testables', 'inactives'})
end

function tests.getAllStates()
  local gameObject = makeTestGameObject()
  local result = gameObject:getAllStates()
  table.sort(result)
  asserts.tablesEQ(result, {'state1', 'state2'})
end

function tests.getSpriteNames()
  local gameObject = makeTestGameObject()
  asserts.tablesEQ(gameObject:getSpriteNames(), {'Sprite1', 'Sprite2'})
end

function tests.getPiece()
  local gameObject = makeTestGameObject()
  simulateUsage(gameObject)
  asserts.GT(gameObject:getPiece(), -1)
end

function tests.setState()
  local gameObject = makeTestGameObject()
  local grid = simulateUsage(gameObject)
  asserts.EQ(gameObject:getState(), 'state1')

  gameObject:setState('state2')
  -- Update the engine so that the change in state is processed
  updateGrid(grid)

  asserts.EQ(gameObject:getState(), 'state2')
  asserts.EQ(gameObject:getLayer(), 'lowerLayer')
  asserts.EQ(gameObject:getSprite(), 'Sprite2')
  asserts.tablesEQ(gameObject:getGroups(), {'testables', 'inactives'})
end

function tests.moveAbs()
  local gameObject = makeTestGameObject()
  local grid = simulateUsage(gameObject)
  updateGrid(grid)

  asserts.tablesEQ(gameObject:getPosition(), {2, 0})

  gameObject:moveAbs('E')
  updateGrid(grid)
  gameObject:preUpdate()

  asserts.tablesEQ(gameObject:getPosition(), {3, 0})
end

function tests.moveRel()
  local gameObject = makeTestGameObject()
  local grid = simulateUsage(gameObject)
  updateGrid(grid)

  asserts.tablesEQ(gameObject:getPosition(), {2, 0})

  gameObject:moveRel('E')
  updateGrid(grid)
  gameObject:preUpdate()

  asserts.tablesEQ(gameObject:getPosition(), {1, 0})
end

function tests.teleport()
  local gameObject = makeTestGameObject()
  local grid = simulateUsage(gameObject)
  updateGrid(grid)

  asserts.tablesEQ(gameObject:getPosition(), {2, 0})
  asserts.EQ(gameObject:getOrientation(), 'S')

  gameObject:teleport({1, 0}, 'W')
  updateGrid(grid)
  gameObject:preUpdate()

  asserts.tablesEQ(gameObject:getPosition(), {1, 0})
  asserts.EQ(gameObject:getOrientation(), 'W')
end

function tests.teleportToGroup()
  local gameObject = makeTestGameObject()
  local grid = simulateUsage(gameObject)
  updateGrid(grid)

  asserts.tablesEQ(gameObject:getPosition(), {2, 0})

  gameObject:teleportToGroup('spawnPoints', 'state2')
  updateGrid(grid)
  gameObject:preUpdate()

  asserts.tablesEQ(gameObject:getPosition(), {4, 0})
  asserts.EQ(gameObject:getState(), 'state2')
end

function tests.teleportToGroupWithOrient()
  local gameObject = makeTestGameObject()
  local grid = simulateUsage(gameObject)
  updateGrid(grid)

  asserts.tablesEQ(gameObject:getPosition(), {2, 0})

  local originalOrient = 'S'
  gameObject:setOrientation(originalOrient)
  updateGrid(grid)
  asserts.EQ(gameObject:getOrientation(), originalOrient)

  gameObject:teleportToGroup(
      'spawnPoints', 'state2',
      grid_world.TELEPORT_ORIENTATION.KEEP_ORIGINAL)
  updateGrid(grid)
  gameObject:preUpdate()

  asserts.EQ(gameObject:getOrientation(), originalOrient)
end

function tests.turn()
  local gameObject = makeTestGameObject()
  local grid = simulateUsage(gameObject)
  updateGrid(grid)

  asserts.EQ(gameObject:getOrientation(), 'S')

  gameObject:update(grid)

  -- Turn number 3 is 90 degrees counterclockwise.
  gameObject:turn(3)
  updateGrid(grid)
  gameObject:preUpdate()

  asserts.EQ(gameObject:getOrientation(), 'E')
end

function tests.setOrientation()
  local gameObject = makeTestGameObject()
  local grid = simulateUsage(gameObject)
  updateGrid(grid)

  asserts.EQ(gameObject:getOrientation(), 'S')

  gameObject:update(grid)

  gameObject:setOrientation('E')
  updateGrid(grid)
  gameObject:preUpdate()

  asserts.EQ(gameObject:getOrientation(), 'E')
end

function tests.reset()
  local gameObject = makeTestGameObject()
  local grid = simulateUsage(gameObject)
  updateGrid(grid)

  -- Assert the initial conditions before changing anything dynamically.
  asserts.EQ(gameObject:getOrientation(), 'S')
  asserts.tablesEQ(gameObject:getPosition(), {2, 0})
  asserts.EQ(gameObject:getState(), 'state1')

  -- Dynamically change some things, simulating gameplay.
  gameObject:setOrientation('E')
  gameObject:moveAbs('E')
  gameObject:setState('state2')
  updateGrid(grid)
  gameObject:preUpdate()

  -- Assert that they did indeed change.
  asserts.EQ(gameObject:getOrientation(), 'E')
  asserts.tablesEQ(gameObject:getPosition(), {3, 0})
  asserts.EQ(gameObject:getState(), 'state2')

  -- Now test that reset actually restores to the initial state.
  gameObject:reset()
  -- We need to call start because position and orientation are undefined before
  -- this call.  We reuse the grid because it is still valid, although for real
  -- resets, we typically recreate the grid itself.
  gameObject:start(grid)
  asserts.EQ(gameObject:getOrientation(), 'S')
  asserts.tablesEQ(gameObject:getPosition(), {2, 0})
  asserts.EQ(gameObject:getState(), 'state1')
end


return test_runner.run(tests)
