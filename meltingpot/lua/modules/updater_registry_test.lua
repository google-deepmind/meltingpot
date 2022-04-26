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

-- Tests for the `updater_registry` module.

local meltingpot = 'meltingpot.lua.modules.'
local base_simulation = require(meltingpot .. 'base_simulation')
local component_library = require(meltingpot .. 'component_library')
local updater_registry = require(meltingpot .. 'updater_registry')
local game_object = require(meltingpot .. 'game_object')

local grid_world = require 'system.grid_world'
local random = require 'system.random'
local tile_set = require 'common.tile_set'
local helpers = require 'common.helpers'
local log = require 'common.log'
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'


local function getEmptyWorldConfig()
  local worldConfig = {
    outOfBoundsSprite = 'OutOfBounds',
    outOfViewSprite = 'OutOfView',
    updateOrder = {},
    renderOrder = {'logic', 'lowerPhysical', 'upperPhysical'},
    customSprites = {},
    hits = {},
    states = {}
  }
  return worldConfig
end

local function makeTestGameObject(id)
  local gameObject = game_object.GameObject{
      id = id,
      components = {
          -- `StateManager` is mandatory. It sets up all the states for
          -- the game object.
          component_library.StateManager{
              initialState = 'state1',
              stateConfigs = {
                  {
                      state = 'state1',
                      groups = {'spawnPoints'},
                  },
                  {
                      state = 'state2',
                      groups = {},
                  },
              }
          },
          -- `Transform` is mandatory. This component manages the location of
          -- the game object as well as its current active piece and sprite.
          component_library.Transform{
              position = {2, 0},
              orientation = 'S'
          },
      }
  }
  return gameObject
end


local tests = {}

function tests.registerSingleSimpleUpdater()
  local registry = updater_registry.UpdaterRegistry()
  registry:registerUpdater{
    updateFn = function() end,
  }
  asserts.tablesEQ(registry:getSortedPriorities(), {100})
end

function tests.registerSingleUpdaterWithPriority()
  local registry = updater_registry.UpdaterRegistry()
  registry:registerUpdater{
    updateFn = function() end,
    priority = 53,
  }
  asserts.tablesEQ(registry:getSortedPriorities(), {53})
end

function tests.registerMultipleUpdatersWithSamePriority()
  local registry = updater_registry.UpdaterRegistry()
  registry:registerUpdater{
    updateFn = function() end,
    priority = 53,
  }
  registry:registerUpdater{
    updateFn = function() end,
    priority = 53,
  }
  registry:registerUpdater{
    updateFn = function() end,
    priority = 53,
  }
  asserts.tablesEQ(registry:getSortedPriorities(), {53})
end

function tests.registerMultipleUpdatersWithDifferentPriority()
  local registry = updater_registry.UpdaterRegistry()
  registry:registerUpdater{
    updateFn = function() end,
  }
  registry:registerUpdater{
    updateFn = function() end,
    priority = 42,
  }
  registry:registerUpdater{
    updateFn = function() end,
    priority = 123,
  }
  registry:registerUpdater{
    updateFn = function() end,
    priority = 42,
  }
  asserts.tablesEQ(registry:getSortedPriorities(), {123, 100, 42})
end

function tests.uniquifyIdsDefault()
  local gameObject = makeTestGameObject('OID_1')
  local registry = updater_registry.UpdaterRegistry()
  registry:registerUpdater{updateFn = function() end}
  registry:uniquifyStatesAndAddGroups(gameObject)
  asserts.tablesEQ(
    gameObject:getGroupsForState('state1'),
    {'spawnPoints', 'UPDATER_GRP_____Updater#0'}
  )
  asserts.tablesEQ(
    gameObject:getGroupsForState('state2'), {'UPDATER_GRP_____Updater#0'}
  )
end

function tests.uniquifyIdsWithGroupPrefix()
  local gameObject = makeTestGameObject('OID_1')
  local registry = updater_registry.UpdaterRegistry()
  registry:setGroupPrefix('my_prefix')
  registry:registerUpdater{updateFn = function() end}
  registry:uniquifyStatesAndAddGroups(gameObject)
  asserts.tablesEQ(
    gameObject:getGroupsForState('state1'),
    {'spawnPoints', 'UPDATER_GRP__my_prefix_Updater#0'}
  )
  asserts.tablesEQ(
    gameObject:getGroupsForState('state2'), {'UPDATER_GRP__my_prefix_Updater#0'}
  )
end

function tests.uniquifyIdsWithMultipleGroupPrefixes()
  local gameObject = makeTestGameObject('OID_1')
  local registry = updater_registry.UpdaterRegistry()
  registry:setGroupPrefix('my_prefix')
  registry:registerUpdater{updateFn = function() end}
  -- Pretend we are in another component, registering updaters
  registry:setGroupPrefix('another')
  registry:registerUpdater{updateFn = function() end}
  registry:uniquifyStatesAndAddGroups(gameObject)
  asserts.tablesEQ(
    gameObject:getGroupsForState('state1'),
    {'spawnPoints', 'UPDATER_GRP__my_prefix_Updater#0',
      'UPDATER_GRP__another_Updater#0'}
  )
  asserts.tablesEQ(
    gameObject:getGroupsForState('state2'),
    {'UPDATER_GRP__my_prefix_Updater#0', 'UPDATER_GRP__another_Updater#0'}
  )
end

function tests.uniquifyIdsWithMultipleGroupPrefixesNoneNew()
  local gameObject = makeTestGameObject('OID_1')
  local registry = updater_registry.UpdaterRegistry()
  registry:registerUpdater{updateFn = function() end, group = 'spawnPoints'}
  -- Pretend we are in another component, registering updaters
  registry:setGroupPrefix('my_prefix')
  registry:registerUpdater{updateFn = function() end, group = 'preexisting'}
  registry:uniquifyStatesAndAddGroups(gameObject)
  asserts.tablesEQ(
    gameObject:getGroupsForState('state1'), {'spawnPoints'}
  )
  asserts.tablesEQ(
    gameObject:getGroupsForState('state2'), {}
  )
end

function tests.uniquifyIdsWithMultipleGroupPrefixesTwoGameObjects()
  local setUpdates = function(gameObject)
    local registry = gameObject:getUpdaterRegistry()
    registry:setGroupPrefix('my_prefix')
    registry:registerUpdater{updateFn = function() end, priority = 90}
    -- Pretend we are in another component, registering updaters
    registry:setGroupPrefix('another')
    registry:registerUpdater{updateFn = function() end}
    registry:uniquifyStatesAndAddGroups(gameObject)
  end

  local gameObject1 = makeTestGameObject('OID_1')
  setUpdates(gameObject1)

  local gameObject2 = makeTestGameObject('OID_2')
  setUpdates(gameObject2)

  asserts.tablesEQ(
    gameObject1:getGroupsForState('state1'),
    {'spawnPoints', 'UPDATER_GRP__my_prefix_Updater#0',
      'UPDATER_GRP__another_Updater#0'}
  )
  asserts.tablesEQ(
    gameObject1:getGroupsForState('state2'),
    {'UPDATER_GRP__my_prefix_Updater#0', 'UPDATER_GRP__another_Updater#0'}
  )
  asserts.tablesEQ(
    gameObject2:getGroupsForState('state1'),
    {'spawnPoints', 'UPDATER_GRP__my_prefix_Updater#0',
      'UPDATER_GRP__another_Updater#0'}
  )
  asserts.tablesEQ(
    gameObject2:getGroupsForState('state2'),
    {'UPDATER_GRP__my_prefix_Updater#0', 'UPDATER_GRP__another_Updater#0'}
  )

  local merged = updater_registry.UpdaterRegistry()
  merged:mergeWith(gameObject1:getUpdaterRegistry())
  merged:mergeWith(gameObject2:getUpdaterRegistry())

  asserts.tablesEQ(merged:getSortedPriorities(), {100, 90})
  local updateOrder = {}
  local expectedOrder = {
    '_priority_100_UPDATER_GRP__another_Updater#0',
    '_priority_90_UPDATER_GRP__my_prefix_Updater#0'}
  merged:addUpdateOrder(updateOrder)
  asserts.tablesEQ(expectedOrder, updateOrder)
end

function tests.uniquifyIdsSomeState()
  local gameObject = makeTestGameObject('OID_1')
  local registry = updater_registry.UpdaterRegistry()
  registry:registerUpdater{
    updateFn = function() end,
    state = 'state1',
  }
  registry:uniquifyStatesAndAddGroups(gameObject)
  asserts.tablesEQ(
    gameObject:getGroupsForState('state1'),
    {'spawnPoints', 'UPDATER_GRP_____Updater#0'}
  )
  asserts.tablesEQ(gameObject:getGroupsForState('state2'), {})
end

function tests.uniquifyIdsAnotherState()
  local gameObject = makeTestGameObject('OID_1')
  local registry = updater_registry.UpdaterRegistry()
  registry:registerUpdater{
    updateFn = function() end,
    state = 'state2',
  }
  registry:uniquifyStatesAndAddGroups(gameObject)
  asserts.tablesEQ(gameObject:getGroupsForState('state1'), {'spawnPoints'})
  asserts.tablesEQ(
    gameObject:getGroupsForState('state2'), {'UPDATER_GRP_____Updater#0'}
  )
end

function tests.uniquifyIdsSomeStates()
  local gameObject = makeTestGameObject('OID_1')
  local registry = updater_registry.UpdaterRegistry()
  registry:registerUpdater{
    updateFn = function() end,
    states = {'state1'},
  }
  registry:uniquifyStatesAndAddGroups(gameObject)
  asserts.tablesEQ(
    gameObject:getGroupsForState('state1'),
    {'spawnPoints', 'UPDATER_GRP_____Updater#0'}
  )
  asserts.tablesEQ(gameObject:getGroupsForState('state2'), {})
end

function tests.uniquifyIdsAnotherStates()
  local gameObject = makeTestGameObject('OID_1')
  local registry = updater_registry.UpdaterRegistry()
  registry:registerUpdater{
    updateFn = function() end,
    states = {'state2'},
  }
  registry:uniquifyStatesAndAddGroups(gameObject)
  asserts.tablesEQ(gameObject:getGroupsForState('state1'), {'spawnPoints'})
  asserts.tablesEQ(
    gameObject:getGroupsForState('state2'), {'UPDATER_GRP_____Updater#0'}
  )
end

function tests.uniquifyIdsAllStates()
  local gameObject = makeTestGameObject('OID_1')
  local registry = updater_registry.UpdaterRegistry()
  registry:registerUpdater{
    updateFn = function() end,
    states = {'state1', 'state2'},
  }
  registry:uniquifyStatesAndAddGroups(gameObject)
  asserts.tablesEQ(
    gameObject:getGroupsForState('state1'),
    {'spawnPoints', 'UPDATER_GRP_____Updater#0'}
  )
  asserts.tablesEQ(
    gameObject:getGroupsForState('state2'), {'UPDATER_GRP_____Updater#0'}
  )
end

function tests.mergeRegistries()
  local gameObject1 = makeTestGameObject('OID_1')
  local gameObject2 = makeTestGameObject('OID_2')

  local registry1 = gameObject1:getUpdaterRegistry()
  registry1:registerUpdater{
    updateFn = function() end,
  }
  registry1:uniquifyStatesAndAddGroups(gameObject1)

  local registry2 = gameObject2:getUpdaterRegistry()
  registry2:registerUpdater{
    updateFn = function() end,
  }
  registry2:uniquifyStatesAndAddGroups(gameObject2)

  local merged = updater_registry.UpdaterRegistry()
  merged:mergeWith(registry1)
  merged:mergeWith(registry2)

  asserts.tablesEQ(merged:getSortedPriorities(), {100})
  local updateOrder = {}
  local expectedNames = {
    ['_priority_100_UPDATER_GRP_____Updater#0'] = 1,
  }
  merged:addUpdateOrder(updateOrder)
  for _, name in ipairs(updateOrder) do
    asserts.EQ(expectedNames[name], 1)
  end
end

function tests.mergeRegistriesGeneral()
  local gameObject1 = makeTestGameObject('OID_1')
  local gameObject2 = makeTestGameObject('OID_2')

  local registry1 = gameObject1:getUpdaterRegistry()
  registry1:registerUpdater{
    updateFn = function() end,
    state = 'state2',
  }
  registry1:registerUpdater{
    updateFn = function() end,
    state = 'state1',
  }
  registry1:uniquifyStatesAndAddGroups(gameObject1)

  local registry2 = gameObject2:getUpdaterRegistry()
  registry2:registerUpdater{
    updateFn = function() end,
    states = {'state2'},
  }
  registry2:uniquifyStatesAndAddGroups(gameObject2)

  local merged = updater_registry.UpdaterRegistry()
  merged:mergeWith(registry1)
  merged:mergeWith(registry2)

  asserts.tablesEQ(merged:getSortedPriorities(), {100})
  local updateOrder = {}
  local expectedNames = {
    ['_priority_100_UPDATER_GRP_____Updater#0'] = 1,
    ['_priority_100_UPDATER_GRP_____Updater#1'] = 1,
  }
  merged:addUpdateOrder(updateOrder)
  for _, name in ipairs(updateOrder) do
    asserts.EQ(expectedNames[name], 1)
  end
end

return test_runner.run(tests)
