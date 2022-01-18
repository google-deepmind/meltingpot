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
local asserts = require 'testing.asserts'
local mocking = require 'testing.mocking'
local test_runner = require 'testing.test_runner'

local mock = mocking.mock
local when = mocking.when
local capture = mocking.capture

local random = mocking.spyLibrary('system.random')
local rngState = capture.anyValue()
local anyValue = capture.anyValue()

when(random).choice(rngState, anyValue).thenCall(function(x, y) return y[2] end)

local meltingpot = 'meltingpot.lua.modules.'
local prefab_utils = require(meltingpot .. 'prefab_utils')

local tests = {}

local function _createPrefab(name)
  return {
    name = name,
    components = {
      {
        component = 'Transform',
        kwargs = {
          position = {-1, -1},
          orientation = '?',
        },
      },
    },
  }
end

local function _createGameObjectConfig(name, x, y, orientation)
  return {
    name = name,
    components = {
      {
        component = 'Transform',
        kwargs = {
          position = {x, y},
          orientation = orientation,
        },
      },
    },
  }
end

function tests.buildGameObjectFromNamedPrefab()
  ascii = 'a'
  prefabs = {prefab = _createPrefab('prefab')}
  charPrefabMap = {a = 'prefab'}
  gameObjects = prefab_utils.buildGameObjectConfigs(
      ascii, prefabs, charPrefabMap)
  asserts.EQ(#gameObjects, 1)
  asserts.tablesEQ(gameObjects[1], _createGameObjectConfig('prefab', 0, 0, 'N'))
end

function tests.buildTwoRowsGameObjectsFromNamedPrefab()
  ascii = 'a\na'
  prefabs = {prefab = _createPrefab('prefab')}
  charPrefabMap = {a = 'prefab'}
  gameObjects = prefab_utils.buildGameObjectConfigs(
      ascii, prefabs, charPrefabMap)
  asserts.EQ(#gameObjects, 2)
  asserts.tablesEQ(gameObjects[1], _createGameObjectConfig('prefab', 0, 0, 'N'))
  asserts.tablesEQ(gameObjects[2], _createGameObjectConfig('prefab', 0, 1, 'N'))
end

function tests.buildTwoColsGameObjectsFromNamedPrefab()
  ascii = 'aa'
  prefabs = {prefab = _createPrefab('prefab')}
  charPrefabMap = {a = 'prefab'}
  gameObjects = prefab_utils.buildGameObjectConfigs(
      ascii, prefabs, charPrefabMap)
  asserts.EQ(#gameObjects, 2)
  asserts.tablesEQ(gameObjects[1], _createGameObjectConfig('prefab', 0, 0, 'N'))
  asserts.tablesEQ(gameObjects[2], _createGameObjectConfig('prefab', 1, 0, 'N'))
end

function tests.buildTwoDifferentGameObjectsFromNamedPrefab()
  ascii = 'ba'
  prefabs = {
      prefabA = _createPrefab('prefabA'),
      prefabB = _createPrefab('prefabB'),
  }
  charPrefabMap = {a = 'prefabA', b = 'prefabB'}
  gameObjects = prefab_utils.buildGameObjectConfigs(
      ascii, prefabs, charPrefabMap)
  asserts.EQ(#gameObjects, 2)
  asserts.tablesEQ(
      gameObjects[1], _createGameObjectConfig('prefabB', 0, 0, 'N'))
  asserts.tablesEQ(
      gameObjects[2], _createGameObjectConfig('prefabA', 1, 0, 'N'))
end

function tests.buildGameObjectNotFoundIgnored()
  -- Request a character not present in the charPrefabMap
  ascii = 'b'
  prefabs = {prefab = _createPrefab('prefab')}
  charPrefabMap = {a = 'prefab'}
  gameObjects = prefab_utils.buildGameObjectConfigs(
      ascii, prefabs, charPrefabMap)
  asserts.EQ(#gameObjects, 0)
end

function tests.buildGameObjectFromAllSpec()
  ascii = 'x'
  prefabs = {
      prefabA = _createPrefab('prefabA'),
      prefabB = _createPrefab('prefabB'),
  }
  charPrefabMap = {x = {type = 'all', list = {'prefabB', 'prefabA'}}}
  gameObjects = prefab_utils.buildGameObjectConfigs(
      ascii, prefabs, charPrefabMap)
  asserts.EQ(#gameObjects, 2)
  asserts.tablesEQ(
      gameObjects[1], _createGameObjectConfig('prefabB', 0, 0, 'N'))
  asserts.tablesEQ(
      gameObjects[2], _createGameObjectConfig('prefabA', 0, 0, 'N'))
end

function tests.buildGameObjectFromChoiceSpec()
  ascii = 'x'
  prefabs = {
      prefabA = _createPrefab('prefabA'),
      prefabB = _createPrefab('prefabB'),
  }
  charPrefabMap = {x = {type = 'choice', list = {'prefabB', 'prefabA'}}}
  gameObjects = prefab_utils.buildGameObjectConfigs(
      ascii, prefabs, charPrefabMap)
  asserts.EQ(#gameObjects, 1)
  asserts.tablesEQ(
      gameObjects[1], _createGameObjectConfig('prefabA', 0, 0, 'N'))
end

function tests.buildGameObjectFromNestedSpecChoiceAll()
  ascii = 'x'
  prefabs = {
      prefabA = _createPrefab('prefabA'),
      prefabB = _createPrefab('prefabB'),
  }
  charPrefabMap = {x = {
      type = 'choice',
      list = {'prefabB', {type = 'all', list = {'prefabA', 'prefabB'}}}}}
  gameObjects = prefab_utils.buildGameObjectConfigs(
      ascii, prefabs, charPrefabMap)
  asserts.EQ(#gameObjects, 2)
  asserts.tablesEQ(
      gameObjects[1], _createGameObjectConfig('prefabA', 0, 0, 'N'))
  asserts.tablesEQ(
      gameObjects[2], _createGameObjectConfig('prefabB', 0, 0, 'N'))
end

function tests.buildGameObjectFromNestedSpecAllChoice()
  ascii = 'x'
  prefabs = {
      prefabA = _createPrefab('prefabA'),
      prefabB = _createPrefab('prefabB'),
      prefabC = _createPrefab('prefabC'),
  }
  charPrefabMap = {x = {
      type = 'all',
      list = {'prefabA', {type = 'choice', list = {'prefabB', 'prefabC'}}}}}
  gameObjects = prefab_utils.buildGameObjectConfigs(
      ascii, prefabs, charPrefabMap)
  asserts.EQ(#gameObjects, 2)
  asserts.tablesEQ(
      gameObjects[1], _createGameObjectConfig('prefabA', 0, 0, 'N'))
  asserts.tablesEQ(
      gameObjects[2], _createGameObjectConfig('prefabC', 0, 0, 'N'))
end

return test_runner.run(tests)

