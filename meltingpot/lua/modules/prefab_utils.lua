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

-- Utility functions to build GameObjects from prefabs and ASCII maps.

local helpers = require 'common.helpers'
local log = require 'common.log'
local tables = require 'common.tables'
local random = require 'system.random'

local meltingpot = 'meltingpot.lua.modules.'
local colors = require(meltingpot .. 'colors')


local function _getFirstNamedComponent(gameObjectConfig, name)
  for _, comp in pairs(gameObjectConfig["components"]) do
    if comp.component == name then
      return comp
    end
  end
  error("No component with name " .. name .. " found.")
end

local function _createGameObjectConfig(prefab, row, col)
  local go = tables.deepCopy(prefab)
  local transform = _getFirstNamedComponent(go, "Transform")
  transform.kwargs.position = {col, row}
  transform.kwargs.orientation = "N"
  return go
end

local function _createPrefabsFromSpec(row, col, prefab, prefabs, gameObjects)
  if type(prefab) == "table" then  -- Prefab spec case.
    assert(
      prefab.type,
      "A prefab specification other than a prefab name must be a table with " ..
      "the following format:\n" ..
      "  {type = '<compound_type>', list = {...}}\n\n" ..
      "Where <compound_type> must be one of 'all' or 'choice'.")
    assert(
      prefab.list,
      "A prefab specification other than a prefab name must be a table with " ..
      "the following format:\n" ..
      "  {type = '<compound_type>', list = {...}}\n\n" ..
      "Where the `list` must be a table whose values are prefab names.")
    log.v(1, "char maps to table: " .. helpers.tostring(prefab))
    if prefab.type == "all" then
      for _, p in pairs(prefab.list) do
        _createPrefabsFromSpec(row, col, p, prefabs, gameObjects)
      end
    elseif prefab.type == "choice" then
      local which = random:choice(prefab.list)
      _createPrefabsFromSpec(row, col, which, prefabs, gameObjects)
    end
  else  -- Named prefab case.
    local actual = rawget(prefabs, prefab)
    assert(actual, "Prefab with name '" .. prefab .. "' not found prefabs.")
    table.insert(gameObjects, _createGameObjectConfig(actual, row, col))
  end
end

--[[ This function processes a single character in the ASCII map, at the given
row & column (indexed from 0). The character, if not found in charPrefabMap,
is ignored
]]
local function _processChar(row, col, char, charPrefabMap, prefabs, gameObjects)
  log.v(2, "Processing char '" .. char .. "' at " .. row .. ", " .. col)
  local prefab = rawget(charPrefabMap, char)
  if prefab == nil then
    if char ~= " " then
      log.warn("Character", char, "not found in the charPrefabMap. Ignoring.")
    end
    return
  end
  _createPrefabsFromSpec(row, col, prefab, prefabs, gameObjects)
end

-- Iterate over all characters in the map, and call the provided function
-- for each character.
local function _visitText(text, func)
  local stripNewLine = 1
  while text:sub(stripNewLine, stripNewLine) == '\n' do
    stripNewLine = stripNewLine + 1
  end
  local row = 0
  local col = 0
  for i = stripNewLine, #text do
    local c = text:sub(i, i)
    if c == '\n' then
      row = row + 1
      col = 0
    else
      func(row, col, c)
      col = col + 1
    end
  end
end

function buildAvatarConfigs(numPlayers, prefabs, playerPalettes)
  local gameObjects = {}

  log.v(1, "Building " .. numPlayers .. " avatars.")
  if #playerPalettes ~= numPlayers then
    log.v(
      1,
      "Player palettes passed (" .. #playerPalettes .. ") don't correspond " ..
      "to the number of players (" .. numPlayers .. "). " ..
      "Creating default player palettes.")
    playerPalettes = {}
    for i = 1, numPlayers do
      log.v(2, "Adding pallette with: " .. helpers.tostring(colors.colors[i]))
      table.insert(playerPalettes, colors.getPalette(colors.colors[i]))
    end
  end

  -- Add avatars as game objects (configs).
  for idx = 1, numPlayers do
    local gameObject = tables.deepCopy(prefabs["avatar"])
    local colorPalette = playerPalettes[idx]
    -- First, modify the prefab's sprite name.
    local spriteName = _getFirstNamedComponent(
        gameObject, "Appearance")["kwargs"]["spriteNames"][1]
    local newSpriteName = spriteName .. idx
    _getFirstNamedComponent(
        gameObject,
        "Appearance")["kwargs"]["spriteNames"][1] = newSpriteName
    -- Second, name the same sprite in the prefab's stateManager.
    local stateConfigs = _getFirstNamedComponent(
        gameObject,
        "StateManager")["kwargs"]["stateConfigs"]
    for _, stateConfig in pairs(stateConfigs) do
      if stateConfig["sprite"] == spriteName then
        stateConfig["sprite"] = newSpriteName
      end
    end
    -- Third, override the prefab's color palette for this sprite.
    _getFirstNamedComponent(
        gameObject, "Appearance")["kwargs"]["palettes"][1] = colorPalette
    -- Fourth, override the avatar's player id.
    _getFirstNamedComponent(
        gameObject, "Avatar")["kwargs"]["index"] = idx
    log.v(
      2, "Avatar " .. idx .. " configuration: " ..
      helpers.tostring(gameObject, '', 6))
    table.insert(gameObjects, gameObject)
  end
  return gameObjects
end

-- Build all avatar and normal game objects based on the config and map.
function buildGameObjectConfigs(asciiMap, prefabs, charPrefabMap)
  -- Procedural generation.
  local gameObjects = {}

  log.v(1, "charPrefabMap: " .. helpers.tostring(charPrefabMap))
  log.v(1, "Parsing map " .. asciiMap)
  _visitText(
    asciiMap,
    function(row, col, char)
      _processChar(row, col, char, charPrefabMap, prefabs, gameObjects)
    end)

  return gameObjects
end

return {
  buildGameObjectConfigs = buildGameObjectConfigs,
  buildAvatarConfigs = buildAvatarConfigs,
}

