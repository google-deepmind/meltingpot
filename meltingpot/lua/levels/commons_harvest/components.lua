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

local args = require 'common.args'
local class = require 'common.class'
local helpers = require 'common.helpers'
local log = require 'common.log'
local random = require 'system.random'
local tensor = require 'system.tensor'

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')


local function concat(table1, table2)
  local resultTable = {}
  for k, v in pairs(table1) do
    table.insert(resultTable, v)
  end
  for k, v in pairs(table2) do
    table.insert(resultTable, v)
  end
  return resultTable
end

local function extractPieceIdsFromObjects(gameObjects)
  local result = {}
  for k, v in ipairs(gameObjects) do
    table.insert(result, v:getPiece())
  end
  return result
end


local Neighborhoods = class.Class(component.Component)

function Neighborhoods:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Neighborhoods')},
  })
  Neighborhoods.Base.__init__(self, kwargs)
end

function Neighborhoods:reset()
  self._variables.pieceToNumNeighbors = {}
end

function Neighborhoods:getPieceToNumNeighbors()
  -- Note: this table is frequently modified by callbacks.
  return self._variables.pieceToNumNeighbors
end

function Neighborhoods:getUpperBoundPossibleNeighbors()
  return self._config.upperBoundPossibleNeighbors
end


local DensityRegrow = class.Class(component.Component)

function DensityRegrow:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('DensityRegrow')},
      {'liveState', args.stringType},
      {'waitState', args.stringType},
      {'radius', args.numberType},
      {'regrowthProbabilities', args.tableType},
      {'canRegrowIfOccupied', args.default(true)},
  })
  DensityRegrow.Base.__init__(self, kwargs)

  self._config.liveState = kwargs.liveState
  self._config.waitState = kwargs.waitState

  self._config.radius = kwargs.radius
  self._config.regrowthProbabilities = kwargs.regrowthProbabilities

  if self._config.radius >= 0 then
    self._config.upperBoundPossibleNeighbors = math.floor(
        math.pi * self._config.radius ^ 2 + 1) + 1
  else
    self._config.upperBoundPossibleNeighbors = 0
  end
  self._config.canRegrowIfOccupied = kwargs.canRegrowIfOccupied

  self._started = false
end

function DensityRegrow:reset()
  self._started = false
end

function DensityRegrow:registerUpdaters(updaterRegistry)
  local function sprout()
    if self._config.canRegrowIfOccupied then
      self.gameObject:setState(self._config.liveState)
    else
      -- Only setState if no player is at the same position.
      local transform = self.gameObject:getComponent('Transform')
      local players = transform:queryDiamond('upperPhysical', 0)
      if #players == 0 then
        self.gameObject:setState(self._config.liveState)
      end
    end
  end
  -- Add an updater for each `wait` regrowth rate category.
  for numNear = 0, self._config.upperBoundPossibleNeighbors - 1 do
    -- Cannot directly index the table with numNear since Lua is 1-indexed.
    local idx = numNear + 1
    -- If more nearby than probabilities declared then use the last declared
    -- probability in the table (normally the high probability).
    local idx = math.min(idx, #self._config.regrowthProbabilities)
    -- Using the `group` kwarg here creates global initiation conditions for
    -- events. On each step, all objects in `group` have the given `probability`
    -- of being selected to call their `updateFn`.
    -- Set updater for each neighborhood category.
    updaterRegistry:registerUpdater{
        updateFn = sprout,
        priority = 10,
        group = 'waits_' .. tostring(numNear),
        state = 'appleWait_' .. tostring(numNear),
        probability = self._config.regrowthProbabilities[idx],
    }
  end
end

function DensityRegrow:start()
  local sceneObject = self.gameObject.simulation:getSceneObject()
  local neighborhoods = sceneObject:getComponent('Neighborhoods')
  self._variables.pieceToNumNeighbors = neighborhoods:getPieceToNumNeighbors()
  self._variables.pieceToNumNeighbors[self.gameObject:getPiece()] = 0
end

function DensityRegrow:postStart()
  self:_beginLive()
  self._started = true
  self._underlyingGrass = self.gameObject:getComponent(
      'Transform'):queryPosition('background')
end

function DensityRegrow:update()
  if self.gameObject:getLayer() == 'logic' then
    self:_updateWaitState()
  end
end

function DensityRegrow:onStateChange(oldState)
  if self._started then
    local newState = self.gameObject:getState()
    local aliveState = self:getAliveState()
    if newState == aliveState then
      self:_beginLive()
    elseif oldState == aliveState then
      self:_endLive()
    end
  end
end

function DensityRegrow:getAliveState()
  return self._config.liveState
end

function DensityRegrow:getWaitState()
  return self._config.waitState
end

--[[ This function updates the state of a potential (wait) apple to correspond
to the correct regrowth probability for its number of neighbors.]]
function DensityRegrow:_updateWaitState()
  if self.gameObject:getState() ~= self._config.liveState then
    local piece = self.gameObject:getPiece()
    local numClose = self._variables.pieceToNumNeighbors[piece]
    local newState = self._config.waitState .. '_' .. tostring(numClose)
    self.gameObject:setState(newState)
    if newState == self._config.waitState .. '_' .. tostring(0) then
      self._underlyingGrass:setState('dessicated')
    else
      self._underlyingGrass:setState('grass')
    end
  end
end

function DensityRegrow:_getNeighbors()
  local transformComponent = self.gameObject:getComponent('Transform')
  local waitNeighbors = extractPieceIdsFromObjects(
      transformComponent:queryDisc('logic', self._config.radius))
  local liveNeighbors = extractPieceIdsFromObjects(
      transformComponent:queryDisc('lowerPhysical', self._config.radius))
  local neighbors = concat(waitNeighbors, liveNeighbors)
  return neighbors, liveNeighbors, waitNeighbors
end

--[[ Function that executes when state gets set to the `live` state.]]
function DensityRegrow:_beginLive()
  -- Increment respawn group assignment for all nearby waits.
  local neighbors, liveNeighbors, waitNeighbors = self:_getNeighbors()
  for _, neighborPiece in ipairs(waitNeighbors) do
    if neighborPiece ~= self.gameObject:getPiece() then
      local closeBy = self._variables.pieceToNumNeighbors[neighborPiece]
      if not closeBy then
        assert(false, 'Neighbors not found when they should exist.')
      end
      self._variables.pieceToNumNeighbors[neighborPiece] =
          self._variables.pieceToNumNeighbors[neighborPiece] + 1
    end
  end
end

--[[ Function that executes when state changed to no longer be `live`.]]
function DensityRegrow:_endLive()
  -- Decrement respawn group assignment for all nearby waits.
  local neighbors, liveNeighbors, waitNeighbors = self:_getNeighbors()
  for _, neighborPiece in ipairs(waitNeighbors) do
    if neighborPiece ~= self.gameObject:getPiece() then
      local closeBy = self._variables.pieceToNumNeighbors[neighborPiece]
      if not closeBy then
        assert(false, 'Neighbors not found when they should exist.')
      end
      self._variables.pieceToNumNeighbors[neighborPiece] =
          self._variables.pieceToNumNeighbors[neighborPiece] - 1
    else
      -- Case where neighbor piece is self.
      self._variables.pieceToNumNeighbors[neighborPiece] = #liveNeighbors
    end
    assert(self._variables.pieceToNumNeighbors[neighborPiece] >= 0,
             'Less than zero neighbors: Something has gone wrong.')
  end
end

local allComponents = {
    Neighborhoods = Neighborhoods,
    DensityRegrow = DensityRegrow,
}

component_registry.registerAllComponents(allComponents)

return allComponents
