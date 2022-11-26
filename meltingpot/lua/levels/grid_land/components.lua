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
local set = require 'common.set'
local tables = require 'common.tables'
local log = require 'common.log'
local events = require 'system.events'
local random = require 'system.random'
local tensor = require 'system.tensor'

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')


local Cell = class.Class(component.Component)

function Cell:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Cell')},
      {'numCellStates', args.numberType},
      {'statesToProperties', args.tableType},
      -- The default radius over which to search for neighbors on every step.
      {'radius', args.numberType},
      -- The default type of query (disc = L2, diamond = L1).
      {'queryType', args.default('disc'), args.oneOf('disc', 'diamond')},
      -- Layers on which to search for neighbors on every step.
      {'interactionLayers', args.tableType},
      -- `stateSpecificQueryConfig` optional, (maps `state` to
      -- {radius = (number), queryType = {'diamond', 'disc'}} to use to query
      -- for neighbors when in specific states.
      -- If `stateSpecificQueryConfig` is specified it overrides `radius`.
      {'stateSpecificQueryConfig', args.default({}), args.tableType},
  })
  self.Base.__init__(self, kwargs)

  self._config.radius = kwargs.radius
  self._config.queryType = kwargs.queryType
  self._config.interactionLayers = kwargs.interactionLayers

  local function _removeNonsense(inputTable)
    local result = {}
    for key, value in pairs(inputTable) do
      if type(value) == 'string' or type(value) == 'number' then
        result[key] = value
      end
    end
    return result
  end

  self._config.stateSpecificQueryConfig = {}
  for state, queryConfig in pairs(kwargs.stateSpecificQueryConfig) do
    self._config.stateSpecificQueryConfig[state] = _removeNonsense(queryConfig)
  end
end

function Cell:onHit(hitterObject, hitName)
  local currentState = self.gameObject:getState()
  if hitName == 'ioHit' then
    -- Ignore immovables.
    local groups = set.Set(self.gameObject:getGroups())
    if not groups['immovables'] then
      local hitterIO = hitterObject:getComponent('IOBeam')
      local vesicle = hitterIO:getVesicleObject():getComponent('AvatarVesicle')
      -- Vesicle must not be activated, cell old enough, and not blocked for IO.
      local blocked = self.gameObject:getComponent('Product'):isBlocked()
      if not vesicle:isBlocked() and not blocked and self:_oldEnough() then
        local next_vesicle_state = self.gameObject:getState()
        local next_ground_state = vesicle:pop()
        vesicle:add(next_vesicle_state)
        self.gameObject:setState(next_ground_state)
        vesicle:block()
        self.gameObject:getComponent('Reactant'):block()
      end
    end
  end
end

function Cell:_oldEnough()
  return self.gameObject:getComponent('Transform'):framesOld() > 1
end

function Cell:_getQueryConfig(state)
  if self._config.stateSpecificQueryConfig[state] then
    local config = self._config.stateSpecificQueryConfig[state]
    local radius = config['radius']
    local queryType = config['queryType']
    return radius, queryType
  end
  return self._config.radius, self._config.queryType
end

function Cell:_queryNeighbors()
  local radius, queryType = self:_getQueryConfig(self.gameObject:getState())
  local neighbors = {}
  for _, layer in ipairs(self._config.interactionLayers) do
    local neighborsThisLayer
    if queryType == 'disc' then
      neighborsThisLayer =
          self.gameObject:getComponent('Transform'):queryDisc(layer, radius)
    elseif queryType == 'diamond' then
      neighborsThisLayer =
          self.gameObject:getComponent('Transform'):queryDiamond(layer, radius)
    else
      assert(false, 'Unrecognized queryType = ' .. helpers.tostring(queryType))
    end
    for _, neighborObject in pairs(neighborsThisLayer) do
      table.insert(neighbors, neighborObject)
    end
  end
  return neighbors
end

function Cell:getNeighbors()
  return self:_queryNeighbors()
end


local ReactionAlgebra = class.Class(component.Component)

function ReactionAlgebra:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('ReactionAlgebra')},
      {'reactions', args.tableType},
  })
  self.Base.__init__(self, kwargs)

  -- Parsing inputs from python produces strange empty tables alongside
  -- the reactions. These must be removed before proceeding.
  local reactions = self:_cleanParsedReactions(kwargs.reactions)

  local function _getListFromStoichiometry(tableStateToNum)
    local list = {}
    for state, numRequired in pairs(tableStateToNum) do
      for i = 1, numRequired do
        table.insert(list, state)
      end
    end
    return list
  end

  local function _getStoichiometryFromList(list)
    local stateToNum = {}
    for _, state in ipairs(list) do
      local timesRepeatedSoFar = stateToNum[state] or 0
      stateToNum[state] = timesRepeatedSoFar + 1
    end
    return stateToNum
  end

  -- Check that all reactions are valid and convert them to a common format.
  self._config.reactions = {}
  for reactionIndex, reaction in ipairs(reactions) do
    self._config.reactions[reactionIndex] = {
        name = reaction.name,
        fixedSwapOrder = reaction.fixedSwapOrder,
        priority = reaction.priority
    }
    if reaction.fixedSwapOrder then
      -- reactants and products will be lists with (possibly) repeated entries.
      self:_checkConfigurationFixedOrder(reaction.reactants, reaction.products)
      self._config.reactions[reactionIndex].reactants = {
          list = reaction.reactants,
          stateToNumRequired = _getStoichiometryFromList(reaction.reactants)
      }
      self._config.reactions[reactionIndex].products = {
          list = reaction.products,
          stateToNumToProduce = _getStoichiometryFromList(reaction.products)
      }
    else
      -- reaction.reactants and reaction.products will be "stochiometries",
      -- i.e. tables mapping state to num molecules either required (in case of
      -- reactants) or to produce (in case of products).
      self:_checkConfiguration(reaction.reactants, reaction.products)
      self._config.reactions[reactionIndex].reactants = {
          list = _getListFromStoichiometry(reaction.reactants),
          stateToNumRequired = reaction.reactants
      }
      self._config.reactions[reactionIndex].products = {
          list = _getListFromStoichiometry(reaction.products),
          stateToNumToProduce = reaction.products
      }
    end
  end
end

function ReactionAlgebra:_cleanParsedReactions(reactions)
  local function _removeNonsenseFromDictMappingAnyToNum(input)
    local result = {}
    for key, value in pairs(input) do
      if type(value) == 'number' then
        result[key] = value
      end
    end
    return result
  end

  local function _convertEmptyToNil(input)
    if type(input) == 'table' and #input == 0 then
      return nil
    end
    return input
  end

  local cleanReactions = {}
  for reactionName, reaction in pairs(reactions) do
    local fixedSwapOrder = _convertEmptyToNil(reaction.fixedSwapOrder)
    local reactants = reaction.reactants
    local products = reaction.products
    if not fixedSwapOrder then
      reactants = _removeNonsenseFromDictMappingAnyToNum(reactants)
      products = _removeNonsenseFromDictMappingAnyToNum(products)
    end
    local reactionDict = {
        name = reactionName,
        fixedSwapOrder = fixedSwapOrder,
        reactants = reactants,
        products = products,
        priority = reaction.priority
    }
    table.insert(cleanReactions, reactionDict)
  end
  return cleanReactions
end

function ReactionAlgebra:_checkConfigurationFixedOrder(reactants, products)
  assert(#reactants == #products,
         'The number of reactants and products must be the same')
end

function ReactionAlgebra:_checkConfiguration(reactants, products)
  local numReactants = 0
  for state, numRequiredThisState in pairs(reactants) do
    numReactants = numReactants + numRequiredThisState
  end
  local numProducts = 0
  for state, numThisState in pairs(products) do
    numProducts = numProducts + numThisState
  end
  assert(numReactants == numProducts,
         'The number of reactants and products must be the same')
end

function ReactionAlgebra:getReactionByIndex(reactionIndex)
  return self._config.reactions[reactionIndex]
end

function ReactionAlgebra:getNumReactions()
  return #self._config.reactions
end

function ReactionAlgebra:getPrioritizedIndices() -- low priority comes first
  local indiciesByPriority = {}
  local maxPriority = 0
  for reactionIndex, reaction in ipairs(self._config.reactions) do
    assert(reaction.priority,
           "must assign priorities to all reactions. Missed: " .. reaction.name)
    assert(reaction.priority > 0,
           "priorities must be positive integers (they cannot be zero)")
    maxPriority = math.max(maxPriority, reaction.priority)
    if not indiciesByPriority[reaction.priority] then
      indiciesByPriority[reaction.priority] = {}
    end
    table.insert(indiciesByPriority[reaction.priority], reactionIndex)
  end
  local flatIndices = {}
  self.numReactionsPerPriority = {}
  -- Note: priorities must be positive integers (they cannot be zero).
  for priority = 1, maxPriority do
    local count = 0
    -- Skip priority bands to which no reactions were assigned.
    if indiciesByPriority[priority] then
      for _, reactionIndex in ipairs(indiciesByPriority[priority]) do
        table.insert(flatIndices, reactionIndex)
        count = count + 1
      end
    end
    self.numReactionsPerPriority[priority] = count
  end
  return flatIndices
end

function ReactionAlgebra:_flipArray(flatIndices)
  local result = {}
  for i = #flatIndices, 1, -1 do
    table.insert(result, flatIndices[i])
  end
  return result
end


local GlobalMetricTracker = class.Class(component.Component)

function GlobalMetricTracker:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('GlobalMetricTracker')},
  })
  self.Base.__init__(self, kwargs)
end

function GlobalMetricTracker:reset()
  self.numActualReactions = 0
end

function GlobalMetricTracker:reportReaction(reactionName)
  -- TODO(b/192927360): track more metrics here.
  self.numActualReactions = self.numActualReactions + 1
  -- Event: reaction: dict `name` (str).
  events:add('reaction', 'dict',
             'name', reactionName)
end


local Reactant = class.Class(component.Component)

function Reactant:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Reactant')},
      {'shuffleReactionCheckOrder', args.default(true), args.booleanType},
      {'reactivities', args.tableType},
      -- priorityMode: If `true` then check reactions according to their
      --   provided priority. If `shuffleReactionCheckOrder` is also `true` then
      --   shuffle within each priority level.
      {'priorityMode', args.default(false), args.booleanType},
  })
  self.Base.__init__(self, kwargs)
  self._config.shuffleReactionCheckOrder = kwargs.shuffleReactionCheckOrder
  self._config.priorityMode = kwargs.priorityMode

  self._config.reactivities = {}
  for reactionGroup, reactivity in pairs(kwargs.reactivities) do
    self._config.reactivities[reactionGroup] = reactivity
  end
end

function Reactant:registerUpdaters(updaterRegistry)
  -- Using the `group` kwarg here creates global initiation conditions for
  -- events. On each step, all objects in `group` have the given `probability`
  -- of being selected to call their `updateFn`.
  for reactivityGroup, rate in pairs(self._config.reactivities) do
    updaterRegistry:registerUpdater{
        updateFn = function() self:_tryReact() end,
        priority = 10,
        group = reactivityGroup,
        probability = rate,
        startFrame = 1
    }
  end
end

function Reactant:reset()
  self._blocked = false
end

function Reactant:postStart()
  local sceneObject = self.gameObject.simulation:getSceneObject()
  self._algebra = sceneObject:getComponent('ReactionAlgebra')
  self._numReactions = self._algebra:getNumReactions()
  if self._config.priorityMode then
    self._reactionIndices = self._algebra:getPrioritizedIndices()
  else
    self._reactionIndices = {}
    for i = 1, self._numReactions do
      table.insert(self._reactionIndices, i)
    end
  end
  self._metricTracker = sceneObject:getComponent('GlobalMetricTracker')
end

function Reactant:_syncReaction(reactionIndex)
  -- Get the reactants and products for the corresponding reaction index from
  -- the global reaction algebra component.
  local reaction = self._algebra:getReactionByIndex(reactionIndex)
  self._reactionName = reaction.name
  self._fixedSwapOrder = reaction.fixedSwapOrder
  self._reactants = reaction.reactants
  self._products = reaction.products
end

function Reactant:_trySpecificReaction(cells)
  -- Instantiate empty tables where we can add references to nearby cells.
  local nearbyReactantCells = {}
  for state, _ in pairs(self._reactants.stateToNumRequired) do
    nearbyReactantCells[state] = {}
  end

  -- Search for reactants in the inputted array of cells.
  for _, cell in ipairs(cells) do
    local state = cell:getState()
    local blocked = cell:getComponent('Product'):isBlocked() or
        cell:getComponent('Reactant'):isBlocked()
    if self._reactants.stateToNumRequired[state] and not blocked then
      table.insert(nearbyReactantCells[state], cell)
    end
  end

  -- Check if a reaction is possible.
  local reactionIsPossible = true
  for state, numRequired in pairs(self._reactants.stateToNumRequired) do
    if #nearbyReactantCells[state] < numRequired then
      reactionIsPossible = false
      break
    end
  end

  -- Make sure the self molecule is part of the reaction.
  if not nearbyReactantCells[self.gameObject:getState()] then
    reactionIsPossible = false
  end

  return reactionIsPossible, nearbyReactantCells
end

function Reactant:_tryReact(grid, piece)
  local cells = self.gameObject:getComponent('Cell'):getNeighbors()

  local reactionIsPossible, nearbyReactantCells
  for _, reactionIndex in ipairs(self._reactionIndices) do
    self:_syncReaction(reactionIndex)
    reactionIsPossible, nearbyReactantCells = self:_trySpecificReaction(cells)
    if reactionIsPossible then
      self._metricTracker:reportReaction(self._reactionName)
      break
    end
  end

  -- If it is possible to react then tell each cell to change to the "activated
  -- state".
  if reactionIsPossible then
    -- Generate the order that reactants will be replaced by products.
    local orderedProducts = self._products.list
    if not self._fixedSwapOrder then
      -- random.shuffle returns a shuffled copy of the original list.
      orderedProducts = random:shuffle(orderedProducts)
    end

    -- Activate each cell involved in the reaction.
    local reactantRepeats = {}
    for i, productState in ipairs(orderedProducts) do
      local pairedReactant = self._reactants.list[i]
      local cellsThisReactant = nearbyReactantCells[pairedReactant]
      local cellIndexToActivate = reactantRepeats[pairedReactant] or 1
      local cell = cellsThisReactant[cellIndexToActivate]
      cell:getComponent('Product'):activate(
          productState, self._reactionName, pairedReactant)
      reactantRepeats[pairedReactant] = cellIndexToActivate + 1
    end
  end
end

function Reactant:update()
  if self._config.shuffleReactionCheckOrder then
    if self._config.priorityMode then
      self:_priorityRespectingShuffle()
    else
      random:shuffleInPlace(self._reactionIndices)
    end
  end
  self._blocked = false
end

function Reactant:block()
  self._blocked = true
end

function Reactant:isBlocked()
  return self._blocked
end

function Reactant:_priorityRespectingShuffle()
  local oldIndices = tensor.Int64Tensor(self._reactionIndices)
  local indexTensorPerPriority = {}
  local start = 1
  -- Shuffle each priority band of the old index.
  for priority, numReactions in ipairs(self._algebra.numReactionsPerPriority) do
    indexTensorPerPriority[priority] = oldIndices:narrow(
      1, start, numReactions):clone():shuffle(random)
    start = start + numReactions
  end
  -- Form the new index table.
  local newIndices = {}
  for _, indexTensor in ipairs(indexTensorPerPriority) do
    for idx = 1, indexTensor:size() do
      table.insert(newIndices, indexTensor(idx):val())
    end
  end
  -- Discard the old reaction indices table, replacing it with the new table.
  self._reactionIndices = newIndices
end


local Product = class.Class(component.Component)

function Product:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Product')},
  })
  self.Base.__init__(self, kwargs)

  self._config.activatedState = 'activated'
end

function Product:reset()
  self._nextState = nil
  self._transitionedThisStep = false

  self._latestReaction = nil
  self._latestReplacedReactant = nil
end

function Product:update()
  self._transitionedThisStep = false
  local state = self.gameObject:getState()
  if state == self._config.activatedState then
    self.gameObject:setState(self._nextState)
    self._transitionedThisStep = true
  end
  self._nextState = nil
end

--[[ Activate the corresponding cell so it will move first to `activatedState`
on the next step and then to `nextState` on the step after that.

Arguments:

*   `nextState` (str): the product compound that will replace a reactant in the
    focal cell.
*   `reactionName` (str): which reaction the focal cell is participating in.
    Note that reaction name is only used for analysis and for providing reaction
    specific rewards, it is not used by the underlying system mechanics.
*   `reactantToReplace` (str): which specific reactant compound will be replaced
    by a product in the focal cell. Note that `reactantToReplace` is only used
    for analysis, it is not needed for the underlying system mechanics.
]]
function Product:activate(nextState, reactionName, reactantToReplace)
  self.gameObject:setState(self._config.activatedState)
  self._nextState = nextState

  self._latestReaction = reactionName
  self._latestReplacedReactant = reactantToReplace
end

function Product:getLatestReaction()
  return self._latestReaction
end

function Product:getLatestReplacedReactant()
  return self._latestReplacedReactant
end

function Product:didTransition()
  return self._transitionedThisStep
end

function Product:isBlocked()
  local state = self.gameObject:getState()
  local activated = self._config.activatedState
  return self._nextState ~= nil or state == activated
end


local IOBeam = class.Class(component.Component)

function IOBeam:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('IOBeam')},
      {'cooldownTime', args.numberType},
  })
  self.Base.__init__(self, kwargs)

  self._config.cooldownTime = kwargs.cooldownTime
  self._config.beamLength = 1
  self._config.beamRadius = 0
end

function IOBeam:reset()
  self._coolingTimer = 0
  self._IOAllowed = true
end

function IOBeam:registerUpdaters(updaterRegistry)
  local toggle = function()
    local playerVolatileVariables = (
        self.gameObject:getComponent('Avatar'):getVolatileData())
    local actions = playerVolatileVariables.actions
    -- Execute the beam if applicable.
    if self._config.cooldownTime >= 0 then
      if self._coolingTimer > 0 then
        self._coolingTimer = self._coolingTimer - 1
      else
        if actions['ioAction'] == 1 and self._IOAllowed then
          self._coolingTimer = self._config.cooldownTime

          local cellBelowCurrentLocation = self:getCellComponentUnderneath()
          cellBelowCurrentLocation:onHit(self.gameObject, 'ioHit')
        end
      end
    end
    self._IOAllowed = true
  end

  updaterRegistry:registerUpdater{
      updateFn = toggle,
      priority = 7,
  }
end

function IOBeam:onStateChange()
  self._respawnTimer = self._config.framesTillRespawn
end

function IOBeam:start()
  -- Set the beam cooldown timer to its `ready` state (i.e. coolingTimer = 0).
  self._coolingTimer = 0
  self._avatarComponent = self.gameObject:getComponent('Avatar')
end

function IOBeam:getVesicleObject()
  -- Assume there will only be one connected object with an AvatarVesicle.
  return self._avatarComponent:getAllConnectedObjectsWithNamedComponent(
    'AvatarVesicle')[1]
end

function IOBeam:getCellComponentUnderneath()
  local transformComponent = self.gameObject:getComponent('Transform')
  -- By default, query position checks the position of the current game object.
  local cellObjectBelow = transformComponent:queryPosition("lowerPhysical")
  return cellObjectBelow:getComponent('Cell')
end

function IOBeam:disallowIO()
  self._IOAllowed = false
end


local AvatarVesicle = class.Class(component.Component)

function AvatarVesicle:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AvatarVesicle')},
      -- `playerIndex` (int): player index for the avatar to connect to.
      {'playerIndex', args.numberType},
      {'preInitState', args.stringType},
      {'initialState', args.stringType},
      {'waitState', args.stringType},
  })
  self.Base.__init__(self, kwargs)
  self._kwargs = kwargs
end

function AvatarVesicle:reset()
  local kwargs = self._kwargs
  self._playerIndex = kwargs.playerIndex
  self._preInitState = kwargs.preInitState
  self._initialState = kwargs.initialState
  self._waitState = kwargs.waitState
  self._blocked = false
  -- The empty state is also the initial state.
  self._emptyState = self._initialState
end

--[[ Note that postStart is called from the avatar manager, after start has been
called on all other game objects, even avatars.]]
function AvatarVesicle:postStart()
  local sim = self.gameObject.simulation
  self._avatarObject = sim:getAvatarFromIndex(self._playerIndex)

  -- Note that it is essential to set the state before teleporting it.
  -- This is because pieces with no assigned layer have no position, and thus
  -- cannot be teleported.
  self.gameObject:setState(self._initialState)
  self.gameObject:teleport(self._avatarObject:getPosition(),
                           self._avatarObject:getOrientation())

  -- Get the avatar component from the game object and connect to it.
  local avatarComponent = self._avatarObject:getComponent('Avatar')
  avatarComponent:connect(self.gameObject)
end

function AvatarVesicle:update()
  local avatarComponent = self._avatarObject:getComponent('Avatar')

  -- Provide rewards based on vesicle contents.
  local productComponent = self.gameObject:getComponent('Product')
  if productComponent:didTransition() then
    local reactionName = productComponent:getLatestReaction()
    local rewardsComponent = self._avatarObject:getComponent(
      'ReactionsToRewards')
    local rewardingReactions = rewardsComponent:getRewardingReactions()
    if rewardingReactions[reactionName] then
      local rewardValue = rewardsComponent:getRewardValue(reactionName)
      avatarComponent:addReward(rewardValue)
    end
    -- Report an event whenever a reaction occurs involving an avatar vesicle.
    local replacedReactant = productComponent:getLatestReplacedReactant()
    events:add('vesicle_reaction', 'dict',
               'player_index', avatarComponent:getIndex(), -- int
               -- Use vesicle_name to disambiguate if we have multiple vesicles.
               'vesicle_name', self.gameObject.name, -- str
               'reaction_name', reactionName, -- str
               'reactant_compound', replacedReactant, -- str
               'product_compound', self.gameObject:getState()) -- str
  end

  -- Prevent avatar movement while still allowing IOBeam actions whenever an
  -- immovable molecule is within the vesicle.
  avatarComponent:allowMovement()
  local groups = set.Set(self.gameObject:getGroups())
  if groups['immovables'] then
    avatarComponent:disallowMovement()
    local ioComponent = self._avatarObject:getComponent('IOBeam')
    ioComponent:disallowIO()
  end

  self._blocked = false
end

function AvatarVesicle:isEmpty()
  return self.gameObject:getState() == self._emptyState
end

function AvatarVesicle:block()
  self._blocked = true
end

function AvatarVesicle:isBlocked()
  return self.gameObject:getComponent('Product'):isBlocked() and
      not self._blocked
end

function AvatarVesicle:add(state)
  self.gameObject:setState(state)
end

function AvatarVesicle:pop()
  local output = self.gameObject:getState()
  return output
end


local ReactionsToRewards = class.Class(component.Component)

function ReactionsToRewards:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('ReactionsToRewards')},
      -- rewardingReactions should be a table mapping reactionName to reward.
      {'rewardingReactions', args.tableType},
  })
  self.Base.__init__(self, kwargs)
  self._kwargs = kwargs
end

function ReactionsToRewards:reset()
  local kwargs = self._kwargs
  self._rewardingReactions = self:_removeEmptyTables(kwargs.rewardingReactions)
end

function ReactionsToRewards:_removeEmptyTables(input)
  local result = {}
  for key, value in pairs(input) do
    if type(value) == 'number' then
      result[key] = value
    end
  end
  return result
end

function ReactionsToRewards:getRewardingReactions()
  return self._rewardingReactions
end

function ReactionsToRewards:getRewardValue(reactionName)
  return self._rewardingReactions[reactionName]
end


local VesicleManager = class.Class(component.Component)

function VesicleManager:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('VesicleManager')},
      {'orderedVesicles', args.tableType},
      {'cytoavatarStates', args.tableType},
  })
  self.Base.__init__(self, kwargs)
  self._orderedVesicles = kwargs.orderedVesicles
  self._cytoavatarStates = kwargs.cytoavatarStates
end

function VesicleManager:reset()
  self._numOccupied = 0
  self._avatarComponent = self.gameObject:getComponent('Avatar')
  self._started = false
end

function VesicleManager:push()
  self._numOccupied = self._numOccupied + 1
end

function VesicleManager:pop()
  self._numOccupied = self._numOccupied - 1
end

function VesicleManager:setInitialCompounds()
  local allVesicles =
      self._avatarComponent:getAllConnectedObjectsWithNamedComponent(
          'AvatarVesicle')
   for _, vesicle in ipairs(allVesicles) do
     vesicle:setState('empty')
   end
end

function VesicleManager:update()
  if not self._started then
    self:setInitialCompounds()
  end
  self._started = true

  -- First update the count of the number of occupied vesicles.
  self._numOccupied = 0
  local allVesicles =
      self._avatarComponent:getAllConnectedObjectsWithNamedComponent(
          'AvatarVesicle')
  for _, vesicle in ipairs(allVesicles) do
    local vesicleComponent = vesicle:getComponent('AvatarVesicle')
    if not vesicleComponent:isEmpty() then
      -- Add 1 to numOccupied if vesicle is nonempty.
      self._numOccupied = self._numOccupied + 1
    end
  end
  -- Set state of the avatar object accordingly to its num occupied vesicles.
  if self._numOccupied == 0 then
    self.gameObject:setState(self._cytoavatarStates['empty'])
  elseif self._numOccupied == 1 then
    self.gameObject:setState(self._cytoavatarStates['holdingOne'])
  else
    assert(False, 'Nonsensical numOccupied: ' .. tostring(self._numOccupied))
  end
end


local allComponents = {
    -- Grid cell components.
    Cell = Cell,
    Reactant = Reactant,
    Product = Product,
    -- Avatar components.
    IOBeam = IOBeam,
    AvatarVesicle = AvatarVesicle,
    ReactionsToRewards = ReactionsToRewards,
    VesicleManager = VesicleManager,
    -- Global components,
    ReactionAlgebra = ReactionAlgebra,
    GlobalMetricTracker = GlobalMetricTracker,
}

component_registry.registerAllComponents(allComponents)

return allComponents
