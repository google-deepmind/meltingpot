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
local set = require 'common.set'
local events = require 'system.events'

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')

local _COMPASS = {'N', 'E', 'S', 'W'}
local _OPPOSITECOMPASS = {N = 'S', E = 'W', S = 'N', W = 'E'}

local function range(length)
  local result = {}
  for i = 1, length do
    table.insert(result, i)
  end
  return result
end


--[[ `FruitType` makes a Harvestable (i.e. a tree) probabilistically yield
either apples or bananas.
]]
local FruitType = class.Class(component.Component)

function FruitType:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('FruitType')},
      {'probabilities', args.tableType},
  })
  FruitType.Base.__init__(self, kwargs)

  self._config.probabilities = kwargs.probabilities
  -- Check that tree spawn probabilities sum to one.
  self:checkFruitTypeProbabilitiesSumToOne()
end

function FruitType:checkFruitTypeProbabilitiesSumToOne()
  local sum = 0
  for _, probability in pairs(self._config.probabilities) do
    sum = sum + probability
  end
  if sum < 1.0 or sum > 1.0 then
    assert(false, 'Fruit tree spawn probabilities must sum to one.')
  end
end

function FruitType:spawn()
  local cuts = {}
  local order = {}
  local cumulativeValue = 0.0
  for key, probability in pairs(self._config.probabilities) do
    table.insert(order, key)
    cumulativeValue = cumulativeValue + probability
    table.insert(cuts, cumulativeValue)
  end

  local rnd = random:uniformReal(0, 1)

  local lowerBound = 0.0
  local upperBound
  for itemIdx, itemKey in ipairs(order) do
    upperBound = cuts[itemIdx]
    if itemKey ~= 'empty' and rnd > lowerBound and rnd < upperBound then
      self.gameObject:setState(itemKey .. 'TreeHarvestable')
      self._fruit = itemKey
    end
    lowerBound = upperBound
  end
end

function FruitType:postStart()
  self:spawn()
  self._ripe = true
end

function FruitType:isRipe()
  return self._ripe
end

function FruitType:setRipe(isRipe)
  -- pass a boolean for `isRipe`.
  self._ripe = isRipe
  -- change to the correct state.
  local harvestableStateName = self._fruit .. 'TreeHarvestable'
  local unripeStateName = self._fruit .. 'TreeUnripe'
  if self._ripe then
    self.gameObject:setState(harvestableStateName)
  else
    self.gameObject:setState(unripeStateName)
  end
end

function FruitType:getFruit()
  return self._fruit
end


--[[ `Harvestable` makes it possible to collect fruit from a tree.
]]
local Harvestable = class.Class(component.Component)

function Harvestable:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Harvestable')},
      -- `regrowthTime` determines how long to wait after a successful harvest
      -- before a tree can be harvested again.
      {'regrowthTime', args.ge(1)},
  })
  Harvestable.Base.__init__(self, kwargs)
  self._config.regrowthTime = kwargs.regrowthTime
end

function Harvestable:reset()
  self._regrowthCounter = self._config.regrowthTime
  -- When an avatar moves onto the location of this Harvestable then point to it
  -- with this variable. Set it back to nil when the avatar successfully
  -- harvests or moves away.
  self._harvestingAvatar = nil
end

function Harvestable:_harvest(harvester)
  local fruitTypeComponent = self.gameObject:getComponent('FruitType')
  -- Set the harvestable object to its unripe state.
  fruitTypeComponent:setRipe(false)
  self._regrowthCounter = self._config.regrowthTime

  -- Add to the harvesting avatar's inventory.
  local inventory = harvester:getComponent('Inventory')
  local specialization = harvester:getComponent('Specialization')
  local fruit = fruitTypeComponent:getFruit()
  local amount = specialization:getHarvestAmount(fruit)
  inventory:add(fruit, amount)

  -- Avatar is no longer trying to harvest since it has already succeeded.
  self._harvestingAvatar = nil
end

function Harvestable:_maybeHarvest(harvester)
  local specialization = harvester:getComponent('Specialization')
  local fruitType = self.gameObject:getComponent('FruitType')
  local fruit = fruitType:getFruit()

  local harvestProbability = specialization:getHarvestProbability(fruit)
  local rnd = random:uniformReal(0, 1)
  if rnd < harvestProbability then
    self:_harvest(harvester)
  end
end

function Harvestable:registerUpdaters(updaterRegistry)
  local resolveHarvesting = function()
    -- Maybe harvest if avatar standing on tree.
    if self._harvestingAvatar then
      self:_maybeHarvest(self._harvestingAvatar)
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = resolveHarvesting,
      priority = 2,  -- Ensures harvesting will execute after avatar movement.
  }
end

function Harvestable:onEnter(enteringObject, contactName)
  local fruitTypeComponent = self.gameObject:getComponent('FruitType')
  local isRipe = fruitTypeComponent:isRipe()
  if contactName == 'avatar' and isRipe then
    self._harvestingAvatar = enteringObject
  end
end

function Harvestable:onExit(exitingObject, contactName)
  if contactName == 'avatar' then
    self._harvestingAvatar = nil
  end
end

function Harvestable:update()
  -- Update ripeness.
  local fruitTypeComponent = self.gameObject:getComponent('FruitType')
  local isRipe = fruitTypeComponent:isRipe()
  if not isRipe then
    self._regrowthCounter = self._regrowthCounter - 1
    if self._regrowthCounter < 1 then
      fruitTypeComponent:setRipe(true)
    end
  end
end


--[[ Prevent stamina recovery while at this location. Used to prevent recovery
while harvesting.]]
local PreventStaminaRecoveryHere = class.Class(component.Component)

function PreventStaminaRecoveryHere:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('PreventStaminaRecoveryHere')},
  })
  PreventStaminaRecoveryHere.Base.__init__(self, kwargs)
end

function PreventStaminaRecoveryHere:onEnter(enteringObject, contactName)
  if contactName == 'avatar' then
    if enteringObject:hasComponent('Stamina') then
      local stamina = enteringObject:getComponent('Stamina')
      stamina:startPreventingRecovery()
    end
  end
end

function PreventStaminaRecoveryHere:onExit(exitingGameObject, contactName)
  if contactName == 'avatar' then
    if exitingGameObject:hasComponent('Stamina') then
      local stamina = exitingGameObject:getComponent('Stamina')
      stamina:stopPreventingRecovery()
    end
  end
end


--[[ `TraversalCost` punishes agents who enter this object's location.
]]
local TraversalCost = class.Class(component.Component)

function TraversalCost:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('TraversalCost')},
      {'penaltyAmount', args.default(0.0), args.ge(0)},  -- The reward penalty.
      {'alsoReduceStamina', args.default(false), args.booleanType},
      -- `staminaPenaltyAmount` is the stamina penalty if applicable.
      {'staminaPenaltyAmount', args.default(0.0), args.ge(0)},
      {'avatarLayer', args.default('upperPhysical'), args.stringType},
  })
  TraversalCost.Base.__init__(self, kwargs)
  self._config.penaltyAmount = kwargs.penaltyAmount
  self._config.alsoReduceStamina = kwargs.alsoReduceStamina
  self._config.staminaPenaltyAmount = kwargs.staminaPenaltyAmount
  self._config.avatarLayer = kwargs.avatarLayer
end

function TraversalCost:applyCost(contactingObject)
  contactingObject:getComponent('Avatar'):addReward(-self._config.penaltyAmount)

  if self._config.alsoReduceStamina then
    local staminaComponent = contactingObject:getComponent('Stamina')
    staminaComponent:addValue(-self._config.staminaPenaltyAmount)
  end
end

function TraversalCost:registerUpdaters(updaterRegistry)
  local transform = self.gameObject:getComponent('Transform')
  local function detectAvatarAndApplyCostIfPresent()
    local contactingObject = transform:queryPosition(self._config.avatarLayer)
    -- Transform.queryPosition returns nil when no object is found so we check
    -- both that an object was found and that it has the avatar component.
    if contactingObject and contactingObject:hasComponent('Avatar') then
      self:applyCost(contactingObject)
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = detectAvatarAndApplyCostIfPresent,
      priority = 3,
  }
end

--[[ `Inventory` keeps track of how many objects each avatar is carrying. It
assumes that agents can carry infinite quantities so this is a kind of inventory
that cannot ever be full.
]]
local Inventory = class.Class(component.Component)

function Inventory:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Inventory')},
      {'itemTypes', args.tableType, args.default({'apple', 'banana'})},
  })
  Inventory.Base.__init__(self, kwargs)
  self._config.itemTypes = kwargs.itemTypes
end

function Inventory:reset()
  self._inventory = {}
  for _, itemType in ipairs(self._config.itemTypes) do
    self._inventory[itemType] = 0
  end
  -- Allocate memory for the tensor representation of the inventory.
  self._tensorInventory = tensor.Int64Tensor(#self._config.itemTypes):fill(0)
end

function Inventory:_add(itemType, number)
  self._inventory[itemType] = self._inventory[itemType] + number
end

function Inventory:_remove(itemType, number)
  if self._inventory[itemType] - number >= 0 then
    self._inventory[itemType] = self._inventory[itemType] - number
  else
    local message = (itemType .. ': Tried to remove ' .. tostring(number) ..
                     ' but inventory contained only ' ..
                     tostring(self._inventory[itemType]))
    assert(false, message)
  end
end

function Inventory:add(itemType, number)
  if number >= 0 then
    self:_add(itemType, number)
  else
    self:_remove(itemType, -number)
  end
end

function Inventory:quantity(itemType)
  return self._inventory[itemType]
end

function Inventory:getInventoryAsTensor(order)
  -- `order` (array of fruit strings). Set order of items in output tensor. For
  -- instance {'apple', 'banana'} --> tensor.Tensor{numApples, numBananas}.
  for idx, fruit in ipairs(order) do
    -- Set index `idx` to value `self._inventory[fruit]`.
    self._tensorInventory(idx):val(self._inventory[fruit])
  end
  return self._tensorInventory
end


--[[ `Eating` endows avatars with the ability to eat items from their inventory
and thereby update a `periodicNeed`.
]]
local Eating = class.Class(component.Component)

function Eating:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Eating')},
      {'edibleKinds', args.default({'apple', 'banana'}), args.tableType},
  })
  Eating.Base.__init__(self, kwargs)
  self._edibleKinds = kwargs.edibleKinds
end

function Eating:registerUpdaters(updaterRegistry)
  local inventory = self.gameObject:getComponent('Inventory')
  local taste = self.gameObject:getComponent('Taste')
  local avatar = self.gameObject:getComponent('Avatar')
  local periodicNeed = self.gameObject:getComponent('PeriodicNeed')
  local eat = function()
    local playerVolatileVariables = avatar:getVolatileData()
    local actions = playerVolatileVariables.actions
    for _, fruit in ipairs(self._edibleKinds) do
      if actions['eat_' .. fruit] == 1 and inventory:quantity(fruit) >= 1 then
        inventory:add(fruit, -1)
        local rewardAmount = taste:getRewardAmount(fruit)
        avatar:addReward(rewardAmount)
        periodicNeed:resetDriveLevel()
      end
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = eat,
      priority = 200,
  }
end


--[[ `Specialization` controls the probability with which an avatar can harvest
a `Harvestable` (tree) per step they stand on it. It also controls the number
of items they obtain when they do harvest successfully.
]]
local Specialization = class.Class(component.Component)

function Specialization:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Specialization')},
      {'specialty', args.oneOf('apple', 'banana')},
      {'strongAmount', args.numberType, args.default(1)},
      {'weakAmount', args.numberType, args.default(1)},
      {'strongProbability', args.numberType, args.default(1.0)},
      {'weakProbability', args.numberType, args.default(1.0)},
  })
  Specialization.Base.__init__(self, kwargs)

  self._config.specialty = kwargs.specialty
  self._config.strongAmount = kwargs.strongAmount
  self._config.weakAmount = kwargs.weakAmount
  self._config.strongProbability = kwargs.strongProbability
  self._config.weakProbability = kwargs.weakProbability
end

function Specialization:getHarvestAmount(fruit)
  if fruit == self._config.specialty then
    return self._config.strongAmount
  end
  return self._config.weakAmount
end

function Specialization:getHarvestProbability(fruit)
  if fruit == self._config.specialty then
    return self._config.strongProbability
  end
  return self._config.weakProbability
end

function Specialization:getSpecialty()
  return self._config.specialty
end


--[[ `Taste` determines how much reward an avatar gets from eating a given
type of item (such as an apple or banana).
]]
local Taste = class.Class(component.Component)

function Taste:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Taste')},
      {'mostTastyFruit', args.oneOf('apple', 'banana')},
      {'mostTastyReward', args.numberType},
      {'defaultReward', args.numberType},
  })
  Taste.Base.__init__(self, kwargs)

  self._config.mostTastyFruit = kwargs.mostTastyFruit
  self._config.mostTastyReward = kwargs.mostTastyReward
  self._config.defaultReward = kwargs.defaultReward
end

function Taste:getRewardAmount(fruit)
  if fruit == self._config.mostTastyFruit then
    return self._config.mostTastyReward
  end
  return self._config.defaultReward
end


--[[ Avatars bearing the `MovementPenalty` component pay a penalty every time
they select one of their `costlyActions`.
]]
local MovementPenalty = class.Class(component.Component)

function MovementPenalty:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('MovementPenalty')},
      {'costlyActions', args.tableType},
      {'penaltyAmount', args.ge(0)},
  })
  MovementPenalty.Base.__init__(self, kwargs)

  self._config.costlyActions = kwargs.costlyActions
  self._config.penaltyAmount = kwargs.penaltyAmount
end

function MovementPenalty:registerUpdaters(updaterRegistry)
  local avatar = self.gameObject:getComponent('Avatar')
  local applyActionCost = function()
    local playerVolatileVariables = (
        self.gameObject:getComponent('Avatar'):getVolatileData())
    local actions = playerVolatileVariables.actions
    for _, costly_action_name in pairs(self._config.costlyActions) do
      if actions[costly_action_name] ~= 0 then
        avatar:addReward(-self._config.penaltyAmount)
      end
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = applyActionCost,
      priority = 5,
  }
end


--[[ The `Trading` component implements all the logic of offering and resolving
trades of items.
]]
local Trading = class.Class(component.Component)

function Trading:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Trading')},
      {'maxOfferQuantity', args.positive},
      {'radius', args.positive},
      {'itemTypes', args.tableType, args.default({'apple', 'banana'})},
  })
  Trading.Base.__init__(self, kwargs)

  self._config.maxOfferQuantity = kwargs.maxOfferQuantity
  self._config.radius = kwargs.radius
  self._config.itemTypes = kwargs.itemTypes
end

function Trading:_getEmptyOffer()
  local offer = {}
  for _, itemType in ipairs(self._config.itemTypes) do
    offer[itemType] = 0
  end
  return offer
end

function Trading:_resetOffer()
  self._offer = self:_getEmptyOffer()
end

function Trading:reset()
  self:_resetOffer()
end

function Trading:registerUpdaters(updaterRegistry)
  local function offer()
    -- Listen for the action and set the appropriate self._offer.
    local playerVolatileVariables = (
        self.gameObject:getComponent('Avatar'):getVolatileData())
    local actions = playerVolatileVariables.actions
    if self.gameObject:getComponent('Avatar'):isAlive() then
      for _, itemType in ipairs(self._config.itemTypes) do
        local offerActionString = 'offer_' .. itemType
        if actions[offerActionString] ~= 0 then
          self._offer[itemType] = actions[offerActionString]
        end
      end
      if actions['offer_cancel'] == 1 then
        self:_resetOffer()
      end
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = offer,
      priority = 250,
  }
end

function Trading:hasEnough(offer)
  -- Check that you have at least as many as you are offering to give.
  for fruitType, quantity in pairs(offer) do
    if quantity < 0 then
      local inventory = self.gameObject:getComponent('Inventory')
      if quantity + inventory:quantity(fruitType) < 0 then
        return false
      end
    end
  end
  return true
end

function Trading:isValid(offer)
  local take, give
  for fruitType, quantity in pairs(offer) do
    if quantity > 0 then
      take = true
    end
    if quantity < 0 then
      give = true
    end
  end
  if take and give then
    return true
  else
    return false
  end
end

function Trading:isCompatible(nearbyOffer)
  -- For everything I want, check they are offering it.
  for fruitType, quantityIWant in pairs(self._offer) do
    if quantityIWant > 0 and quantityIWant + nearbyOffer[fruitType] > 0 then
      return false
    end
  end
  return true
end

function Trading:assertValidTrade(offerPackage, theirPackage)
  -- TODO(b/260153645): check that offers are compatible.
  local myInventory = offerPackage.avatarObject:getComponent('Inventory')
  local theirInventory = theirPackage.avatarObject:getComponent('Inventory')

  -- Check both players have enough items to trade.
  local validTradeForMe, validTradeForThem
  -- The following check for having enough fruit to trade should never come out
  -- false. This is because we already checked for that possibility earlier in
  -- the process. It's important to check that early because otherwise agents
  -- could strategically mess up one another's trades by advertising more than
  -- they can deliver.
  for _, fruitType in ipairs(self._config.itemTypes) do
    -- My inventory contains at least as many items as I am selling.
    if -myInventory:quantity(fruitType) <= offerPackage.offer[fruitType] then
      validTradeForMe = true
    else
      validTradeForMe = false
    end
    -- Their inventory contains at least as many items as I am selling.
    if -theirInventory:quantity(fruitType) <= theirPackage.offer[fruitType] then
      validTradeForThem = true
    else
      validTradeForThem = false
    end
    assert(validTradeForMe and validTradeForThem,
           'Somehow an invalid trade slipped past previous checks.')
  end
end

function Trading:resolve(offerPackage, theirPackage)
  -- Make sure that both players have enough items to trade.
  self:assertValidTrade(offerPackage, theirPackage)

  local myInventory = offerPackage.avatarObject:getComponent('Inventory')
  local theirInventory = theirPackage.avatarObject:getComponent('Inventory')
  local myExecutedTrade = {}
  local theirExecutedTrade = {}
  for _, fruitType in ipairs(self._config.itemTypes) do
    -- Ensure the minimal possible number of items actually change hands.
    if offerPackage.offer[fruitType] >= theirPackage.offer[fruitType] then
      myExecutedTrade[fruitType] = math.min(
          offerPackage.offer[fruitType],
          math.abs(theirPackage.offer[fruitType])
      )
      theirExecutedTrade[fruitType] = -myExecutedTrade[fruitType]
    else
      theirExecutedTrade[fruitType] = math.min(
          theirPackage.offer[fruitType],
          math.abs(offerPackage.offer[fruitType])
      )
      myExecutedTrade[fruitType] = -theirExecutedTrade[fruitType]
    end
    assert(myExecutedTrade[fruitType] == -theirExecutedTrade[fruitType],
           'Trades do not match.')

    -- Update the inventories.
    myInventory:add(fruitType, myExecutedTrade[fruitType])
    theirInventory:add(fruitType, theirExecutedTrade[fruitType])
  end

  -- Report the trade as an event for debug/analysis.
  local item0 = self._config.itemTypes[1]  -- e.g. 'apple'
  local item1 = self._config.itemTypes[2]  -- e.g. 'banana'
  events:add('trade', 'dict',
             'item_0', item0,
             'item_1', item1,
             'player_a_index', offerPackage.partnerId,
             'player_b_index', theirPackage.partnerId,
             'player_a_offered_item_0', offerPackage.offer[item0],
             'player_a_offered_item_1', offerPackage.offer[item1],
             'player_a_traded_item_0', myExecutedTrade[item0],
             'player_a_traded_item_1', myExecutedTrade[item1],
             'player_b_offered_item_0', theirPackage.offer[item0],
             'player_b_offered_item_1', theirPackage.offer[item1],
             'player_b_traded_item_0', theirExecutedTrade[item0],
             'player_b_traded_item_1', theirExecutedTrade[item1]
  )

  -- If trade was successful then cancel both partners' offers.
  offerPackage.trading:cancelOffer()
  theirPackage.trading:cancelOffer()
  return true
end

function Trading:getNearbyAvatars()
  local transform = self.gameObject:getComponent('Transform')
  local nearbyObjects = transform:queryDisc('upperPhysical',
                                            self._config.radius)
  -- Only return nearby objects that are avatars.
  local nearbyAvatars = {}
  for _, object in ipairs(nearbyObjects) do
    if object:hasComponent('Avatar') then
      table.insert(nearbyAvatars, object)
    end
  end
  return nearbyAvatars
end

function Trading:_isStrictlyBetterOffer(offerA, offerB)
  local possiblyBetter = false
  for fruitType, _ in pairs(offerA.offer) do
    if offerA.offer[fruitType] < offerB.offer[fruitType] then
      possiblyBetter = true
    end
    if offerA.offer[fruitType] > offerB.offer[fruitType] then
      return false
    end
  end
  return possiblyBetter
end

function Trading:getPossiblePartners()
  local nearbyAvatars = self:getNearbyAvatars()
  -- One pass over the avatars to collect the relevant data.
  local nearbyCompatibleOffers = {}
  for _, avatarObject in ipairs(nearbyAvatars) do
    local trading = avatarObject:getComponent('Trading')
    local nearbyOffer = trading:getPublicOffer()
    if self:isCompatible(nearbyOffer) and trading:hasEnough(nearbyOffer) and
        trading:isCompatible(self._offer) then
      local nearbyOfferPackage = {
          offer = nearbyOffer,
          avatarObject = avatarObject,
          trading = trading,
          partnerId = avatarObject:getComponent('Avatar'):getIndex(),
          dominated = false,
      }
      table.insert(nearbyCompatibleOffers, nearbyOfferPackage)
    end
  end

  -- Loop through all offers to see if any dominate each other.
  for _, offerA in ipairs(nearbyCompatibleOffers) do
    for _, offerB in ipairs(nearbyCompatibleOffers) do
      if self:_isStrictlyBetterOffer(offerA, offerB) then
        offerB.dominated = true
      end
    end
  end

  -- Collect all offers that were not dominated.
  local nearbyCompatibleNonDominatedOffers = {}
  for _, offer in ipairs(nearbyCompatibleOffers) do
    if not offer.dominated then
      table.insert(nearbyCompatibleNonDominatedOffers, offer)
    end
  end

  -- TODO(b/260155059): use subset of the offers from equally close players.

  return nearbyCompatibleNonDominatedOffers
end

function Trading:callResolveIfPossible()
  if self:isValid(self._offer) and self:hasEnough(self._offer) then
    -- Get all my offers.
    local myPossiblePartners = self:getPossiblePartners()

    -- For each of my possible partners, check that I am their possible partner.
    local myId = self.gameObject:getComponent('Avatar'):getIndex()
    for _, offerPackage in pairs(myPossiblePartners) do
      local theirPossiblePartners = offerPackage.trading:getPossiblePartners()
      -- Iterate over all their offers to find I am in it.
      for _, theirPackage in pairs(theirPossiblePartners) do
        if theirPackage.partnerId == myId then
          -- Return once a compatible offer has been found.
          return self:resolve(offerPackage, theirPackage)
        end
      end
    end
  end
  return false
end

function Trading:getOffer()
  return self._offer
end

function Trading:getTradeRadius()
  return self._config.radius
end

--[[ Trading:getPublicOffer() returns the current advertised offer.

Rturns {apple = 0, banana = 0} when the current offer is invalid i.e. the player
would not have enough items in its inventory to fulfil its part of the deal if
a trade were to occur in which it would have to give up the maximal amount it
offered.
]]
function Trading:getPublicOffer()
  if self:isValid(self._offer) and self:hasEnough(self._offer) then
    return self:getOffer()
  else
    return self:_getEmptyOffer()
  end
end

function Trading:cancelOffer()
  self:_resetOffer()
end


local InventoryObserver = class.Class(component.Component)

function InventoryObserver:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('InventoryObserver')},
      {'order', args.default({'apple', 'banana'}), args.tableType},
  })
  InventoryObserver.Base.__init__(self, kwargs)

  self._config.order = kwargs.order
  self._config.numFruitTypes = #self._config.order
end

function InventoryObserver:addObservations(tileSet,
                                           world,
                                           observations,
                                           avatarCount)
  local playerIdx = self.gameObject:getComponent('Avatar'):getIndex()
  local inventory = self.gameObject:getComponent('Inventory')
  observations[#observations + 1] = {
      name = tostring(playerIdx) .. '.INVENTORY',
      type = 'tensor.Int64Tensor',
      shape = {self._config.numFruitTypes},
      func = function(grid)
        return inventory:getInventoryAsTensor(self._config.order)
      end
  }
end


local MyOfferObserver = class.Class(component.Component)

function MyOfferObserver:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('MyOfferObserver')},
      {'order', args.default({'apple', 'banana'}), args.tableType},
  })
  MyOfferObserver.Base.__init__(self, kwargs)

  self._config.order = kwargs.order
  self._config.numFruitTypes = #self._config.order
end

function MyOfferObserver:reset()
  -- Allocate memory for the tensor representation of the offer.
  self._tensorMyOffer = tensor.Int64Tensor(self._config.numFruitTypes):fill(0)
end

function MyOfferObserver:getMyPublicOfferAsTensor(order)
  -- `order` (array of fruit strings). Set order of items in output tensor. For
  -- instance {'apple', 'banana'} --> tensor.Tensor{numApples, numBananas}.
  local trading = self.gameObject:getComponent('Trading')
  local offer = trading:getPublicOffer()
  for idx, fruit in ipairs(order) do
    -- Set index `idx` to value `self._inventory[fruit]`.
    self._tensorMyOffer(idx):val(offer[fruit])
  end
  return self._tensorMyOffer
end

function MyOfferObserver:addObservations(tileSet,
                                         world,
                                         observations,
                                         avatarCount)
  local playerIdx = self.gameObject:getComponent('Avatar'):getIndex()
  observations[#observations + 1] = {
      name = tostring(playerIdx) .. '.MY_OFFER',
      type = 'tensor.Int64Tensor',
      shape = {self._config.numFruitTypes},
      func = function(grid)
        -- Note: This always returns the public offer.
        return self:getMyPublicOfferAsTensor(self._config.order)
      end
  }
end


local AllOffersObserver = class.Class(component.Component)

function AllOffersObserver:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AllOffersObserver')},
      {'order', args.default({'apple', 'banana'}), args.tableType},
      -- If `flatten` is true then output a flat vector, otherwise the output
      -- observation is a matrix of size (num_players - 1, num_fruit_types).
      {'flatten', args.default(false), args.booleanType},
  })
  AllOffersObserver.Base.__init__(self, kwargs)

  self._config.order = kwargs.order
  self._config.numFruitTypes = #self._config.order
  self._config.flatten = kwargs.flatten
end

function AllOffersObserver:setMaxPossibleOffers()
  local radius = self.gameObject:getComponent('Trading'):getTradeRadius()
  -- Note: this assumes queryDisc was used in Trading.getNearbyAvatars.
  self._maxPossibleOffers = math.ceil(math.pi * radius * radius)
end

function AllOffersObserver:reset()
  -- Allocate memory for the tensor representation of all nearby offers.
  self:setMaxPossibleOffers()
  self._tensorOffers = tensor.Int64Tensor(self._maxPossibleOffers,
                                          self._config.numFruitTypes):fill(0)
end

function AllOffersObserver:getNearbyAvatars()
  local trading = self.gameObject:getComponent('Trading')
  local avatars = trading:getNearbyAvatars()
  return avatars
end

function AllOffersObserver:getPublicOffersAsTensor(order)
  -- First zero the offers tensor from the previous timestep.
  self._tensorOffers:fill(0)
  -- `order` (array of fruit strings). Set order of items in output tensor. For
  -- instance {'apple', 'banana'} --> tensor.Tensor{numApples, numBananas}.
  local selfIdx = self.gameObject:getComponent('Avatar'):getIndex()
  -- Note: there is no guarantee that avatars will be in any specific order. For
  -- instance, is NOT guaranteed to be in slot id or joining order.
  local avatars = self:getNearbyAvatars()
  local arbitrary_idx = 1
  for _, avatarObject in pairs(avatars) do
    -- Exclude the offer that would correspond to the self offer since it will
    -- end up coming out in an entirely different observation channel.
    if avatarObject:getComponent('Avatar'):getIndex() ~= selfIdx then
      local offer = avatarObject:getComponent('Trading'):getPublicOffer()
      for idx, fruit in ipairs(order) do
        -- Set index `idx` to value `self._inventory[fruit]`.
        self._tensorOffers(arbitrary_idx, idx):val(offer[fruit])
      end
      arbitrary_idx = arbitrary_idx + 1
    end
  end

  return self._tensorOffers
end

function AllOffersObserver:addObservations(tileSet,
                                           world,
                                           observations,
                                           avatarCount)
  self:setMaxPossibleOffers()
  local shape, formatOutput
  if self._config.flatten then
    shape = {self._maxPossibleOffers * self._config.numFruitTypes}
    formatOutput = function(publicOffersTensor)
      local flatOffers = publicOffersTensor:reshape(shape)
      return flatOffers
    end
  else
    shape = {self._maxPossibleOffers, self._config.numFruitTypes}
    formatOutput = function(publicOffersTensor)
      return publicOffersTensor
    end
  end

  local playerIdx = self.gameObject:getComponent('Avatar'):getIndex()
  observations[#observations + 1] = {
      name = tostring(playerIdx) .. '.OFFERS',
      type = 'tensor.Int64Tensor',
      shape = shape,
      func = function(grid)
        return formatOutput(self:getPublicOffersAsTensor(self._config.order))
      end
  }
end


local HungerObserver = class.Class(component.Component)

function HungerObserver:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('HungerObserver')},
      {'needComponent', args.default('PeriodicNeed'), args.stringType},
  })
  HungerObserver.Base.__init__(self, kwargs)
  self._config.needComponent = kwargs.needComponent
end

function HungerObserver:addObservations(tileSet, world, observations)
  local playerIndex = self.gameObject:getComponent('Avatar'):getIndex()
  local needComponent = self.gameObject:getComponent(self._config.needComponent)
  observations[#observations + 1] = {
      name = tostring(playerIndex) .. '.HUNGER',
      type = 'Doubles',
      shape = {},
      func = function(grid)
        return needComponent:getNeed()
      end
  }
end


local TradeManager = class.Class(component.Component)

function TradeManager:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('TradeManager')},
  })
  TradeManager.Base.__init__(self, kwargs)
end

function TradeManager:registerUpdaters(updaterRegistry)
  local simulation = self.gameObject.simulation
  local numPlayers = simulation:getNumPlayers()

  local function resolveTrades()
    local order = random:shuffle(range(numPlayers))

    for _, playerIndex in ipairs(order) do
      local avatarObject = simulation:getAvatarFromIndex(playerIndex)
      local trading = avatarObject:getComponent('Trading')
      local success = trading:callResolveIfPossible()
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = resolveTrades,
      priority = 2,
  }
end

local allComponents = {
    -- Fruit tree components.
    FruitType = FruitType,
    Harvestable = Harvestable,
    PreventStaminaRecoveryHere = PreventStaminaRecoveryHere,

    -- Water components.
    TraversalCost = TraversalCost,

    -- Avatar components.
    Inventory = Inventory,
    Eating = Eating,
    Specialization = Specialization,
    Taste = Taste,
    MovementPenalty = MovementPenalty,
    Trading = Trading,

    -- Avatar observer components.
    InventoryObserver = InventoryObserver,
    MyOfferObserver = MyOfferObserver,
    AllOffersObserver = AllOffersObserver,
    HungerObserver = HungerObserver,

    -- Scene components.
    TradeManager = TradeManager,
}

component_registry.registerAllComponents(allComponents)

return allComponents
