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
local events = require 'system.events'
local helpers = require 'common.helpers'
local log = require 'common.log'
local random = require 'system.random'
local tensor = require 'system.tensor'

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')


local FixedRateRegrow = class.Class(component.Component)

function FixedRateRegrow:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('FixedRateRegrow')},
      {'liveState', args.stringType},
      {'waitState', args.stringType},
      {'regrowRate', args.ge(0.0), args.le(1.0)},
  })
  FixedRateRegrow.Base.__init__(self, kwargs)

  self._config.liveState = kwargs.liveState
  self._config.waitState = kwargs.waitState
  self._config.regrowRate = kwargs.regrowRate
end

function FixedRateRegrow:update()
  if self.gameObject:getState() == self._config.waitState then
    if random:uniformReal(0, 1) < self._config.regrowRate then
      local transform = self.gameObject:getComponent('Transform')
      local maybeAvatar = transform:queryPosition('upperPhysical')
      if not maybeAvatar then
        self.gameObject:setState(self._config.liveState)
      end
    end
  end
end

local Pickable = class.Class(component.Component)

function Pickable:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Pickable')},
      {'liveState', args.stringType},
      {'waitState', args.stringType},
      {'rewardForPicking', args.numberType},
  })
  Pickable.Base.__init__(self, kwargs)

  self._config.rewardForPicking = kwargs.rewardForPicking

  self._config.liveState = kwargs.liveState
  self._config.waitState = kwargs.waitState
end

function Pickable:onEnter(enteringGameObject, contactName)
  if contactName == 'avatar' and
      self.gameObject:getState() == self._config.liveState then
    -- Add reward for picking up object.
    enteringGameObject:getComponent('Avatar'):addReward(
        self._config.rewardForPicking)
    -- Add to the player's inventory as lowest refinement.
    enteringGameObject:getComponent('Inventory'):addTokens(1, 1)
    enteringGameObject:getComponent('TokenTracker').collectedToken = (
        enteringGameObject:getComponent('TokenTracker').collectedToken + 1)
    -- Replace the appple with an invisible appleWait.
    self.gameObject:setState(self._config.waitState)
  end
end

--[[ The `GiftBeam` component endows an avatar with the ability to fire a beam
and be hit by the gift beams of other avatars.
]]
local GiftBeam = class.Class(component.Component)

function GiftBeam:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('GiftBeam')},
      {'cooldownTime', args.numberType},
      {'beamLength', args.numberType},
      {'beamRadius', args.numberType},
      {'agentRole', args.stringType},
      {'giftMultiplier', args.numberType},
      {'successfulGiftReward', args.numberType},
      {'roleRewardForGifting', args.tableType},
  })
  GiftBeam.Base.__init__(self, kwargs)

  self._config.cooldownTime = kwargs.cooldownTime
  self._config.beamLength = kwargs.beamLength
  self._config.beamRadius = kwargs.beamRadius
  self._config.agentRole = kwargs.agentRole
  self._config.giftMultiplier = kwargs.giftMultiplier
  self._config.successfulGiftReward = kwargs.successfulGiftReward
  self._config.roleRewardForGifting = kwargs.roleRewardForGifting

  self._coolingTimer = 0
end

function GiftBeam:addHits(worldConfig)
  worldConfig.hits['gift'] = {
      layer = 'beamGift',
      sprite = 'beamGift',
  }
  table.insert(worldConfig.renderOrder, 'beamGift')
end

function GiftBeam:addSprites(tileSet)
  -- This color is pink.
  tileSet:addColor('beamGift', {255, 202, 202})
end

function GiftBeam:getAgentRole()
  return self._config.agentRole
end

function GiftBeam:onHit(hitterGameObject, hitName)
  if hitName == 'gift' then
    local hitAvatar = self.gameObject:getComponent('Avatar')
    local hitRole = self.gameObject:getComponent('GiftBeam'):getAgentRole()
    local hitIndex = hitAvatar:getIndex()
    local hitterAvatar = hitterGameObject:getComponent('Avatar')
    local hitterRole = hitterGameObject:getComponent('GiftBeam'):getAgentRole()
    local hitterIndex = hitterAvatar:getIndex()
    local amount = self._config.roleRewardForGifting[hitterRole]
    if amount ~= nil then
      hitterAvatar:addReward(amount)
    end
    local hitterInventory = hitterGameObject:getComponent('Inventory')
    local hitInventory = self.gameObject:getComponent('Inventory')
    -- return `true` to prevent the beam from passing through a hit player.
    local srcType = hitterInventory:getHighestTypeAvailable()
    -- Only add tokens if any is available. Gift the most refined first.
    if srcType > 0 then
      local dstAmount = self._config.giftMultiplier
      local dstType = srcType + 1
      -- If at most refined, don't multiply nor increase refinement.
      if srcType + 1 > hitInventory:getNumTokenTypes() then
        dstType = hitInventory:getNumTokenTypes()
        dstAmount = 1
      else
        hitterAvatar:addReward(amount * self._config.successfulGiftReward)
      end
      hitterInventory:removeTokens(srcType, 1)
      local actual = hitInventory:addTokens(dstType, dstAmount)
      self.gameObject:getComponent(
        'TokenTracker').giftsReceived(hitterIndex, srcType):add(actual)
      self.gameObject:getComponent(
        'TokenTracker').giftsReceivedFromAny = self.gameObject:getComponent(
            'TokenTracker').giftsReceivedFromAny + actual
      hitterGameObject:getComponent(
          'TokenTracker').giftsGiven(hitIndex, srcType):add(actual)
      hitterGameObject:getComponent(
        'TokenTracker').giftsGivenToAny = hitterGameObject:getComponent(
            'TokenTracker').giftsGivenToAny + actual
      events:add("gift", "dict",
          "gifter_index", hitterIndex,
          "gifter_role", hitterRole,
          "receipient_index", hitIndex,
          "receipient_role", hitRole,
          "source_type", srcType,
          "received_amount", actual)
      end
    return true
  end
end

function GiftBeam:registerUpdaters(updaterRegistry)

  local gift = function()
    local playerVolatileVariables = (
        self.gameObject:getComponent('Avatar'):getVolatileData())
    local actions = playerVolatileVariables.actions
    -- Execute the beam if applicable.
    if self.gameObject:getComponent('Avatar'):isAlive() then
      if actions.refineAndGift == 1 and self._coolingTimer <= 0 then
        self._coolingTimer = self._config.cooldownTime
        self.gameObject:hitBeam(
            'gift', self._config.beamLength, self._config.beamRadius)
        -- TODO(b/260154384): reward for even attempting to gift?
      end
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = gift,
      priority = 140,
  }
end

-- Integrate all signals that affect whether it is possible to fire the gift
-- beam into a single float between 0 and 1. It is only possible to use the beam
-- action when 1 is returned. The GiftBeam will be restored sooner the closer to
-- 1 the signal becomes.
function GiftBeam:readyToShoot()
  local normalizedTimeTillReady = self._coolingTimer / self._config.cooldownTime
  if self.gameObject:getComponent('Avatar'):isAlive() then
    return math.max(1 - normalizedTimeTillReady, 0)
  else
    return 0
  end
end

function GiftBeam:update()
  if self._coolingTimer > 0 then
    self._coolingTimer = self._coolingTimer - 1
  end
end

function GiftBeam:start()
  -- Set the beam cooldown timer to its `ready` state (i.e. coolingTimer = 0).
  self._coolingTimer = 0
  end

--[[ The inventory carrying tokens. Also provides the action to consume them.

Items in the inventory have a type, starting from 1, and up to numTokenTypes
(both inclusive). When getting tokens, we can request the one with the highest
type, or the lowest, or one of a specific type.
]]
local Inventory = class.Class(component.Component)

function Inventory:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Inventory')},
      {'capacityPerType', args.numberType},
      {'numTokenTypes', args.numberType},
      {'consumptionCooldown', args.default(0), args.numberType},
  })
  Inventory.Base.__init__(self, kwargs)

  self._config.capacityPerType = kwargs.capacityPerType
  self._config.numTokenTypes = kwargs.numTokenTypes
  self._config.consumptionCooldown = kwargs.consumptionCooldown
  self:emptyInventory()
end

function Inventory:reset()
  self:emptyInventory()
end

function Inventory:start()
  self._consumeCooldownTimer = 0
  self.tokensConsumed = self.gameObject:getComponent(
      'TokenTracker').tokensConsumed
end

function Inventory:getNumTokenTypes()
  return self._config.numTokenTypes
end

function Inventory:emptyInventory()
  self.inventory = tensor.DoubleTensor(
      self._config.numTokenTypes):fill(0)
end

-- This function returns the highest type of token for which the inventory has
-- at least one. If the inventory is empty, this function returns 0 (types are
-- 1 to numTokenTypes).
function Inventory:getHighestTypeAvailable()
  local tokenType = 0
  for tType = 1, self._config.numTokenTypes do
    local count = self.inventory(tType):val()
    if count > 0 then
      tokenType = tType
    end
  end
  return tokenType
end

-- This function returns the lowest type of token for which the inventory has
-- at least one. If the inventory is empty, this function returns 0 (types are
-- 1 to numTokenTypes).
function Inventory:getLowestTypeAvailable()
  for tType = 1, self._config.numTokenTypes do
    local count = self.inventory(tType):val()
    if count > 0 then
      return tType
    end
  end
  return 0
end

--[[ Attempt to add a certain amount of tokens of the specified type. If the
current tokens of this type in the inventory + the amount requested is larger
than the capacity of the inventory, we will add as many tokens as possible. The
function returns how many tokens were actually added.
]]
function Inventory:addTokens(tokenType, amount)
  assert(tokenType >= 1 and tokenType <= self._config.numTokenTypes)
  local value = math.min(
    self.inventory(tokenType):val() + amount,
    self._config.capacityPerType)
  self.inventory(tokenType):val(value)
  return value
end

--[[ Attempt to remove a certain amount of tokens of the specified type. If the
amount requested is larger than the available tokens, we will remove as many
tokens as possible. The function returns how many tokens were actually removed.
]]
function Inventory:removeTokens(tokenType, amount)
  assert(tokenType >= 1 and tokenType <= self._config.numTokenTypes)
  local actual = math.min(self.inventory(tokenType):val(), amount)
  self.inventory(tokenType):val(math.max(
      self.inventory(tokenType):val() - amount, 0))
  return actual
end

function Inventory:update()
  local state = self.gameObject:getComponent('Avatar'):getVolatileData()
  local actions = state.actions
  -- Execute the beam if applicable.
  if actions.consumeTokens == 1 and self._consumeCooldownTimer <= 0 then
    local avatar = self.gameObject:getComponent('Avatar')
    local amount = 0
    for tokenType = 1, self._config.numTokenTypes do
      local value = self.inventory(tokenType):val()
      amount = amount + value
      self.tokensConsumed(tokenType):add(value)
    end
    avatar:addReward(amount)
    self:emptyInventory()
    self._consumeCooldownTimer = self._config.consumptionCooldown
    -- Sate the hunger drive if applicable.
    if self.gameObject:hasComponent('Hunger') and amount > 0 then
      self.gameObject:getComponent('Hunger'):resetDriveLevel()
    end
  end
  self._consumeCooldownTimer = self._consumeCooldownTimer - 1
end


--[[ Token tracker keeps track of the picking up, gifting and consumption of
tokens by players. These can then be used for cumulants and debug observations.
]]
local TokenTracker = class.Class(component.Component)

function TokenTracker:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('TokenTracker')},
      {'numPlayers', args.numberType},
      {'numTokenTypes', args.numberType},
  })
  TokenTracker.Base.__init__(self, kwargs)

  self._config.numPlayers = kwargs.numPlayers
  self._config.numTokenTypes = kwargs.numTokenTypes
end

function TokenTracker:reset()
  self.giftsGiven = tensor.Int32Tensor(
      self._config.numPlayers,
      self._config.numTokenTypes)
  self.giftsGivenToAny = 0.0
  self.giftsReceived = tensor.Int32Tensor(
      self._config.numPlayers,
      self._config.numTokenTypes)
  self.giftsReceivedFromAny = 0.0
  self.tokensConsumed = tensor.Int32Tensor(self._config.numTokenTypes)
  self.collectedToken = 0.0
end

function TokenTracker:preUpdate()
  self.giftsGiven:fill(0)
  self.giftsGivenToAny = 0.0
  self.giftsReceived:fill(0)
  self.giftsReceivedFromAny = 0.0
  self.tokensConsumed:fill(0)
  self.collectedToken = 0.0
end


local allComponents = {
    -- Coin components
    FixedRateRegrow = FixedRateRegrow,
    Pickable = Pickable,

    -- Avatar components
    GiftBeam = GiftBeam,
    Inventory = Inventory,
    TokenTracker = TokenTracker,
}

component_registry.registerAllComponents(allComponents)

return allComponents
