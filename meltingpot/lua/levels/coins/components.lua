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
local events = require 'system.events'
local random = require 'system.random'
local tensor = require 'system.tensor'

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')

local _COMPASS = {'N', 'E', 'S', 'W'}


-- A component that keeps track of a parameterized coin type.
local PlayerCoinType = class.Class(component.Component)

function PlayerCoinType:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('PlayerCoinType')},
      {'coinType', args.stringType},
  })
  PlayerCoinType.Base.__init__(self, kwargs)

  self._config.coinType = kwargs.coinType
end

function PlayerCoinType:getCoinType()
  return self._config.coinType
end


-- Coins switch state when touched by an avatar, and provide rewards to players
-- based on the (mis)match between its type and the collecting players's coin
-- type. It can be used in combination with ChoiceCoinRegrow.
local Coin = class.Class(component.Component)

function Coin:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Coin')},
      {'waitState', args.stringType},
      {'rewardSelfForMatch', args.numberType},
      {'rewardSelfForMismatch', args.numberType},
      {'rewardOtherForMatch', args.numberType},
      {'rewardOtherForMismatch', args.numberType},
      {'terminateEpisode', args.booleanType, args.default(false)},
      {'coinsToTerminateEpisode', args.numberType, args.default(-1)},
  })
  Coin.Base.__init__(self, kwargs)

  self._config.waitState = kwargs.waitState
  self._config.rewardSelfForMatch = kwargs.rewardSelfForMatch
  self._config.rewardOtherForMatch = kwargs.rewardOtherForMatch
  self._config.rewardSelfForMismatch = kwargs.rewardSelfForMismatch
  self._config.rewardOtherForMismatch = kwargs.rewardOtherForMismatch
  self._config.terminateEpisode = kwargs.terminateEpisode
  self._config.coinsToTerminateEpisode = kwargs.coinsToTerminateEpisode
end

function Coin:reset()
  self._waitState = self._config.waitState
end

function Coin:rewardOthers(amountToReward, avatarIndexToSkip)
  local simulation = self.gameObject.simulation
  -- Iterate through avatars.
  for _, object in pairs(simulation:getAvatarGameObjects()) do
    local avatarComponent = object:getComponent('Avatar')
    -- Skip the player who collected the coin.
    if avatarComponent:getIndex() ~= avatarIndexToSkip then
      -- Add reward.
      avatarComponent:addReward(amountToReward)
    end
  end
end

function Coin:onEnter(enteringGameObject, contactName)
  local simulation = self.gameObject.simulation
  assert(
    simulation:getNumPlayers() <= 2, 'We only allow 1 or 2 players in Coins.')

  if contactName == 'avatar' then
    local coinState = self.gameObject:getState()
    if coinState ~= self._waitState then
      -- Prepare to record collection event.
      local coinsCollected = simulation:getSceneObject():getComponent(
        "GlobalCoinCollectionTracker").coinsCollected

      -- Get collecting player's coin type.
      local avatarComponent = enteringGameObject:getComponent('Avatar')
      local roleComponent = enteringGameObject:getComponent('Role')
      local partnerTracker = enteringGameObject:getComponent('PartnerTracker')
      local playerIndex = avatarComponent:getIndex()
      local playerCoinTypeComponent = enteringGameObject:getComponent(
        'PlayerCoinType')
      local playerCoinType = playerCoinTypeComponent:getCoinType()

      -- Check for match between this coin's type and collecting player's type.
      if playerCoinType == coinState then
        -- Reward collecting player and others for match.
        local selfReward = roleComponent:getRewardSelfForMatch(
            self._config.rewardSelfForMatch)
        avatarComponent:addReward(selfReward)
        local otherReward = roleComponent:getRewardOtherForMatch(
            self._config.rewardOtherForMatch)
        self:rewardOthers(otherReward, playerIndex)
        -- Record collection event.
        coinsCollected(playerIndex, playerIndex):val(1)
        partnerTracker:reportMatch()
      else
        -- Reward collecting player and others for mismatch.
        local selfReward = roleComponent:getRewardSelfForMismatch(
            self._config.rewardSelfForMismatch)
        avatarComponent:addReward(selfReward)
        local otherReward = roleComponent:getRewardOtherForMismatch(
            self._config.rewardOtherForMismatch)
        self:rewardOthers(otherReward, playerIndex)
        -- Record collection event.
        coinIndex = playerIndex % 2 + 1
        coinsCollected(playerIndex, coinIndex):val(1)
        partnerTracker:reportMismatch()
      end

      -- Record events.
      events:add('coin_consumed', 'dict',
                 'player_index', playerIndex,                 -- int
                 'player_coin_type', playerCoinType,          -- str
                 'coin_type', coinState)                      -- str

      -- Change the coin to its wait (disabled) state.
      self.gameObject:setState(self._waitState)

      -- Track cumulative collections for each player.
      local cumulativeCoinsCollected = simulation:getSceneObject():getComponent(
        "GlobalCoinCollectionTracker").cumulativeCoinsCollected
      cumulativeCoinsCollected(playerIndex):add(1)

      -- If the level is configured to end after a certain number of
      -- collections...
      if self._config.terminateEpisode then
        -- ... count coins and compare against the configured threshold.
        coinCount = cumulativeCoinsCollected(playerIndex):val()
        if coinCount >= self._config.coinsToTerminateEpisode then
          -- End episode.
          self.gameObject.simulation:endEpisode()
        end
      end
    end
  end
end


--[[ The `ChoiceCoinRegrow` component enables a game object that is in a
particular (traditionally thought of as "dormant") state to change its state
probabilistically (at a fixed rate). Used primarily for respawning objects.
]]
local ChoiceCoinRegrow = class.Class(component.Component)

function ChoiceCoinRegrow:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('ChoiceCoinRegrow')},
      {'liveStateA', args.stringType},
      {'liveStateB', args.stringType},
      {'waitState', args.stringType},
      {'regrowRate', args.ge(0.0), args.le(1.0)},
  })
  ChoiceCoinRegrow.Base.__init__(self, kwargs)

  self._config.liveStates = {kwargs.liveStateA, kwargs.liveStateB}
  self._config.waitState = kwargs.waitState
  self._config.regrowRate = kwargs.regrowRate
end

function ChoiceCoinRegrow:registerUpdaters(updaterRegistry)
  -- Registers an update with high priority that only gets called when the
  -- object is in the `waitState` state.
  updaterRegistry:registerUpdater{
    state = self._config.waitState,
    probability = self._config.regrowRate,
    updateFn = function()
      self.gameObject:setState(random:choice(self._config.liveStates))
    end,
  }
end


--[[ The GlobalCoinCollectionTracker keeps track of coin collections.]]
local GlobalCoinCollectionTracker = class.Class(component.Component)

function GlobalCoinCollectionTracker:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('GlobalCoinCollectionTracker')},
      {'numPlayers', args.numberType},
  })
  GlobalCoinCollectionTracker.Base.__init__(self, kwargs)

  self._config.numPlayers = kwargs.numPlayers
end

function GlobalCoinCollectionTracker:reset()
  self.coinsCollected = tensor.Int32Tensor(self._config.numPlayers, 2)
  self.cumulativeCoinsCollected = tensor.Int32Tensor(self._config.numPlayers)
  self.cumulativeCoinsCollected:fill(0)
end

function GlobalCoinCollectionTracker:preUpdate()
  self.coinsCollected:fill(0)
end


local Role = class.Class(component.Component)

function Role:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Role')},
      {'multiplyRewardSelfForMatch', args.default(1.0), args.numberType},
      {'multiplyRewardSelfForMismatch', args.default(1.0), args.numberType},
      {'multiplyRewardOtherForMatch', args.default(1.0), args.numberType},
      {'multiplyRewardOtherForMismatch', args.default(1.0), args.numberType},
  })
  Role.Base.__init__(self, kwargs)

  self._config.multiplyRewardSelfForMatch = kwargs.multiplyRewardSelfForMatch
  self._config.multiplyRewardSelfForMismatch =
      kwargs.multiplyRewardSelfForMismatch
  self._config.multiplyRewardOtherForMatch = kwargs.multiplyRewardOtherForMatch
  self._config.multiplyRewardOtherForMismatch =
      kwargs.multiplyRewardOtherForMismatch
end

function Role:reset()
  -- Dynamic frame-by-frame tracking of coin collection.
  self.cumulantCollectedMatch = 0
  self.cumulantCollectedMismatch = 0
end

function Role:getRewardSelfForMatch(baseReward)
  self.cumulantCollectedMatch = 1
  local result = baseReward * self._config.multiplyRewardSelfForMatch
  return result
end

function Role:getRewardSelfForMismatch(baseReward)
  self.cumulantCollectedMismatch = 1
  local result = baseReward * self._config.multiplyRewardSelfForMismatch
  return result
end

function Role:getRewardOtherForMatch(baseReward)
  local result = baseReward * self._config.multiplyRewardOtherForMatch
  return result
end

function Role:getRewardOtherForMismatch(baseReward)
  local result = baseReward * self._config.multiplyRewardOtherForMismatch
  return result
end

function Role:preUpdate()
  self.cumulantCollectedMatch = 0
  self.cumulantCollectedMismatch = 0
end


local PartnerTracker = class.Class(component.Component)

function PartnerTracker:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('PartnerTracker')},
  })
  PartnerTracker.Base.__init__(self, kwargs)
end

function PartnerTracker:reset()
  -- Dynamic frame-by-frame tracking of the other player's coin collection.
  self.partnerCollectedMatch = 0
  self.partnerCollectedMismatch = 0
end

function PartnerTracker:postStart()
  self._trackerOfPartner = self:getTrackerOfPartner()
end

function PartnerTracker:preUpdate()
  self.partnerCollectedMatch = 0
  self.partnerCollectedMismatch = 0
end

function PartnerTracker:getTrackerOfPartner()
  local selfIndex = self.gameObject:getComponent('Avatar'):getIndex()
  local partnerIndex
  if selfIndex == 1 then
    -- If I am 1 then you are 2.
    partnerIndex = 2
  elseif selfIndex == 2 then
    -- If I am 2 then you are 1.
    partnerIndex = 1
  else
    assert(false, 'Unrecognized self index. Coins only supports two players.')
  end
  local otherAvatarObject = self.gameObject.simulation:getAvatarFromIndex(
      partnerIndex)
  return otherAvatarObject:getComponent('PartnerTracker')
end

function PartnerTracker:reportMatch()
  self._trackerOfPartner.partnerCollectedMatch = 1
end

function PartnerTracker:reportMismatch()
  self._trackerOfPartner.partnerCollectedMismatch = 1
end


local allComponents = {
    PlayerCoinType = PlayerCoinType,
    Coin = Coin,
    ChoiceCoinRegrow = ChoiceCoinRegrow,
    GlobalCoinCollectionTracker = GlobalCoinCollectionTracker,
    Role = Role,
    PartnerTracker = PartnerTracker,
}
component_registry.registerAllComponents(allComponents)

return allComponents
