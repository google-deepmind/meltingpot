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
      {'liveStates', args.tableType},
      {'liveRates', args.tableType},
      {'waitState', args.stringType},
  })
  self.Base.__init__(self, kwargs)

  self._config.liveStates = kwargs.liveStates
  self._config.liveRates = kwargs.liveRates
  self._config.waitState = kwargs.waitState
end

function FixedRateRegrow:registerUpdaters(updaterRegistry)
  for i, rate in ipairs(self._config.liveRates) do
    updaterRegistry:registerUpdater{
      priority = 200,
      state = self._config.waitState,
      probability = rate,
      updateFn = function()
        local transform = self.gameObject:getComponent('Transform')
        local maybeAvatar = transform:queryPosition('upperPhysical')
        if not maybeAvatar then
          self.gameObject:setState(self._config.liveStates[i])
        end
      end,
    }
  end
end

local Ore = class.Class(component.Component)

function Ore:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Ore')},
      {'rawState', args.stringType},
      {'waitState', args.stringType},
      {'partialState', args.stringType},
      {'minNumMiners', args.numberType},
      {'miningWindow', args.numberType},
  })
  self.Base.__init__(self, kwargs)

  self._config.rawState = kwargs.rawState
  self._config.waitState = kwargs.waitState
  self._config.partialState = kwargs.partialState
  self._config.minNumMiners = kwargs.minNumMiners
  self._config.miningWindow = kwargs.miningWindow
end

function Ore:currentMiners()
  local count = 0
  for k, v in pairs(self._miners) do
    count = count + 1
  end
  return count
end

function Ore:reset()
  -- Table tracking which players have attempted mining this resource.
  self._miners = {}
  self._miningCountdown = 0
  if self.gameObject:getState() ~= self._config.waitState then
    self.gameObject:setState(self._config.rawState)
  end
end

function Ore:update()
  self._miningCountdown = self._miningCountdown - 1
  if self._miningCountdown == 0 then
    -- Clean miners
    self:reset()
  end
end

function Ore:addMiner(minerId)
  self._miningCountdown = self._config.miningWindow
  self._miners[minerId] = 1
  self.gameObject:setState(self._config.partialState)
end

function Ore:onHit(hitterGameObject, hitName)
  if hitName == 'mine' and
      (self.gameObject:getState() == self._config.rawState or
       self.gameObject:getState() == self._config.partialState) then
    local hitterIndex = hitterGameObject:getComponent('Avatar'):getIndex()
    self:addMiner(hitterIndex)

    local hitterMineBeam = hitterGameObject:getComponent('MineBeam')
    hitterMineBeam:processRoleMineEvent(self._config.minNumMiners)
    -- If the Ore has enough miners, process rewards.
    if self:currentMiners() == self._config.minNumMiners then
      for id, _ in pairs(self._miners) do
        local avatarGO = self.gameObject.simulation:getAvatarFromIndex(id)
        avatarGO:getComponent('MineBeam'):processRoleExtractEvent(
            self._config.minNumMiners)
        for otherId, _ in pairs(self._miners) do
          if otherId ~= id then
            avatarGO:getComponent('MineBeam'):processRolePairExtractEvent(
                otherId, self._config.minNumMiners)
          end
        end
      end
      self:reset()
      self.gameObject:setState(self._config.waitState)
    end
    -- return `true` to prevent the beam from passing through a hit ore.
    return true
  end
  -- Other beams, or if in state not raw nor partial can pass through.
  return false
end

--[[ The `MineBeam` component endows an avatar with the ability to fire a beam.
]]
local MineBeam = class.Class(component.Component)

function MineBeam:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('MineBeam')},
      {'cooldownTime', args.numberType},
      {'beamLength', args.numberType},
      {'beamRadius', args.numberType},
      {'agentRole', args.stringType},
      -- These two must be tables indexed by [role][oreType]
      {'roleRewardForMining', args.tableType},
      {'roleRewardForExtracting', args.tableType},
  })
  self.Base.__init__(self, kwargs)

  self._config.cooldownTime = kwargs.cooldownTime
  self._config.beamLength = kwargs.beamLength
  self._config.beamRadius = kwargs.beamRadius
  self._config.agentRole = kwargs.agentRole
  self._config.roleRewardForMining = kwargs.roleRewardForMining
  self._config.roleRewardForExtracting = kwargs.roleRewardForExtracting

  self._coolingTimer = 0
end

function MineBeam:readyToShoot()
  local normalizedTimeTillReady = self._coolingTimer / self._config.cooldownTime
  return 1 - normalizedTimeTillReady
end

function MineBeam:addHits(worldConfig)
  worldConfig.hits['mine'] = {
      layer = 'beamMine',
      sprite = 'beamMine',
  }
  table.insert(worldConfig.renderOrder, 'beamMine')
end

function MineBeam:addSprites(tileSet)
  -- This color is pink.
  tileSet:addColor('beamMine', {255, 202, 202})
end

function MineBeam:processRoleMineEvent(oreType)
  local amount = self._config.roleRewardForMining[
      self._config.agentRole][oreType]
  local avatar = self.gameObject:getComponent('Avatar')
  avatar:addReward(amount)

  events:add("mining", "dict",
      "player", avatar:getIndex(),
      "ore_type", oreType)

  self.playerMined(oreType):add(1)
end

function MineBeam:processRoleExtractEvent(oreType)
  local amount = self._config.roleRewardForExtracting[
      self._config.agentRole][oreType]
  local avatar = self.gameObject:getComponent('Avatar')
  avatar:addReward(amount)
  local index = avatar:getIndex()

  events:add("extraction", "dict",
      "player", index,
      "ore_type", oreType)

  self.playerExtracted(oreType):add(1)
end

function MineBeam:processRolePairExtractEvent(otherId, oreType)
  local index = self.gameObject:getComponent('Avatar'):getIndex()

  events:add("extraction_pair", "dict",
  "player_a", index,
  "player_b", otherId,
  "ore_type", oreType)
  
  self.coExtracted(otherId, oreType):add(1)
end

function MineBeam:update()
  if self._coolingTimer > 0 then
    self._coolingTimer = self._coolingTimer - 1
  end

  -- TODO(b/260338825): It would be good to factor out the firing logic to be in
  -- an updater so we can control the exact order of execution within a frame.
  -- Right now it depends on the Lua table order that the components are added.
  local state = self.gameObject:getComponent('Avatar'):getVolatileData()
  local actions = state.actions
  -- Execute the beam if applicable.
  if actions.mine == 1 and self:readyToShoot() >= 1 then
    self._coolingTimer = self._config.cooldownTime
    self.gameObject:hitBeam(
        'mine', self._config.beamLength, self._config.beamRadius)
  end
end

function MineBeam:start()
  -- Set the beam cooldown timer to its `ready` state (i.e. coolingTimer = 0).
  self._coolingTimer = 0
  self.playerMined = self.gameObject:getComponent('MiningTracker').playerMined
  self.playerExtracted = self.gameObject:getComponent(
      'MiningTracker').playerExtracted
  self.coExtracted = self.gameObject:getComponent('MiningTracker').coExtracted
end

--[[ The MiningTracker keeps track of the mining metrics.]]
local MiningTracker = class.Class(component.Component)

function MiningTracker:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('MiningTracker')},
      {'numPlayers', args.numberType},
      {'numOreTypes', args.numberType},
  })
  self.Base.__init__(self, kwargs)

  self._config.numPlayers = kwargs.numPlayers
  self._config.numOreTypes = kwargs.numOreTypes
end

function MiningTracker:reset()
  self.playerMined = tensor.Int32Tensor(self._config.numOreTypes)
  self.playerExtracted = tensor.Int32Tensor(self._config.numOreTypes)
  self.coExtracted = tensor.Int32Tensor(
      self._config.numPlayers,
      self._config.numOreTypes)
end

function MiningTracker:preUpdate()
  self.playerMined:fill(0)
  self.playerExtracted:fill(0)
  self.coExtracted:fill(0)
end

local allComponents = {
    FixedRateRegrow = FixedRateRegrow,
    Ore = Ore,
    MineBeam = MineBeam,
    MiningTracker = MiningTracker,
}

component_registry.registerAllComponents(allComponents)

return allComponents
