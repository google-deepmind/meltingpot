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

local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'
local tests = {}

local components = require 'meltingpot.lua.levels.territory.components'

local function resource(health)
  local part = components.Resource{
    initialHealth = health, destroyedState = 'destroyed', reward = 2,
    rewardRate = 1, rewardDelay = 0, delayTillSelfRepair = 0,
    selfRepairProbability = 1,
  }
  local object = {state = 'unclaimed', changes = 0}
  function object:getState() return self.state end
  function object:setState(state)
    self.pending = state
    self.changes = self.changes + 1
  end
  local texture = {changes = 0}
  function texture:setState(_) self.changes = self.changes + 1 end
  local indicator = {changes = 0}
  function indicator:setState(_) self.changes = self.changes + 1 end
  part.gameObject = object
  part:reset()
  part._numPlayers = 2
  part._texture_object = texture
  part._associatedDamageIndicator = indicator
  return part, object, texture, indicator
end

local function player(index)
  local avatar = {rewards = 0, getIndex = function() return index end}
  function avatar:addReward(reward) self.rewards = self.rewards + reward end
  function avatar:isWait() return false end
  local object = {
    getComponent = function(_, name) if name == 'Avatar' then return avatar end end,
    hasComponent = function() return false end,
  }
  avatar.gameObject = object
  return object, avatar
end

for _, health in ipairs({1, 2, 3}) do
  tests['destroyedResourcesStayTransparent' .. health] = function()
    local part, object, texture, indicator = resource(health)
    local hitter = player(1)
    for n = 1, health do
      asserts.EQ(part:onHit(hitter, 'zapHit'), n < health)
    end
    asserts.EQ(object:getState(), 'unclaimed')
    local pending, changes = object.pending, object.changes
    for _ = 1, 2 * health do
      asserts.EQ(part:onHit(hitter, 'zapHit'), false)
    end
    asserts.EQ(object.pending, pending)
    asserts.EQ(object.changes, changes)
    asserts.EQ(texture.changes, 1)
    asserts.EQ(indicator.changes, 1)
    asserts.EQ(part._health, health)
  end
end

function tests.claimsCannotReplaceDestroyedOwnership()
  local part = resource(1)
  local owner, avatar = player(1)
  local stranger = player(2)
  part:onHit(owner, 'claimBeam_1')
  part:onHit(owner, 'zapHit')
  asserts.EQ(part:onHit(stranger, 'claimBeam_2'), false)
  asserts.EQ(part:onHit(stranger, 'directionHit2'), false)
  asserts.EQ(part._claimedByAvatarComponent, avatar)
  asserts.EQ(part:getRewardingStatus(), 'inactive')
end

function tests.destroyedResourcesDoNotPayBeforeTheQueuedStateApplies()
  local part = resource(1)
  local owner, avatar = player(1)
  part:onHit(owner, 'claimBeam_1')
  local updates = {}
  part:registerUpdaters({registerUpdater = function(_, spec)
    updates[#updates + 1] = spec
  end})
  updates[1].updateFn()
  asserts.EQ(avatar.rewards, 2)
  part:onHit(owner, 'zapHit')
  updates[1].updateFn()
  asserts.EQ(avatar.rewards, 2)
  asserts.EQ(part:getRewardingStatus(), 'inactive')
end

function tests.nonlethalDamageStillBlocksAndRepairs()
  local part, object = resource(3)
  local hitter = player(1)
  asserts.EQ(part:onHit(hitter, 'zapHit'), true)
  asserts.EQ(part._health, 2)
  part:update()
  asserts.EQ(part._health, 3)
  asserts.EQ(part._destroyed, false)
  asserts.EQ(object.changes, 0)
end

function tests.otherBeamsAndResetKeepOriginalBehavior()
  local part, object = resource(1)
  local hitter = player(1)
  asserts.EQ(part:onHit(hitter, 'unknown'), false)
  asserts.EQ(part._health, 1)
  asserts.EQ(object.changes, 0)
  part:onHit(hitter, 'zapHit')
  part:reset()
  asserts.EQ(part._destroyed, false)
  asserts.EQ(part:onHit(hitter, 'zapHit'), false)
  asserts.EQ(object.changes, 2)
end

return test_runner.run(tests)
