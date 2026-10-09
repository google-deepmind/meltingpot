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

local components = require 'meltingpot.lua.levels.paintball.shared_components'
local tensor = require 'system.tensor'

local function player(index, withTaste)
  local object = {}
  local avatar = {reward = 0, gameObject = object}
  function avatar:getIndex() return index end
  function avatar:addReward(value) self.reward = self.reward + value end
  local taste = {calls = 0}
  function taste:zap(_) self.calls = self.calls + 1; avatar:addReward(5) end
  function object:getComponent(name)
    if name == 'Avatar' then return avatar end
    if name == 'Taste' and withTaste then return taste end
  end
  function object:hasComponent(name) return name == 'Taste' and withTaste end
  return object, avatar, taste
end

local function victim(health)
  local object = {state = 'health1', changes = 0}
  local avatar = {reward = 0, gameObject = object}
  function avatar:getIndex() return 3 end
  function avatar:addReward(value) self.reward = self.reward + value end
  function avatar:getAliveState() return 'health1' end
  function avatar:getWaitState() return 'playerWait' end
  function avatar:getSpawnGroup() return 'spawn' end
  function avatar:allowMovement() end
  local transform = {queryPosition = function() return nil end}
  function object:getComponent(name) return name == 'Avatar' and avatar or transform end
  function object:getState() return self.state end
  function object:setState(state) self.pending = state; self.changes = self.changes + 1 end
  function object:flush() self.state = self.pending; self.pending = nil end
  function object:teleportToGroup(_, state) self.state = state end
  local part = components.ZappedByColor{
    team = 'blue', allTeamNames = {'red', 'blue'}, framesTillRespawn = 2,
    penaltyForBeingZapped = -2, rewardForZapping = 3, healthRegenerationRate = 0,
    maxHealthOnGround = health, maxHealthOnOwnColor = health, maxHealthOnEnemyColor = health,
  }
  part.gameObject = object
  part:reset()
  part.playerZapMatrix = tensor.Int32Tensor(3, 3):fill(0)
  local updates = {}
  part:registerUpdaters{registerUpdater = function(_, entry) updates[entry.priority] = entry.updateFn end}
  return part, object, avatar, updates
end

for _, order in ipairs({{1, 2}, {2, 1}}) do
  for _, health in ipairs({1, 2, 3}) do
    tests['oneElimination_' .. order[1] .. '_' .. health] = function()
      local part, object, target = victim(health)
      local p1, a1 = player(1, false)
      local p2, a2 = player(2, false)
      local players, avatars = {p1, p2}, {a1, a2}
      for _ = 1, health do part:onHit(players[order[1]], 'red') end
      asserts.EQ(object:getState(), 'health1')  -- setState has not been applied.
      asserts.EQ(target.reward, -2)
      part:onHit(players[order[2]], 'red')
      part:onHit(players[order[1]], 'red')
      asserts.EQ(target.reward, -2)
      asserts.EQ(avatars[order[1]].reward, 3)
      asserts.EQ(avatars[order[2]].reward, 0)
      asserts.EQ(part.zapperIndex, order[1])
      asserts.EQ(part.playerZapMatrix(3, order[1]):val(), 1)
      asserts.EQ(part.playerZapMatrix(3, order[2]):val(), 0)
      asserts.EQ(object.changes, 1)
    end
  end
end

function tests.tasteRewardIsAlsoDeliveredOnce()
  local part, _, target = victim(1)
  local hitter, avatar, taste = player(1, true)
  for _ = 1, 4 do part:onHit(hitter, 'red') end
  asserts.EQ(taste.calls, 1)
  asserts.EQ(avatar.reward, 5)
  asserts.EQ(target.reward, -2)
end

function tests.respawnAllowsAnotherElimination()
  local part, object, target, update = victim(1)
  local hitter, attacker = player(1, false)
  part:onHit(hitter, 'red')
  object:flush()
  part:onHit(hitter, 'red')
  asserts.EQ(target.reward, -2)
  update[135]()  -- Actual registered respawn callback restores health.
  part:onHit(hitter, 'red')
  asserts.EQ(target.reward, -4)
  asserts.EQ(attacker.reward, 6)
  asserts.EQ(part.playerZapMatrix(3, 1):val(), 2)
end

function tests.regenerationInWaitDoesNotMakeTheAvatarHittable()
  local part, object, target, update = victim(1)
  local hitter, attacker = player(1, false)
  part:onHit(hitter, 'red')
  object:flush()
  update[2]()  -- Existing regeneration can run while the avatar is waiting.
  part:onHit(hitter, 'red')
  asserts.EQ(target.reward, -2)
  asserts.EQ(attacker.reward, 3)
end

function tests.friendlyAndUnrecognizedBeamsDoNotDamage()
  local part, object, target = victim(2)
  local hitter, attacker = player(1, false)
  part:onHit(hitter, 'blue')
  part:onHit(hitter, 'other')
  part:onHit(hitter, 'red')
  asserts.EQ(target.reward, 0)
  asserts.EQ(object.changes, 0)
  part:onHit(hitter, 'red')
  asserts.EQ(target.reward, -2)
  asserts.EQ(attacker.reward, 3)
end

function tests.episodeResetRestoresHealth()
  local part, object, target = victim(1)
  local hitter, attacker = player(1, false)
  part:onHit(hitter, 'red')
  part:reset()
  object.state = 'health1'
  part:onHit(hitter, 'red')
  asserts.EQ(target.reward, -4)
  asserts.EQ(attacker.reward, 6)
end

return test_runner.run(tests)
