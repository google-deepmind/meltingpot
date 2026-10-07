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

local components = require 'meltingpot.lua.levels.predator_prey.components'
local stamina = require 'meltingpot.lua.levels.stamina.shared_components'
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'
local tests = {}

local function fixture(victimPredator, protected)
  local players = {}
  for index = 1, 4 do
    local state = {reward = 0, transitions = 0, freezes = 0, arms = 0}
    local isPredator = index == 1 or index == 3 or
        (index == 2 and victimPredator)
    local role = components.Role{isPredator = isPredator}
    local energy = stamina.Stamina{
        maxStamina = 20, amountInvisible = 18, amountGreen = 1,
        amountYellow = 1, amountRed = 1, costlyActions = {'move'},
        classConfig = {name = 'test', greenFreezeTime = 0,
                       yellowFreezeTime = 0, redFreezeTime = 0},
    }
    energy:reset()
    local edible = components.AvatarEdible{predatorRewardForEating = 2.5}
    edible:reset()
    local avatar = {
        getIndex = function() return index end,
        getAliveState = function() return 'alive' end,
        getWaitState = function() return 'wait' end,
        getSpawnGroup = function() return 'spawn' end,
        addReward = function(_, amount) state.reward = state.reward + amount end,
        disallowMovementUntil = function() state.freezes = state.freezes + 1 end,
    }
    local members = {
        Role = role, Stamina = energy, Avatar = avatar, AvatarEdible = edible,
        InteractEatAcorn = not isPredator and
            {isEating = function() return false end} or nil,
        AvatarAnimation = {armsUp = function() state.arms = state.arms + 1 end},
        PredatorInteractBeam = {showForDuration = function() end},
        Transform = {queryDisc = function()
          if protected then return {players[1].object, players[2].object,
                                    players[4].object} end
          return {players[1].object, players[2].object}
        end},
    }
    local object = {
        getLayer = function() return 'upperPhysical' end,
        getComponent = function(_, key) return members[key] end,
        hasComponent = function(_, key) return members[key] ~= nil end,
        setState = function(_, value)
          -- Native state changes are deferred; alive() changes immediately.
          state.transitions = state.transitions + 1
          state.requestedState = value
        end,
        teleportToGroup = function() state.requestedState = 'alive' end,
    }
    for _, member in pairs(members) do member.gameObject = object end
    players[index] = {object = object, edible = edible, stamina = energy,
                      state = state}
  end
  return players
end

for _, victimPredator in ipairs({false, true}) do
  for _, sameSource in ipairs({false, true}) do
    local name = tostring(victimPredator) .. '_' .. tostring(sameSource)
    tests['duplicateHits_' .. name] = function()
      local p = fixture(victimPredator, false)
      p[2].edible:onHit(p[1].object, 'predator')
      p[2].edible:onHit(p[sameSource and 1 or 3].object, 'predator')
      asserts.EQ(p[2].state.transitions, 1)
      asserts.EQ(p[2].edible:alive(), false)
      if victimPredator then
        asserts.EQ(p[1].stamina:getValue(), 16)
        asserts.EQ(p[3].stamina:getValue(), 20)
        asserts.EQ(p[1].state.reward, 0)
      else
        asserts.EQ(p[1].state.reward, 2.5)
        asserts.EQ(p[3].state.reward, 0)
        asserts.EQ(p[1].state.freezes, 1)
        asserts.EQ(p[3].state.freezes, 0)
      end
    end
  end
  tests['respawnAllowsAnotherConsumption_' .. tostring(victimPredator)] = function()
    local p = fixture(victimPredator, false)
    p[2].edible:onHit(p[1].object, 'predator')
    local respawner = components.AvatarRespawn{framesTillRespawn = 2}
    respawner.gameObject = p[2].object
    local callback
    respawner:registerUpdaters{registerUpdater = function(_, updater)
      callback = updater.updateFn
    end}
    callback()
    asserts.EQ(p[2].edible:alive(), true)
    p[2].edible:onHit(p[3].object, 'predator')
    asserts.EQ(p[2].state.transitions, 2)
    if victimPredator then
      asserts.EQ(p[1].stamina:getValue(), 16)
      asserts.EQ(p[3].stamina:getValue(), 16)
    else
      asserts.EQ(p[1].state.reward, 2.5)
      asserts.EQ(p[3].state.reward, 2.5)
    end
  end
end

function tests.protectedPreyRemainAliveAndUnrewarded()
  local p = fixture(false, true)
  for _ = 1, 2 do p[2].edible:onHit(p[1].object, 'predator') end
  asserts.EQ(p[2].edible:alive(), true)
  asserts.EQ(p[2].state.transitions, 0)
  asserts.EQ(p[1].state.reward, 0)
  asserts.EQ(p[2].state.arms, 2)
  asserts.EQ(p[4].state.arms, 2)
end

function tests.unrelatedBeamsDoNotConsumePlayers()
  local p = fixture(false, false)
  asserts.EQ(p[2].edible:onHit(p[1].object, 'other'), nil)
  asserts.EQ(p[2].state.transitions, 0)
  asserts.EQ(p[2].edible:alive(), true)
end

return test_runner.run(tests)
