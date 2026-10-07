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

local components = require 'meltingpot.lua.levels.the_matrix.components'
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'
local tests = {}

local function fixture()
  local resolved = {}
  local matrix = components.TheMatrix{
      matrix = {{1, 0}, {0, 1}},
      resultIndicatorColorIntervals = {{-100, 100}},
  }
  local simulation = {
      getNumPlayers = function() return 4 end,
      getSceneObject = function()
        return {getComponent = function() return matrix end}
      end,
  }
  matrix.gameObject = {simulation = simulation}
  matrix:reset()
  local players = {}
  for index = 1, 4 do
    matrix.playerCollectedAtLeastOneResource[index] = true
    local avatar = {
        getIndex = function() return index end,
        getAliveState = function() return 'alive' end,
        getWaitState = function() return 'wait' end,
        addReward = function() end,
    }
    local zapper = components.GameInteractionZapper{
        cooldownTime = 2, beamLength = 3, beamRadius = 0,
        framesTillRespawn = 2, numResources = 2,
    }
    local object = {
        simulation = simulation,
        hasComponent = function() return false end,
        getComponent = function(_, name)
          if name == 'Avatar' then return avatar end
          if name == 'GameInteractionZapper' then return zapper end
          error('Unexpected component: ' .. name)
        end,
    }
    avatar.gameObject = object
    zapper.gameObject = object
    zapper:start()
    zapper._resolve = function(_, row, column)
      resolved[#resolved + 1] = {row, column}
    end
    local callbacks = {}
    zapper:registerUpdaters{registerUpdater = function(_, updater)
      callbacks[updater.priority] = updater.updateFn
    end}
    callbacks[890]()
    players[index] = {object = object, zapper = zapper,
                      nextFrame = callbacks[890]}
  end
  return players, resolved
end

for _, targetBusy in ipairs({false, true}) do
  for _, sourceBusy in ipairs({false, true}) do
    local name = 'reservation_' .. tostring(targetBusy) .. '_' ..
        tostring(sourceBusy)
    tests[name] = function()
      local p = fixture()
      p[1].zapper.interactedThisStep = targetBusy
      p[2].zapper.interactedThisStep = sourceBusy
      local blocked = p[1].zapper:_preventExtraSimultaneousInteraction(
          p[2].object)
      asserts.EQ(blocked, targetBusy or sourceBusy)
      if blocked then
        asserts.EQ(p[1].zapper.interactedThisStep, targetBusy)
        asserts.EQ(p[2].zapper.interactedThisStep, sourceBusy)
      else
        asserts.EQ(p[1].zapper.interactedThisStep, true)
        asserts.EQ(p[2].zapper.interactedThisStep, true)
      end
    end
  end
end

local function checkDisjointPairs(a, b, c, d)
  local p, resolved = fixture()
  asserts.EQ(p[b].zapper:onHit(p[a].object, 'gameInteraction'), true)
  asserts.EQ(p[c].zapper:onHit(p[a].object, 'gameInteraction'), true)
  asserts.EQ(#resolved, 1)
  asserts.EQ(p[c].zapper.interacted_this_step, 0)
  asserts.EQ(p[c].zapper:onHit(p[d].object, 'gameInteraction'), true)
  asserts.tablesEQ(resolved, {{a, b}, {d, c}})
  for _, player in ipairs(p) do
    asserts.EQ(player.zapper.interacted_this_step, 1)
  end
end

function tests.rejectedBeamDoesNotConsumeAnUninvolvedPlayer()
  checkDisjointPairs(1, 2, 3, 4)
end

function tests.playerPermutationsPreserveDisjointInteractions()
  for a = 1, 4 do
    for b = 1, 4 do
      for c = 1, 4 do
        for d = 1, 4 do
          if a ~= b and a ~= c and a ~= d and
              b ~= c and b ~= d and c ~= d then
            checkDisjointPairs(a, b, c, d)
          end
        end
      end
    end
  end
end

function tests.duplicateAndReverseHitsStayBlockedUntilNextFrame()
  local p, resolved = fixture()
  p[2].zapper:onHit(p[1].object, 'gameInteraction')
  p[2].zapper:onHit(p[1].object, 'gameInteraction')
  p[1].zapper:onHit(p[2].object, 'gameInteraction')
  asserts.EQ(#resolved, 1)
  for _, player in ipairs(p) do
    player.nextFrame()
    player.zapper:update()
  end
  p[1].zapper:onHit(p[2].object, 'gameInteraction')
  asserts.tablesEQ(resolved, {{1, 2}, {2, 1}})
end

function tests.unrelatedBeamsDoNotReservePlayers()
  local p, resolved = fixture()
  asserts.EQ(p[2].zapper:onHit(p[1].object, 'otherBeam'), nil)
  asserts.EQ(#resolved, 0)
  asserts.EQ(p[1].zapper.interactedThisStep, false)
  asserts.EQ(p[2].zapper.interactedThisStep, false)
end

return test_runner.run(tests)
