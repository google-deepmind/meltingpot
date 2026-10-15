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

-- Matrix inventories remain visible until the scheduled interaction effects.
local components = require 'meltingpot.lua.levels.the_matrix.components'
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'
local tests = {}

local function fixture(rowWins, delay, zeroInitial, resetWinner, resetLoser)
  local players = {}
  local matrix = components.TheMatrix{
      matrix = {{0, 3}, {1, 0}},
      columnPlayerMatrix = {{0, 1}, {3, 0}},
      zeroInitialInventory = zeroInitial,
      resultIndicatorColorIntervals = {{-1e6, 1e6}},
  }
  local scene = {getComponent = function(_, name)
    asserts.EQ(name, 'TheMatrix')
    return matrix
  end}
  local simulation = {
      getNumPlayers = function() return 2 end,
      getSceneObject = function() return scene end,
      getAvatarFromIndex = function(_, index) return players[index] end,
  }
  matrix.gameObject = {simulation = simulation}
  matrix:reset()
  local zappers, avatars, effects = {}, {}, {}
  for index = 1, 2 do
    local avatar = {reward = 0, frozenFor = 0}
    function avatar:addReward(amount) self.reward = self.reward + amount end
    function avatar:disallowMovementUntil(frames) self.frozenFor = frames end
    function avatar:getIndex() return index end
    function avatar:getAliveState() return 'alive' end
    function avatar:getWaitState() return 'wait' end
    local zapper = components.GameInteractionZapper{
        cooldownTime = 2, beamLength = 3, beamRadius = 1,
        framesTillRespawn = 2, numResources = 2,
        freezeOnInteraction = delay,
        reset_winner_inventory = resetWinner,
        reset_loser_inventory = resetLoser,
        losingPlayerDies = false, winningPlayerDies = false,
    }
    -- The same taste on both players isolates reset timing from dispatch.
    local taste = components.InteractionTaste{
        mostTastyResourceClass = 1, extraReward = 10,
    }
    taste:reset()
    local members = {Avatar = avatar, GameInteractionZapper = zapper,
                     InteractionTaste = taste}
    local object = {
        simulation = simulation,
        hasComponent = function(_, name) return members[name] ~= nil end,
        getComponent = function(_, name) return assert(members[name]) end,
    }
    avatar.gameObject = object
    zapper.gameObject = object
    players[index] = object
    avatars[index] = avatar
    zappers[index] = zapper
    zapper:start()
    zapper:postStart()
    zapper:registerUpdaters({registerUpdater = function(_, updater)
      if updater.priority == 4 then effects[index] = updater.updateFn end
    end})
  end
  matrix:getPlayerInventory(1):val(rowWins and {7, 0} or {0, 7})
  matrix:getPlayerInventory(2):val(rowWins and {0, 7} or {7, 0})
  matrix.playerCollectedAtLeastOneResource = {true, true}
  return matrix, players, avatars, zappers, effects
end

-- Cover either winner, both reset flags, both initialization policies, and delays.
for _, rowWins in ipairs({false, true}) do
  for _, delay in ipairs({0, 1, 3}) do
    for _, zeroInitial in ipairs({false, true}) do
      for _, resetWinner in ipairs({false, true}) do
        for _, resetLoser in ipairs({false, true}) do
          local label = string.format(
              'rowWins_%s_delay_%d_zero_%s_winner_%s_loser_%s',
              tostring(rowWins), delay, tostring(zeroInitial),
              tostring(resetWinner), tostring(resetLoser))
          tests[label] = function()
            local matrix, players, avatars, zappers, effects = fixture(
                rowWins, delay, zeroInitial, resetWinner, resetLoser)
            local before = {
                matrix:getPlayerInventory(1):clone():val(),
                matrix:getPlayerInventory(2):clone():val(),
            }
            zappers[2]:_resolve(1, 2, players[1])
            for index = 1, 2 do
              asserts.tablesEQ(
                  matrix:getPlayerInventory(index):val(), before[index])
              asserts.EQ(matrix.playerCollectedAtLeastOneResource[index], true)
              asserts.EQ(avatars[index].reward, 0)
              asserts.EQ(avatars[index].frozenFor, delay + 2)
            end
            for _ = 1, delay do
              effects[2]()
              effects[1]()
              for index = 1, 2 do
                asserts.tablesEQ(
                  matrix:getPlayerInventory(index):val(), before[index])
                asserts.EQ(avatars[index].reward, 0)
              end
            end
            effects[2]()
            effects[1]()
            -- Unique preferred resource grants a bonus before either reset.
            asserts.EQ(avatars[1].reward, rowWins and 13 or 1)
            asserts.EQ(avatars[2].reward, rowWins and 1 or 13)
            for index = 1, 2 do
              local won = (index == 1) == rowWins
              local reset = (won and resetWinner) or (not won and resetLoser)
              local initial = zeroInitial and {0, 0} or {1, 1}
              asserts.tablesEQ(matrix:getPlayerInventory(index):val(),
                  reset and initial or before[index])
              asserts.EQ(matrix.playerCollectedAtLeastOneResource[index], not reset)
            end
            -- Scheduled effects are consumed exactly once.
            effects[2]()
            effects[1]()
            asserts.EQ(avatars[1].reward, rowWins and 13 or 1)
            asserts.EQ(avatars[2].reward, rowWins and 1 or 13)
          end
        end
      end
    end
  end
end

return test_runner.run(tests)
