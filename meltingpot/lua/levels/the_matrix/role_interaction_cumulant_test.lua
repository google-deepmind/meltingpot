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
local library = require 'meltingpot.lua.modules.component_library'
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'
local tests = {}

local function fixture(sourceRole, targetRole)
  local resolved = {}
  local matrix = components.TheMatrix{
      matrix = {{2, 0}, {0, 2}},
      resultIndicatorColorIntervals = {{-100, 100}},
  }
  local scene = {getComponent = function() return matrix end}
  local simulation = {getSceneObject = function() return scene end,
                      getNumPlayers = function() return 2 end}
  matrix.gameObject = {simulation = simulation}
  matrix:reset()
  matrix.playerCollectedAtLeastOneResource = {true, true}
  local players = {}
  for index, role in ipairs({sourceRole, targetRole}) do
    local avatar = {
        getIndex = function() return index end,
        addReward = function(_, amount) asserts.EQ(amount, 0) end,
    }
    local zapper = components.GameInteractionZapper{
        cooldownTime = 2, beamLength = 3, beamRadius = 0,
        framesTillRespawn = 2, numResources = 2,
    }
    local members = {Avatar = avatar, GameInteractionZapper = zapper}
    if role ~= 'absent' then
      members.DyadicRole = components.DyadicRole{rowPlayer = role == 'row'}
    end
    local object = {
        simulation = simulation,
        getComponent = function(_, name) return members[name] end,
        hasComponent = function(_, name) return members[name] ~= nil end,
    }
    for _, member in pairs(members) do member.gameObject = object end
    zapper:start()
    zapper.interactedThisStep = false
    zapper._resolve = function(_, row, column)
      for _, player in ipairs(players) do
        asserts.EQ(player.zapper.interacted_this_step, 1)
      end
      resolved[#resolved + 1] = {row, column}
    end
    local reporter = library.AvatarMetricReporter{metrics = {{
        name = 'INTERACTED', type = 'tensor.Int32Tensor', shape = {},
        component = 'GameInteractionZapper', variable = 'interacted_this_step',
    }}}
    reporter.gameObject = object
    local observations = {}
    reporter:addObservations(nil, nil, observations)
    players[index] = {object = object, zapper = zapper,
                      observation = observations[1].func}
  end
  return players, resolved, matrix
end

for _, source in ipairs({'row', 'column', 'absent'}) do
  for _, target in ipairs({'row', 'column', 'absent'}) do
    tests['roles_' .. source .. '_' .. target] = function()
      local p, resolved, matrix = fixture(source, target)
      local before = matrix.playerResources:clone()
      local blocked = source == target and source ~= 'absent'
      asserts.EQ(p[2].zapper:onHit(p[1].object, 'gameInteraction'), true)
      asserts.EQ(#resolved, blocked and 0 or 1)
      for _, player in ipairs(p) do
        asserts.EQ(player.zapper.interacted_this_step, blocked and 0 or 1)
        asserts.EQ(player.observation(), blocked and 0 or 1)
      end
      if not blocked then
        if source == 'column' and target == 'row' then
          asserts.tablesEQ(resolved[1], {2, 1})
        else
          asserts.tablesEQ(resolved[1], {1, 2})
        end
      end
      asserts.tablesEQ(matrix.playerResources:val(), before:val())
    end
  end
end

function tests.unreadyPlayersStillDoNotReportAnInteraction()
  local p, resolved, matrix = fixture('row', 'column')
  matrix.disallowUnreadyInteractions = true
  matrix.playerCollectedAtLeastOneResource[1] = false
  p[2].zapper:onHit(p[1].object, 'gameInteraction')
  asserts.EQ(#resolved, 0)
  asserts.EQ(p[1].observation(), 0)
  asserts.EQ(p[2].observation(), 0)
end

function tests.frozenTargetsStillDoNotReportAnInteraction()
  local p, resolved = fixture('column', 'row')
  p[2].zapper._framesTillScheduledEffects = 2
  p[2].zapper:onHit(p[1].object, 'gameInteraction')
  asserts.EQ(#resolved, 0)
  asserts.EQ(p[1].observation(), 0)
  asserts.EQ(p[2].observation(), 0)
end

function tests.unrelatedBeamsDoNotReportAnInteraction()
  local p, resolved = fixture('row', 'column')
  asserts.EQ(p[2].zapper:onHit(p[1].object, 'other'), nil)
  asserts.EQ(#resolved, 0)
  asserts.EQ(p[1].observation(), 0)
  asserts.EQ(p[2].observation(), 0)
end

return test_runner.run(tests)
