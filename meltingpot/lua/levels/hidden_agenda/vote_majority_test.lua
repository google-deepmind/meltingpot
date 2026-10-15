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

local components = require 'meltingpot.lua.levels.hidden_agenda.components'
local tensor = require 'system.tensor'
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'
local tests = {}

local function fixture(activeCount)
  local progress = components.Progress{
      num_players = 5, potential_pseudorewards = false,
      teleport_spawn_group = 'spawn', voting_params = {type = 'continuous'},
  }
  local roles = {}
  for index = 1, 5 do
    local role = components.Role{
        role = index == 5 and 'impostor' or 'crewmate', frozenState = 'frozen',
    }
    if index > activeCount then role:inactivatePlayer() end
    roles[index] = role
    local avatar = {getIndex = function() return index end}
    progress.avatars[index] = {getComponent = function(_, name)
      if name == 'Role' then return role end
      if name == 'Avatar' then return avatar end
      error('Unexpected component: ' .. name)
    end}
  end
  progress.votingMatrix = tensor.DoubleTensor(5, 7):fill(0)
  progress.votingCounter = {0, 0, 0, 0, 0}
  for index = 1, 5 do
    progress:submitVote(index, index <= activeCount and 6 or 7, 'disable')
  end
  return progress, roles
end

for activeCount = 0, 5 do
  for votes = 0, activeCount do
    tests['active_' .. activeCount .. '_votes_' .. votes] = function()
      local progress = fixture(activeCount)
      for voter = 1, votes do progress:submitVote(voter, 1, 'vote') end
      local before = progress.votingMatrix:clone()
      local expected = votes > activeCount - votes and 1 or 0
      asserts.EQ(progress:getPlayerVotedOff(), expected)
      asserts.tablesEQ(progress.votingMatrix:val(), before:val())
    end
  end
end

function tests.splitVoteDoesNotFavorTheLowerPlayerIndex()
  for left = 1, 4 do
    for right = 1, 4 do
      if left ~= right then
        local progress = fixture(4)
        for voter = 1, 4 do
          progress:submitVote(voter, voter <= 2 and left or right, 'vote')
        end
        asserts.EQ(progress:getPlayerVotedOff(), 0)
      end
    end
  end
end

function tests.activeRoleChangesUpdateTheDenominator()
  local progress, roles = fixture(5)
  roles[2]:inactivatePlayer()
  progress:submitVote(2, 7, 'disable')
  progress:submitVote(1, 3, 'vote')
  progress:submitVote(3, 3, 'vote')
  asserts.EQ(progress:getPlayerVotedOff(), 0)
  roles[4]:inactivatePlayer()
  progress:submitVote(4, 7, 'disable')
  asserts.EQ(progress:getPlayerVotedOff(), 3)
  roles[2]:activatePlayer()
  progress:submitVote(2, 6, 'disable')
  asserts.EQ(progress:getPlayerVotedOff(), 0)
end

function tests.abstentionsDoNotBecomeCandidates()
  local progress = fixture(5)
  for index = 1, 5 do progress:submitVote(index, 6, 'vote') end
  asserts.EQ(progress:getPlayerVotedOff(), 0)
end

return test_runner.run(tests)
