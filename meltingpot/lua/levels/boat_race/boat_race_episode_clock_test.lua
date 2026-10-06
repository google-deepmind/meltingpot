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

local components = require 'meltingpot.lua.levels.boat_race.components'

local function manager(earlyExitOnAny)
  local players = {{state = 'player'}, {state = 'player'}}
  for _, player in ipairs(players) do
    function player:getState() return self.state end
  end
  local simulation = {ends = 0}
  function simulation:getGameObjectsByName(_) return players end
  function simulation:endEpisode() self.ends = self.ends + 1 end
  local part = components.EpisodeManager{checkInterval = 5, earlyExitOnAny = earlyExitOnAny}
  part.gameObject = {simulation = simulation}
  local update
  part:registerUpdaters{registerUpdater = function(_, entry) update = entry.updateFn end}
  local function reset()
    -- Match GameObject.reset: reset is an optional component lifecycle hook.
    if part.reset then part:reset() end
    simulation.ends = 0
  end
  reset()
  return part, players, simulation, update, reset
end

for _, any in ipairs({false, true}) do
  for _, length in ipairs({1, 2, 4, 6}) do
    tests['resetClock_' .. tostring(any) .. '_' .. length] = function()
      local _, players, simulation, update, reset = manager(any)
      for _ = 1, length do update() end
      players[1].state = 'playerWait'
      players[2].state = 'playerWait'
      reset()
      update()
      asserts.EQ(simulation.ends, 1)
      for _ = 1, 4 do update() end
      asserts.EQ(simulation.ends, 1)
      update()
      asserts.EQ(simulation.ends, 2)
    end
  end
end

function tests.uninterruptedEpisodeKeepsItsExistingSchedule()
  local _, players, simulation, update = manager(false)
  players[1].state, players[2].state = 'playerWait', 'playerWait'
  for step = 1, 12 do
    update()
    asserts.EQ(simulation.ends, math.floor((step - 1) / 5) + 1)
  end
end

function tests.livePlayersDoNotEndAnAllDisqualifiedEpisode()
  local _, players, simulation, update, reset = manager(false)
  players[1].state = 'playerWait'
  for _ = 1, 7 do update() end
  reset()
  for _ = 1, 7 do update() end
  asserts.EQ(simulation.ends, 0)
end

function tests.anyDisqualifiedModeRemainsDistinct()
  local _, players, simulation, update, reset = manager(true)
  players[1].state = 'playerWait'
  update()
  asserts.EQ(simulation.ends, 1)
  reset()
  update()
  asserts.EQ(simulation.ends, 1)
end

return test_runner.run(tests)
