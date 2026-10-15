--[[ Copyright 2020 DeepMind Technologies Limited.

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

-- The need signal saturates without changing hunger penalties.
local avatars = require 'meltingpot.lua.modules.avatar_library'
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'
local tests = {}

local function fixture(delay, reward)
  local state = {alive = true, rewards = {}}
  local need = avatars.PeriodicNeed{delay = delay, reward = reward}
  need.gameObject = {getComponent = function()
    return {
        isAlive = function() return state.alive end,
        addReward = function(_, value) table.insert(state.rewards, value) end,
    }
  end}
  need:reset()
  return need, state
end

function tests.needSaturatesAfterTheConfiguredDelay()
  for _, delay in ipairs({1, 2, 5, 2.5}) do
    local need = fixture(delay, -1)
    for tick = 0, 20 do
      local expected = math.min(tick / delay, 1)
      asserts.LE(math.abs(need:getNeed() - expected), 1e-12)
      asserts.GE(need:getNeed(), 0)
      asserts.LE(need:getNeed(), 1)
      need:update()
    end
  end
end

function tests.penaltiesContinueOnEveryHungryFrame()
  for _, reward in ipairs({-3, 0, 2}) do
    local need, state = fixture(3, reward)
    for tick = 1, 10 do
      need:update()
      need:getNeed()
      need:getNeed()  -- Reading the signal must not change the reward schedule.
      asserts.EQ(#state.rewards, math.max(tick - 2, 0))
    end
    for _, actual in ipairs(state.rewards) do asserts.EQ(actual, reward) end
    asserts.EQ(need:getNeed(), 1)
  end
end

function tests.satisfyingTheNeedRestartsTheWholeCountdown()
  local need, state = fixture(2, -1)
  for _ = 1, 7 do need:update() end
  asserts.EQ(need:getNeed(), 1)
  asserts.EQ(#state.rewards, 6)
  need:resetDriveLevel()
  asserts.EQ(need:getNeed(), 0)
  need:update()
  asserts.EQ(need:getNeed(), 0.5)
  asserts.EQ(#state.rewards, 6)
  need:update()
  asserts.EQ(need:getNeed(), 1)
  asserts.EQ(#state.rewards, 7)
  need:reset()
  asserts.EQ(need:getNeed(), 0)
end

function tests.inactiveAvatarsDoNotExposeTheNeedSignal()
  local need, state = fixture(2, -1)
  for _ = 1, 5 do need:update() end
  state.alive = false
  asserts.EQ(need:getNeed(), 0)
  state.alive = true
  asserts.EQ(need:getNeed(), 1)
end

return test_runner.run(tests)
