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

-- The preference bonus requires a unique maximum over every resource.
local components = require 'meltingpot.lua.levels.the_matrix.components'
local tensor = require 'system.tensor'
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'
local tests = {}

local function taste(preferred, zeroDefault, extraReward)
  local result = components.InteractionTaste{
      mostTastyResourceClass = preferred,
      extraReward = extraReward or 7,
      zeroDefaultInteractionReward = zeroDefault or false,
  }
  result:reset()
  return result
end

function tests.earlierLargerResourcePreventsBonus()
  local inventory = tensor.DoubleTensor{2, 5, 1}
  asserts.EQ(taste(1):getExtraRewardForInteraction(3, inventory), 3)
end

function tests.earlierTiedResourcePreventsBonus()
  local inventory = tensor.DoubleTensor{2, 2, 1}
  asserts.EQ(taste(1):getExtraRewardForInteraction(3, inventory), 3)
end

function tests.zeroDefaultDoesNotTurnALosingPreferenceIntoABonus()
  local inventory = tensor.DoubleTensor{2, 5, 1}
  asserts.EQ(taste(1, true):getExtraRewardForInteraction(3, inventory), 0)
end

function tests.resourcePermutationsPreserveReward()
  -- The preferred resource always has quantity two and loses to quantity five.
  for _, case in ipairs({
      {{2, 5, 1}, 1}, {{2, 1, 5}, 1},
      {{5, 2, 1}, 2}, {{1, 2, 5}, 2},
      {{5, 1, 2}, 3}, {{1, 5, 2}, 3},
  }) do
    asserts.EQ(taste(case[2]):getExtraRewardForInteraction(
        3, tensor.DoubleTensor(case[1])), 3)
  end
end

function tests.uniqueMaximumGetsPositiveOrNegativeBonus()
  for _, extra in ipairs({7, -4}) do
    local inventory = tensor.DoubleTensor{5, 2, 1}
    local before = inventory:clone()
    asserts.EQ(taste(1, false, extra):getExtraRewardForInteraction(
        3, inventory), 3 + extra)
    asserts.EQ(taste(1, true, extra):getExtraRewardForInteraction(
        3, inventory), extra)
    asserts.tablesEQ(inventory:val(), before:val())
  end
end

function tests.disabledPreferenceAndResetRemainUnchanged()
  local preference = taste(-1, true)
  asserts.EQ(preference:getExtraRewardForInteraction(
      3, tensor.DoubleTensor{0, 0, 0}), 3)
  preference = taste(2)
  for _ = 1, 2 do
    preference:reset()
    asserts.EQ(preference:getExtraRewardForInteraction(
        3, tensor.DoubleTensor{1, 5, 1}), 10)
  end
end

function tests.exhaustiveSmallInventoriesMatchUniqueMaximumOracle()
  -- Independent oracle: count occurrences of the global maximum.
  for numResources = 2, 4 do
    for encoded = 0, 3 ^ numResources - 1 do
      local remaining = encoded
      local values = {}
      for index = 1, numResources do
        values[index] = remaining % 3
        remaining = math.floor(remaining / 3)
      end
      local maximum = math.max(unpack(values))
      local numMaxima = 0
      for _, value in ipairs(values) do
        if value == maximum then numMaxima = numMaxima + 1 end
      end
      for preferred = 1, numResources do
        for _, zeroDefault in ipairs({false, true}) do
          local baseReward = zeroDefault and 0 or 3
          local getsBonus = numMaxima == 1 and values[preferred] == maximum
          local expected = baseReward + (getsBonus and 7 or 0)
          local actual = taste(preferred, zeroDefault):getExtraRewardForInteraction(
              3, tensor.DoubleTensor(values))
          asserts.EQ(actual, expected)
        end
      end
    end
  end
end

return test_runner.run(tests)
