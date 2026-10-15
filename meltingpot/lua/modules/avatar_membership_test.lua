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

-- Avatar ID observations represent membership, not query hit counts.
local avatars = require 'meltingpot.lua.modules.avatar_library'
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'
local tests = {}

local function actor(index)
  return {
      hasComponent = function(_, name) return name == 'Avatar' end,
      getComponent = function() return {getIndex = function() return index end} end,
  }
end

local function fixture()
  local state = {ids = {2, 2, 3}, layer = 'upperPhysical', hits = {}}
  local avatar = {
      getIndex = function() return 1 end,
      queryPartialObservationWindow = function(_, layer)
        return state.hits[layer] or {}
      end,
  }
  local zapper = {getZappablePlayerIndices = function() return state.ids end}
  local object = {
      getLayer = function() return state.layer end,
      getComponent = function(_, name)
        if name == 'Avatar' then return avatar else return zapper end
      end,
      simulation = {getNumPlayers = function() return 4 end},
  }
  local view = avatars.AvatarIdsInViewObservation{layers = {'upperPhysical'}}
  local range = avatars.AvatarIdsInRangeToZapObservation{}
  view.gameObject = object
  range.gameObject = object
  range:reset()
  return view, range, state
end

function tests.repeatedViewHitsAndLayersRemainBinary()
  local view, _, state = fixture()
  local second = actor(2)
  state.hits.upperPhysical = {second, second, actor(3)}
  state.hits.lowerPhysical = {second}
  local result = view:_getQueryResult(
      {'upperPhysical', 'upperPhysical', 'lowerPhysical'})
  asserts.tablesEQ(result:val(), {0, 1, 1, 0})
  asserts.EQ(#state.hits.upperPhysical, 3)
end

function tests.viewIgnoresNonAvatarsAndClearsMissingIds()
  local view, _, state = fixture()
  state.hits.upperPhysical = {actor(1), actor(4),
      {hasComponent = function() return false end}}
  asserts.tablesEQ(view:_getQueryResult({'upperPhysical'}):val(), {1, 0, 0, 1})
  state.hits.upperPhysical = {}
  asserts.tablesEQ(view:_getQueryResult({'upperPhysical'}):val(), {0, 0, 0, 0})
  asserts.tablesEQ(view:_getQueryResult({}):val(), {0, 0, 0, 0})
end

function tests.repeatedZapRaysRemainBinaryAndDoNotChangeTheQuery()
  local _, range, state = fixture()
  asserts.tablesEQ(range:_getQueryResult():val(), {0, 1, 1, 0})
  asserts.tablesEQ(state.ids, {2, 2, 3})
  state.ids = {4}
  asserts.tablesEQ(range:_getQueryResult():val(), {0, 0, 0, 1})
  state.ids = {}
  asserts.tablesEQ(range:_getQueryResult():val(), {0, 0, 0, 0})
end

function tests.inactiveZapObservationIsZeroAndResetIsReusable()
  local _, range, state = fixture()
  state.layer = ''
  asserts.tablesEQ(range:_getQueryResult():val(), {0, 0, 0, 0})
  state.layer = 'upperPhysical'
  range:reset()
  asserts.tablesEQ(range:_getQueryResult():val(), {0, 1, 1, 0})
end

function tests.registeredCallbacksExposeTheBinaryInt32Vectors()
  local view, range, state = fixture()
  state.hits.upperPhysical = {actor(2), actor(2)}
  local observations = {}
  view:addObservations(nil, nil, observations)
  range:addObservations(nil, nil, observations)
  asserts.EQ(observations[1].name, '1.AVATAR_IDS_IN_VIEW')
  asserts.EQ(observations[2].name, '1.AVATAR_IDS_IN_RANGE_TO_ZAP')
  for _, observation in ipairs(observations) do
    asserts.EQ(observation.type, 'tensor.Int32Tensor')
    asserts.tablesEQ(observation.shape, {4})
  end
  asserts.tablesEQ(observations[1].func():val(), {0, 1, 0, 0})
  asserts.tablesEQ(observations[2].func():val(), {0, 1, 1, 0})
end

return test_runner.run(tests)
