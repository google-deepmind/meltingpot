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

local library = require 'meltingpot.lua.modules.component_library'
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'
local tests = {}

for _, pair in ipairs({{1, 5}, {3, 5}, {5, 5}, {7, 5}, {10, 3}, {2, 1}}) do
  local minimum, interval = unpack(pair)
  tests['minimum_' .. minimum .. '_interval_' .. interval] = function()
    local ending = library.StochasticIntervalEpisodeEnding{
        minimumFramesPerEpisode = minimum, intervalLength = interval,
        probabilityTerminationPerInterval = 1.0,
    }
    local frame, ended = 0, nil
    ending.gameObject = {simulation = {endEpisode = function()
      ended = frame
    end}}
    local updater
    ending:registerUpdaters{registerUpdater = function(_, value)
      updater = value
    end}
    asserts.EQ(updater.startFrame, minimum)
    local expected = math.ceil(minimum / interval) * interval
    for episode = 1, 2 do
      ending:reset()
      ended = nil
      frame = 0
      for tick = 1, expected do
        frame = tick
        ending:update()
        if frame >= updater.startFrame then updater.updateFn() end
        if frame < expected then asserts.EQ(ended, nil) end
      end
      asserts.EQ(ended, expected)
    end
  end
end

return test_runner.run(tests)
