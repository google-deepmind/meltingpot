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

-- Tests for the `error_handling` module.
local meltingpot = 'meltingpot.lua.modules.'
local error_handling = require(meltingpot .. 'error_handling')

local strings = require 'common.strings'
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'


local tests = {}

local function _makeError(badFn, handler)
  -- Causes an error and returns the trace (up to this point in the stack.)
  current_trace = handler('')
  status, trace = xpcall(function() badFn() end, handler)
  -- Remove info for frames above the current one.
  ct = #current_trace
  t = #trace
  while current_trace:sub(ct, ct) == trace:sub(t, t) and t > 0 and ct > 0 do
    ct = ct - 1
    t = t - 1
  end
  return status, trace:sub(1, t)
end

function tests.noAttachedInfoMatchesOriginalTrace()
  local badFn = function() error('cool error') end

  error_handling.useCustomTraceback(false)
  status_original, error_original = _makeError(badFn, debug.traceback)
  asserts.EQ(status_original, false)

  error_handling.useCustomTraceback(true)
  status_custom, error_custom = _makeError(badFn, debug.traceback)
  asserts.EQ(status_custom, false)

  asserts.EQ(error_original, error_custom)
end

function tests.attachedInfoIncludedInTrace()
  local badFn = function()
    error_handling.tryWithErrorInfo(
        function() error('this is an error!') end,
        function() return 'some debug info' end)
  end

  error_handling.useCustomTraceback(false)
  status_original, error_original = _makeError(badFn, debug.traceback)
  asserts.EQ(status_original, false)

  error_handling.useCustomTraceback(true)
  status_custom, error_custom = _makeError(badFn, debug.traceback)
  asserts.EQ(status_custom, false)

  asserts.NE(error_original, error_custom)
  asserts.hasSubstr(error_custom, 'some debug info')
end

return test_runner.run(tests)
