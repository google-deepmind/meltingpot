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

local error_handling = {}

--[[ Error handling functionality for Melting Pot.

This module replaces the DMLab2D traceback with one that offers additional
debugging functionality.

`error_handling` offers `tryWithErrorInfo` which allows the user to attach
a function that provides additional troubleshooting information to stack traces,
invoking the function only when an error occurs.

Usage example:

  error_handling = require('error_handling')

  local someLocalVariable = 3
  error_handling.tryWithErrorInfo(
      function() assert(someLocalVariable == 4) end,
      function() return "Some Local Variable: " .. someLocalVariable end
  )

In the above example, when the assertion is triggered the full stack trace
will be printed with the message "Some Local Variable: 3" included in the
call stack.

If you see the error "error in error handling", that is likely a bug in this
module. To disable this and revert to default behavior, call
`error_handling.useCustomTraceback(false)` anywhere before the error is
raised.
]]

local _ROOT_MATCHER = 'org_deepmind_lab2d/(.*)'

local function _makeError(msg)
  local ESCAPE = string.char(27)
  local RED = ESCAPE .. '[0;31m'
  local CLEAR = ESCAPE .. '[0;0m'
  return string.format("%sERROR:%s %s", RED, CLEAR, msg)
end

local function _shortenPath(path)
  return string.match(path, _ROOT_MATCHER) or path
end

local function _traceInfoAsString(traceinfo, errorInfo)
  -- Convert getinfo structure into a string for the traceback.
  local loc = ''
  local src = traceinfo.source
  if src:sub(1, 1) ~= '=' then
    loc = string.format("%s:%d:", _shortenPath(src), traceinfo.currentline)
  else
    loc = src:sub(2) .. ':'
  end
  if traceinfo.name then
    loc = loc .. string.format(' in function \'%s\'', traceinfo.name)
  end
  if errorInfo then
    return _makeError(loc) .. ' (' .. errorInfo .. ')'
  else
    return _makeError(loc)
  end
end

function error_handling.tryWithErrorInfo(tryFn, errorInfoFn)
  --[[ In the event of an error attaches additional info to the stack trace.

  This function passes simply passes through to the called function. In the
  event of an exception, `error_handling.traceback` extracts and calls
  `errorInfoFn`.
  ]]
  result = tryFn()
  -- Note: Spreading this over two lines ensures that this is not converted
  -- to a tail call which would make errorInfoFn inaccessible.
  return result
end

function error_handling.traceback(msg, level)
  --[[ Replacement function for `debug.traceback`.

  Prints out relevant information on the callstack, including any error
  information attached by tryWithErrorInfo.
  ]]

  -- Note: skip the first level. There will always be an extra entry in the
  -- stack for the error_handling.traceback function itself, which doesn't need
  -- to be printed as part of the traceback.
  level = (level or 1) + 1

  -- Collect the trace information at each level.
  traceinfo = {}
  errorinfo = {}
  while true do
    local levelinfo = debug.getinfo(level, 'Slnf')
    if levelinfo == nil then
      break
    end
    traceinfo[level] = levelinfo

    if levelinfo.func == error_handling.tryWithErrorInfo then
      -- Look for the arg `errorInfoFn` at this level of the stack.
      local name, errorInfoFn = debug.getlocal(level, 2)
      if name ~= 'errorInfoFn' then
        errorinfo[level + 1] = function() return 'error in tryWithErrorInfo' end
      elseif not errorInfoFn or type(errorInfoFn) ~= 'function' then
        errorinfo[level + 1] = function() return 'invalid errorInfoFn' end
      else
        errorinfo[level + 1] = errorInfoFn
      end
    end

    level = level + 1
  end

  -- Print line numbers, file info, etc for each line in the trace.
  -- If there is any debug info captured by tryWithErrorInfo, print that
  -- and skip the empty `tryWithErrorInfo` call.
  trace = {'stack trace-back:'}
  for level, levelinfo in pairs(traceinfo) do
    local errorInfoString = nil
    if errorinfo[level] ~= nil then
      -- Call `errorInfoFn` to collect the error info
      local _, valueOrError = pcall(errorinfo[level])
      errorInfoString = tostring(valueOrError)
    end
    -- If the original error message contains a path to an originating file,
    -- match the path formatting for other error lines.
    if levelinfo.source ~= levelinfo.short_src then
      msg = msg:gsub(levelinfo.short_src, _shortenPath(levelinfo.source))
    end

    table.insert(trace, _traceInfoAsString(levelinfo, errorInfoString))
  end

  return '\n' .. _makeError(msg) .. '\n' .. table.concat(trace, '\n')
end

local originalTraceback = debug.traceback
function error_handling.useCustomTraceback(useCustom)
  if useCustom then
    debug.traceback = error_handling.traceback
  else
    debug.traceback = originalTraceback
  end
end

-- Default behavior is to use this traceback as long as the module is imported.
error_handling.useCustomTraceback(true)

return error_handling
