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

-- Tests for the `component_registry` module.
local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')

local class = require 'common.class'
local helpers = require 'common.helpers'
local log = require 'common.log'
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'


local tests = {}

function tests.testUsage()
  asserts.tablesEQ(component_registry.getAllComponents(), {})
  asserts.shouldFail(
      function() component_registry.getComponent('Something') end)

  local component_library = require 'modules.component_library'
  local component = component_registry.getComponent('AdditionalSprites')

  asserts.tablesEQ(component, component_library.AdditionalSprites)

  component = class.Class(component.Component)
  component_registry.registerComponent('Empty', component)
  asserts.tablesEQ(component_registry.getComponent('Empty'), component)

  -- Test that the registry actually contains _all_ the components exported by
  -- the library, as well as the manually added one.
  local allComponents = component_library
  allComponents['Empty'] = component
  asserts.tablesEQ(component_registry.getAllComponents(), allComponents)

end

return test_runner.run(tests)
