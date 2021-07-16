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

--[[ This module is used to register a global dictionary of components.  This is
used by the Base Simulation to initialise components whenever a new one is
requested by name.
]]

local ComponentRegistry = {}

-- Register a component in the component registry.
local function registerComponent(name, component)
  ComponentRegistry[name] = component
end

-- Register all components in the table into the component registry.
local function registerAllComponents(component_map)
  for name, component in pairs(component_map) do
    ComponentRegistry[name] = component
  end
end

-- Gets a previously registered component from the registry.
local function getComponent(componentName)
  if ComponentRegistry[componentName] ~= nil then
    return ComponentRegistry[componentName]
  end
  error('The component ' .. componentName .. ' does not exist in the registry.')
end

-- Get all components in the registry.
local function getAllComponents()
  return ComponentRegistry
end

return {
  registerComponent = registerComponent,
  registerAllComponents = registerAllComponents,
  getComponent = getComponent,
  getAllComponents = getAllComponents,
}
