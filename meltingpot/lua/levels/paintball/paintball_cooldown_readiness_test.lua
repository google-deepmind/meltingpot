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

local components = require 'meltingpot.lua.levels.paintball.shared_components'

local function zapper(primary, secondary)
  local part = components.ColorZapper{
    team = 'red', color = {255, 0, 0}, cooldownTime = primary,
    beamLength = 3, beamRadius = 1, secondaryBeamCooldownTime = secondary,
    secondaryBeamLength = 6, secondaryBeamRadius = 0, aliveStates = {'health2'},
  }
  local action = {fireZap = 0}
  local object = {shots = 0, position = {1, 1}}
  local transform = {
    getPosition = function() return {object.position[1], object.position[2]} end,
    queryPosition = function() return nil end,
  }
  local avatar = {getVolatileData = function() return {actions = action} end}
  function object:getComponent(name)
    return name == 'Transform' and transform or avatar
  end
  function object:getState() return 'health2' end
  function object:hitBeam(_, length) self.shots = self.shots + 1; self.length = length end
  part.gameObject = object
  part:reset()
  local update
  part:registerUpdaters{registerUpdater = function(_, entry) update = entry.updateFn end}
  local function step(value)
    action.fireZap = value
    update()
    return part:readyToShoot()
  end
  step(0)  -- The environment initializes the previous position at reset.
  return part, object, step
end

function tests.secondaryCooldownUsesItsOwnDuration()
  local part, object, step = zapper(2, 4)
  asserts.EQ(step(2), 0)
  for i = 1, 4 do
    asserts.EQ(step(2), i / 4)
    asserts.EQ(object.shots, 1)
  end
  asserts.EQ(step(2), 0)
  asserts.EQ(object.shots, 2)
end

function tests.primaryCooldownIsUnchanged()
  local part, object, step = zapper(2, 4)
  asserts.EQ(step(1), 0)
  asserts.EQ(step(1), 0.5)
  asserts.EQ(step(1), 1)
  asserts.EQ(object.shots, 1)
  asserts.EQ(step(1), 0)
  asserts.EQ(object.shots, 2)
end

function tests.switchingBeamTypesSwitchesTheNormalization()
  local part, object, step = zapper(2, 4)
  step(2)
  for _ = 1, 4 do step(0) end
  asserts.EQ(step(1), 0)
  asserts.EQ(step(0), 0.5)
  asserts.EQ(step(0), 1)
  asserts.EQ(step(2), 0)
  asserts.EQ(step(0), 0.25)
  asserts.EQ(object.shots, 3)
end

function tests.episodeResetRestoresReadiness()
  local part, _, step = zapper(2, 4)
  step(2)
  part:reset()
  asserts.EQ(part:readyToShoot(), 1)
  step(0)
  asserts.EQ(step(1), 0)
  asserts.EQ(step(0), 0.5)
end

function tests.movingStillPreventsLongRangeShots()
  local part, object, step = zapper(2, 4)
  object.position = {2, 1}
  asserts.EQ(step(2), 1)
  asserts.EQ(object.shots, 0)
  asserts.EQ(step(2), 0)
  asserts.EQ(object.shots, 1)
end

function tests.disabledAndInstantCooldownsRemainFinite()
  local disabled, object, step = zapper(-1, 4)
  asserts.EQ(disabled:readyToShoot(), 0)
  for _ = 1, 3 do step(1); step(2) end
  asserts.EQ(object.shots, 0)
  local instant, shots, advance = zapper(0, 0)
  asserts.EQ(instant:readyToShoot(), 1)
  for _ = 1, 3 do asserts.EQ(advance(1), 1) end
  asserts.EQ(shots.shots, 3)
end

function tests.equalDurationsKeepTheOriginalProgress()
  local _, _, step = zapper(4, 4)
  asserts.EQ(step(2), 0)
  for i = 1, 4 do asserts.EQ(step(0), i / 4) end
end

return test_runner.run(tests)
