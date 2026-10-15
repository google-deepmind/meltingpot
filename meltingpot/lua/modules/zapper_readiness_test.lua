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

-- Readiness must agree with zero-cooldown and disabled firing.
local avatars = require 'meltingpot.lua.modules.avatar_library'
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'
local tests = {}

local function fixture(cooldown)
  local state = {alive = true, hits = 0}
  local avatar = {
      isAlive = function() return state.alive end,
      getAliveState = function() return 'alive' end,
      getWaitState = function() return 'wait' end,
      getVolatileData = function() return {actions = {fireZap = 1}} end,
  }
  local zapper = avatars.Zapper{
      cooldownTime = cooldown, beamLength = 3, beamRadius = 1,
      framesTillRespawn = 2, penaltyForBeingZapped = 0, rewardForZapping = 0,
  }
  zapper.gameObject = {
      getComponent = function() return avatar end,
      hitBeam = function() state.hits = state.hits + 1 end,
      simulation = {getSceneObject = function()
        return {hasComponent = function() return false end}
      end},
  }
  zapper:reset()
  zapper:start()
  local updaters = {}
  zapper:registerUpdaters({registerUpdater = function(_, updater)
    updaters[updater.priority] = updater.updateFn
  end})
  return zapper, state, updaters[140]
end

function tests.zeroCooldownReportsReadyAndFiresEveryFrame()
  local zapper, state, fire = fixture(0)
  for frame = 1, 5 do
    asserts.EQ(zapper:readyToShoot(), 1)
    fire()
    asserts.EQ(state.hits, frame)
    zapper:update()
    asserts.EQ(zapper:readyToShoot(), 1)
  end
end

function tests.negativeCooldownReportsDisabledAndDoesNotFire()
  for _, cooldown in ipairs({-1, -5}) do
    local zapper, state, fire = fixture(cooldown)
    for _ = 1, 4 do
      asserts.EQ(zapper:readyToShoot(), 0)
      fire()
      zapper:update()
    end
    asserts.EQ(state.hits, 0)
  end
end

function tests.positiveCooldownRetainsItsProgressSignal()
  local zapper, state, fire = fixture(4)
  asserts.EQ(zapper:readyToShoot(), 1)
  fire()
  asserts.EQ(state.hits, 1)
  for tick = 0, 4 do
    asserts.EQ(zapper:readyToShoot(), tick / 4)
    if tick < 4 then fire() end
  end
  asserts.EQ(state.hits, 1)
  fire()
  asserts.EQ(state.hits, 2)
end

function tests.inactiveAvatarsReportZeroForEveryCooldown()
  for _, cooldown in ipairs({-1, 0, 4}) do
    local zapper, state, fire = fixture(cooldown)
    state.alive = false
    asserts.EQ(zapper:readyToShoot(), 0)
    fire()
    asserts.EQ(state.hits, 0)
  end
end

function tests.zeroCooldownRespectsRemainingPreventionTimer()
  local zapper, state, fire = fixture(0)
  zapper:disallowZapping()
  zapper:update()
  asserts.EQ(zapper:readyToShoot(), 0)
  zapper:allowZapping()
  fire()  -- Consume the existing prevention timer without firing.
  asserts.EQ(state.hits, 0)
  asserts.EQ(zapper:readyToShoot(), 1)
  fire()
  asserts.EQ(state.hits, 1)
end

return test_runner.run(tests)
