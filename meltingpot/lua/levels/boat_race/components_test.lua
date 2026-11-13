--[[ Copyright 2026 DeepMind Technologies Limited.

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

local components = require 'meltingpot.lua.levels.boat_race.components'
local test_runner = require 'testing.test_runner'

local tests = {}

local function makeAvatar()
  local avatar = {state = 'player', blocked = 0}
  local avatarComponent = {
      disallowMovement = function() avatar.blocked = avatar.blocked + 1 end,
  }
  local rowingComponent = {
      setSeat = function(self, seat) self.seat = seat end,
  }
  function avatar:getState() return self.state end
  function avatar:setState(state) self.state = state end
  function avatar:getComponent(name)
    if name == 'Avatar' then return avatarComponent end
    if name == 'Rowing' then return rowingComponent end
  end
  return avatar, rowingComponent
end

local function makeBoat()
  local boat = {rowers = {}, reports = 0, fullTransitions = 0}
  function boat:reportRower(side, avatar)
    self.rowers[side] = avatar
    self.reports = self.reports + 1
  end
  function boat:isSeatTaken(side) return self.rowers[side] ~= nil end
  function boat:isFull() return self.rowers.L and self.rowers.R end
  function boat:setBoatStateFull()
    self.fullTransitions = self.fullTransitions + 1
  end
  function boat:reset() self.rowers = {} end
  return boat
end

local function makeSeat(boat, side)
  local seat = setmetatable({
      _manager = boat,
      _seatSide = side,
      _config = {rowingState = 'rowing'},
  }, {__index = components.Seat})
  local gameObject = {state = 'seat', connected = {}}
  function gameObject:getState() return self.state end
  function gameObject:setState(state) self.pendingState = state end
  function gameObject:connect(avatar)
    table.insert(self.connected, avatar)
  end
  seat.gameObject = gameObject
  return seat
end

function tests.firstRowerRetainsSeatBeforeQueuedStateChange()
  local boat = makeBoat()
  local left = makeSeat(boat, 'L')
  local first = makeAvatar()
  local second, rowing = makeAvatar()

  left:onEnter(first, 'avatar')
  left:onEnter(second, 'avatar')  -- Engine has not applied seatTaken yet.

  assert(boat.rowers.L == first)
  assert(boat.reports == 1)
  assert(#left.gameObject.connected == 1)
  assert(second.state == 'player')
  assert(second.blocked == 0)
  assert(rowing.seat == nil)

  boat:reset()
  left:onEnter(second, 'avatar')
  assert(boat.rowers.L == second)
  assert(boat.reports == 2)
end

function tests.seatMembershipResetsWithBoat()
  local boat = setmetatable({_rowers = {}}, {__index = components.BoatManager})
  assert(not boat:isSeatTaken('L'))
  assert(not boat:isSeatTaken('R'))
  boat._rowers.L = {}
  assert(boat:isSeatTaken('L'))
  assert(not boat:isSeatTaken('R'))
  boat:reset()
  assert(not boat:isSeatTaken('L'))
end

function tests.fullBoatRegistersOnlyOnce()
  local boat = makeBoat()
  local left = makeSeat(boat, 'L')
  local right = makeSeat(boat, 'R')
  left:onEnter(makeAvatar(), 'avatar')
  right:onEnter(makeAvatar(), 'avatar')
  left:onEnter(makeAvatar(), 'avatar')
  assert(boat.fullTransitions == 1)
end

return test_runner.run(tests)
