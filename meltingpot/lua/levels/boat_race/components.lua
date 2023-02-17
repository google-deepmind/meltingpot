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

local args = require 'common.args'
local class = require 'common.class'
local helpers = require 'common.helpers'
local log = require 'common.log'
local random = require 'system.random'
local tensor = require 'system.tensor'
local events = require 'system.events'

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')


-- The BoatManager allows handling the objects comprising the boat as a single
-- entity. This component is added to the left-seat, and connects all the needed
-- objects so they can move as a unit. It also manages the effects of rowing.
local BoatManager = class.Class(component.Component)

function BoatManager:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('BoatManager')},
      {'flailEffectiveness', args.default(0.1), args.numberType},
      {'mismatchStrokePenalty', args.default(-0.5), args.numberType},
      {'mismatchRolePseudoreward', args.default(-5.0), args.numberType},
      {'matchRolePseudoreward', args.default(2.0), args.numberType},
  })
  BoatManager.Base.__init__(self, kwargs)

  self._otherSide = {L = 'R', R = 'L'}
  self._config.flailEffectiveness = kwargs.flailEffectiveness
  self._config.mismatchStrokePenalty = kwargs.mismatchStrokePenalty
  self._config.mismatchRolePseudoreward = kwargs.mismatchRolePseudoreward
  self._config.matchRolePseudoreward = kwargs.matchRolePseudoreward
end

function BoatManager:reset()
  self._rowers = {}
  self._strokes = {}
  self._seats = {}
end

function BoatManager:postStart()
  --  /\    Front of boat
  -- p;:q   Oars and seats
  --  LJ    Back of boat
  local transform = self.gameObject:getComponent('Transform')
  local position = transform:getPosition()
  -- Store the reference of seats on the boat manager
  table.insert(self._seats, self.gameObject)
  table.insert(self._seats, transform:queryPosition('lowerPhysical',
    {position[1] + 1, position[2]}))
  -- Calculate the key coordination of the boat
  local upperLeft = {position[1] - 1, position[2] - 1}
  local lowerRight = {position[1] + 2, position[2] + 1}
  -- Oars are on upperPhysical
  local oarsBoat = transform:queryRectangle('overlay', upperLeft, lowerRight)
  -- Hull is on lowerPhysical
  self._lowerBoat = transform:queryRectangle(
      'lowerPhysical', upperLeft, lowerRight)
  for _, go in pairs(oarsBoat) do
    self.gameObject:connect(go)
  end
  for _, go in pairs(self._lowerBoat) do
    self.gameObject:connect(go)
  end
  local scene = self.gameObject.simulation:getSceneObject()
  if scene:hasComponent("RaceManager") then
    self._raceManager = scene:getComponent("RaceManager")
  end
end

function BoatManager:reportRower(seatSide, avatar)
  self._rowers[seatSide] = avatar
  log.v(1, "Added rower to side", seatSide, "to BoatManager.")
  events:add('player_added_to_side', 'dict',
    'player_index', avatar:getComponent("Avatar"):getIndex(),
    'seatSide', seatSide)
end

function BoatManager:_reportAndClearStrokes()
  local globalStrokes = self.gameObject.simulation:getSceneObject():getComponent(
      "GlobalRaceTracker")
  for side, stroke in pairs(self._strokes) do
    local index = self._rowers[side]:getComponent('Avatar'):getIndex()
    local avatarStrokes = self._rowers[side]:getComponent(
        "StrokesTracker")
    globalStrokes:countStroke(index, stroke)
    avatarStrokes:countStroke(stroke)
  end
  self._strokes = {}  -- Empty strokes
end

function BoatManager:registerUpdaters(updaterRegistry)
  updaterRegistry:registerUpdater{
    updateFn = function()
      -- Apply agent role pseudo-rewards
      for side, stroke in pairs(self._strokes) do
        local role = self._rowers[side]:getComponent('Rowing'):getRole()
        log.v(2, "Processing role pseudo-rewards: ", side, stroke, role)
        if role ~= 'none' and role ~= stroke then
          self._rowers[side]:getComponent('Avatar'):addReward(
              self._config.mismatchRolePseudoreward)
        elseif role == stroke then
          self._rowers[side]:getComponent('Avatar'):addReward(
              self._config.matchRolePseudoreward)
        end
      end
      if self._strokes.L == 'row' and self._strokes.R == 'row' then
        -- Both efficient rowing causes deterministic movement.
        self.gameObject:moveAbs(self._raceManager:getRaceDirection())
      elseif self._strokes.L == 'flail' or self._strokes.R == 'flail' then
        -- There are 2 independent things that happen if either player flails:
        -- 1. The boat might move forward (stochastically)
        if random:uniformReal(0, 1) < self._config.flailEffectiveness then
          self.gameObject:moveAbs(self._raceManager:getRaceDirection())
        end
        -- 2. Another player rowing gets penalised for mismatched stroke
        for side, stroke in pairs(self._strokes) do
          if stroke == 'row' then
            self._rowers[side]:getComponent('Avatar'):addReward(
                self._config.mismatchStrokePenalty)
          end
        end
      end
      self:_reportAndClearStrokes()
    end,
  }
end

function BoatManager:isFull()
  return self._rowers["L"] and self._rowers["R"]
end

function BoatManager:_registerPair(pairsTensor)
  if self._rowers["L"] == nil or self._rowers["R"] == nil then
    return
  end
  local numPlayers = self.gameObject.simulation:getNumPlayers()
  local posX = self._rowers["L"]:getPosition()[1]
  -- Find out which boat is this pair at.
  -- TODO(b/260154779): Is there a more elegant way to do this? We'd have to
  -- know too much from the ASCII map to do it programatically. :/
  local boat = 1
  if posX >= 10 and posX < 14 then
    boat = 2
  elseif posX >= 14 then
    boat = 3
  end
  events:add('pair_started_race', 'dict',
    'race_number', self._raceManager:getRaceNumber(),
    'boat_index', boat,
    'player_L_index', self._rowers["L"]:getComponent("Avatar"):getIndex(),
    'player_R_index', self._rowers["R"]:getComponent("Avatar"):getIndex())
  -- Register the pair on the appropritate boat spot in the tensor.
  pairsTensor(boat, 1):val(
      self._rowers["L"]:getComponent("Avatar"):getIndex())
  pairsTensor(boat, 2):val(
      self._rowers["R"]:getComponent("Avatar"):getIndex())
end

function BoatManager:setBoatStateFull()
  for _, boat_part in pairs(self._lowerBoat) do
    if boat_part:getState() == "boat" then
     boat_part:setState("boatFull")
    end
  end
  -- Register the pair starting the race.
  local raceStart = self.gameObject.simulation:getSceneObject():getComponent(
      "GlobalRaceTracker").raceStart
  -- TODO(b/260154977): Sometimes this can happen twice in a single race.
  -- I suspect it is due to other agents entering the seat once it's taken.
  -- Should be fixed by adding a blocking goal object.
  self:_registerPair(raceStart)
end

function BoatManager:setBoatStateNormal()
  for _, boat_part in pairs(self._lowerBoat) do
    if boat_part:getState() == "boatFull" then
      boat_part:setState("boat")
    end
  end
end

function BoatManager:oarAction(seatSide, style)
  self._strokes[seatSide] = style
end

function BoatManager:disembarkRowers(targetY)
  for side, rower in pairs(self._rowers) do
    log.v(1, "Disembark rower: ", side)
    rower:getComponent("Rowing"):setSeat(nil)
    if rower:hasComponent("Crown") then
      local crown = rower:getComponent("Crown"):getCrownOverlay()
      crown:disconnect()
      rower:disconnect()
      rower:connect(crown)
    else
      rower:disconnect()
    end
    local targetPosition = {}
    targetPosition[1] = rower:getPosition()[1]
    targetPosition[2] = targetY
    rower:teleport(targetPosition, rower:getOrientation())
  end
end

-- The race manager lives in the Scene object, and manages the start of a race
-- by coordinating the barriers and semaphores.
local RaceManager = class.Class(component.Component)

function RaceManager:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('RaceManager')},
      -- Total time of race = raceStartTime + raceDuration.
      -- Time for the barriers (and semaphore) to lift ans start the race.
      {'raceStartTime', args.default(75), args.numberType},
      -- Time that a light change in the semaphore should remain.
      {'semaphoreTimer', args.default(5), args.numberType},
      -- The duration of the race once the race starts.
      {'raceDuration', args.default(225), args.numberType},
      -- The direction of the race.
      {'raceInitialDirection', args.default('N'), args.stringType},
  })
  RaceManager.Base.__init__(self, kwargs)

  self._otherSide = {N = 'S', S = 'N'}
  self._config.raceStartTime = kwargs.raceStartTime
  self._config.semaphoreTimer = kwargs.semaphoreTimer
  self._config.raceDuration = kwargs.raceDuration
  self._config.raceInitialDirection = kwargs.raceInitialDirection
  self._race_number = 0
end

function RaceManager:reset()
  self._raceDirection = self._config.raceInitialDirection
end

function RaceManager:getRaceNumber()
  return self._race_number
end

function RaceManager:registerUpdaters(updaterRegistry)
  local barriersToggle = function(state)
    local barriers = self.gameObject.simulation:getGameObjectsByName("barrier")
    for _, barrier in pairs(barriers) do
      if barrier:getState() ~= "on" then
        barrier:setState("on")
      else
        barrier:setState("off")
      end
    end
  end

  local semaphoreChange = function(color)
    local semaphores = self.gameObject.simulation:getGameObjectsByName(
        "semaphore")
    for _, semaphore in pairs(semaphores) do
      semaphore:setState(color)
    end
    self.gameObject:setState("semaphore_" .. color)
  end

  local disconnectComponentSafely = function(component, newState)
    component:disconnect()
    component:setState(newState)
    component:teleport({0, 0}, "N")
  end

  local playerClean = function()
    local players = self.gameObject.simulation:getGameObjectsByName("avatar")
    for i, player in pairs(players) do
      if player:getState() ~= "landed" then
        log.v(1, "Disqualifying player:", i, 'with state:', player:getState())
        events:add('player_disqualified', 'dict',
          'race_number', self._race_number,
          'player_index', i)
        if player:hasComponent("Crown") then
          local crown = player:getComponent("Crown"):getCrownOverlay()
          disconnectComponentSafely(crown, "crownWait")
        end
        disconnectComponentSafely(player, "playerWait")
        player:getComponent("Avatar"):disallowMovement()
      else
        events:add('player_ended_race', 'dict',
          'race_number', self._race_number,
          'player_index', i)
        player:setState("player")
      end
    end
  end

  local boatReset = function()
    local l_seats = self.gameObject.simulation:getGameObjectsByName("seat_L")
    local r_seats = self.gameObject.simulation:getGameObjectsByName("seat_R")
    for _, seat in pairs(l_seats) do
      seat:setState("seat")
      local boat_manager = seat:getComponent("BoatManager")
      assert(boat_manager ~= nil, "The boat manager is not set!")
      boat_manager:reset()
    end
    for _, seat in pairs(r_seats) do
      seat:setState("seat")
    end
  end

  local appleSpawnBankFlip = function()
    -- We need to modify both the current apple state as well as its state if
    -- the Edible onEnter is triggered because the apple will spawn if there
    -- is a player standing on apple when its state gets updated.
    local apples = self.gameObject.simulation:getGameObjectsByName("apple")
    for _, apple in pairs(apples) do
      if apple:getState() ~= "applePause" then
        apple:setState("applePause")
        apple:getComponent("Edible"):setWaitState("applePause")
      else
        apple:setState("apple")
        apple:getComponent("Edible"):setWaitState("appleWait")
      end
    end
    local single_apples = self.gameObject.simulation:getGameObjectsByName(
      "single_apple")
    for _, single_apple in pairs(single_apples) do
      single_apple:setState("apple")
    end
  end

  local goalReset = function()
    local goals = self.gameObject.simulation:getGameObjectsByName("water_goal")
    for _, goal in pairs(goals) do
      local transform = goal:getComponent('Transform')
      local position = transform:getPosition()
      local boat = transform:queryPosition('lowerPhysical', position)
      local water_goal = goal:getComponent('WaterGoal')
      if boat ~= nil and
        self:getRaceDirection() ~= water_goal._config.bank_side then
        log.v(1, "Found boat object with state at start: ", boat:getState())
        goal:setState("goalNonBlocking")
      else
        log.v(1, "Not found boat or goal at start, block entrance...")
        goal:setState("goalBlocking")
      end
    end
  end

  local raceReady = function() semaphoreChange("yellow") end

  local raceStart = function()
    semaphoreChange("green")
    barriersToggle()
    self.gameObject:setState("boatRace")
    -- Sentinel value to signal start of race
    self.gameObject:getComponent("GlobalRaceTracker").raceStart:fill(-1)
    -- increase number of races
    self._race_number = self._race_number + 1
    events:add('race_start', 'dict', 'race_number', self._race_number)
  end

  local raceEnd = function()
    semaphoreChange("red")
    playerClean()
  end

  local raceReset = function()
    self._raceDirection = self._otherSide[self._raceDirection]
    boatReset()
    goalReset()
    appleSpawnBankFlip()
    self.gameObject:setState("partnerChoice")
  end

  local forceEmbark = function()
    local players = {}
    for _, player in pairs(
      self.gameObject.simulation:getAvatarGameObjects("avatar")) do
      table.insert(players, player)
    end
    random:shuffleInPlace(players)
    local l_seats = self.gameObject.simulation:getGameObjectsByName("seat_L")
    for i, l_seat in pairs(l_seats) do
      local boat_manager = l_seat:getComponent("BoatManager")
      local seat = random:choice(boat_manager._seats)
      local transform = seat:getComponent('Transform')
      local position = transform:getPosition()
      players[i]:teleport({position[1], position[2]}, "N")
    end
    self.gameObject:setState("partnerChoice")
  end

  updaterRegistry:registerUpdater{
    state = "ForceEmbark",
    startFrame = 1,
    updateFn = forceEmbark,
  }
  updaterRegistry:registerUpdater{
    state = "partnerChoice",
    startFrame = self._config.raceStartTime - 2 * self._config.semaphoreTimer,
    updateFn = raceReady,
  }
  updaterRegistry:registerUpdater{
    state = "semaphore_yellow",
    startFrame = self._config.semaphoreTimer,
    updateFn = raceStart,
  }
  updaterRegistry:registerUpdater{
    state = "boatRace",
    startFrame = self._config.raceDuration,
    updateFn = raceEnd,
    priority = 200,
  }
  updaterRegistry:registerUpdater{
    state = "semaphore_red",
    updateFn = raceReset,
    priority = 50,
  }
end

function RaceManager:getRaceDirection()
  return self._raceDirection
end


-- The `EpisodeManager` will periodically monitor whether all (or any) players
-- have been disqualified and terminate the episode early if this is the case.
-- Episode lengths will always be a multiple of the `checkInterval`
-- parameter.
local EpisodeManager = class.Class(component.Component)

function EpisodeManager:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('EpisodeManager')},
      {'checkInterval', args.numberType},
      -- End episode if _any_ player is disqualified.
      {'earlyExitOnAny', args.default(false), args.booleanType},
  })
  EpisodeManager.Base.__init__(self, kwargs)
  self._config.checkInterval = kwargs.checkInterval
  self._config.earlyExitOnAny = kwargs.earlyExitOnAny
  self._step = 0
end

function EpisodeManager:registerUpdaters(updaterRegistry)
  local earlyExit = function()
    if self._step % self._config.checkInterval == 0 then
      local players = self.gameObject.simulation:getGameObjectsByName("avatar")
      local anyDisqualified = false
      local allDisqualified = true
      for i, player in pairs(players) do
        if player:getState() == "playerWait" then
          anyDisqualified = true
        end
        if player:getState() ~= "playerWait" then
          allDisqualified = false
        end
      end
      if allDisqualified and not self._config.earlyExitOnAny then
        log.v(1, "All players disqualified, ending episode.")
        self.gameObject.simulation:endEpisode()
      end
      if anyDisqualified and self._config.earlyExitOnAny then
        log.v(1, "Some players disqualified, ending episode.")
        self.gameObject.simulation:endEpisode()
      end
    end

    self._step = self._step + 1

  end

  updaterRegistry:registerUpdater{
      updateFn = earlyExit,
      priority = 100,
  }

end


--[[ The `Crown` component keeps track of the number of times an avatar has
rowed and flailed. If the ratio of rowing to flailing is within the config
thresholds, a crown overlay is displayed on top of the avatar. The crown
sprite can be configured to be invisible.
]]
local Crown = class.Class(component.Component)
function Crown:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Crown')},
      -- Min rowing to flailing ratio to turn on crown
      {'turnOnThreshold', args.numberType},
      -- Max rowing to flailing ratio to turn off crown
      {'turnOffThreshold', args.numberType},
      -- weighting decrease for the exponential moving average
      {'alpha', args.numberType},
      -- weighting decrease for the exponential moving average
      {'beta', args.numberType},
  })
  Crown.Base.__init__(self, kwargs)
  assert(kwargs.turnOnThreshold > kwargs.turnOffThreshold,
         "Crown turnOnThreshold should be strictly larger than turnOnThreshold")
  self._config.turnOnThreshold = kwargs.turnOnThreshold
  self._config.turnOffThreshold = kwargs.turnOffThreshold
  self._config.alpha = kwargs.alpha
  self._config.beta = kwargs.beta
  self._num_rows = 0
  self._num_flails = 0
  self._num_actions = 0
  self._mean = 0
end

function Crown:start()
  self._avatarComponent = self.gameObject:getComponent('Avatar')
end

function Crown:recordAction(action)
  local action_value = 0
  if action == 'flail' then
    self._num_flails = self._num_flails + 1
  elseif action == 'row' then
    self._num_rows = self._num_rows + 1
    action_value = 1
  else
    return
  end
  self._mean = self._config.alpha * action_value
      + (1 - self._config.alpha) * self._mean
  self._num_actions = self._num_actions + 1
end

function Crown:getCrownOverlay()
  local overlayObject = self._avatarComponent
    :getAllConnectedObjectsWithNamedComponent('AvatarConnector')
  if #overlayObject == 1 then
    return overlayObject[1]
  end
end

function Crown:update()
  if self._crown == nil then
    self._crown = self:getCrownOverlay()
  end
  self._mean = self._mean * (1 - self._config.beta)
  if self._mean > self._config.turnOnThreshold and
    self._crown:getState() == "crownOff" then
    events:add("turn_on_crown", "avg paddle", self._mean,
      "num_rows", self._num_rows, "num_flails", self._num_flails,
      "player_index", self._avatarComponent:getIndex())
    self._crown:setState("crownOn")
  elseif self._mean < self._config.turnOffThreshold and
    self._crown:getState() == "crownOn" then
    events:add("turn_off_crown", "avg paddle", self._mean,
      "num_rows", self._num_rows, "num_flails", self._num_flails,
      "player_index", self._avatarComponent:getIndex())
    self._crown:setState("crownOff")
  end
end

--[[ The `Rowing` component endows an avatar with the ability to row the boat.
There are two actions, 'row' and 'flail'. Flailing means a secure progress but
without much efficiency. Rowing means that if both players perform it, movement
is much faster. Mismatches result in the agent rowing potentially incurring a
negative reward.
]]
local Rowing = class.Class(component.Component)

function Rowing:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Rowing')},
      -- Cooldown for the rowing action.
      {'cooldownTime', args.numberType},
      {'playerRowingState', args.default('rowing'), args.stringType},
      {'playerRole', args.default('none'), args.stringType},
  })
  Rowing.Base.__init__(self, kwargs)

  self._config.cooldownTime = kwargs.cooldownTime
  self._config.playerRowingState = kwargs.playerRowingState
  self._config.playerRole = kwargs.playerRole
end

function Rowing:postStart()
  local scene = self.gameObject.simulation:getSceneObject()
  self._raceManager = scene:getComponent("RaceManager")
end

function Rowing:getRole()
  return self._config.playerRole
end

function Rowing:registerUpdaters(updaterRegistry)
  local flailing = function()
    local playerVolatileVariables = (
        self.gameObject:getComponent('Avatar'):getVolatileData())
    local actions = playerVolatileVariables.actions
    -- Execute the beam if applicable.
    if self.gameObject:getState() == self._config.playerRowingState then
      if actions['flail'] == 1 and self._seat ~= nil then
        local moved = self._seat:moveOar('flail')
        if moved then
          events:add('player_flailed', 'dict',
            'race_number', self._raceManager:getRaceNumber(),
            'player_index', self.gameObject:getComponent("Avatar"):getIndex())
          if self.gameObject:hasComponent('Crown') then
            local crown = self.gameObject:getComponent('Crown')
            crown:recordAction('flail')
          end
        end
      end
    end
  end

  local rowing = function()
    local playerVolatileVariables = (
        self.gameObject:getComponent('Avatar'):getVolatileData())
    local actions = playerVolatileVariables.actions
    -- Execute the beam if applicable.
    if self.gameObject:getState() == self._config.playerRowingState then
      if self._coolingTimer > 0 then
        self._coolingTimer = self._coolingTimer - 1
        if self._coolingTimer == 0 and self._seat ~= nil then
          self._seat:moveOar(nil)
        end
      end
      if self._coolingTimer == 0 and actions['row'] == 1 then
        self._coolingTimer = self._config.cooldownTime
        if self._seat ~= nil then
          local moved = self._seat:moveOar('row')
          if moved then
            events:add('player_rowed', 'dict',
             'race_number', self._raceManager:getRaceNumber(),
             'player_index', self.gameObject:getComponent("Avatar"):getIndex())
            if self.gameObject:hasComponent('Crown') then
              local crown = self.gameObject:getComponent('Crown')
              crown:recordAction('row')
            end
          end
        end
      end
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = rowing,
      priority = 140,
  }
  updaterRegistry:registerUpdater{
      updateFn = flailing,
      priority = 130,
  }
end

function Rowing:start()
  -- Set the beam cooldown timer to its `ready` state (i.e. coolingTimer = 0).
  self._coolingTimer = 0
end

function Rowing:setSeat(seat)
  self._seat = seat
end


local Seat = class.Class(component.Component)

function Seat:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Seat')},
      {'rowingState', args.default('rowing'), args.stringType},
  })
  Seat.Base.__init__(self, kwargs)

  self._config.rowingState = kwargs.rowingState
end

function Seat:postStart()
  local transform = self.gameObject:getComponent('Transform')
  --  /\    Front of boat
  -- p;:q   Oars and seats
  --  LJ    Back of boat
  -- Oars are on overlay, rest of boat is lowerPhysical.
  local oar = transform:queryDiamond('overlay', 1)
  assert(#oar == 1,
         "Exactly one 'overlay' object expected around the seat, " .. #oar
         .. " found instead. There should only be one Oar!")
  self._oar = oar[1]
  if self.gameObject:hasComponent("BoatManager") then
    self._manager = self.gameObject:getComponent("BoatManager")
  else
    -- The manager is on the seat tot he right
    local position = transform:getPosition()
    position[1] = position[1] - 1
    self._manager = transform:queryPosition(
        'lowerPhysical', position):getComponent("BoatManager")
  end
  self._seatSide = string.gsub(self.gameObject.name, "seat_", "")
end

function Seat:update()
  if self._oar:getState() == "oarUp_row" or
      self._oar:getState() == "oarUp_flail" then
    self._oar:setState("oarDown")
  end
end

function Seat:getOar()
  return self._oar
end

function Seat:moveOar(style)
  if not self._manager:isFull() then
    -- did not move oar
    return false
  end
  if style then  -- If not nil style, change state
    self._oar:setState("oarUp_" .. style)
  end
  -- The game object's name for the seat is "seat_L" or "seat_R".
  self._manager:oarAction(self._seatSide, style)
  return true
end

function Seat:releaseOar()
  self._oar:setState("oarDown")
end

function Seat:onEnter(enteringObject, contactName)
  if contactName == 'avatar' and self.gameObject:getState() == "seat" and
      enteringObject:getState() == "player" then
    self._manager:reportRower(self._seatSide, enteringObject)
    enteringObject:getComponent("Avatar"):disallowMovement()
    enteringObject:getComponent("Rowing"):setSeat(self)
    enteringObject:setState(self._config.rowingState)
    self.gameObject:connect(enteringObject)
    self.gameObject:setState("seatTaken")

    if self._manager:isFull() then
      self._manager:setBoatStateFull()
    end
  end
end


local WaterGoal = class.Class(component.Component)

function WaterGoal:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('WaterGoal')},
      {'bank_side', args.stringType},
      {'rowingPlayerState', args.default('rowing'), args.stringType},
      {'landPlayerState', args.default('landed'), args.stringType},
      {'usedSeatState', args.default('seatUsed'), args.stringType},
  })
  WaterGoal.Base.__init__(self, kwargs)
  self._config.rowingPlayerState = kwargs.rowingPlayerState
  self._config.landPlayerState = kwargs.landPlayerState
  self._config.usedSeatState = kwargs.usedSeatState
  self._config.bank_side = kwargs.bank_side
end

function WaterGoal:postStart()
  local scene = self.gameObject.simulation:getSceneObject()
  if scene:hasComponent("RaceManager") then
    self._raceManager = scene:getComponent("RaceManager")
  end
end

function WaterGoal:isGoalReached()
  return self._raceManager:getRaceDirection() == self._config.bank_side
end

function WaterGoal:getGoalTeleportY()
  local offset = 0
  if self._config.bank_side == "N" then
    offset = -3
  else
    offset = 3
  end
  return self.gameObject:getComponent('Transform'):getPosition()[2] + offset
end

function WaterGoal:onEnter(enteringObject, contactName)
  if self:isGoalReached() then
    if contactName == 'boat' and
       enteringObject:getState() ~= self._config.usedSeatState then
      if enteringObject:hasComponent("BoatManager") then
        log.v(1, "Boat arrived to shore")
        enteringObject:getComponent("BoatManager"):setBoatStateNormal()
        enteringObject:getComponent("BoatManager"):disembarkRowers(
            self:getGoalTeleportY())
      end
      enteringObject:getComponent("Seat"):releaseOar()
      enteringObject:setState(self._config.usedSeatState)
    end
  end
end

function WaterGoal:onExit(leavingObject, contactName)
  if self:isGoalReached() then
    if contactName == 'avatar' and
       self._raceManager.gameObject:getState() == 'boatRace' and
       leavingObject:getState() == self._config.rowingPlayerState then
      leavingObject:getComponent("Avatar"):allowMovement()
      leavingObject:setState('landed')
      log.v(1, "Successfully disembarked rower.")
    end
  else
    log.v(1, "Raise watergoal. Object exited:", leavingObject.name)
    self.gameObject:setState("goalBlocking")
  end
end


--[[ The GlobalRaceTracker keeps track of the gift metric.]]
local GlobalRaceTracker = class.Class(component.Component)

function GlobalRaceTracker:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('GlobalRaceTracker')},
      {'numPlayers', args.numberType},
  })
  GlobalRaceTracker.Base.__init__(self, kwargs)

  self._config.numPlayers = kwargs.numPlayers
end

function GlobalRaceTracker:reset()
  self.raceStart = tensor.Int32Tensor(self._config.numPlayers / 2, 2)
  self.strokes = tensor.Int32Tensor(self._config.numPlayers)
end

function GlobalRaceTracker:preUpdate()
  self.raceStart:fill(0)
  self.strokes:fill(0)
end

function GlobalRaceTracker:countStroke(index, stroke)
  if stroke == "flail" then
    self.strokes(index):val(1)
  elseif stroke == "row" then
    self.strokes(index):val(2)
  end
end


--[[ The StrokesTracker keeps track of the stroke style of the player.]]
local StrokesTracker = class.Class(component.Component)

function StrokesTracker:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('StrokesTracker')},
  })
  StrokesTracker.Base.__init__(self, kwargs)
end

function StrokesTracker:reset()
  self.strokes = tensor.Int32Tensor(2)
end

function StrokesTracker:preUpdate()
  self.strokes:fill(0)
end

function StrokesTracker:countStroke(stroke)
  if stroke == "flail" then
    self.strokes(1):val(1)
  elseif stroke == "row" then
    self.strokes(2):val(1)
  end
end


local allComponents = {
    BoatManager = BoatManager,
    RaceManager = RaceManager,
    EpisodeManager = EpisodeManager,
    Rowing = Rowing,
    Crown = Crown,
    Seat = Seat,
    WaterGoal = WaterGoal,
    GlobalRaceTracker = GlobalRaceTracker,
    StrokesTracker = StrokesTracker,
}

component_registry.registerAllComponents(allComponents)

return allComponents
