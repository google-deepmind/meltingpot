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
local set = require 'common.set'
local events = require 'system.events'
local random = require 'system.random'

local meltingpot = 'meltingpot.lua.'
local mp_modules = meltingpot .. 'modules.'
local paintball = meltingpot .. 'levels.paintball.'

local component = require(mp_modules .. 'component')
local shared_components = require(paintball .. 'shared_components')
local component_registry = require(mp_modules .. 'component_registry')


--[[ King of the Hill game uses a special Ground component to handle Hill logic,
different from the Ground component used by other paintball type games.
]]
local GroundOrHill = class.Class(shared_components.Ground)

function GroundOrHill:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('GroundOrHill')},
      {'teamNames', args.tableType},
      {'isHill', args.default(false), args.booleanType},
  })
  GroundOrHill.Base.__init__(self, {name = kwargs.name,
                                    teamNames = kwargs.teamNames})
  self._teamNames = set.Set(kwargs.teamNames)
  self._isHill = kwargs.isHill
end

function GroundOrHill:onHit(hittingGameObject, hittingTeam)
  -- Assume teamNames are identical to color states e.g. red, blue etc.
  local passThrough = GroundOrHill.Base.onHit(
    self, hittingGameObject, hittingTeam)
  if self._teamNames[hittingTeam] then
    local oldTeam = self.gameObject:getState()
    if self._isHill then
      -- Provide reward for painting the hill if applicable.
      if hittingGameObject:hasComponent('Taste') then
        hittingGameObject:getComponent('Taste'):paintedHillSquare(oldTeam)
      end
    end
  end
  return passThrough
end

function GroundOrHill:capture(capturingTeam)
  local oldTeam = self.gameObject:getState()
  self.gameObject:setState(capturingTeam)
end

function GroundOrHill:start()
  local sceneObject = self.gameObject.simulation:getSceneObject()
  self._hillManager = sceneObject:getComponent('HillManager')
  if self._isHill then
    self._hillManager:registerSquare(self)
  end
end


local ControlIndicator = class.Class(component.Component)

function ControlIndicator:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('ControlIndicator')},
  })
  ControlIndicator.Base.__init__(self, kwargs)
end

function ControlIndicator:registerUpdaters(updaterRegistry)
  local sceneObject = self.gameObject.simulation:getSceneObject()
  local hillManager = sceneObject:getComponent('HillManager')

  local function displayColorOfControllingTeam()
    local teamInControl = hillManager:getTeamInControl()
    self.gameObject:setState(teamInControl)
  end

  updaterRegistry:registerUpdater{
      updateFn = displayColorOfControllingTeam,
      priority = 3,
  }
end


local TeamMember = class.Class(component.Component)

function TeamMember:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('TeamMember')},
      {'team', args.stringType},
  })
  TeamMember.Base.__init__(self, kwargs)
  self._config.team = kwargs.team
end

function TeamMember:start()
  local sceneObject = self.gameObject.simulation:getSceneObject()
  local avatarComponent = self.gameObject:getComponent('Avatar')
  sceneObject:getComponent('HillManager'):registerAvatar(avatarComponent,
                                                         self._config.team)
end

function TeamMember:getTeam()
  return self._config.team
end


local Taste = class.Class(component.Component)

function Taste:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Taste')},
      {'mode', args.default('none'), args.oneOf('none',
                                                'control_hill',
                                                'paint_hill',
                                                'zap_while_in_control')},
      {'rewardAmount', args.default(0.0), args.numberType},
      {'zeroMainReward', args.default(false), args.booleanType},
      {'minFramesBetweenHillRewards', args.default(0), args.ge(0)},
  })
  Taste.Base.__init__(self, kwargs)
  self._config.mode = kwargs.mode
  self._config.rewardAmount = kwargs.rewardAmount
  self._config.zeroMainReward = kwargs.zeroMainReward
  self._config.minFramesBetweenHillRewards = kwargs.minFramesBetweenHillRewards
end

function Taste:reset()
  self._hillManager = self.gameObject.simulation:getSceneObject():getComponent(
    'HillManager')
  self._framesSinceEvent = 0
end

function Taste:registerUpdaters(updaterRegistry)
  local function incrementFramesSinceEvent()
    self._framesSinceEvent = self._framesSinceEvent + 1
  end
  updaterRegistry:registerUpdater{
      updateFn = incrementFramesSinceEvent,
      priority = 300,
  }
end

function Taste:paintedHillSquare(oldTeamOfThisSquare)
  local teamInControl = self._hillManager:getTeamInControl()
  local ownTeam = self.gameObject:getComponent('TeamMember'):getTeam()
  if self._framesSinceEvent > self._config.minFramesBetweenHillRewards then
    if self._config.mode == 'paint_hill' then
      self.gameObject:getComponent('Avatar'):addReward(
        self._config.rewardAmount)
    end
    if self._config.mode == 'control_hill' then
      if teamInControl == 'uncontrolled' then
        self.gameObject:getComponent('Avatar'):addReward(
          self._config.rewardAmount)
      end
    end
    self._framesSinceEvent = 0
  end
  -- Record event for player coloring the hill.
  if ownTeam ~= oldTeamOfThisSquare then
    local avatarComponent = self.gameObject:getComponent('Avatar')
    events:add('hill_square_colored', 'dict',
               'player_index', avatarComponent:getIndex(),  -- int
               'team', ownTeam) -- string
  end
end

function Taste:zap(unusedAvatarThatGotZapped)
  local teamInControl = self._hillManager:getTeamInControl()
  local ownTeam = self.gameObject:getComponent('TeamMember'):getTeam()
  if self._config.mode == 'zap_while_in_control' then
    if teamInControl == ownTeam then
      self.gameObject:getComponent('Avatar'):addReward(
        self._config.rewardAmount)
    end
  end
end

function Taste:mainReward(avatar, isInControl)
  if not self._config.zeroMainReward then
    self._hillManager:sendMainReward(avatar, isInControl)
  end
end


local HillManager = class.Class(component.Component)

function HillManager:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('HillManager')},
      {'percentToCapture', args.ge(51), args.le(100)},
      {'rewardPerStepInControl', args.numberType},
  })
  HillManager.Base.__init__(self, kwargs)
  self._config.percentToCapture = kwargs.percentToCapture
  self._config.rewardPerStepInControl = kwargs.rewardPerStepInControl
end

function HillManager:reset()
  self._registeredSquares = {}
  self._avatarsByTeam = {}
  self._teamCurrentlyInControl = "uncontrolled"
  self._totalHillSize = 0
end

function HillManager:_getPercentFilledByMaximalTeam()
  local maximalTeam = "uncontrolled"
  local maximalTeamAmount = 0

  local simulation = self.gameObject.simulation
  local numColoredByTeam = {
      red = simulation:getGroupCount("hill_red"),
      blue = simulation:getGroupCount("hill_blue"),
      uncontrolled = simulation:getGroupCount("hill_clean"),
  }

  for team, numColored in pairs(numColoredByTeam) do
    if numColored > maximalTeamAmount then
      maximalTeam = team
      maximalTeamAmount = numColored
    end
  end
  local percentFilled = (maximalTeamAmount / self._totalHillSize) * 100
  return percentFilled, maximalTeam
end

function HillManager:_getPercentOwnedByTeamInControl()
  local simulation = self.gameObject.simulation
  local numColored = simulation:getGroupCount(
      "hill_" .. self._teamCurrentlyInControl)
  local percentFilled = (numColored / self._totalHillSize) * 100
  return percentFilled
end

function HillManager:_capture(capturedByTeam)
  for _, square in ipairs(self._registeredSquares) do
    square:capture(capturedByTeam)
  end
  -- Record the hill capture event.
  events:add('hill_captured', 'dict',
             'team', capturedByTeam)  -- string
end

function HillManager:registerUpdaters(updaterRegistry)
  local function updateHill()
    local percentFilled, maximalTeam = self:_getPercentFilledByMaximalTeam()

    -- Change of control logic.
    if maximalTeam and maximalTeam ~= self._teamCurrentlyInControl then
      if percentFilled >= self._config.percentToCapture then
        -- Successful capture
        self:_capture(maximalTeam)
        self._teamCurrentlyInControl = maximalTeam
      end
    end

    -- Loss of control logic.
    if self._teamCurrentlyInControl ~= "uncontrolled" then
      local percentOwnedByTeamInControl = self:_getPercentOwnedByTeamInControl()
      if percentOwnedByTeamInControl < 50 then
        self._teamCurrentlyInControl = "uncontrolled"
      end
    end

    -- Scoring logic.
    if self._teamCurrentlyInControl ~= "uncontrolled" then
      self:provideRewards(self._teamCurrentlyInControl, true)
      local teamNotInControl = self:getOtherTeam(self._teamCurrentlyInControl)
      self:provideRewards(teamNotInControl, false)
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = updateHill,
      priority = 5,
  }
end


function HillManager:registerSquare(groundComponent)
  table.insert(self._registeredSquares, groundComponent)
  self._totalHillSize = self._totalHillSize + 1
end

function HillManager:registerAvatar(avatar, team)
  if not self._avatarsByTeam[team] then
    self._avatarsByTeam[team] = {}
  end
  table.insert(self._avatarsByTeam[team], avatar)
end

function HillManager:provideRewards(team, isInControl)
  for _, avatar in ipairs(self._avatarsByTeam[team]) do
    if avatar.gameObject:hasComponent('Taste') then
      -- Skip the main reward if applicable.
      avatar.gameObject:getComponent('Taste'):mainReward(avatar, isInControl)
    else
      self:sendMainReward(avatar, isInControl)
    end
  end
end

function HillManager:sendMainReward(avatar, isInControl)
  if isInControl then
    avatar:addReward(self._config.rewardPerStepInControl)
  else
    avatar:addReward(-self._config.rewardPerStepInControl)
  end
end

function HillManager:getTeamInControl()
  return self._teamCurrentlyInControl
end

function HillManager:getOtherTeam(team)
  if team == 'red' then
    return 'blue'
  elseif team == 'blue' then
    return 'red'
  end
end


local allComponents = {
    -- Object components.
    GroundOrHill = GroundOrHill,
    ControlIndicator = ControlIndicator,

    -- Avatar components.
    TeamMember = TeamMember,
    Taste = Taste,

    -- Scene components.
    HillManager = HillManager,
}

component_registry.registerAllComponents(allComponents)

return allComponents
