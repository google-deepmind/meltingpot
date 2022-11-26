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
local tables = require 'common.tables'
local events = require 'system.events'
local random = require 'system.random'

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')

local Flag = class.Class(component.Component)

function Flag:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Flag')},
      {'team', args.stringType},
  })
  Flag.Base.__init__(self, kwargs)
  self._config.team = kwargs.team
end

function Flag:registerUpdaters(updaterRegistry)
  local _resetToStartStateAndPosition = function()
    local home = self._flagManager:getHomeTilePosition(self._config.team)
    self.gameObject:teleport(home, 'N')
    self.gameObject:setState('dropped')
  end
  updaterRegistry:registerUpdater{
      updateFn = _resetToStartStateAndPosition,
      priority = 3,
      state = 'wait',
  }
end

function Flag:reset()
  local sceneObject = self.gameObject.simulation:getSceneObject()
  self._flagManager = sceneObject:getComponent('FlagManager')
  self._carryingAvatarObject = nil
end

function Flag:start()
  -- Register this flag object with the global flag manager.
  self._flagManager:registerFlag(self._config.team, self.gameObject)
end

function Flag:onEnter(enteringGameObject, contactName)
  if contactName == 'avatar' and self.gameObject:getState() == 'dropped' then
    local avatarComponent = enteringGameObject:getComponent('Avatar')
    local teamMemberComponent = enteringGameObject:getComponent('TeamMember')
    local tasteComponent = enteringGameObject:getComponent('Taste')
    local enteringTeam = teamMemberComponent:getTeam()
    if enteringTeam == self._config.team then
      if not self:onHomeTile() then
        -- Teleport own team's flag back to home if encountering it away from
        -- the home location. This happens when it gets dropped after a player
        -- carrying it was zapped.
        local home = self._flagManager:getHomeTilePosition(self._config.team)
        self.gameObject:teleport(home, 'N')
        tasteComponent:flagReturned()
        -- Record the flag return event.
        events:add('flag_returned', 'dict',
                   'player_index', avatarComponent:getIndex(),  -- int
                   'team', self._config.team)  -- string
      end
      if self:onHomeTile() and teamMemberComponent:isCarryingFlag() then
        -- Successfully capture the carried flag when an avatar carries the flag
        -- of the opposing team onto their own flag while it is on their own
        -- team's home tile.
        self._flagManager:capture(self._config.team, enteringGameObject)
        -- Record the flag capture event.
        events:add('flag_captured', 'dict',
                   'player_index', avatarComponent:getIndex(),  -- int
                   'team', self._config.team)  -- string
      end
    else
      -- Case of a player picking up the opposing team's flag.
      self.gameObject:setState('carried')
      self._carryingAvatarObject = enteringGameObject
      self._carryingAvatarObject:getComponent('Avatar'):connect(self.gameObject)
      teamMemberComponent:pickUpFlag(self.gameObject)
      tasteComponent:flagPickedUp()
      -- Record the flag pick up event.
      events:add('flag_picked_up', 'dict',
                 'player_index', avatarComponent:getIndex(),  -- int
                 'team', teamMemberComponent:getTeam())  -- string
    end
  end
end

function Flag:onHomeTile()
  local home = self._flagManager:getHomeTilePosition(self._config.team)
  -- Compare serialized locations.
  return tables.tostring(self.gameObject:getPosition()) == tables.tostring(home)
end

--[[ Disconnect from connected avatar, if no avatar connected then do nothing.
]]
function Flag:disconnectFromAvatar()
  if self._carryingAvatarObject then
    self._carryingAvatarObject:getComponent('TeamMember'):dropFlag()
    self._carryingAvatarObject:getComponent('Avatar'):disconnect(
      self.gameObject)
    self._carryingAvatarObject = nil
  end
end

function Flag:avatarStateChange(behavior)
  if behavior == 'die' then
    self:disconnectFromAvatar()
    self.gameObject:setState('dropped')
  end
end


local HomeTile = class.Class(component.Component)

function HomeTile:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('HomeTile')},
      {'team', args.stringType},
  })
  HomeTile.Base.__init__(self, kwargs)
  self._config.team = kwargs.team
end

function HomeTile:postStart()
  local sceneObject = self.gameObject.simulation:getSceneObject()
  local flagManager = sceneObject:getComponent('FlagManager')
  flagManager:registerHomeTile(self._config.team, self.gameObject:getPosition())
end

function HomeTile:getTeam()
  return self._config.team
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
  local flagManager = sceneObject:getComponent('FlagManager')

  local function updateColorIndicator()
    -- Show the color of a team that can score because their flag is on their
    -- own home tile. If both teams have their flag on their home tile then show
    -- purple. If neither, then show black.
    local flagControlState = flagManager:getFlagControlState()
    self.gameObject:setState(flagControlState)
  end

  updaterRegistry:registerUpdater{
        updateFn = updateColorIndicator,
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

function TeamMember:reset()
  self._carryingFlag = false
  local sceneObject = self.gameObject.simulation:getSceneObject()
  self._flagManager = sceneObject:getComponent('FlagManager')
end

function TeamMember:start()
  self._flagManager:registerAvatar(self.gameObject, self._config.team)
end

function TeamMember:getTeam()
  return self._config.team
end

function TeamMember:isCarryingFlag()
  return self._carryingFlag
end

function TeamMember:pickUpFlag(flagObject)
  self._carryingFlag = true
end

function TeamMember:dropFlag()
  self._carryingFlag = false
end


local Taste = class.Class(component.Component)

function Taste:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Taste')},
      {'defaultTeamReward', args.ge(0.0)},
      {'rewardForZapping', args.default(0.0), args.numberType},
      {'extraRewardForZappingFlagCarrier', args.default(0.0), args.numberType},
      {'rewardForReturningFlag', args.default(0.0), args.numberType},
      {'rewardForPickingUpOpposingFlag', args.default(0.0), args.numberType},
  })
  Taste.Base.__init__(self, kwargs)
  self._config.defaultTeamReward = kwargs.defaultTeamReward
  self._config.rewardForZapping = kwargs.rewardForZapping
  self._config.extraRewardForZappingFlagCarrier = (
    kwargs.extraRewardForZappingFlagCarrier)
  self._config.rewardForReturningFlag = kwargs.rewardForReturningFlag
  self._config.rewardForPickingUpOpposingFlag = (
    kwargs.rewardForPickingUpOpposingFlag)
end

function Taste:reset()
  self._avatarComponent = self.gameObject:getComponent('Avatar')
end

function Taste:addRewardForSuccessfulCapture(capturingAvatarObject)
  self._avatarComponent:addReward(self._config.defaultTeamReward)
end

function Taste:addRewardForOpposingTeamCapture()
  -- The default penalty for letting the other team capture the flag is
  -- -1 * the default reward for capturing the flag.
  self._avatarComponent:addReward(-self._config.defaultTeamReward)
end

function Taste:zap(avatarThatGotZapped)
  local gotFlagCarrier = avatarThatGotZapped:getComponent(
    'TeamMember'):isCarryingFlag()
  if gotFlagCarrier then
    self._avatarComponent:addReward(
      self._config.extraRewardForZappingFlagCarrier)
    -- Record an event for zapping a flag carrier.
    events:add('zapped_flag_carrier', 'dict',
               'player_index', self._avatarComponent:getIndex())  -- int
  end
  self._avatarComponent:addReward(self._config.rewardForZapping)
end

function Taste:flagReturned()
  -- Deliver an individual return for returning own team's flag (default 0.0).
  self._avatarComponent:addReward(self._config.rewardForReturningFlag)
end

function Taste:flagPickedUp()
  -- Deliver individual return for picking up opposing flag (default 0.0).
  self._avatarComponent:addReward(self._config.rewardForPickingUpOpposingFlag)
end


local FlagManager = class.Class(component.Component)

function FlagManager:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('FlagManager')},
  })
  FlagManager.Base.__init__(self, kwargs)
  self._homeTilePositions = {}
end

function FlagManager:registerUpdaters(updaterRegistry)
  local _getFlagControlState = function()
    -- This function determines what color to show in the on-map indicators.
    local teamWithFlagOnHomeTile = nil
    local numTeamsWithFlagOnHomeTile = 0
    for team, flagObject in pairs(self._flagObjectsByTeam) do
      if flagObject:getComponent('Flag'):onHomeTile() then
        teamWithFlagOnHomeTile = team
        numTeamsWithFlagOnHomeTile = numTeamsWithFlagOnHomeTile + 1
      end
    end
    if numTeamsWithFlagOnHomeTile == 1 then
      self._currentFlagControlState = teamWithFlagOnHomeTile
    elseif numTeamsWithFlagOnHomeTile == 2 then
      self._currentFlagControlState = 'both'
    elseif numTeamsWithFlagOnHomeTile == 0 then
      self._currentFlagControlState = 'neither'
    end
  end
  updaterRegistry:registerUpdater{
      updateFn = _getFlagControlState,
      priority = 175,
  }
end

function FlagManager:reset()
  self._flagObjectsByTeam = {}
  self._avatarObjects = {}
  self._avatarObjectsByTeam = {}
  self._currentFlagControlState = nil
end

function FlagManager:registerFlag(team, flagObject)
  self._flagObjectsByTeam[team] = flagObject
end

function FlagManager:registerHomeTile(team, position)
  self._homeTilePositions[team] = position
end

function FlagManager:registerAvatar(avatar, team)
  table.insert(self._avatarObjects, avatar)
  if not self._avatarObjectsByTeam[team] then
    self._avatarObjectsByTeam[team] = {}
  end
  table.insert(self._avatarObjectsByTeam[team], avatar)
end

function FlagManager:getHomeTilePosition(team)
  return self._homeTilePositions[team]
end

function FlagManager:resetFlags()
  -- Teleport all flags back to their starting location and state.
  for team, flagObject in pairs(self._flagObjectsByTeam) do
    flagObject:getComponent('Flag'):disconnectFromAvatar()
    -- All avatars drop their flags.
    for _, avatarObject in ipairs(self._avatarObjectsByTeam[team]) do
      avatarObject:getComponent('TeamMember'):dropFlag()
    end
    flagObject:setState('wait')
  end
end

--[[ Call this function upon successfully capturing of the opposing team's flag.
]]
function FlagManager:capture(capturingTeam, capturingAvatarObject)
  -- Teleport the captured team's flag back to its team's home tile.
  self:resetFlags()
  -- Provide rewards and penalties for the flag capture.
  for _, avatarObject in ipairs(self._avatarObjects) do
    local team = avatarObject:getComponent('TeamMember'):getTeam()
    if team == capturingTeam then
      avatarObject:getComponent('Taste'):addRewardForSuccessfulCapture(
        capturingAvatarObject)
    else
      avatarObject:getComponent('Taste'):addRewardForOpposingTeamCapture()
    end
  end
end

function FlagManager:getFlagControlState()
  return self._currentFlagControlState
end


local allComponents = {
    -- Object components.
    Flag = Flag,
    HomeTile = HomeTile,
    ControlIndicator = ControlIndicator,

    -- Avatar components.
    TeamMember = TeamMember,
    Taste = Taste,

    -- Scene components.
    FlagManager = FlagManager,
}

component_registry.registerAllComponents(allComponents)

return allComponents
