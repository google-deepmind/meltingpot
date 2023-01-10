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
local tensor = require 'system.tensor'

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')
local component_library = require(meltingpot .. 'component_library')

-- For Lua 5.2 compatibility.
local unpack = unpack or table.unpack

--[[ Progress component for the Scene object, tracks task completion,
  useful methods and metrics for analysis, and interfaces with the
  voting components of each avatar.
]]--
local Progress = class.Class(component.Component)

function Progress:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Progress')},
      -- The number of total players in the game.
      {'num_players', args.numberType},
      -- The number of gems that the Crewmates must deposit to win.
      {'goal', args.default(32), args.numberType},
      -- Whether the pseudoreward is implemented using a potential-based approach.
      {'potential_pseudorewards', args.default(False), args.booleanType},
      -- Rewards for the Crewmates winning by tasks..
      {'crewmate_task_reward', args.default(4), args.numberType},
      {'impostor_task_reward', args.default(-4), args.numberType},
      -- Rewards for the Impostors winning by tagging.
      {'crewmate_tag_reward', args.default(-4), args.numberType},
      {'impostor_tag_reward', args.default(4), args.numberType},
      -- Rewards for the Crewmates winning by voting.
      {'crewmate_vote_reward', args.default(4), args.numberType},
      {'impostor_vote_reward', args.default(-4), args.numberType},
      -- Rewards for voting out a member of your group or a member of the other.
      {'incorrect_vote_reward', args.default(-1), args.numberType},
      {'correct_vote_reward', args.default(1), args.numberType},
      -- Small negative reward at every time step to encourage efficiency.
      {'step_reward', args.default(0), args.numberType},
      -- Teleport spawn group (location for after being tagged or voted out).
      {'teleport_spawn_group', args.stringType},
      -- Extra parameters to help with the voting interface.
      {'voting_params', args.default({}), args.tableType},
  })
  Progress.Base.__init__(self, kwargs)
  self.numPlayers = kwargs.num_players
  self.goal = kwargs.goal
  self.teleportSpawnGroup = kwargs.teleport_spawn_group
  self.votingValues = kwargs.voting_params

  self.potential_pseudorewards = kwargs.potential_pseudorewards
  self.crewmate_task_reward = kwargs.crewmate_task_reward
  self.impostor_task_reward = kwargs.impostor_task_reward
  self.crewmate_tag_reward = kwargs.crewmate_tag_reward
  self.impostor_tag_reward = kwargs.impostor_tag_reward
  self.impostor_vote_reward = kwargs.impostor_vote_reward
  self.crewmate_vote_reward = kwargs.crewmate_vote_reward
  self.incorrect_vote_reward = kwargs.incorrect_vote_reward
  self.correct_vote_reward = kwargs.correct_vote_reward
  self.step_reward = kwargs.step_reward
  self.goalCounter = 0
  self.avatars = {}
  self.crewmatesLeftForImpostorWin = -1

  -- Variables to track information to expose as observations.
  self.progress_bar = tensor.DoubleTensor({0})
  self.identity_tensor = tensor.DoubleTensor(self.numPlayers)
  self.depositSquares = {}
end

function Progress:start()
  -- Cache gameObjects for each player.
  self.votingMatrix = tensor.DoubleTensor(self.numPlayers, self.numPlayers+2)
  self:cachePlayers()
  -- Use votingValues to initialize various voting parameters.
  if self.votingValues.type == 'deliberation' then
    assert(self.votingValues.votingPhaseCooldown ~= nil and
            self.votingValues.votingFrameFrequency ~= nil and
            self.votingValues.taggingTriggerVoting ~= nil and
            self.votingValues.votingPhaseCooldown >= 0 and
            self.votingValues.votingFrameFrequency >= 0)
    -- time in the voting phase
    self._votingPhaseCooldown = self.votingValues.votingPhaseCooldown
    -- time between voting phases
    self._votingCooldown = self.votingValues.votingFrameFrequency
    -- flag for whether the game is in a voting round or not
    self.inVotingRound = false
  end
  -- create the voting matrix if the voting type requires a global matrix.
  if self.votingValues.type == 'deliberation' or
    self.votingValues.type == 'continuous' then
    -- Voting matrix starts with all active agents voting for "no vote".
    self:resetVotingMatrix()
  end
  -- Initialize the step counter.
  self.stepCounter = 0

  if self.votingValues.type == 'continuous' then
    assert(self.votingValues.votingDelay ~= nil and
           self.votingValues.votingDelay >= 0)
    -- Check if there is a voting delay at the beginning of the episode.
    if self.votingValues.votingDelay > 0 then
      local activeAvatars = self:getActivePlayers()
      for _, object in pairs(activeAvatars) do
        local roleComponent = object:getComponent("Role")
        -- Prevent players from voting.
        roleComponent:changePlayerStatus(true, true, false)
      end
    end
  end
end

function Progress:getVotingRoundStatus()
  return self.inVotingRound
end

function Progress:setVotingRoundStatus(voting)
  self.inVotingRound = voting
end

function Progress:doesTaggingTriggerVoting()
  return self.votingValues.taggingTriggerVoting
end

function Progress:getTeleportSpawnGroup()
  return self.teleportSpawnGroup
end

function Progress:getAvatars()
  return self.avatars
end

function Progress:getDepositSquares()
  return self.depositSquares
end

function Progress:addRewardByGroup(impostorReward, crewmateReward)
  -- Add rewards to impostors and crewmates.
  for _, object in pairs(self.avatars) do
    local roleComponent = object:getComponent("Role")
    local avatarComponent = object:getComponent('Avatar')
    if roleComponent:getRole() == "impostor" then
      avatarComponent:addReward(impostorReward)
    elseif roleComponent:getRole() == "crewmate" then
      avatarComponent:addReward(crewmateReward)
    end
  end
end

function Progress:gameEnd(impostorReward, crewmateReward)
  -- Function to add final rewards and end the episode.
  for _, object in pairs(self.avatars) do
    local roleComponent = object:getComponent("Role")
    local avatarComponent = object:getComponent('Avatar')
    if roleComponent:getRole() == "impostor" then
      avatarComponent:addReward(impostorReward)
    elseif roleComponent:getRole() == "crewmate" then
      avatarComponent:addReward(crewmateReward)
    end

    if self.potential_pseudorewards then
      --[[ If using the potential-based approach for pseudorewards, decrement
      the final reward based on pseudorewards accumulated from
      gem collecting / depositing over the course of the episode.]]--
      avatarComponent:addReward(-1*roleComponent:gemsCollectedReward())
      avatarComponent:addReward(-1*roleComponent:gemsDepositedReward())
    end
    roleComponent:changePlayerStatus(false, false, false)
  end
  self.gameObject.simulation:endEpisode()
end

function Progress:cacheDepositSquares()
  --[[ If they haven't been saved already, save the gameObjects of all the
    deposit squares in the game into self.depositSquares for easy access.
    ]]--
  if (#self.depositSquares == 0) then
    local simulation = self.gameObject.simulation
    self.depositSquares = simulation:getAllGameObjectsWithComponent("Deposit")
  end
  assert(#self.depositSquares > 0)
end

function Progress:cachePlayers()
  --[[ If they haven't been saved already, save the gameObjects of all the
  avatars in the game in self.avatars and count the number of Impostors to
  determine the win condition for the Impostors.
  ]]--
  if (#self.avatars == 0) then
    local numImpostors = 0
    local avatarObjects = self.gameObject.simulation:getAvatarGameObjects()
    self.impostors = {}
    for _, object in pairs(avatarObjects) do
      table.insert(self.avatars, object)
      local roleComponent = object:getComponent("Role")
      local avatarComponent = object:getComponent("Avatar")
      if roleComponent:getRole() == "impostor" then
        if object:hasComponent("Tagger") then
          object:getComponent("Tagger"):allowTagging()
        end
        table.insert(self.impostors, object)
        numImpostors = numImpostors + 1
        -- Set the one-hot encoding of who the Impostor is.
        self.identity_tensor(avatarComponent:getIndex()):val(1)
      elseif roleComponent:getRole() == "crewmate" then
        if object:hasComponent("Tagger") then
          object:getComponent("Tagger"):disallowTagging()
        end
      end
    end
    self.crewmatesLeftForImpostorWin = numImpostors
    assert(#self.avatars > 0, "Not enough avatars.")
    assert(#self.impostors > 0, "Not enough impostors.")
  end
end

function Progress:update()
  -- Check whether the Crewmates have completed all the tasks.
  self:checkCrewmateTaskWin()

  -- A small negative reward at each step to encourage efficiency.
  for _, object in pairs(self.avatars) do
    local avatarComponent = object:getComponent('Avatar')
    avatarComponent:addReward(self.step_reward)
  end

  if self.votingValues.type == 'deliberation' then
    -- Check whether the game is in a voting round.
    if self:getVotingRoundStatus() == true then
      -- Count down the time in the voting round.
      self._votingPhaseCooldown = self._votingPhaseCooldown - 1
      if self._votingPhaseCooldown <= 0 then
        -- The voting phase has ended.
        -- Check the Crewmate voting win condition.
        self:checkCrewmateVoteWin()
        -- Not in the voting phase anymore, respawn avatars to starting location.
        self:resetVotingMatrix()
        self._votingPhaseCooldown = self.votingValues.votingPhaseCooldown
        self:setVotingRoundStatus(false)
        self:triggerRespawnEvent()
      end
    elseif self:getVotingRoundStatus() == false then
      -- Cooldown until the next voting round starts.
      self._votingCooldown = self._votingCooldown - 1
      if self._votingCooldown <= 0 then
        -- Time to start the voting round.
        self:triggerVotingEvent()
        self:setVotingRoundStatus(true)
        self._votingCooldown = self.votingValues.votingFrameFrequency
      end
    end
  elseif self.votingValues.type == 'continuous' then
    -- Check to re-enable voting after initial voting delay.
    if self.stepCounter >= self.votingValues.votingDelay then
      local activeAvatars = self:getActivePlayers()
      for _, object in pairs(activeAvatars) do
        local roleComponent = object:getComponent("Role")
        roleComponent:changePlayerStatus(true, true, true)
      end
    end
    -- Check whether the voting win condition is met.
    self:checkCrewmateVoteWin()
  elseif self.votingValues.type == 'sus' then
    -- Check whether the voting win condition is met.
    self:checkCrewmateVoteWin()
  end
  -- Update the step counter.
  self.stepCounter = self.stepCounter + 1
end

function Progress:getStepCounter()
  return self.stepCounter
end

function Progress:getProgress()
  return self.goalCounter/self.goal
end

function Progress:updateProgress(amount)
  -- Update the goalCounter and the progress_bar observation.
  self.goalCounter = self.goalCounter + amount
  self.progress_bar = tensor.DoubleTensor({self:getProgress()})
end

function Progress:getImpostors()
  return self.impostors
end

function Progress:getActivePlayers()
  -- Return a table with the active players (not tagged out or voted out).
  local activeAvatars = {}
  for _, object in pairs(self.avatars) do
    local roleComponent = object:getComponent("Role")
    if roleComponent:checkActive() then
      table.insert(activeAvatars, object)
    end
  end
  return activeAvatars
end

function Progress:getIdxActivePlayers()
  -- Return the avatar idxs for the active players.
  local active = {}
  for _, object in pairs(self.avatars) do
    local roleComponent = object:getComponent("Role")
    if roleComponent:checkActive() then
      local avatarComponent = object:getComponent('Avatar')
      table.insert(active, avatarComponent:getIndex())
    end
  end
  return active
end

function Progress:numActiveImpostors()
  -- Return the number of active Impostors (not tagged out or voted out).
  local activeImpostors = 0
  for _, object in pairs(self.avatars) do
    local roleComponent = object:getComponent("Role")
    if roleComponent:getRole() == "impostor" then
      if roleComponent:checkActive() then
        activeImpostors = activeImpostors + 1
      end
    end
  end
  return activeImpostors
end

function Progress:numActiveCrewmates()
  -- Return the number of active Crewmates (not tagged out or voted out).
  local activeCrewmates = 0
  for _, object in pairs(self.avatars) do
    local roleComponent = object:getComponent("Role")
    if roleComponent:getRole() == "crewmate" then
      if roleComponent:checkActive() then
        activeCrewmates = activeCrewmates + 1
      end
    end
  end
  return activeCrewmates
end

function Progress:checkImpostorTagWin()
  -- Check whether the Impostors won by tagging and end the episode.
  if (self:numActiveCrewmates() <= self.crewmatesLeftForImpostorWin) then
    -- Add to events as a win condition.
    events:add('win', 'dict',
               'condition', 'impostor tag win') -- string
    self:gameEnd(self.impostor_tag_reward, self.crewmate_tag_reward)
    return true
  end
  return false
end

function Progress:checkCrewmateTaskWin()
  -- Check whether the Crewmates won by finishing tasks and end the episode.
  if(self:getProgress() >= 1.0) then
    -- Add to events as a win condition.
    events:add('win', 'dict',
               'condition', 'crewmate task win') -- string
    self:gameEnd(self.impostor_task_reward, self.crewmate_task_reward)
    return true
  end
  return false
end

function Progress:getPlayerVotedOff()
  -- Identify the player (if any) to be voted out by group consensus.
  local numActiveAvatars = #self:getActivePlayers()
  for player = 1, self.numPlayers do
    local col = self.votingMatrix:select(2, player)
    if col:sum() >= math.ceil(numActiveAvatars/2) then
      return player
    end
  end
  -- No players were voted off.
  return 0
end

function Progress:checkCrewmateVoteWin()
  -- Check whether the Crewmates won by voting out the Impostor
  -- with other functionality to handle all voting, and end the episode.
  if self.votingValues.type == 'sus' then
    for _, object in pairs(self.impostors) do
      local votingComponent = object:getComponent("Voting")
      -- Check if suspicion is above the threshold.
      if (votingComponent:getSuspicion() >= 1) then
        -- inactivate player
        local roleComponent = object:getComponent("Role")
        local aliveState = object:getComponent("Avatar"):getAliveState()
        roleComponent:inactivatePlayer()
        local teleportSpawnGroup = self:getTeleportSpawnGroup()
        object:teleportToGroup(teleportSpawnGroup, aliveState)
        roleComponent:changePlayerStatus(false, false, false)
        -- check win condition
        if self:numActiveImpostors() == 0 then
          -- add to events as a win condition
          events:add('win', 'dict',
                     'condition', 'crewmate vote win') -- string
          -- all Impostors voted off, game ends
          self:gameEnd(self.impostor_vote_reward, self.crewmate_vote_reward)
          return true
        end
      end
    end
  else
    -- Get index of player who was voted off (0 if no player was voted off).
    local votedOffIdx = self:getPlayerVotedOff()
    if votedOffIdx > 0 then
      local activeAvatars = self:getActivePlayers()
      for _, object in pairs(activeAvatars) do
        local roleComponent = object:getComponent("Role")
        local avatarComponent = object:getComponent("Avatar")
        local aliveState = avatarComponent:getAliveState()
        if avatarComponent:getIndex() == votedOffIdx then
          -- Add to events when a player is voted out.
          events:add('vote_removed', 'dict',
                     'player_index', votedOffIdx, -- int
                     'player_identity', roleComponent:getRole(), -- string
                     'matrix', self.votingMatrix) -- tensor
          -- Inactivate player
          roleComponent:inactivatePlayer()
          local teleportSpawnGroup = self:getTeleportSpawnGroup()
          object:teleportToGroup(teleportSpawnGroup, aliveState)
          roleComponent:changePlayerStatus(false, false, false)
          -- Check win conditions.
          if roleComponent:getRole() == "impostor" then
            -- The Impostor was voted off.
            if self:numActiveImpostors() == 0 then
              -- Game ends as all Impostors were voted off.
              -- Add to events as a win condition.
              events:add('win', 'dict',
                         'condition', 'crewmate vote win') -- string
              self:gameEnd(self.impostor_vote_reward, self.crewmate_vote_reward)
              return true
            end
            -- Impostors get negative reward for voting another Impostor off.
            self:addRewardByGroup(self.incorrect_vote_reward,
                                  self.correct_vote_reward)
          else
            -- Crewmate voted off, potential Impostor win.
            self:checkImpostorTagWin()
            -- Crewmates get negative reward for voting another Crewmate off.
            self:addRewardByGroup(self.correct_vote_reward,
                                  self.incorrect_vote_reward)
          end
        else
          -- Player was not voted off, unfreeze player.
          if self.votingValues.type == 'deliberation' then
            roleComponent:changePlayerStatus(true, true, false)
          elseif self.votingValues.type == 'continuous' then
            roleComponent:changePlayerStatus(true, true, true)
          end
          -- Respawn player to inital start location.
          local spawnGroup = avatarComponent:getSpawnGroup()
          object:teleportToGroup(spawnGroup, aliveState)
        end
      end
    end
  end
  return false
end

function Progress:triggerVotingEvent()
  -- Functionality for starting the voting event in the 'deliberation' mode.
  local activePlayers = self:getActivePlayers()
  for _, object in pairs(activePlayers) do
    -- Freeze all actions other than voting.
    local roleComponent = object:getComponent("Role")
    roleComponent:changePlayerStatus(false, false, true)
    -- Teleport the players to the voting location.
    local votingComponent = object:getComponent("Voting")
    local avatarComponent = object:getComponent('Avatar')
    object:teleportToGroup(votingComponent:getVotingSpawnGroup(),
                           avatarComponent:getAliveState())
  end
  -- Add to events at the beginning of the voting round.
  events:add('vote_round', 'dict',
             'step', tensor.DoubleTensor({self.step_counter})) -- int
end

function Progress:triggerRespawnEvent()
  -- Functionality to teleport the players back to their original spawn
  -- locations after the voting round in the 'deliberation' mode.
  local activePlayers = self:getActivePlayers()
  for _, object in pairs(activePlayers) do
    -- Unfreeze all actions other than voting.
    local roleComponent = object:getComponent("Role")
    roleComponent:changePlayerStatus(true, true, false)
    -- Reset tagger cooldown so Impostors can't immediately tag Crewmates.
    if object:hasComponent("Tagger") then
      local taggerComponent = object:getComponent("Tagger")
      taggerComponent:resetCoolingTimer()
    end
    -- Teleport the avatars to the spawn location.
    local avatarComponent = object:getComponent("Avatar")
    local aliveState = avatarComponent:getAliveState()
    object:teleportToGroup(avatarComponent:getSpawnGroup(), aliveState)
  end
end

function Progress:_resetVotingMatrix(numPlayers)
  -- Initialize a blank voting matrix (all players set to tagged out).
  local activePlayersIdx = self:getIdxActivePlayers()
  self.votingMatrix:fill(0)
  -- Voting for each player, no vote, and player tagged out.
  for x=1,numPlayers do
    -- Set each player to player tagged out.
    self.votingMatrix(x, numPlayers+2):val(1)
  end
  for _, idx in pairs(activePlayersIdx) do
    -- Set active players to no-vote.
    self.votingMatrix(idx, numPlayers+2):val(0)
    self.votingMatrix(idx, numPlayers+1):val(1)
  end
end

function Progress:getVotingMatrix()
  assert(self.votingValues.type == 'deliberation' or
         self.votingValues.type == 'continuous')
  return self.votingMatrix
end

function Progress:resetVotingMatrix()
  -- Reset the voting matrix by setting active players to no-vote and
  -- inactive players to tagged out.
  self:_resetVotingMatrix(self.numPlayers)
  self.votingCounter = {}
  for x=1,self.numPlayers do
    self.votingCounter[x] = 0
  end
end

function Progress:submitVote(playerIdx, lastVote, submissionType)
  -- Helper method for the Voting component of each player to interface with the
  -- global voting matrix.
  for i=1, self.numPlayers+2 do
    if self.votingMatrix(playerIdx,i):val() == 1 then
      self.votingMatrix(playerIdx,i):val(0)
    end
  end
  self.votingMatrix(playerIdx,lastVote):val(1)

  if submissionType == 'vote' then
    self.votingCounter[playerIdx] = self.votingCounter[playerIdx] + 1
  end
end

function Progress:getNumAvatarsSaw(callingGameObject)
  local selfAvatarComponent = callingGameObject:getComponent('Avatar')
  local selfPlayerIdx = tostring(selfAvatarComponent:getIndex())

  local visibleFromAvatar = 0

  for _, avatarObject in pairs(self:getActivePlayers()) do
    -- Get the players visible from each active player.
    local avatarComponent = avatarObject:getComponent('Avatar')
    local avatarsObservable = avatarComponent:queryPartialObservationWindow(callingGameObject:getLayer())
    -- Check if the callingGameObject was visible from the player.
    for _, visibleObject in pairs(avatarsObservable) do
      if (visibleObject:hasComponent('Avatar')) then
        local avatarPlayerIdx = tostring(visibleObject:getComponent('Avatar'):getIndex())
        if (avatarPlayerIdx == selfPlayerIdx) then
          visibleFromAvatar = visibleFromAvatar + 1
        end
      end
    end
  end
  -- Subtract 1 to account for the callingGameObject itself.
  return visibleFromAvatar-1
end

-- Inventory component tracks collected gems
local Inventory = class.Class(component.Component)

function Inventory:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Inventory')},
      -- Maximum number of gems a player can hold at a given time.
      {'max_gems', args.default(1), args.numberType},
  })
  Inventory.Base.__init__(self, kwargs)
  self.maxGems = kwargs.max_gems
  self.currentGems = 0
end

function Inventory:pickUpGem()
  -- Increment the number of gems in the inventory.
  self.currentGems = self.currentGems + 1
end

function Inventory:clearGems()
  -- Reset the number of gems the player has in their inventory.
  self.currentGems = 0
end

function Inventory:extraSpace()
  -- Determine whether there is room in the inventory.
  return self.currentGems < self.maxGems
end

function Inventory:getNumGems()
  -- Return the number of gems currently in the inventory.
  return self.currentGems
end

function Inventory:fractionFull()
  -- Return the number of gems in the inventory as a fraction of the capacity.
  return self.currentGems / (self.maxGems)
end

--[[ Deposit component tracks depositing of the gems and updating of inventory.
Touching the Deposit object deposits all gems, which are removed from inventory.
]]--
local Deposit = class.Class(component.Component)

function Deposit:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Deposit')},
      -- Reward a Crewmate recieves from depositing each gem.
      {'crewmateReward', args.default(0.5), args.numberType},
      -- Reward an Impostor recieves from depositing each gem.
      {'impostorReward', args.default(0), args.numberType},
    })
  Deposit.Base.__init__(self, kwargs)
  self.crewmateReward = kwargs.crewmateReward
  self.impostorReward = kwargs.impostorReward
end

function Deposit:onEnter(enteringGameObject, contactName)
  local sceneObject = self.gameObject.simulation:getSceneObject()
  local progressComponent = sceneObject:getComponent('Progress')

  if(enteringGameObject:hasComponent('Avatar')) then
    local avatarComponent = enteringGameObject:getComponent('Avatar')
    local inventoryComponent = enteringGameObject:getComponent('Inventory')
    local roleComponent = enteringGameObject:getComponent('Role')
    -- If there are any gems in the inventory, clear the inventory and
    -- increase the progress bar.
    if(inventoryComponent:fractionFull() > 0) then
      local numGems = inventoryComponent:getNumGems()
      progressComponent:updateProgress(numGems)
      -- Add to events upon deposit.
      events:add('deposit', 'dict',
                 'player_index', avatarComponent:getIndex(), -- int
                 'num_gems', numGems) -- int
      if roleComponent:getRole() == "crewmate" then
        avatarComponent:addReward(self.crewmateReward*numGems)
        roleComponent:rewardForGemsDeposited(self.crewmateReward*numGems)
      elseif roleComponent:getRole() == "impostor" then
        avatarComponent:addReward(self.impostorReward*numGems)
        roleComponent:rewardForGemsDeposited(self.impostorReward*numGems)
      end
      inventoryComponent:clearGems()
    end
  end
end

-- Collectable component adds to inventory. Based on the Edible component.
local Collectable = class.Class(component.Component)

function Collectable:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Collectable')},
      {'liveState', args.stringType},
      {'waitState', args.stringType},
      -- The reward a Crewmate recieves for collecting a gem.
      {'rewardForCollecting_crewmate', args.default(0.5), args.numberType},
      -- The reward an Impostor recieves for collecting a gem.
      {'rewardForCollecting_impostor', args.default(0), args.numberType},
      -- The rate at which the gems regrow.
      {'regrowRate', args.default(0.01), args.numberType}
  })
  Collectable.Base.__init__(self, kwargs)

  self.liveState = kwargs.liveState
  self.waitState = kwargs.waitState
  self.rewardForCollecting_crewmate = kwargs.rewardForCollecting_crewmate
  self.rewardForCollecting_impostor = kwargs.rewardForCollecting_impostor
  self.regrowRate = kwargs.regrowRate
end

function Collectable:onEnter(enteringGameObject, contactName)
  if (enteringGameObject:hasComponent('Avatar')) then
    -- Functionality to pick up a gem and add it to inventory.
    local avatarComponent = enteringGameObject:getComponent('Avatar')
    if (self.gameObject:getState() == self.liveState) then
      local inventoryComponent = enteringGameObject:getComponent('Inventory')
      local roleComponent = enteringGameObject:getComponent("Role")
      if (inventoryComponent:extraSpace()) then
        if roleComponent:getRole() == "crewmate" then
          avatarComponent:addReward(self.rewardForCollecting_crewmate)
          roleComponent:rewardForGemsCollected(self.rewardForCollecting_crewmate)
        elseif roleComponent:getRole() == "impostor" then
          avatarComponent:addReward(self.rewardForCollecting_impostor)
          roleComponent:rewardForGemsCollected(self.rewardForCollecting_impostor)
        end
        -- Add to events upon gem pickup.
        events:add('gem', 'dict',
                   'player_index', avatarComponent:getIndex(), -- int
                   'inventory', inventoryComponent:getNumGems()) -- int
        inventoryComponent:pickUpGem()
        self.gameObject:setState(self.waitState)
      end
    else
      -- Add to events upon unsuccessful gem pickup.
      events:add('gem_unsuccessful', 'dict',
                 'player_index', avatarComponent:getIndex()) -- int
    end
  end
end

function Collectable:registerUpdaters(updaterRegistry)
  -- Updater to control the regrowth of the gems.
  updaterRegistry:registerUpdater{
    state = self.waitState,
    probability = self.regrowRate,
    updateFn = function() self.gameObject:setState(self.liveState) end,
  }
end

-- AdditionalObserver handles exposing several additional observations.
local AdditionalObserver = class.Class(component.Component)

function AdditionalObserver:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AdditionalObserver')},
      {'num_players', args.numberType},
  })
  AdditionalObserver.Base.__init__(self, kwargs)
  self.numPlayers = kwargs.num_players
end

function AdditionalObserver:getNumPlayers()
  return self.numPlayers
end

function AdditionalObserver:addObservations(tileSet,
                                           world,
                                           observations,
                                           avatarCount)
  local avatarComponent = self.gameObject:getComponent('Avatar')
  local stringPlayerIdx = tostring(avatarComponent:getIndex())

  -- Observation for each agent's private inventory.
  local inventoryComponent = self.gameObject:getComponent('Inventory')
  observations[#observations + 1] = {
      name = stringPlayerIdx .. '.INVENTORY',
      type = 'tensor.DoubleTensor',
      shape = {1},
      func = function(grid)
        return tensor.DoubleTensor({inventoryComponent:fractionFull()})
      end
  }

  -- Observations to enable voting.
  if self.gameObject:hasComponent('Voting') then
    local votingComponent = self.gameObject:getComponent('Voting')
    if votingComponent:getVotingMethod() == 'sus' then
      -- Observation for each player's private suspicion counter.
      observations[#observations + 1] = {
        name = stringPlayerIdx .. '.SUS_COUNTER',
        type = 'tensor.DoubleTensor',
        shape = {1},
        func = function(grid)
          return tensor.DoubleTensor({votingComponent:getSuspicion()})
        end
      }
    elseif votingComponent:getVotingMethod() == 'deliberation' or
        votingComponent:getVotingMethod() == 'continuous' then
      -- Observation to display the voting matrix for each player.
      local sceneObject = self.gameObject.simulation:getSceneObject()
      local progressComponent = sceneObject:getComponent('Progress')
      observations[#observations + 1] = {
        name = stringPlayerIdx .. '.VOTING',
        type = 'tensor.DoubleTensor',
        shape = {self.numPlayers, self.numPlayers+2},
        func = function(grid)
          return progressComponent:getVotingMatrix()
        end
      }
    end
  end
end

--[[ The `Tagger` component, based on the Zapper, allows the Impostors to tag
Crewmates and for the Crewmates to be tagged. Mechanisms are the same as Zapper,
but added update functions tied to Hidden Agenda game dynamics.
]]
local Tagger = class.Class(component.Component)

function Tagger:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Tagger')},
      {'cooldownTime', args.numberType},
      {'beamLength', args.numberType},
      {'beamRadius', args.numberType},
      {'framesTillRespawn', args.default(1000), args.numberType},
      -- Reward to the Crewmate for being tagged by an Impostor.
      {'penaltyForBeingTagged', args.default(-5), args.numberType},
      -- Reward to the Impostor for tagging a Crewmate.
      {'rewardForTagging', args.default(5), args.numberType},
      -- Three methods to remove the hit player (remove, freeze, teleport).
      {'removeHitPlayer', args.default('freeze'), args.stringType},
  })
  Tagger.Base.__init__(self, kwargs)

  self._config.cooldownTime = kwargs.cooldownTime
  self._config.beamLength = kwargs.beamLength
  self._config.beamRadius = kwargs.beamRadius
  self._config.framesTillRespawn = kwargs.framesTillRespawn
  self._config.penaltyForBeingTagged = kwargs.penaltyForBeingTagged
  self._config.rewardForTagging = kwargs.rewardForTagging
  assert(kwargs.removeHitPlayer == 'remove' or
         kwargs.removeHitPlayer == 'freeze' or
         kwargs.removeHitPlayer == 'teleport' )
  self._config.removeHitPlayer = kwargs.removeHitPlayer
  self.playerRespawnedThisStep = false
  self._noTaggingCounter = 0
end

function Tagger:start()
  -- Initialize where the players get teleported to when tagged.
  if self._config.removeHitPlayer == 'teleport' then
    local sceneObject = self.gameObject.simulation:getSceneObject()
    local progressComponent = sceneObject:getComponent('Progress')
    self.teleportSpawnGroup = progressComponent:getTeleportSpawnGroup()
  end
end

function Tagger:addHits(worldConfig)
  worldConfig.hits['tagHit'] = {
      layer = 'beamZap',
      sprite = 'BeamZap',
  }
  component.insertIfNotPresent(worldConfig.renderOrder, 'beamZap')
end

function Tagger:addSprites(tileSet)
  -- Fires a yellow beam.
  tileSet:addColor('BeamZap', {252, 252, 106})
end

--[[ Get all players in reach of being tagged by the focal player in one step.

Return an array of avatars who are in reach and can be zapped by the focal
player on this timestep (if called before actions are applied) or the next
timestep (if called after actions are applied).

Note: This function assumes avatars cannot move and shoot on the same frame.

Args:
  `layer` (string): the layer on which to search for avatar objects. By default,
      the layer will be set to the current layer of the focal game object.

Returns an unsorted array of avatar objects (game objects bearing the avatar
component).
]]
function Tagger:getWhoTaggable()
  local layer = self.gameObject:getLayer()
  local transform = self.gameObject:getComponent('Transform')
  local taggableAvatars = {}

  -- Shift left and right by the correct amounts and create forward rays.
  local rays = {}
  for x = -self._config.beamRadius, self._config.beamRadius do
    local ray = {
        positionOffset = {x, 0},
        length = self._config.beamLength - math.abs(x),
        direction = nil
    }
    table.insert(rays, ray)
  end
  -- Send rays to the left and right.
  local leftVector = {-self._config.beamRadius, 0}
  local leftRay = {
      positionOffset = {0, 0},
      length = self._config.beamRadius,
      direction = transform:getAbsoluteDirectionFromRelative(leftVector),
  }
  local rightVector = {self._config.beamRadius, 0}
  local rightRay = {
      positionOffset = {0, 0},
      length = self._config.beamRadius,
      direction = transform:getAbsoluteDirectionFromRelative(rightVector),
  }
  table.insert(rays, leftRay)
  table.insert(rays, rightRay)

  -- Use one ray per horizontal offset (relative coordinate frame).
  for _, ray in ipairs(rays) do
    local success, foundObject, offset = transform:rayCastDirection(
      layer, ray.length, ray.direction, ray.positionOffset)
    if success and foundObject ~= nil and foundObject:hasComponent('Avatar') then
      table.insert(taggableAvatars, foundObject)
    end
  end

  return taggableAvatars
end

function Tagger:registerUpdaters(updaterRegistry)
  local aliveState = self:getAliveState()
  local waitState = self:getWaitState()

  local sceneObject = self.gameObject.simulation:getSceneObject()
  local progressComponent = sceneObject:getComponent('Progress')
  local avatarComponent = self.gameObject:getComponent('Avatar')

  local zap = function()
    local playerVolatileVariables = (avatarComponent:getVolatileData())
    local actions = playerVolatileVariables.actions
    -- Execute the beam if applicable.
    if avatarComponent:isAlive() then
      if self._config.cooldownTime >= 0 then
        if self._coolingTimer > 0 then
          self._coolingTimer = self._coolingTimer - 1
        else
          -- Add that the tagger can be fired this step.
          events:add('tagReady', 'dict',
                     'player_index', avatarComponent:getIndex(),
                     'num_taggable', #self:getWhoTaggable())  -- int
          if actions['tag'] == 1 then
            -- Add tagging action to events.
            events:add('tag', 'dict',
               'player_index', avatarComponent:getIndex())  -- int
            -- Execute the tagging beam.
            self._coolingTimer = self._config.cooldownTime
            self.gameObject:hitBeam(
                'tagHit', self._config.beamLength, self._config.beamRadius)
            if (self.gameObject:hasComponent('Voting')) then
              local votingComponent = self.gameObject:getComponent('Voting')
              if votingComponent:getVotingMethod() == 'sus' then
                -- Update suspicion and check voting.
                votingComponent:updateSuspicion('tag')
              elseif votingComponent:getVotingMethod() == 'deliberation' then
                if (progressComponent:doesTaggingTriggerVoting()) then
                  local numTaggable = #self:getWhoTaggable()
                  -- If the number who saw were the number who were tagged, no
                  -- alive player saw the tagging.
                  if progressComponent:getNumAvatarsSaw(self.gameObject) > numTaggable then
                    -- Time to start the voting round.
                    progressComponent:triggerVotingEvent()
                    progressComponent:setVotingRoundStatus(true)
                    progressComponent._votingCooldown = progressComponent.votingValues.votingFrameFrequency
                  end
                end
              end
            end
          end
        end
      end
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = zap,
      priority = 140,
  }

  -- Not currently used since tagged players do not respawn.
  local respawn = function()
    local spawnGroup = self.gameObject:getComponent('Avatar'):getSpawnGroup()
    self.gameObject:teleportToGroup(spawnGroup, aliveState)
    self.playerRespawnedThisStep = true
  end

  updaterRegistry:registerUpdater{
      updateFn = respawn,
      priority = 135,
      state = waitState,
      startFrame = self._config.framesTillRespawn
  }
end

function Tagger:onHit(hittingGameObject, hitName)
  if hitName == 'tagHit' then
    local taggedAvatar = self.gameObject:getComponent('Avatar')
    local taggedIndex = taggedAvatar:getIndex()
    local taggerAvatar = hittingGameObject:getComponent('Avatar')
    local taggerIndex = taggerAvatar:getIndex()

    local taggedRoleComponent = self.gameObject:getComponent("Role")
    -- The tag only counts if the player is active.
    if taggedRoleComponent:checkActive() then
      if self.playerTagMatrix then
        self.playerTagMatrix(taggedIndex, taggerIndex):add(1)
      end
      -- Add tagger hitting to events.
      events:add('tagHit', 'dict',
                 'source', taggerAvatar:getIndex(),  -- int
                 'target', taggedAvatar:getIndex())  -- int
      -- Give the tagger and tagged agents reward.
      taggedAvatar:addReward(self._config.penaltyForBeingTagged)
      taggerAvatar:addReward(self._config.rewardForTagging)

      local roleComponent = self.gameObject:getComponent("Role")

      roleComponent:inactivatePlayer()
      if self._config.removeHitPlayer == 'remove' then
        self.gameObject:setState(self:getWaitState())
        -- Make the player disappear from the map.
        roleComponent:changePlayerStatus(false, false, false)
      elseif self._config.removeHitPlayer == 'freeze' then
        self.gameObject:setState(roleComponent:getFrozenState())
        -- Freeze the player in place (no transport to jail).
        roleComponent:changePlayerStatus(false, false, false)
      elseif self._config.removeHitPlayer == 'teleport' then
        roleComponent:changePlayerStatus(false, false, false)
        -- Teleport the player to jail.
        local aliveState = self:getAliveState()
        self.gameObject:teleportToGroup(self.teleportSpawnGroup, aliveState)
      end
    end
    -- Check for Impostor tagging win condition.
    local sceneObject = self.gameObject.simulation:getSceneObject()
    local progressComponent = sceneObject:getComponent('Progress')
    progressComponent:checkImpostorTagWin()

    -- Temporarily store the index of the zapper avatar in state so it can
    -- be observed elsewhere.
    self.taggerIndex = taggerIndex
    -- return `true` to prevent the beam from passing through a hit player.
    return true
  end
end

function Tagger:onStateChange()
  self._respawnTimer = self._config.framesTillRespawn
end

function Tagger:start()
  local scene = self.gameObject.simulation:getSceneObject()
  self.playerTagMatrix = nil
  if scene:hasComponent('GlobalMetricHolder') then
    self.playerTagMatrix = scene:getComponent(
        'GlobalMetricHolder').playerTagMatrix
  end
  -- Set the beam cooldown timer to full cooldown time.
  self:resetCoolingTimer()
end

function Tagger:_handleTimedTaggingPrevention()
  local oldFreezeCounter = self._noTaggingCounter
  self._noTaggingCounter = math.max(self._noTaggingCounter - 1, 0)
  if oldFreezeCounter == 1 then
    self:allowTagging()
  end
end

function Tagger:resetCoolingTimer()
  self._coolingTimer = self._config.cooldownTime + 1
end

function Tagger:update()
  -- Metrics must be read from preUpdate since they will get reset in update.
  self.playerRespawnedThisStep = false
  self.taggerIndex = nil
  -- Note: After zapping is allowed again after having been disallowed, players
  -- still need to wait another `cooldownTime` frames before they can zap again.
  if self._disallowTagging then
    self:resetCoolingTimer()
  end
  self:_handleTimedTaggingPrevention()
end

function Tagger:getAliveState()
  return self.gameObject:getComponent('Avatar'):getAliveState()
end

function Tagger:getWaitState()
  return self.gameObject:getComponent('Avatar'):getWaitState()
end

function Tagger:getTeleportState()
  return self._config.teleportState
end

function Tagger:getTeleportSpawnGroup()
  return self.teleportSpawnGroup
end

function Tagger:readyToShoot()
  local normalizedTimeTillReady = self._coolingTimer / self._config.cooldownTime
  if self.gameObject:getComponent('Avatar'):isAlive() then
    return math.max(1 - normalizedTimeTillReady, 0)
  else
    return 0
  end
end

function Tagger:disallowTagging()
  self._disallowTagging = true
end

function Tagger:allowTagging()
  self._disallowTagging = false
end

function Tagger:disallowTaggingUntil(numFrames)
  self:disallowTagging()
  self._noTaggingCounter = numFrames
end

local AdditionalPlayerSprites = class.Class(component_library.AdditionalSprites)

function AdditionalPlayerSprites:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AdditionalPlayerSprites')},
      -- All sprites added by this component share the same render mode. If you
      -- need to mix and match render modes then you can just add two instances
      -- of this component.
      {'renderMode',
        args.default('colored_square'),
        args.oneOf('invisible', 'colored_square', 'ascii_shape')},
      {'customSpriteNames', args.default({}), args.tableType},
      {'customSpriteRGBColors', args.default({}), args.tableType},
      {'customSpriteShapes', args.default({}), args.tableType},
      {'customPalettes', args.default({}), args.tableType},
      -- Boolean flags that determine whether or not to rotate each sprite.
      {'customNoRotates', args.default({}), args.tableType},
  })
  print("INIT ADD SPRITES")
  AdditionalPlayerSprites.Base.__init__(self, kwargs)
end

-- Role handles shared functionality for the Crewmate and Impostor.
local Role = class.Class(component.Component)

function Role:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Role')},
      {'role', args.stringType},
      {'frozenState', args.stringType},
  })
  Role.Base.__init__(self, kwargs)
  self.playerActive = true
  self.frozenState = kwargs.frozenState
  assert(kwargs.role == 'crewmate' or kwargs.role == 'impostor',
         "role must be either `crewmate` or `impostor`, got `" .. kwargs.role ..
         "`.")
  self.role = kwargs.role
  self.gemsCollectedReward = 0
  self.gemsDepositedReward = 0

  self.crewVisionSteps = 0
  self.impVisionSteps = 0
end

function Role:awake()
  local object = self.gameObject
  if self:getRole() == "impostor" then
    -- Use spriteMap to create a "self" view of identity.
    -- "Player" to "Player_impostor"
    local avatarComponent = object:getComponent("Avatar")
    local appearanceComponent = object:getComponent("Appearance")
    local additionalSpritesComponent = object:getComponent("AdditionalPlayerSprites")
    local spriteNames = appearanceComponent:getSpriteNames()
    local additionalSpriteNames = additionalSpritesComponent._config.customSpriteNames
    if string.find(spriteNames[1], "Player") and
        string.find(additionalSpriteNames[1], "Player_impostor") then
      -- This should not execute in Melting Pot since the sprite map is handled
      -- in Python and the avatar sprite names are different.
      local spriteMap = {}
      spriteMap[spriteNames[1]] = "Player_impostor" .. avatarComponent:getIndex()
      avatarComponent:setSpriteMap(spriteMap)
    end
  end
end

function Role:rewardForGemsCollected(reward)
  self.gemsCollectedReward = self.gemsCollectedReward + reward
end

function Role:rewardForGemsDeposited(reward)
  self.gemsDepositedReward = self.gemsDepositedReward + reward
end

function Role:gemsDepositedReward()
  return self.gemsDepositedReward
end

function Role:gemsCollectedReward()
  return self.gemsCollectedReward
end

function Role:getRole()
  return self.role
end

function Role:getFrozenState()
  return self.frozenState
end

function Role:checkActive()
  return self.playerActive
end

function Role:inactivatePlayer()
  self.playerActive = false
end

function Role:activatePlayer()
  self.playerActive = true
end

function Role:visionTime()
  -- Determine whether the player is in view of a Crewmate or an Impostor, and
  -- increase the appropriate counter.
  local sceneObject = self.gameObject.simulation:getSceneObject()
  local progressComponent = sceneObject:getComponent('Progress')

  local selfAvatarComponent = self.gameObject:getComponent('Avatar')
  local selfPlayerIdx = tostring(selfAvatarComponent:getIndex())

  local visibleFromCrew = 0
  local visibleFromImp = 0

  for _, avatarObject in pairs(progressComponent:getActivePlayers()) do
    -- Get the avatars visible from each active player.
    local avatarComponent = avatarObject:getComponent('Avatar')
    local avatarRoleComponent = avatarObject:getComponent("Role")
    local avatarsObservable = avatarComponent:queryPartialObservationWindow(avatarObject:getLayer())
    -- Check if self was visible from the player.
    for _, visibleObject in pairs(avatarsObservable) do
      if (visibleObject:hasComponent('Avatar')) then
        local visiblePlayerIdx = tostring(visibleObject:getComponent('Avatar'):getIndex())
        if ((visiblePlayerIdx == selfPlayerIdx) and
            (selfPlayerIdx - avatarComponent:getIndex() ~= 0)) then
          if avatarRoleComponent:getRole() == 'crewmate' then
            -- Seen by a Crewmate.
            visibleFromCrew = visibleFromCrew + 1
          elseif avatarRoleComponent:getRole() == 'impostor' then
            -- Seen by an Impostor.
            visibleFromImp = visibleFromImp + 1
          end
        end
      end
    end
  end
  if visibleFromCrew > 0 then
    self.crewVisionSteps = self.crewVisionSteps + 1
  end
  if visibleFromImp > 0 then
    self.impVisionSteps = self.impVisionSteps + 1
  end
end

function Role:crewVisionTime()
  local sceneObject = self.gameObject.simulation:getSceneObject()
  local progressComponent = sceneObject:getComponent('Progress')
  return self.crewVisionSteps
end

function Role:impVisionTime()
  local sceneObject = self.gameObject.simulation:getSceneObject()
  local progressComponent = sceneObject:getComponent('Progress')
  return self.impVisionSteps
end

function Role:update()
  self:visionTime()
end

function Role:distanceFromDepositSquare()
  local transformComponent = self.gameObject:getComponent("Transform")
  local selfPosition = transformComponent:getPosition()
  local selfIdx = self.gameObject:getComponent("Avatar"):getIndex()

  local sceneObject = self.gameObject.simulation:getSceneObject()
  local progressComponent = sceneObject:getComponent('Progress')

  -- Cache gameObjects for each player.
  progressComponent:cachePlayers()

  -- Cache Deposit squares.
  progressComponent:cacheDepositSquares()

  local distanceTable = {}
  for _, object in pairs(progressComponent:getDepositSquares()) do
    if object:hasComponent("Deposit") then
      -- Calculate the distance from this player to the Deposit square.
      local playerTransformComponent = object:getComponent("Transform")
      local depositPosition = playerTransformComponent:getPosition()
      local distanceToDeposit = math.sqrt(
        math.pow((depositPosition[1]-selfPosition[1]), 2)+
        math.pow((depositPosition[2]-selfPosition[2]), 2))
      table.insert(distanceTable, distanceToDeposit)
    end
  end
  return math.min(unpack(distanceTable))
end

function Role:getPlayerDistances()
  local transformComponent = self.gameObject:getComponent("Transform")
  local selfPosition = transformComponent:getPosition()
  local selfIdx = self.gameObject:getComponent("Avatar"):getIndex()

  local sceneObject = self.gameObject.simulation:getSceneObject()
  local progressComponent = sceneObject:getComponent('Progress')

  -- Cache gameObjects for each player.
  progressComponent:cachePlayers()

  local distanceTable = {}
  for _, object in pairs(progressComponent:getAvatars()) do
    local roleComponent = object:getComponent("Role")
    local playerIdx = object:getComponent("Avatar"):getIndex()
    if selfIdx == playerIdx then
      distanceTable[playerIdx] = 0.001
    elseif roleComponent:checkActive() then
      -- If the player is active, calculate the distance from this player to
      -- the other player and add to the table.
      local playerTransformComponent = object:getComponent("Transform")
      local playerPosition = playerTransformComponent:getPosition()
      local distanceToPlayer = math.sqrt(
        math.pow((playerPosition[1]-selfPosition[1]), 2)+
        math.pow((playerPosition[2]-selfPosition[2]), 2))
      distanceTable[playerIdx] = distanceToPlayer
    else
      distanceTable[playerIdx] = -1.0
    end
  end
  return distanceTable
end

function Role:getImpostorDistance()
  if self:getRole() == "impostor" then
    return 0.001
  end

  local transformComponent = self.gameObject:getComponent("Transform")
  local selfPosition = transformComponent:getPosition()

  local sceneObject = self.gameObject.simulation:getSceneObject()
  local progressComponent = sceneObject:getComponent('Progress')

  -- Cache gameObjects for each player.
  progressComponent:cachePlayers()

  local distance_table = {}
  for _, object in pairs(progressComponent:getImpostors()) do
    local roleComponent = object:getComponent("Role")
    if roleComponent:checkActive() then
      -- If the impostor is active, calculate the distance from this player to
      -- the impostor and add to the table.
      local impTransformComponent = object:getComponent("Transform")
      local impPosition = impTransformComponent:getPosition()
      local distance_to_imp = math.sqrt(
        math.pow((impPosition[1]-selfPosition[1]), 2)+
        math.pow((impPosition[2]-selfPosition[2]), 2))
      table.insert(distance_table, distance_to_imp)
    end
  end
  if #distance_table > 0 then
    -- Return the distance to the closest Impostor.
    return math.min(unpack(distance_table))
  else
    -- Reaches this point only in the last observation as the Impostor
    -- is voted out.
    return -1.0
  end
end

function Role:changePlayerStatus(move, tag, vote)
  local object = self.gameObject
  local avatarComponent = object:getComponent('Avatar')
  -- Whether or not the player can move.
  if move == true then
    avatarComponent:allowMovement()
  elseif move == false then
    avatarComponent:disallowMovement()
  end
  -- Whether or not the player can tag.
  if (object:hasComponent('Tagger') and self:getRole() == "impostor") then
    local taggerComponent = object:getComponent('Tagger')
    if tag == true then
      taggerComponent:allowTagging()
    elseif tag == false then
      taggerComponent:disallowTagging()
    end
  end
  -- Whether or not the player can vote.
  if (object:hasComponent('Voting')) then
    local votingComponent = object:getComponent('Voting')
    local playerIdx = avatarComponent:getIndex()
    local obsComponent = self.gameObject:getComponent("AdditionalObserver")
    local numPlayers = obsComponent:getNumPlayers()
    local sceneObject = self.gameObject.simulation:getSceneObject()
    local progressComponent = sceneObject:getComponent('Progress')
    if votingComponent:getVotingMethod() == 'deliberation' or
       votingComponent:getVotingMethod() == 'continuous' then
      if self:checkActive() then
        if (votingComponent:checkVotingAllowed() == false and vote == true) or
           (vote == false) then
          -- Set vote to no-vote if voting status change or voting disallowed.
          progressComponent:submitVote(playerIdx, numPlayers+1, "disable")
        end
      else
        -- Set to tagged out if player not active.
        progressComponent:submitVote(playerIdx, numPlayers+2, "disable")
      end
    end
    if vote == true then
      votingComponent:allowVoting()
    elseif vote == false then
      votingComponent:disallowVoting()
    end
  end
end


-- Voting handles functionality for the Crewmate and Impostor voting
local Voting = class.Class(component.Component)

function Voting:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Voting')},
      {'spawnGroup', args.default('votingSpawnPoints'), args.stringType},
      {'votingActive', args.default(true), args.booleanType},
      {'votingMethod', args.default('sus'), args.stringType},
      {'votingValues', args.default({}), args.tableType},
  })
  Voting.Base.__init__(self, kwargs)
  self.votingSpawnGroup = kwargs.spawnGroup
  self._allowVoting = kwargs.votingActive
  self.votingMethod = kwargs.votingMethod
  self.votingValues = kwargs.votingValues
  assert(self.votingMethod == 'sus' or self.votingMethod == 'deliberation' or
          self.votingMethod == 'continuous')
  self.suspicionValue = 0
end

function Voting:start()
  if self.votingMethod == 'sus' then
    assert(self.votingValues.taggingVisible ~= nil and
           self.votingValues.susDecay ~= nil)
  elseif self.votingMethod == 'deliberation' then
    self._lastVote = nil
  elseif self.votingMethod == 'continuous' then
    self._lastVote = nil
  end
end

function Voting:getVotingMethod()
  return self.votingMethod
end

function Voting:getSuspicion()
  return self.suspicionValue
end

function Voting:changeSuspicion(delta)
  self.suspicionValue = self.suspicionValue + delta
  self.suspicionValue = math.max(self.suspicionValue, 0)
end

function Voting:getVotingSpawnGroup()
  return self.votingSpawnGroup
end

function Voting:getLastVote()
  return self._lastVote
end

function Voting:update()
  local sceneObject = self.gameObject.simulation:getSceneObject()
  local progressComponent = sceneObject:getComponent('Progress')
  local inVotingRound = progressComponent:getVotingRoundStatus()
  if (self.votingMethod == 'continuous' or
      self.votingMethod == 'deliberation') then
    if inVotingRound == true or self.votingMethod == 'continuous' then
      -- Voting has not completed yet (deliberation) or
      -- always available (continuous).
      local avatarComponent = self.gameObject:getComponent('Avatar')
      local actions = avatarComponent:getVolatileData().actions
      local obsComponent = self.gameObject:getComponent("AdditionalObserver")
      local numPlayers = obsComponent:getNumPlayers()
      -- Check if vote is new and can be submitted (numPlayers+1 is no vote).
      if (self:checkVotingAllowed() and
          actions['vote'] > 0 and
          actions['vote'] ~= self._lastVote) then
        self._lastVote = actions['vote']
        -- Submit the vote.
        local playerIdx = avatarComponent:getIndex()
        progressComponent:submitVote(playerIdx, self._lastVote, "vote")
        -- Add vote to the events.
        events:add('vote', 'dict',
                   'player_index', avatarComponent:getIndex(), -- int
                   'vote_index', self._lastVote)  -- int
      end
    else
      -- Voting is not currently in session.
      self._lastVote = nil
    end
  elseif self.votingMethod == 'sus' then
    self:changeSuspicion(self.votingValues.susDecay)
  end
end

function Voting:updateSuspicion(event)
  if (self.votingMethod == 'sus' and self:checkVotingAllowed() == true) then
    local sceneObject = self.gameObject.simulation:getSceneObject()
    local progressComponent = sceneObject:getComponent('Progress')

    if (self.votingValues.taggingVisible > 0 and event == 'tag') then
      local visibleFromAvatar = progressComponent:getNumAvatarsSaw(self.gameObject)
      local taggerComponent = self.gameObject:getComponent("Tagger")
      local numTaggable = #taggerComponent:getWhoTaggable()
      if visibleFromAvatar > numTaggable then
        self:changeSuspicion(self.votingValues.taggingVisible*(visibleFromAvatar-numTaggable))
        events:add('vote_sussed', 'dict',
                   'method', 'taggingVisible', -- string
                   'visibleFromAvatar', visibleFromAvatar, -- int
                   'updatedSuspicion', self:getSuspicion()) -- int
      end
    end
  end
end

function Voting:checkVotingAllowed()
    return self._allowVoting
  end

function Voting:disallowVoting()
  self._allowVoting = false
end

function Voting:allowVoting()
  self._allowVoting = true
end

local allComponents = {
    Progress = Progress,
    Inventory = Inventory,
    Deposit = Deposit,
    Collectable = Collectable,
    Tagger = Tagger,
    Voting = Voting,
    AdditionalObserver = AdditionalObserver,
    Role = Role,
    AdditionalPlayerSprites = AdditionalPlayerSprites,
}
component_registry.registerAllComponents(allComponents)
return allComponents
