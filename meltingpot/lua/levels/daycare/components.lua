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
local set = require 'common.set'
local log = require 'common.log'
local events = require 'system.events'
local random = require 'system.random'
local tensor = require 'system.tensor'

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')

local _GRASP_PRIORITY = 140
local _MOVE_PRIORITY = 150
local _AVATAR_RESPAWN = 160
local _APPLE_SPAWN = 200
local _APPLE_RESPAWN = 180
local _APPLE_EAT = 190

local _DIRECTION = {
    N = tensor.Tensor({0, -1}),
    E = tensor.Tensor({1, 0}),
    S = tensor.Tensor({0, 1}),
    F = tensor.Tensor({0, 3}),
    W = tensor.Tensor({-1, 0}),
}

local Role = class.Class(component.Component)

function Role:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Role')},
      {'isChild', args.booleanType},
  })
  Role.Base.__init__(self, kwargs)
  self._config.isChild = kwargs.isChild
end

function Role:isChild()
  return self._config.isChild
end

function Role:isParent()
  return not self._config.isChild
end

-- The `Graspable` component enables an object to be grasped by PlayerGrasp.
local Graspable = class.Class(component.Component)

function Graspable:__init__(kwargs)
  kwargs = args.parse(kwargs, {{'name', args.default('Graspable')}})
  Graspable.Base.__init__(self, kwargs)
end

function Graspable:onHit(hitterObject, hitName)
  if string.sub(hitName, 1, #'grasp_') == 'grasp_' then
    hitterObject:getComponent('PlayerGrasp'):grasp(self.gameObject)
    self._setIsGrasped = true
  end
end

function Graspable:isGrasped()
  return self._setIsGrasped
end

function Graspable:reset()
  self._setIsGrasped = false
end

function Graspable:registerUpdaters(updaterRegistry)
  local grasp = function()
    self._setIsGrasped = false

  end

  updaterRegistry:registerUpdater{
      updateFn = grasp,
      priority = _GRASP_PRIORITY - 1,
  }
end

-- The `PlayerGrasp` component endows an avatar with the ability to grasp an
-- object in the direction they are facing.

local PlayerGrasp = class.Class(component.Component)

function PlayerGrasp:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('PlayerGrasp')},
      {'shape', args.stringType},
      {'palette', args.tableType},
      {'canGraspTree', args.booleanType},
      {'graspSuccessProbability', args.ge(0.0), args.le(1.0), args.default(1.0)},
      --parent gets reward if they pick and drop something the child pointed at
      {'attentiveParentPseudoreward', args.default(0.0), args.numberType},
      --parent gets reward for anything they pick and drop
      {'droppingParentPseudoreward', args.default(0.0), args.numberType},
      --child gets reward for failed graps
      {'tryingChildPseudoreward', args.default(0.0), args.numberType},
      --child gets reward for failed graps at banana trees
      {'tryingChildBananaPseudoreward', args.default(0.0), args.numberType},
  })

  PlayerGrasp.Base.__init__(self, kwargs)
  self._config.shape = kwargs.shape
  self._config.palette = kwargs.palette
  self._config.canGraspTree = kwargs.canGraspTree
  self._config.graspSuccessProbability = kwargs.graspSuccessProbability
  self._config.attentiveParentPseudoreward = kwargs.attentiveParentPseudoreward
  self._config.droppingParentPseudoreward = kwargs.droppingParentPseudoreward
  self._config.tryingChildPseudoreward = kwargs.tryingChildPseudoreward
  self._config.tryingChildBananaPseudoreward = kwargs.tryingChildBananaPseudoreward
end

function PlayerGrasp:awake()
  self._hitName = 'grasp_' .. self:avatarIndex()
end

function PlayerGrasp:reset()
  self._lastGraspAction = false
  self._avatar = self.gameObject:getComponent('Avatar')
  self._avatar_idx = self._avatar:getIndex()
  if self.gameObject:getComponent('Role'):isChild() then
    self._avatar_role = 'child'
  else
    self._avatar_role = 'parent'
  end
  self._graspedObject = 'empty'
end

function insertAtIfNotPresent(tbl, insert_at, element)
  for _, value in pairs(tbl) do
    if value == element then
      return
    end
  end
  table.insert(tbl, insert_at, element)
end

function PlayerGrasp:addHits(worldConfig)
  -- Add the grasp beam underneath the overlay layer.
  local insert_at = 0
  for index, layer in pairs(worldConfig.renderOrder) do
    if layer == 'overlay' then
      insert_at = index
      break
    end
  end
  insertAtIfNotPresent(worldConfig.renderOrder, insert_at, self._hitName)
  worldConfig.hits[self._hitName] = {
      layer = self._hitName,
      sprite = self._hitName,
  }
end

function PlayerGrasp:addSprites(tileSet)
  tileSet:addShape(self._hitName, {
      palette = self._config.palette,
      text = self._config.shape})
end

-- Ungrasping logic: don't allow ungrasping on top of trees
-- Give pseudorewards to parent when ungrasping for the first time.
function PlayerGrasp:unGrasp()
    local transform = self._graspedObject:getComponent('Transform')
    local position = transform:getPosition()
    local maybe_empty_tree = transform:queryPosition('lowerPhysical', position)
    local maybe_full_tree = transform:queryPosition('upperPhysical', position)
    if maybe_empty_tree == nil and maybe_full_tree == nil then
      local fruit = self._graspedObject:getComponent('FruitType')
      if fruit.initial_dropper_role == nil then
        events:add("ungrasp", "dict",
        "player_index", self._avatar_idx,
        "fruit_type", fruit.fruitType,
        "height", fruit.originalHeight,
        "player_role", self._avatar_role,
        "child_attempted", tostring(fruit.child_attempted_grasp))
        fruit.initial_dropper_role = self._avatar_role
        if self._avatar_role == 'parent' then
          self._avatar:addReward(
            self._config.droppingParentPseudoreward)
        end
        if self._avatar_role == 'parent' and fruit.child_attempted_grasp then
          self._avatar:addReward(
            self._config.attentiveParentPseudoreward)
        end
      end
      self.gameObject:connect(self._graspedObject)
      self.gameObject:disconnect()
      self._graspedObject = 'empty'
    end
end

function PlayerGrasp:registerUpdaters(updaterRegistry)
  local grasp = function()
    local playerVolatileVariables = self._avatar:getVolatileData()
    local graspAction = playerVolatileVariables.actions['grasp'] == 1

    -- If the player is holding an object continue holding it until the player
    -- tries grasping again.
    -- If the player is not holding an object, attempt to grasp when the grasp
    -- action is sent.
    local isHoldingObject = self._graspedObject ~= 'empty'
    local changeGrasp = graspAction and not self._lastGraspAction

    if isHoldingObject and changeGrasp then
      self:unGrasp()
    end

    if isHoldingObject ~= changeGrasp then
      self.gameObject:hitBeam(self._hitName, 1, 0)
    end

    self._lastGraspAction = graspAction
  end

  local rotate_grasped = function()
    -- Only rotate the grasped object if there is one connected.
    if self._graspedObject == 'empty' then
      return
    end

    -- Only rotate the grasped object if the player rotates.
    local playerVolatileVariables = self._avatar:getVolatileData()
    local turn_action = playerVolatileVariables.actions['turn']
    if turn_action == 0 then
      return
    end

    -- Determine the direction the player will be facing on the next turn.
    local _COMPASS = {N = 1, E = 2, S = 3, W = 4}
    local _COMPASSKEYS = {'N', 'E', 'S', 'W'}
    local function rotate(facing, turn)
      return _COMPASSKEYS[(_COMPASS[facing] - 1 + turn + 4) % 4 + 1]
    end

    local playerDir = rotate(self.gameObject:getOrientation(), turn_action)
    local objectDir = rotate(self._graspedObject:getOrientation(), turn_action)

    -- Teleport the object to the location the player will be facing.
    local offsetPosition = tensor.Tensor(self.gameObject:getPosition()):cadd(
        _DIRECTION[playerDir]):val()
    self._graspedObject:disconnect()
    self._graspedObject:teleport(offsetPosition, objectDir)
    self.gameObject:connect(self._graspedObject)
  end

  updaterRegistry:registerUpdater{
      updateFn = grasp,
      priority = _GRASP_PRIORITY,
  }
  updaterRegistry:registerUpdater{
      updateFn = rotate_grasped,
      priority = _MOVE_PRIORITY - 1,
  }
end

function PlayerGrasp:avatarIndex()
  return self.gameObject:getComponent('Avatar'):getIndex()
end

function PlayerGrasp:failedGrasp(fruit)
  events:add("failed_grasp", "dict",
     "height", fruit.originalHeight,
     "fruit_type", fruit.fruitType,
     "player_index", self.gameObject:getComponent('Avatar'):getIndex(),
     "player_role", self._avatar_role)
   fruit.child_attempted_grasp = true
  self.gameObject:getComponent('Avatar'):addReward(self._config.tryingChildPseudoreward)
  if fruit:isBanana() then
    self.gameObject:getComponent('Avatar')
    :addReward(self._config.tryingChildBananaPseudoreward)
  end
 end

function PlayerGrasp:grasp(gameObject)
  if self._graspedObject ~= 'empty' then
    return self._graspedObject
  end
  local graspProbability = self._config.graspSuccessProbability
  local failedGrasp = random:uniformReal(0, 1) > graspProbability
  local fruit = gameObject:getComponent('FruitType')
  if fruit:isShrub() and failedGrasp then
    self:failedGrasp(fruit)
    return nil
  end
  local canNotGraspTree = not self._config.canGraspTree

  if fruit:isTree() and canNotGraspTree then
    self:failedGrasp(fruit)
    return nil
  end
  local picked = fruit:pickFruitFromTree()
  if picked then
    fruit.initial_picker_role = self._avatar_role
    events:add("success_grasp", "dict",
      "height", fruit.originalHeight,
      "fruit_type", fruit.fruitType,
      "initial_picker_role", fruit.initial_picker_role,
      "player_index", self.gameObject:getComponent('Avatar'):getIndex(),
      "player_role", self._avatar_role)
  end
  -- disconnect from everyone else currently holding the same object
  local players = self.gameObject.simulation:getGameObjectsByName("avatar")
  for i, player in pairs(players) do
    local go = player:getComponent('PlayerGrasp')._graspedObject
    if go == gameObject then
      player:disconnect()
      player:getComponent('PlayerGrasp')._graspedObject = 'empty'
    end
  end
  self._graspedObject = gameObject
  self.gameObject:connect(self._graspedObject)
  return self._graspedObject
end

--[[ `TreeType` makes a tree probabilistically yield
either apples or bananas.
]]
local TreeType = class.Class(component.Component)

function TreeType:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('TreeType')},
      {'probabilities', args.tableType},
  })
  TreeType.Base.__init__(self, kwargs)

  self._config.probabilities = kwargs.probabilities
  -- Normalize tree spawn probabilities to one.
  self:normalizeTreeTypeProbabilitiesSumToOne()
end

function TreeType:normalizeTreeTypeProbabilitiesSumToOne()
  local sum = (self._config.probabilities['empty'] +
               self._config.probabilities['appleTree'] +
               self._config.probabilities['bananaTree'] +
               self._config.probabilities['appleShrub'] +
               self._config.probabilities['bananaShrub'])
  self._config.probabilities['empty'] = self._config.probabilities['empty'] / sum
  self._config.probabilities['appleTree'] = self._config.probabilities['appleTree'] / sum
  self._config.probabilities['bananaTree'] = self._config.probabilities['bananaTree'] / sum
  self._config.probabilities['appleShrub'] = self._config.probabilities['appleShrub'] / sum
  self._config.probabilities['bananaShrub'] = self._config.probabilities['bananaShrub'] / sum
end

function TreeType:_weighted_random_choice(list)
  local choice = random:uniformReal(0, 1)
  for idx=1, #list do
    choice = choice - list[idx][1]
    if choice <= 0.0 then
      return list[idx][2]
    end
  end
  assert(false, 'Weighted probabilities must sum to 1.')
end

function TreeType:spawn()
  local states = {'empty', 'appleTree',
    'bananaTree', 'appleShrub', 'bananaShrub'}
  local weighted_states_list = {}
  for idx=1, #states do
    proba = self._config.probabilities[states[idx]]
    table.insert(weighted_states_list, {proba, states[idx]})
  end
  state = TreeType:_weighted_random_choice(weighted_states_list)
  if state ~= 'empty' then
    self.gameObject:setState(state)
  end
end

function TreeType:start()
  self:spawn()
end


--[[ `FruitType` keeps information about a fruit.
]]
local FruitType = class.Class(component.Component)

function FruitType:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('FruitType')},
      {'framesTillAppleRespawn', args.positive},
  })
  FruitType.Base.__init__(self, kwargs)
  self._config.framesTillAppleRespawn = kwargs.framesTillAppleRespawn
end

function FruitType:saveInitialPosition(position)
  self._position = position
  -- the lua idx of the player who first picked the apple
end

--save information about who attempts to pick the tree
function FruitType:initializePickingInfo()
  --Who picked the fruit first: nil, parent or child
  self.initial_picker_role = nil
 --Who picked and dropped the fruit first: nil, parent or child
  self.initial_dropper_role = nil
  --true when child tries and fails to grasp
  self.child_attempted_grasp = false
end

function FruitType:getInitialPosition()
  return self._position
end

function FruitType:setInfoFromTreeState(treeTypeState)
  fruits = {'apple', 'banana'}
  heights = {'Tree', 'Shrub'}
  for fidx=1, #fruits do
    for hidx=1, #heights do
      state = fruits[fidx] .. heights[hidx]
      if treeTypeState == state then
          self.liveState = fruits[fidx] .. "In" .. heights[hidx]
          self.gameObject:setState(self.liveState)
          self.originalHeight = heights[hidx]
          self.fruitType = fruits[fidx]
        return
      end
    end
  end
end


function FruitType:registerUpdaters(updaterRegistry)
    local spawn = function()
      local transform = self.gameObject:getComponent('Transform')
      local position = transform:getPosition()
      local potentialTree = transform:queryPosition('lowerPhysical', position)
      if potentialTree ~= nil and self.gameObject:getState() == 'fruitWait' then
        local treeTypeState = potentialTree:getState()
        self:setInfoFromTreeState(treeTypeState)
        self:saveInitialPosition(position)
        self:initializePickingInfo()
      end

    end

    local respawn = function()
      self.gameObject:setState(self.liveState)
      position = self:getInitialPosition()
      self.gameObject:teleport({position[1], position[2]}, "N")
      self:initializePickingInfo()
    end

  updaterRegistry:registerUpdater{
      updateFn = spawn,
      priority = _APPLE_SPAWN,
      state = "fruitWait",
  }
    updaterRegistry:registerUpdater{
      updateFn = respawn,
      priority = _APPLE_RESPAWN,
      state = "fruitEaten",
      startFrame = self._config.framesTillAppleRespawn
  }
end

function FruitType:isTree()
  return string.sub(self.gameObject:getState(), -#'Tree', -1) == 'Tree'
end

function FruitType:isShrub()
  return string.sub(self.gameObject:getState(), -#'Shrub', -1) == 'Shrub'
end

function FruitType:isPicked()
  return string.sub(self.gameObject:getState(), -#'Picked', -1) == 'Picked'
end

function FruitType:isApple()
  return string.sub(self.gameObject:getState(), 1, #'apple') == 'apple'
end

function FruitType:isBanana()
  return string.sub(self.gameObject:getState(), 1, #'banana') == 'banana'
end


function FruitType:pickFruitFromTree()
  local state = self.gameObject:getState()
  if state == "appleInTree" or state == "appleInShrub" then
    self.gameObject:setState('applePicked')
    return true
  elseif state == "bananaInTree" or state == "bananaInShrub" then
    self.gameObject:setState('bananaPicked')
    return true
  end
  -- fruit was already picked
  return false
end


--[[ `Eating` endows avatars with the ability to eat items and get rewards and
additionally fulfill hunger for children.
]]
local Eating = class.Class(component.Component)

function Eating:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Eating')},
      -- Eating rewards.
      {'bananaReward', args.numberType},
      {'appleReward', args.numberType},
  })
  Eating.Base.__init__(self, kwargs)
  self._config.bananaReward = kwargs.bananaReward
  self._config.appleReward = kwargs.appleReward
end

function Eating:isChildPresent()
  local players = self.gameObject.simulation:getGameObjectsByName("avatar")
  for i, player in pairs(players) do
    if player:getComponent("Role"):isChild() and
      player:getState() ~= 'playerWait' then
        return true
    end
  end
  return false
end

function Eating:calculateReward(isBanana, isChild)
  -- parent only receives reward if child is present
  if isChild or self:isChildPresent() then
    if isBanana then
      return self._config.bananaReward
    else
      return self._config.appleReward
    end
  end
  return 0
end


function Eating:registerUpdaters(updaterRegistry)
  local avatar = self.gameObject:getComponent('Avatar')
  local grasp = self.gameObject:getComponent('PlayerGrasp')
  local eat = function()
    local playerVolatileVariables = (
        self.gameObject:getComponent('Avatar'):getVolatileData())
    local actions = playerVolatileVariables.actions
    if actions['eat'] == 1 and grasp._graspedObject ~= 'empty' then
        grasp._graspedObject:setState("fruitEaten")
        local fruit = grasp._graspedObject:getComponent('FruitType')
        local isBanana = fruit:isBanana()
        local isChild = self.gameObject:getComponent("Role"):isChild()
        local reward = self:calculateReward(isBanana, isChild)
        events:add('fruit_eaten', 'dict',
          'height',fruit.originalHeight,
          'fruit_type', fruit.fruitType,
          'player_index', avatar:getIndex(),
          "player_role", grasp._avatar_role,
          'eater_is_child', tostring(isChild),
          'initial_picker_role', fruit.initial_picker_role,
          'initial_dropper_role', tostring(fruit.initial_dropper_role),
          'child_attempted', tostring(fruit.child_attempted_grasp))
        avatar:addReward(reward)
        if isBanana and isChild then
            self.gameObject:getComponent('Hunger'):resetDriveLevel()
        end
       self.gameObject:disconnect()
       self.gameObject:getComponent('PlayerGrasp')._graspedObject = 'empty'
    end
end

  updaterRegistry:registerUpdater{
      updateFn = eat,
      priority = _APPLE_EAT,
  }
end

--[[ The `Hunger` component keeps track of the child's hugner level.
]]
local Hunger = class.Class(component.Component)
function Hunger:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Hunger')},
      {'framesTillHungry', args.numberType},
  })
  Hunger.Base.__init__(self, kwargs)
  self._config.framesTillHungry = kwargs.framesTillHungry
end

function Hunger:reset()
  self._hungerTimer = self._config.framesTillHungry
end

-- Call this function to reset the countdown, i.e., to "satisfy" the need.
function Hunger:resetDriveLevel()
  self._hungerTimer = self._config.framesTillHungry
  events:add("player_hunger_reset", "dict", "hunger_timer", self._hungerTimer,
    "player_index", self.gameObject:getComponent('Avatar'):getIndex())
end

function Hunger:update()
  local isChild = self.gameObject:getComponent("Role"):isChild()
  if not isChild then
    return
  end
  if self.gameObject:getState() ~= 'playerWait' then
    self._hungerTimer = self._hungerTimer - 1
    if self._hungerTimer == 0 then
      local graspedObject = self.gameObject:getComponent("PlayerGrasp")._graspedObject
      -- mark grasped fruit as eaten so it can respawn if child is in wait state
      if graspedObject  ~= 'empty' then
        self.gameObject:connect(graspedObject)
        self.gameObject:disconnect()
        graspedObject:setState('fruitEaten')
        self.gameObject:getComponent("PlayerGrasp")._graspedObject = 'empty'
        self.gameObject:getComponent("PlayerGrasp")._lastGraspAction = false
      end
      self.gameObject:setState('playerWait')
      events:add("player_wait", "dict", "hunger_timer", self._hungerTimer,
        "player_index", self.gameObject:getComponent('Avatar'):getIndex())
    end
  end
end

function Hunger:getNeed()
  local isChild = self.gameObject:getComponent("Role"):isChild()
  if not isChild then
    return 1
  end
  local normalizedHunger = self._hungerTimer / self._config.framesTillHungry
  if self.gameObject:getState() ~= 'playerWait' then
    return normalizedHunger
  else
    return 0
  end
end

local HungerObserver = class.Class(component.Component)

function HungerObserver:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('HungerObserver')},
      {'needComponent', args.default('Hunger'), args.stringType},
  })
  HungerObserver.Base.__init__(self, kwargs)
  self._config.needComponent = kwargs.needComponent
end

function HungerObserver:addObservations(tileSet, world, observations)
  local playerIndex = self.gameObject:getComponent('Avatar'):getIndex()
  local needComponent = self.gameObject:getComponent(self._config.needComponent)
  observations[#observations + 1] = {
      name = tostring(playerIndex) .. '.HUNGER',
      type = 'Doubles',
      shape = {},
      func = function(grid)
        return needComponent:getNeed()
      end
  }
end

local AvatarRespawn = class.Class(component.Component)

function AvatarRespawn:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AvatarRespawn')},
      {'framesTillRespawn', args.positive},
  })
  AvatarRespawn.Base.__init__(self, kwargs)
  self._config.framesTillRespawn = kwargs.framesTillRespawn
end

function AvatarRespawn:registerUpdaters(updaterRegistry)
  local avatar = self.gameObject:getComponent('Avatar')
  local aliveState = avatar:getAliveState()
  local waitState = avatar:getWaitState()
  local respawn = function()
    -- find parent position
    local transform = nil
    local players = self.gameObject.simulation:getGameObjectsByName("avatar")
    for i, player in pairs(players) do
      if not player:getComponent("Role"):isChild() then
        transform = player:getComponent('Transform')
      end
    end
    assert(transform ~= nil, "The parent avatar was not found!")

    -- try to respawn on nearby ground
    local nearby = transform:queryDisc('background', 1)
    assert(#nearby > 0, "There is no nearby background to respawn on!")

    local nearby_object = nearby[1]
    local pos = nearby_object:getComponent("Transform"):getPosition()
    local teleport_position = {pos[1], pos[2]}
    self.gameObject:setState(aliveState)
    self.gameObject:teleport(teleport_position, 'N')

    self.gameObject:getComponent('Hunger'):reset()
    events:add("player_respawned", "dict", "hunger_timer",
      "player_index", avatar:getIndex())

  end

  updaterRegistry:registerUpdater{
      updateFn = respawn,
      priority = _AVATAR_RESPAWN,
      state = waitState,
      startFrame = self._config.framesTillRespawn
  }
end

local allComponents = {
  -- Avatar components.
  Role = Role,
  PlayerGrasp = PlayerGrasp,
  Eating = Eating,
  Hunger = Hunger,
  HungerObserver = HungerObserver,
  AvatarRespawn = AvatarRespawn,
  -- Object components.
  Graspable = Graspable,
  -- Fruit tree components.
  FruitType = FruitType,
  TreeType = TreeType,
}

component_registry.registerAllComponents(allComponents)

return allComponents
