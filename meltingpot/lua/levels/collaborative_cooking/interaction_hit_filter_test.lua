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

local components = require(
    'meltingpot.lua.levels.collaborative_cooking.components')
local asserts = require 'testing.asserts'
local test_runner = require 'testing.test_runner'
local tests = {}

local function inventory(item)
  return {item = item,
          getHeldItem = function(self) return self.item end,
          setHeldItem = function(self, value) self.item = value end}
end

local function player(item, hitName)
  local inv = inventory(item)
  local avatar = {reward = 0, getIndex = function() return 1 end,
                 addReward = function(self, value)
                   self.reward = self.reward + value
                 end}
  local beam = {getHitName = function() return hitName end,
                getAvatarsInventory = function() return inv end}
  local cumulants = components.AvatarCumulants{}
  local object = {
      getComponent = function(_, name)
        if name == 'Avatar' then return avatar end
        if name == 'InteractBeam' then return beam end
        if name == 'AvatarCumulants' then return cumulants end
        error('Unexpected component: ' .. name)
      end,
      hasComponent = function(_, name) return name == 'AvatarCumulants' end,
  }
  return object, inv, avatar, cumulants
end

for _, name in ipairs({'interact', 'customInteraction'}) do
  for _, other in ipairs({'zapHit', 'otherInteraction'}) do
    local suffix = name .. '_' .. other
    tests['container_' .. suffix] = function()
      local object, inv = player('empty', name)
      local stored = inventory('tomato')
      local container = components.Container{startingItem = 'tomato'}
      container:attachInventory(stored)
      container:onHit(object, other)
      asserts.EQ(inv.item, 'empty')
      asserts.EQ(stored.item, 'tomato')
      asserts.EQ(container._usedThisStep, false)
      container:onHit(object, name)
      asserts.EQ(inv.item, 'tomato')
      asserts.EQ(stored.item, 'empty')
      asserts.EQ(container._usedThisStep, true)
    end
    tests['receiver_' .. suffix] = function()
      local object, inv, avatar = player('soup', name)
      local receiver = components.Receiver{acceptedItems = 'soup', reward = 20}
      receiver:onHit(object, other)
      asserts.EQ(inv.item, 'soup')
      asserts.EQ(avatar.reward, 0)
      receiver:onHit(object, name)
      asserts.EQ(inv.item, 'empty')
      asserts.EQ(avatar.reward, 20)
    end
    tests['potIngredient_' .. suffix] = function()
      local object, inv, avatar, cumulants = player('tomato', name)
      local pot = components.CookingPot{
          reward = 2, acceptedItems = {'tomato'},
          customStateNames = {'tomato_empty_empty', 'empty_empty_empty'},
      }
      local calls = 0
      pot.gameObject = {setState = function(_, state)
        asserts.EQ(state, 'tomato_empty_empty'); calls = calls + 1
      end}
      pot:reset()
      pot:onHit(object, other)
      asserts.EQ(inv.item, 'tomato')
      asserts.EQ(#pot._containedItems, 0)
      asserts.EQ(avatar.reward, 0)
      asserts.EQ(cumulants.addedIngredientToCookingPot, 0)
      asserts.EQ(calls, 0)
      pot:onHit(object, name)
      asserts.EQ(inv.item, 'empty')
      asserts.EQ(#pot._containedItems, 1)
      asserts.EQ(avatar.reward, 2)
      asserts.EQ(cumulants.addedIngredientToCookingPot, 1)
    end
    tests['potCollection_' .. suffix] = function()
      local object, inv, avatar, cumulants = player('dish', name)
      local pot = components.CookingPot{
          reward = 2, customStateNames = {'empty_empty_empty'},
      }
      pot.gameObject = {setState = function(_, state)
        asserts.EQ(state, 'empty_empty_empty')
      end}
      pot:reset()
      pot._cooked = true
      pot._containedItems = {'tomato', 'tomato', 'tomato'}
      pot:onHit(object, other)
      asserts.EQ(inv.item, 'dish'); asserts.EQ(pot:isCooked(), true)
      asserts.EQ(avatar.reward, 0)
      asserts.EQ(cumulants.collectedSoupFromCookingPot, 0)
      pot:onHit(object, name)
      asserts.EQ(inv.item, 'soup'); asserts.EQ(pot:isCooked(), false)
      asserts.EQ(avatar.reward, 2)
      asserts.EQ(cumulants.collectedSoupFromCookingPot, 1)
    end
  end
end

function tests.matchedContainerHitStillRespectsPerStepUse()
  local object, inv = player('empty', 'interact')
  local stored = inventory('tomato')
  local container = components.Container{startingItem = 'tomato'}
  container:attachInventory(stored)
  container:onHit(object, 'interact')
  container:onHit(object, 'interact')
  asserts.EQ(inv.item, 'tomato')
  asserts.EQ(stored.item, 'empty')
end

function tests.matchedReceiverStillRejectsWrongFood()
  local object, inv, avatar = player('tomato', 'interact')
  local receiver = components.Receiver{acceptedItems = 'soup', reward = 20}
  receiver:onHit(object, 'interact')
  asserts.EQ(inv.item, 'tomato')
  asserts.EQ(avatar.reward, 0)
end

return test_runner.run(tests)
