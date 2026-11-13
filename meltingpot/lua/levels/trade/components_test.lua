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

local components = require 'meltingpot.lua.levels.trade.components'
local test_runner = require 'testing.test_runner'

local tests = {}

local function makeInventory(items)
  local inventory = {items = items}
  function inventory:quantity(item)
    return self.items[item]
  end
  function inventory:add(item, amount)
    self.items[item] = self.items[item] + amount
    assert(self.items[item] >= 0)
  end
  return inventory
end

local function makePackage(offer, items, playerIndex)
  local inventory = makeInventory(items)
  local trading = setmetatable({
      _config = {itemTypes = {'apple', 'banana', 'pear'}},
      _offer = offer,
      canceled = 0,
  }, {__index = components.Trading})
  function trading:cancelOffer()
    self.canceled = self.canceled + 1
  end
  local avatar = {}
  function avatar:getComponent(name)
    assert(name == 'Inventory')
    return inventory
  end
  trading.gameObject = avatar
  local package = {
      offer = offer,
      avatarObject = avatar,
      trading = trading,
      partnerId = playerIndex,
  }
  return package, inventory
end

function tests.bothSellersKeepUnrequestedFruit()
  local a, inventoryA = makePackage(
      {apple = -5, banana = 3, pear = -1},
      {apple = 5, banana = 0, pear = 1}, 1)
  local b, inventoryB = makePackage(
      {apple = -2, banana = -3, pear = 1},
      {apple = 2, banana = 3, pear = 0}, 2)

  assert(a.trading:isValid(a.offer))
  assert(b.trading:isValid(b.offer))
  assert(a.trading:isCompatible(b.offer))
  assert(b.trading:isCompatible(a.offer))
  assert(a.trading:hasEnough(a.offer))
  assert(b.trading:hasEnough(b.offer))

  a.trading:resolve(a, b)

  -- Neither trader requested apples. Only bananas and pears should move.
  assert(inventoryA:quantity('apple') == 5)
  assert(inventoryB:quantity('apple') == 2)
  assert(inventoryA:quantity('banana') == 3)
  assert(inventoryB:quantity('banana') == 0)
  assert(inventoryA:quantity('pear') == 0)
  assert(inventoryB:quantity('pear') == 1)
  assert(a.trading.canceled == 1)
  assert(b.trading.canceled == 1)
end

function tests.opposingOffersStillExchangeItems()
  local a, inventoryA = makePackage(
      {apple = -2, banana = 2, pear = 0},
      {apple = 2, banana = 0, pear = 0}, 1)
  local b, inventoryB = makePackage(
      {apple = 2, banana = -2, pear = 0},
      {apple = 0, banana = 2, pear = 0}, 2)

  a.trading:resolve(a, b)

  assert(inventoryA:quantity('apple') == 0)
  assert(inventoryB:quantity('apple') == 2)
  assert(inventoryA:quantity('banana') == 2)
  assert(inventoryB:quantity('banana') == 0)
end

return test_runner.run(tests)
