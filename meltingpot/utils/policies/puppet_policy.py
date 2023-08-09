# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Puppet policy implementation."""

from typing import Generic, Tuple, TypeVar

import dm_env
from meltingpot.utils.policies import policy
from meltingpot.utils.puppeteers import puppeteer as puppeteer_lib

PuppeteerState = TypeVar('PuppeteerState')
PolicyState = TypeVar('PolicyState')


class PuppetPolicy(policy.Policy[Tuple[PuppeteerState, PolicyState]],
                   Generic[PuppeteerState, PolicyState]):
  """A puppet policy controlled by a puppeteer function."""

  def __init__(
      self,
      puppeteer: puppeteer_lib.Puppeteer[PuppeteerState],
      puppet: policy.Policy[PolicyState]) -> None:
    """Creates a new PuppetBot.

    Args:
      puppeteer: Puppeteer that will be called at every step to modify the
        timestep forwarded to the underlying puppet.
      puppet: The puppet policy. Will be closed with this wrapper.
    """
    self._puppeteer = puppeteer
    self._puppet = puppet

  def step(
      self,
      timestep: dm_env.TimeStep,
      prev_state: Tuple[PuppeteerState, PolicyState],
  ) -> Tuple[int, Tuple[PuppeteerState, PolicyState]]:
    """See base class."""
    puppeteer_state, puppet_state = prev_state
    puppet_timestep, puppeteer_state = self._puppeteer.step(
        timestep, puppeteer_state)
    action, puppet_state = self._puppet.step(puppet_timestep, puppet_state)
    next_state = (puppeteer_state, puppet_state)
    return action, next_state

  def initial_state(self) -> Tuple[PuppeteerState, PolicyState]:
    """See base class."""
    return (self._puppeteer.initial_state(), self._puppet.initial_state())

  def close(self) -> None:
    """See base class."""
    self._puppet.close()
