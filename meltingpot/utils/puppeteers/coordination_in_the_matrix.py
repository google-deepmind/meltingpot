# Copyright 2022 DeepMind Technologies Limited.
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
"""Puppeteers for *_coordination_in_the_matrix."""

from typing import Iterable

from meltingpot.utils.puppeteers import in_the_matrix


class CoordinateWithPrevious(in_the_matrix.RespondToPrevious):
  """Puppeteer to use in pure/rationalizable coordination in the matrix.

  This bot will always play the same strategy to whatever its partner played in
  the previous interaction. So if its last partner played resource A then it
  will target resource A, if its last partner played resource B then it
  will target resource B, and so on.

  Important note: this puppeteer does not discriminate between coplayers. It may
  not make sense to use this beyond two-player substrates.
  """

  def __init__(
      self,
      resources: Iterable[in_the_matrix.Resource],
      margin: int,
  ) -> None:
    """Initializes the puppeteer.

    Args:
      resources: The collectible resources to coordinate on.
      margin: Try to collect `margin` more of the target resource than the other
        resource before interacting.
    """
    responses = {resource: resource for resource in resources}
    super().__init__(responses, margin)
