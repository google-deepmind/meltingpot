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
"""Puppeteers for running_with_scissors_in_the_matrix."""

from meltingpot.utils.puppeteers import in_the_matrix


class CounterPrevious(in_the_matrix.RespondToPrevious):
  """Puppeteer for a running with scissors bot.

  This bot will always play the best response strategy to whatever its
  partner played in the previous interaction. So if its partner last played
  rock then it will play paper. If its partner last played paper then it will
  play scissors. If its partner last played scissors then it will play rock.

  Important note: this puppeteer does not discriminate between coplayers. So it
  only makes sense in two-player substrates (e.g.
  `running_with_scissors_in_the_matrix__repeated`).
  """

  def __init__(
      self,
      rock_resource: in_the_matrix.Resource,
      paper_resource: in_the_matrix.Resource,
      scissors_resource: in_the_matrix.Resource,
      margin: int,
  ) -> None:
    """Initializes the puppeteer.

    Args:
      rock_resource: The rock resource.
      paper_resource: The paper resource.
      scissors_resource: The scissors resource.
      margin: Try to collect `margin` more of the target resource than the other
        resource before interacting.
    """
    responses = {
        rock_resource: paper_resource,
        paper_resource: scissors_resource,
        scissors_resource: rock_resource,
    }
    super().__init__(responses, margin)
