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
"""Tools to help with parsing and procedurally generating ascii maps."""

from collections.abc import Mapping, Sequence
from typing import Any, Union


def a_or_b_with_odds(a_descriptor: Union[str, Mapping[str, Any]],
                     b_descriptor: Union[str, Mapping[str, Any]],
                     odds: Sequence[int]) -> Mapping[str, Any]:
  """Return a versus b with specified odds.

  Args:
      a_descriptor: One possibility. May be either a string or a dict that can
          be read by the map parser.
      b_descriptor: The other possibility. May be either a string or a dict that
          can be read by the map parser.
      odds: odds[0] is the number of outcomes where a is returned. odds[1] is
          the number of outcomes where b is returned. Thus the probability of
          returning a is odds[0] / sum(odds) and the probability of returning
          b is odds[1] / sum(odds).

  Returns:
      The dict descriptor that can be used with the map parser to sample either
      a or b at the specified odds.
  """
  a_odds, b_odds = odds
  choices = [a_descriptor] * a_odds + [b_descriptor] * b_odds
  return {"type": "choice", "list": choices}
