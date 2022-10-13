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
"""A set of commonly used ASCII art shape and helper functions for DMLab2D."""

import colorsys
from typing import Dict, Optional, Tuple, Union

ColorRGBA = Tuple[int, int, int, int]
ColorRGB = Tuple[int, int, int]
Color = Union[ColorRGB, ColorRGBA]

VEGETAL_GREEN = (100, 120, 0, 255)
LEAF_GREEN = (64, 140, 0, 255)
ALPHA = (0, 0, 0, 0)
WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)
DARK_GRAY = (60, 60, 60, 255)
TREE_BROWN = (128, 92, 0, 255)
DARK_FLAME = (226, 88, 34, 255)
LIGHT_FLAME = (226, 184, 34, 255)
DARK_STONE = (153, 153, 153, 255)
LIGHT_STONE = (204, 204, 204, 255)


def rgb_to_rgba(rgb: ColorRGB, alpha: int = 255) -> ColorRGBA:
  return (rgb[0], rgb[1], rgb[2], alpha)


def scale_color(color_tuple: ColorRGBA, factor: float,
                alpha: Optional[int] = None) -> ColorRGBA:
  """Scale an RGBA color tuple by a given factor.

  This function scales, multiplicatively, the RGB values of a color tuple by the
  given amount, clamped to a maximum of 255. The alpha channel is either
  overwritten by the specified one, or if none is specified, it is inherited by
  the original color.

  Args:
    color_tuple: The original color to scale.
    factor: The factor to multiplicatively scale the RGB channels by.
    alpha: If provided, the new color will have this alpha, otherwise, inherit
      from original color_tuple.

  Returns:
    A new color tuple, with its RGB channels scaled.
  """
  if len(color_tuple) == 3:
    color_tuple = rgb_to_rgba(color_tuple)  # pytype: disable=wrong-arg-types
  scaled = [int(min(x * factor, 255)) for x in color_tuple]
  scaled[3] = alpha if alpha is not None else color_tuple[-1]
  return tuple(scaled)


# LINT.IfChange
def get_palette(color: Color) -> Dict[str, ColorRGBA]:
  """Convert provided color to a palette suitable for the player text shape.

  The overall palette is:

  'x' Transparent
  ',' Black
  'O' Dark gray
  'o' 45% darker color than the base palette color
  '&' 25% darker color than the base palette color
  '*' The base color of the palette
  '@' 25% lighter color than the base palette color
  '#' White
  'r' A rotation of the main color: RGB -> RBG
  'R' A 25% lighter color than the rotation of the main color: RGB -> RBG

  Args:
    color (tuple of length 4): Red, Green, Blue, Alpha (transparency).

  Returns:
    palette (dict): maps palette symbols to suitable colors.
  """
  palette = {
      "*": (color[0], color[1], color[2], 255),
      "&": scale_color(color, 0.75, 255),
      "o": scale_color(color, 0.55, 255),
      "!": scale_color(color, 0.65, 255),
      "~": scale_color(color, 0.9, 255),
      "@": scale_color(color, 1.25, 255),
      "r": (color[0], color[2], color[1], 255),
      "R": scale_color((color[0], color[2], color[1], 255),
                       1.25, 255),
      "%": (178, 206, 234, 255),
      "#": WHITE,
      "O": DARK_GRAY,
      ",": BLACK,
      "x": ALPHA,
  }
  return palette
# LINT.ThenChange(//meltingpot/lua/modules/colors.lua)


def flip_horizontal(sprite: str) -> str:
  flipped = ""
  for line in sprite.split("\n"):
    flipped += line[::-1] + "\n"
  return flipped[:-1]


def flip_vertical(sprite: str) -> str:
  flipped = ""
  for line in sprite[1:].split("\n"):
    flipped = line + "\n" + flipped
  return flipped


def convert_rgb_to_rgba(rgb_tuple: ColorRGB) -> ColorRGBA:
  rgba_tuple = (rgb_tuple[0], rgb_tuple[1], rgb_tuple[2], 255)
  return rgba_tuple


def adjust_color_brightness(
    color_tuple: Union[ColorRGB, ColorRGBA],
    factor: float) -> ColorRGBA:
  """Adjust color brightness by first converting to hsv and then back to rgb."""
  hsv = colorsys.rgb_to_hsv(color_tuple[0], color_tuple[1], color_tuple[2])
  adjusted_hsv = (hsv[0], hsv[1], hsv[2] * factor)
  adjusted_rgb = colorsys.hsv_to_rgb(*adjusted_hsv)
  if len(color_tuple) == 3:
    output_color = (adjusted_rgb[0], adjusted_rgb[1], adjusted_rgb[2], 255)
  elif len(color_tuple) == 4:
    output_color = (
        adjusted_rgb[0], adjusted_rgb[1], adjusted_rgb[2], color_tuple[3])
  return tuple([int(x) for x in output_color])


def get_diamond_palette(
    base_color: ColorRGB) -> Dict[str, ColorRGBA]:
  return {
      "x": ALPHA,
      "a": (252, 252, 252, 255),
      "b": convert_rgb_to_rgba(base_color),
      "c": adjust_color_brightness(base_color, 0.25),
      "d": convert_rgb_to_rgba(base_color)
  }

HD_AVATAR_N = """
xxxxxxxxxxxxxxxx
xxxx*xxxxxxx*xxx
xxxxx*xxxxx*xxxx
xxxxx*&xxx*&xxxx
xxxx@**&@**&@xxx
xx@x@@*&@*&@*x@x
xx@&@@@@@@@@*&*x
xx*x@@@@@@@**x&x
xxxx@@@@@****xxx
xxxxx@******xxxx
xxxxxxooOOOxxxxx
xxxxx*@@@**&xxxx
xxx@@x@@@**x&*xx
xxxx*xOOOOOx*xxx
xxxxxxx&xoxxxxxx
xxxxx@**x@**xxxx
"""

HD_AVATAR_E = """
xxxxxxxxxxxxxxxx
xxxxxx*xxxx*xxxx
xxxxxxx*xx*xxxxx
xxxxxxx*&x&xxxxx
xxxxx@@@@@@*xxxx
xxx@*@@@RRRr*xxx
xxx**&o@R,r,*&xx
xxx@&o@@R,r,&xxx
xxxx@@@*Rrrr&xxx
xxxxxx****o*xxxx
xxxxxx&&OOOxxxxx
xxxxx&*@@**xxxxx
xxxx&&o*@**&xxxx
xxxxoxoOOOO&xxxx
xxxxxxx&xoxxxxxx
xxxxxxx@**@*xxxx
"""

HD_AVATAR_S = """
xxxxxxxxxxxxxxxx
xxxx*xxxxxxx*xxx
xxxxx*xxxxx*xxxx
xxxxx*&xxx*&xxxx
xxxx@@@@@@@@*xxx
xx@x@RRRRRRr*x@x
xx@&@R,RRR,r*&*x
xx*x@R,RRR,r*x&x
xxxx@RRRRrrr*xxx
xxxx@@@ooo***xxx
xxxxxxooOOOxxxxx
xxxxx*@@@**&xxxx
xxx@@x@@@**x&*xx
xxxx*xOOOOOx*xxx
xxxxxxx&xoxxxxxx
xxxxx@**x@**xxxx
"""

HD_AVATAR_W = """
xxxxxxxxxxxxxxxx
xxxxx*xxxx*xxxxx
xxxxxx*xx*xxxxxx
xxxxxx&x*&xxxxxx
xxxxx@@@@***xxxx
xxxx@RRRr**&@&xx
xxx*@,R,r*&@**xx
xxxx@,R,r**&*&xx
xxxx@Rrrr**o&xxx
xxxxx@o@**ooxxxx
xxxxxx&&&ooxxxxx
xxxxxx@@***&xxxx
xxxxx&@@**&&&xxx
xxxxx&OOOO&xoxxx
xxxxxxx&xoxxxxxx
xxxxx@*@**xxxxxx
"""

HD_AVATAR = [HD_AVATAR_N, HD_AVATAR_E, HD_AVATAR_S, HD_AVATAR_W]

HD_AVATAR_N_W_BADGE = """
xxxxxxxxxxxxxxxx
xxxx*xxxxxxx*xxx
xxxxx*xxxxx*xxxx
xxxxx*&xxx*&xxxx
xxxx@**&@**&@xxx
xx@x@@*&@*&@*x@x
xx@&@@@@@@@@*&*x
xx*x@@@@@@@**x&x
xxxx@@@@@****xxx
xxxxx@******xxxx
xxxxxxooOOOxxxxx
xxxxx*@ab**&xxxx
xxx@@x@cd**x&*xx
xxxx*xOOOOOx*xxx
xxxxxxx&xoxxxxxx
xxxxx@**x@**xxxx
"""

HD_AVATAR_E_W_BADGE = """
xxxxxxxxxxxxxxxx
xxxxxx*xxxx*xxxx
xxxxxxx*xx*xxxxx
xxxxxxx*&x&xxxxx
xxxxx@@@@@@*xxxx
xxx@*@@@RRRr*xxx
xxx**&o@R,r,*&xx
xxx@&o@@R,r,&xxx
xxxx@@@*Rrrr&xxx
xxxxxx****o*xxxx
xxxxxx&&OOOxxxxx
xxxxx&*ab**xxxxx
xxxx&&ocd**&xxxx
xxxxoxoOOOO&xxxx
xxxxxxx&xoxxxxxx
xxxxxxx@**@*xxxx
"""

HD_AVATAR_S_W_BADGE = """
xxxxxxxxxxxxxxxx
xxxx*xxxxxxx*xxx
xxxxx*xxxxx*xxxx
xxxxx*&xxx*&xxxx
xxxx@@@@@@@@*xxx
xx@x@RRRRRRr*x@x
xx@&@R,RRR,r*&*x
xx*x@R,RRR,r*x&x
xxxx@RRRRrrr*xxx
xxxx@@@ooo***xxx
xxxxxxooOOOxxxxx
xxxxx*@ab**&xxxx
xxx@@x@cd**x&*xx
xxxx*xOOOOOx*xxx
xxxxxxx&xoxxxxxx
xxxxx@**x@**xxxx
"""

HD_AVATAR_W_W_BADGE = """
xxxxxxxxxxxxxxxx
xxxxx*xxxx*xxxxx
xxxxxx*xx*xxxxxx
xxxxxx&x*&xxxxxx
xxxxx@@@@***xxxx
xxxx@RRRr**&@&xx
xxx*@,R,r*&@**xx
xxxx@,R,r**&*&xx
xxxx@Rrrr**o&xxx
xxxxx@o@**ooxxxx
xxxxxx&&&ooxxxxx
xxxxxx@ab**&xxxx
xxxxx&@cd*&&&xxx
xxxxx&OOOO&xoxxx
xxxxxxx&xoxxxxxx
xxxxx@*@**xxxxxx
"""

HD_AVATAR_W_BADGE = [HD_AVATAR_N_W_BADGE, HD_AVATAR_E_W_BADGE,
                     HD_AVATAR_S_W_BADGE, HD_AVATAR_W_W_BADGE]

CUTE_AVATAR_N = """
xxxxxxxx
xx*xx*xx
xx****xx
xx&&&&xx
x******x
x&****&x
xx****xx
xx&xx&xx
"""

CUTE_AVATAR_E = """
xxxxxxxx
xx*x*xxx
xx****xx
xx*O*Oxx
x**##*&x
x&****&x
xx****xx
xx&&x&xx
"""

CUTE_AVATAR_S = """
xxxxxxxx
xx*xx*xx
xx****xx
xxO**Oxx
x&*##*&x
x&****&x
xx****xx
xx&xx&xx
"""

CUTE_AVATAR_W = """
xxxxxxxx
xxx*x*xx
xx****xx
xxO*O*xx
x&*##**x
x&****&x
xx****xx
xx&x&&xx
"""

CUTE_AVATAR = [CUTE_AVATAR_N, CUTE_AVATAR_E, CUTE_AVATAR_S, CUTE_AVATAR_W]

CUTE_AVATAR_ALERT_SPRITE = """
xxxxxxxx
xx*xx*xx
xx****xx
x&O**O&x
x&*##*&x
xx****xx
xx****xx
xx&xx&xx
"""

CUTE_AVATAR_ALERT = [CUTE_AVATAR_ALERT_SPRITE] * 4

CUTE_AVATAR_SIT_SPRITE = """
xxxxxxxx
xxxxxxxx
xx*xx*xx
xx****xx
xxO**Oxx
x&*##*&x
x&****BB
xx*&&*bb
"""

CUTE_AVATAR_SIT = [CUTE_AVATAR_SIT_SPRITE] * 4

CUTE_AVATAR_EAT_SPRITE = """
xxxxxxxx
xxxxxxxx
xx*xx*xx
xx****xx
xxO**Oxx
x&*##*&x
x&*BB*&x
xx*bb*xx
"""

CUTE_AVATAR_EAT = [CUTE_AVATAR_EAT_SPRITE] * 4

CUTE_AVATAR_FIRST_BITE_SPRITE = """
xxxxxxxx
xxxxxxxx
xx*xx*xx
xx****xx
xxO**Oxx
x&*BB*&x
x&*bb*&x
xx*&&*xx
"""

CUTE_AVATAR_FIRST_BITE = [CUTE_AVATAR_FIRST_BITE_SPRITE] * 4

CUTE_AVATAR_SECOND_BITE_SPRITE = """
xxxxxxxx
xxxxxxxx
xx*xx*xx
xx****xx
xxO**Oxx
x&*bb*&x
x&****&x
xx*&&*xx
"""

CUTE_AVATAR_SECOND_BITE = [CUTE_AVATAR_SECOND_BITE_SPRITE] * 4

CUTE_AVATAR_LAST_BITE_SPRITE = """
xxxxxxxx
xxxxxxxx
xx*xx*xx
xx****xx
xxO**Oxx
x&*##*&x
x&****&x
xx*&&*xx
"""

CUTE_AVATAR_LAST_BITE = [CUTE_AVATAR_LAST_BITE_SPRITE] * 4

CUTE_AVATAR_W_SHORTS_N = """
xxxxxxxx
xx*xx*xx
xx****xx
xx&&&&xx
x******x
x&****&x
xxabcdxx
xx&xx&xx
"""

CUTE_AVATAR_W_SHORTS_E = """
xxxxxxxx
xx*x*xxx
xx****xx
xx*O*Oxx
x**##*&x
x&****&x
xxabcdxx
xx&&x&xx
"""

CUTE_AVATAR_W_SHORTS_S = """
xxxxxxxx
xx*xx*xx
xx****xx
xxO**Oxx
x&*##*&x
x&****&x
xxabcdxx
xx&xx&xx
"""

CUTE_AVATAR_W_SHORTS_W = """
xxxxxxxx
xxx*x*xx
xx****xx
xxO*O*xx
x&*##**x
x&****&x
xxabcdxx
xx&x&&xx
"""

CUTE_AVATAR_W_SHORTS = [CUTE_AVATAR_W_SHORTS_N, CUTE_AVATAR_W_SHORTS_E,
                        CUTE_AVATAR_W_SHORTS_S, CUTE_AVATAR_W_SHORTS_W]

PERSISTENCE_PREDATOR_N = """
xxexxexx
xxhhhhxx
xhhhhhhx
shhhhhhs
slhlhlha
aullllua
xauuuuax
xxexxexx
"""

PERSISTENCE_PREDATOR_E = """
xxexxxex
xxsssssx
xshyhhys
shhhhhhh
slhlhlhl
aulllllu
xauuuuua
xxexxxex
"""

PERSISTENCE_PREDATOR_S = """
xxexxexx
xxssssxx
xsyhhysx
shhhhhhs
ahlhlhls
aullllua
xauuuuax
xxexxexx
"""

PERSISTENCE_PREDATOR_W = """
xexxxexx
xsssssxx
syhhyhsx
hhhhhhhs
lhlhlhls
ulllllua
auuuuuax
xexxxexx
"""

PERSISTENCE_PREDATOR = [PERSISTENCE_PREDATOR_N, PERSISTENCE_PREDATOR_E,
                        PERSISTENCE_PREDATOR_S, PERSISTENCE_PREDATOR_W]

AVATAR_DEFAULT = """
xxxx@@@@@@@@xxxx
xxxx@@@@@@@@xxxx
xxxx@@@@@@@@xxxx
xxxx@@@@@@@@xxxx
xxxx********xxxx
xxxx********xxxx
xx@@**####**@@xx
xx@@**####**@@xx
xx************xx
xx************xx
xx************xx
xx************xx
xxxx**xxxx**xxxx
xxxx**xxxx**xxxx
xxxx**xxxx**xxxx
xxxx**xxxx**xxxx
"""

AVATAR_BIMANUAL = """
xx@@xxxxxxxx@@xx
xx@@xxxxxxxx@@xx
xx@@xx@@@@xx@@xx
xx@@xx@@@@xx@@xx
xx@@xx****xx@@xx
xx@@xx****xx@@xx
xx@@@@####@@@@xx
xx@@@@####@@@@xx
xxxx********xxxx
xxxx********xxxx
xxxx********xxxx
xxxx********xxxx
xxxx**xxxx**xxxx
xxxx**xxxx**xxxx
xxxx**xxxx**xxxx
xxxx**xxxx**xxxx
"""

UNRIPE_BERRY = """
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxx#xxxxxxxx
xxxxx******xxxxx
xxxx********xxxx
xxx**********xxx
xxxxx******xxxxx
xxxxxx****xxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
"""

BERRY = """
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxx#xxxxxxx
xxxxxxx##xxxxxxx
xxx****##****xxx
xx************xx
x**************x
x***@**@*******x
xx***@********xx
xxx**********xxx
xxxx********xxxx
xxxxx******xxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
"""

LEGACY_APPLE = """
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxx##xxxxx
xxxxxxxx##xxxxxx
xxxxxx@##@xxxxxx
xxxxx@@@@@@xxxxx
xxx&&&&&&&&&&xxx
xxx&*&&&&&&&&xxx
xxx&***&&&&&&xxx
xxx**********xxx
xxxxx******xxxxx
xxxxxxx***xxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
"""

HD_APPLE = """
xxxxxxxxxxxxxxxx
xx&&&&xxxxxxxxxx
xxxxoo&xxxxxxxxx
xxxxxxxoxOOxxxxx
xxxxxxxxOOxxxxxx
xxxx@@xxOx@*xxxx
xx@@***O&&***&xx
x@@*#*&O&****&&x
x@*#***&*****&&x
x@*#********&&ox
xx*********&&oxx
xx********&&&oxx
xxx***&&*&&&oxxx
xxxx&ooxx&ooxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
"""

BADGE = """
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
x#####xxxxxxxxxx
x#####xxxxxxxxxx
x#####xxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
"""

COIN = """
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxx@###xxxxxx
xxxxx@@@@##xxxxx
xxxx&&&@@@@#xxxx
xxx&&&&&&&@@#xxx
xxx&*&&&&&&&&xxx
xxx&***&&&&&&xxx
xxx**********xxx
xxxx********xxxx
xxxxx******xxxxx
xxxxxx****xxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
"""

WALL = """
&&&&##&&&&&&&&&&
&@@@##@@@@@@@@@@
****##**********
****##**********
################
################
&&&@@@@@@@##@@&&
&&@@@@@@@@##@@@&
**********##****
**********##****
################
################
****##**********
****##**********
@@@@##@@@@@@@@@@
&&&&##@@@@@@@@@@
"""

TILE = """
otooooxoxooootoo
tttooxoooxoottto
ttttxoooooxttttt
tttxtoooootxttto
otxtttoootttxtoo
oxtttttotttttxoo
xootttoootttooxo
ooootoooootoooox
xootttoootttooxo
oxtttttotttttxoo
otxtttoootttxtoo
tttxtoooootxttto
ttttxoooooxttttt
tttooxoooxoottto
otooooxoxooootoo
oooooooxoooooooo
"""

TILE1 = """
otooooxo
tttooxoo
ttttxooo
tttxtooo
otxtttoo
oxttttto
xootttoo
ooootooo
"""

TILE2 = """
xooootoo
oxoottto
ooxttttt
ootxttto
otttxtoo
tttttxoo
otttooxo
ootoooox
"""

BRICK_WALL_NW_CORNER = """
iiiiiiii
iicccccc
iccccccc
iccooooo
iccoobbb
iccobooo
iccoboob
iccobobo
"""

BRICK_WALL_NE_CORNER = """
iiiiiiii
ccccccii
ccccccci
ooooocci
bbboocci
ooobocci
boobocci
obobocci
"""

BRICK_WALL_SE_CORNER = """
obobocci
boobocci
ooobocci
bbboocci
ooooocci
ccccccci
ccccccii
iiiiiiii
"""

BRICK_WALL_SW_CORNER = """
iccobobo
iccoboob
iccobooo
iccoobbb
iccooooo
iccccccc
iicccccc
iiiiiiii
"""

BRICK_WALL_INNER_NW_CORNER = """
oooooooo
oobbobbb
oboooooo
oboobbob
oboboooo
oooboccc
oboboccc
oboooccc
"""
BRICK_WALL_INNER_NE_CORNER = """
oooooooo
bbobbboo
oooooobo
bobboobo
oooobooo
cccooobo
cccobobo
cccobobo
"""

BRICK_WALL_INNER_SW_CORNER = """
oboboccc
oboooccc
oboboccc
oooboooo
oboobobb
oboooooo
oobbbbob
oooooooo
"""

BRICK_WALL_INNER_SE_CORNER = """
cccobobo
cccobooo
cccooobo
oooobobo
bobboobo
oooooobo
bbbobboo
oooooooo
"""

BRICK_WALL_NORTH = """
iiiiiiii
cccccccc
cccccccc
oooooooo
bbbbobbb
oooooooo
bobbbbob
oooooooo
"""

BRICK_WALL_EAST = """
obobocci
ooobocci
obobocci
obooocci
obobocci
obobocci
ooobocci
obobocci
"""

BRICK_WALL_SOUTH = """
oooooooo
bobbbbob
oooooooo
bbbobbbb
oooooooo
cccccccc
cccccccc
iiiiiiii
"""

BRICK_WALL_WEST = """
iccobobo
iccobooo
iccobobo
iccooobo
iccobobo
iccobobo
iccobooo
iccobobo
"""

FILL = """
iiiiiiii
iiiiiiii
iiiiiiii
iiiiiiii
iiiiiiii
iiiiiiii
iiiiiiii
iiiiiiii
"""

TILED_FLOOR_GREY = """
ooo-ooo-
ooo-ooo-
ooo-ooo-
--------
ooo-ooo-
ooo-ooo-
ooo-ooo-
--------
"""

ACORN = """
xxxxxxxx
xxoooxxx
xoooooxx
xo***oxx
xx@*@xxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

GRASS_STRAIGHT = """
********
*@*@****
*@*@****
********
*****@*@
*****@*@
********
********
"""

GRASS_STRAIGHT_N_EDGE = """
****x*x*
*@*@****
*@*@****
********
*****@*@
*****@*@
********
********
"""

GRASS_STRAIGHT_E_EDGE = """
********
*@*@****
*@*@***x
********
*****@*@
*****@*@
*******x
********
"""
GRASS_STRAIGHT_S_EDGE = """
********
*@*@****
*@*@****
********
*****@*@
*****@*@
********
**x*x***
"""

GRASS_STRAIGHT_W_EDGE = """
********
x@*@****
*@*@****
********
x****@*@
*****@*@
x*******
********
"""

GRASS_STRAIGHT_NW_CORNER = """
x***x***
*@*@****
*@*@****
x*******
*****@*@
*****@*@
********
********
"""

GRASS_STRAIGHT_NE_CORNER = """
****x**x
*@*@****
*@*@****
*******x
*****@*@
*****@*@
********
********
"""

GRASS_STRAIGHT_SE_CORNER = """
********
*@*@****
*@*@***x
********
*****@*@
*****@*@
********
***x***x
"""

GRASS_STRAIGHT_SW_CORNER = """
********
*@*@****
*@*@****
x*******
*****@*@
*****@*@
********
x***x***
"""

BUTTON = """
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xx************xx
xx************xx
xx**########**xx
xx**########**xx
xx**########**xx
xx**########**xx
xx**########**xx
xx**########**xx
xx**########**xx
xx**########**xx
xx************xx
xx************xx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
"""

PLUS_IN_BOX = """
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xx************xx
xx************xx
xx**##@@@@##**xx
xx**##@@@@##**xx
xx**@@@@@@@@**xx
xx**@@@@@@@@**xx
xx**@@@@@@@@**xx
xx**@@@@@@@@**xx
xx**##@@@@##**xx
xx**##@@@@##**xx
xx************xx
xx************xx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
"""

TREE = """
xx@@@@@@@@@@@@xx
xx@@@@@@@@@@@@xx
xx@@@@@@@@@@@@xx
xx@@@@@@@@@@@@xx
xx@@@@@@@@@@@@xx
xx@@@@@@@@@@@@xx
xxxx@@****@@xxxx
xxxx@@****@@xxxx
xxxxxx****xxxxxx
xxxxxx****xxxxxx
xxxxxx****xxxxxx
xxxxxx****xxxxxx
xxxxxx****xxxxxx
xxxxxx****xxxxxx
xxxxxx****xxxxxx
xxxxxx****xxxxxx
"""

POTATO_PATCH = """
xx@@xxxxxxxxxx@@
xx@@xxxxxxxxxx@@
xxxx@@xxxxxxxx@@
xxxx@@xxxxxx@@xx
xxxxxx@@@@xx@@xx
xxxxxx@@@@xx@@xx
@@@@@@****@@xxxx
@@@@@@****@@xxxx
xxxx@@****@@xxxx
xxxx@@****@@xxxx
xx@@xx@@@@xx@@xx
xx@@xx@@@@xx@@@@
@@xxxxxx@@xx@@@@
@@xxxxxx@@xxxxxx
@@xxxxxxxx@@xxxx
@@xxxxxxxx@@xxxx
"""

FIRE = """
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxx&&&&xx&&xx
xxxxxx&&&&xx&&xx
xx&&xx****xx**xx
xx&&xx****xx**xx
xx************xx
xx************xx
xx@@@@****@@@@xx
xx@@@@****@@@@xx
xxxx@@@@@@@@xxxx
xxxx@@@@@@@@xxxx
xx@@@@xxxx@@@@xx
xx@@@@xxxx@@@@xx
"""

STONE_QUARRY = """
@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@
@@xx##xxxxxx##@@
@@xx##xxxxxx##@@
@@xxxx##xx##xx@@
@@xxxx##xx##xx@@
@@xx##xxxxxx##@@
@@xx##xxxxxx##@@
@@##xxxx##xxxx@@
@@##xxxx##xxxx@@
@@xx##xxxxxx##@@
@@xx##xxxxxx##@@
@@##xxxxxx##xx@@
@@##xxxxxx##xx@@
@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@
"""

WATER_1 = """
**~~*ooo~~~oo~**
~~~o**~~~~~~~**o
ooo~***~~~~~***~
o~~~~**~~*****~~
~~~~*****@@**~~o
o~**********~oo~
o**~~~~~~***o~~~
*oo~~~~~~o**~~~~
~~~ooooooo~**~**
*~~~~oooo~~*@~**
**~~~~oo~~~~**~~
~**~~~~oo~~~**~~
~*@*~~~~oo~~**~~
~~*@**~~~~o**~~~
~~~~********~~~~
~~**~~~~ooo~***~
"""


WATER_2 = """
*~~*~oo~~~~oo~~*
~~oo*~~~~~~~~**~
oo~~~**~~~***~~o
~~~*********~~~~
~~~****@@**~~~oo
o~**********oo~~
~***~~~~~~***~~~
*~~oooo~ooo**~~~
~~~~~~oooo~~*@**
*~~~~~~~~oo~***~
~**~~~~~~~o~**~~
~~**~~~~~~o**~~~
~~*@**~~~~**~~~~
~~~~********~~~~
~~~**~~~~oo***~~
~***~~~oo~~~~**~
"""

WATER_3 = """
***oooo~~~oo**~*
oo~**~~~~~~~**oo
~~~***~~~~~***~~
o~~~~********ooo
~ooo~*@@*****~~~
~~o*****oo****~~
~~**~~oooo~***~~
~*~~~~~~~oo~**~~
*~~~~~~~~~oo*@**
*~~~~~~~~~~***~~
*~~~~~~~~~**o~~~
~**~~~~~~**~oo~~
~*@**~~~**~~~o~~
~~*@******~~o~~~
~~**~~~~~***~~~~
~**~~~~ooo~~***~
"""

WATER_4 = """
*~~*~oo~~ooo~~~*
~ooo*~~~~~~~***o
o~~~~**~~~**~~~~
~~~**@******~~~~
o~~***@@@**~~~oo
~o**********oo~~
~***~~~~~o***~~~
*~oooo~oooo**~~~
~~~~~oooo~~~*@**
*~~~~~~ooo~~***~
~**~~~~~~oo~**~~
~~**~~~~~~o***~~
~~**~~~~~~o**~~~
~~~*@@*~~~**o~~~
~~~~**@******~~~
~***~~~oo~~~~**~
"""

BOAT_FRONT_L = """
xxxxxxxxxxxxx***
xxxxxxxxxxxx*@@@
xxxxxxxxxxx**ooo
xxxxxxxxxx*&*@@@
xxxxxxxx**@&*@@@
xxxxxxx*@@o@&***
xxxxxx*@@o@***&&
xxxxx*@@o@*&&*&&
xxxx*@@o@*&&&*&&
xxxx*@@@*&&&&&*&
xxx*@o@*&&&***@*
xx*@@o*&&***@o@*
xx*@@o***@@*o@@*
x*@@@***o@@*o@@*
x*@@@*@*@o@*****
*@@@*@@*@o@*@@o*
"""

BOAT_FRONT_R = """
***xxxxxxxxxxxxx
@@@*xxxxxxxxxxxx
ooo**xxxxxxxxxxx
@@@*&*xxxxxxxxxx
@@@*&@**xxxxxxxx
***&@o@@*xxxxxxx
&&***@o@@*xxxxxx
&&*&&*@o@@*xxxxx
&&*&&&*@o@@*xxxx
&*&&&&&*@@@*xxxx
@@***&&&*@o@*xxx
@o@@***&&*o@@*xx
@@@@*@@***o@@*xx
@@oo*@@@***o@@*x
@o@@*****@*@o@*x
@o@@*@o@*@@*o@@*
"""

BOAT_REAR_L = """
*@@o*@o*@o@*@@@*
x**@@*@*@o@*****
x*@*****@o@*@@@*
xx*&o@***@@*@@@*
xx*&&o@@@***@@@*
xxx*&&ooo@@*****
xxxx*&&@@oo@*@@@
xxxx*&&&@@@o*ooo
xxxxx*&&&@@@*@@@
xxxxxx*&&&&@*ooo
xxxxxxx*&&&&*@@@
xxxxxxxx**&&*&&&
xxxxxxxxxx*&*&&&
xxxxxxxxxxx**&&&
xxxxxxxxxxxx*&&&
xxxxxxxxxxxxx***
"""

BOAT_REAR_R = """
@o@*@@o*@o@*@o@*
@o@*@@o*o@*@o**x
@o@**********&*x
@@o*@@****o@&*xx
@@o****@@o@&&*xx
*****@@oo@&&*xxx
@@@*@oo@@&&*xxxx
ooo*o@@@&&&*xxxx
@@@*@@@&&&*xxxxx
ooo*@&&&&*xxxxxx
@@@*&&&&*xxxxxxx
&&&*&&**xxxxxxxx
&&&*&*xxxxxxxxxx
&&&**xxxxxxxxxxx
&&&*xxxxxxxxxxxx
***xxxxxxxxxxxxx
"""

BOAT_SEAT_L = """
*@@o*@@o*@@@*@o*
*@@o*o@o*@o@*@o*
*@@o*@@o*@o@****
*@@o*@o@*@o@*@@*
*@@o*******@*o@*
*@o@*@oo@@@*****
*@o@*@@@oooooo@@
*@o@******@@@oo@
*@o@*&&&&&******
*@o@*****&&&&&&&
*o@@*@@@********
*o@@*&&&*&&@*@@*
*o@@*&&&*&&&*&&*
*o@@*****&&&*&&*
*@@@*@@@*&&&*&&*
*@@o*@o@*o@@*@o*
"""

BOAT_SEAT_R = """
o@@*@@@*@o@*o@@*
o@@*@@@*@@@*o@@*
@o@*****o@o*@@@*
@o@*@@@*ooo*@@@*
@@@*@*******@@o*
*****ooo@o@*@@o*
@@o@o@@@o@@*@@o*
@@@@@@******@o@*
******&&&&&*@o@*
&&&&&&&*****@o@*
********@o@*@@o*
@o@*o@&*&&&*o@o*
****&&&*&&&*@o@*
&&&*&&&*****@o@*
&&&*&&&*@o@*@o@*
@@@*@@o*@o@*@o@*
"""

OAR_DOWN_L = """
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxx****
xxxxx#xxx***#@&&
xx##xx***#@@&***
xxxxx*#@&&***xxx
xx#xxx****xx#xxx
xxx##xxxxxx#xxxx
x#xxx###x##xxxxx
xxxxxxxxxxxxx#xx
xx##xxxxxxx##xxx
xxxxxx###xxxxxxx
"""

OAR_UP_L = """
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xx****xxxxxxxxxx
x*@@##**xxxxxxxx
*&@@@@#**xxxxxxx
*&&@@@@@#****xxx
x*&&&***&@@@#***
xx***xxx****&@@#
xxxxxxxxxxxx****
xxxxxxxxxxxxxxxx
xx#xx#xxxxxxxxxx
xxx##xxxx#xxxxxx
#xxxxxxx#xxxxxxx
xx##xx#xxxx##xxx
xxxxxxxx##xxxxxx
xx####xxxxxxxxxx
"""

OAR_DOWN_R = """
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
****xxxxxxxxxxxx
&&@#***xxx#xxxxx
***&@@#***xx##xx
xxx***&&@#*xxxxx
xxx#xx****xxx#xx
xxxx#xxxxxx##xxx
xxxxx##x###xxx#x
xx#xxxxxxxxxxxxx
xxx##xxxxxxx##xx
xxxxxxx###xxxxxx
"""

OAR_UP_R = """
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxx****xx
xxxxxxxx**##@@*x
xxxxxxx**#@@@@&*
xxx****#@@@@@&&*
***#@@@&***&&&*x
#@@&****xxx***xx
****xxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxx#xx#xx
xxxxxx#xxxx##xxx
xxxxxxx#xxxxxxx#
xxx##xxxx#xx##xx
xxxxxx##xxxxxxxx
xxxxxxxxxx####xx
"""

BARRIER_ON = """
x*xxxxxxxxxxxxxx
*#*xxxxxxxxxxxxx
*@*xxxxxxxxxxxxx
*@*xxxxxxxxxxxxx
*@*xxxxxxxxxxxxx
*@*xxxxxxxxxxxxx
***************x
*&@@@@@@@@@@@##*
*&&&@@@@@@@@@@&*
***************x
*&*xxxxxxxxxxxxx
*@*xxxxxxxxxxxxx
*@*xxxxxxxxxxxxx
*@*xxxxxxxxxxxxx
*&*xxxxxxxxxxxxx
***xxxxxxxxxxxxx
"""

BARRIER_OFF = """
x*x**xxxxxxxxxxx
*#*##*xxxxxxxxxx
*@*@#*xxxxxxxxxx
*&*@@*xxxxxxxxxx
**@@&*xxxxxxxxxx
**@@*xxxxxxxxxxx
**@@*xxxxxxxxxxx
*@@&*xxxxxxxxxxx
*&&*xxxxxxxxxxxx
****xxxxxxxxxxxx
*&*xxxxxxxxxxxxx
*@*xxxxxxxxxxxxx
*@*xxxxxxxxxxxxx
*@*xxxxxxxxxxxxx
*&*xxxxxxxxxxxxx
***xxxxxxxxxxxxx
"""

FLAG = """
xO@@xxxx
xO**@xxx
xO***xxx
xOxx&&xx
xOxxxoox
xOxxxxxx
xOxxxxxx
xxxxxxxx
"""

FLAG_HELD_N = """
xO@@@xxx
xO***xxx
xO**&&xx
xOxxx&&x
xxxxxxox
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

FLAG_HELD_E = """
xxxx@*Ox
xx@***Ox
x&***oOx
*&oxxxOx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

FLAG_HELD_S = """
x@xxxxxx
xx&*x@Ox
xxx&**Ox
xxxxo&Ox
xxxxxxOx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

FLAG_HELD_W = """
xxxO@xxx
xxxOO*@x
xxxxOo&*
xxxxOOx*
xxxxxOxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

FLAG_HELD = [FLAG_HELD_N, FLAG_HELD_E, FLAG_HELD_S, FLAG_HELD_W]

ROCK = """
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxllllllllxxxx
xxxlr***kkkrrxxx
xxlr*****kkkksxx
xxrr****kkkkksxx
xxr****kkkkkksxx
xxr*****kkkkksxx
xxr******kksssxx
xxr*****kkksssxx
xxr****kkkssssxx
xxrr***ssspspsxx
xxxlspspppppsxxx
xxxxlsssssssxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
"""

PAPER = """
xxxxxxxxxxxxxxxx
x**************x
x@@@***@@**@@@*x
x@**@*@**@*@**@x
x@@@**@@@@*@@@*x
x@****@**@*@***x
x@****@**@*@***x
x**************x
x**************x
x**@@@@**@@@***x
x**@*****@**@**x
x**@@@***@@@***x
x**@*****@**@**x
x**@@@@**@**@**x
x**************x
xxxxxxxxxxxxxxxx
"""

SCISSORS = """
xx##xxxxxxxxx##x
xx*x#xxxxxxx#x*x
xx*xx#xxxxx#xx*x
xx*xxx#xxx#xxx*x
xx*xxx##xx#xxx*x
xxx****##xx***xx
xxxxxxx>##xxxxxx
xxxxxxxx>##xxxxx
xxxxx#xxx>##xxxx
xxxx##>xxx>##xxx
xxx##>xxxxx>##xx
xx##>xxxxxxx>##x
x##>xxxxxxxxx>##
x#>xxxxxxxxxxx>#
x>xxxxxxxxxxxxx>
xxxxxxxxxxxxxxxx
"""

SPACER_N = """
xx****xx
x*****~x
x**&!~~x
**&&!o~~
~~o!!o~~
~~oooo~~
x~~~~~~x
x~~xx~~x
"""

SPACER_E = """
xxx****x
xx*****~
&&**#%%%
&!**%%%%
!o~*****
oo~~***~
xx~~~~~~
xx~~xx~~
"""

SPACER_S = """
xx***~xx
x*****~x
x*#%%%~x
**%%%%~~
~*****~~
~~~**~~~
x~~~~~~x
x~~xx~~x
"""

SPACER_W = """
x***~xxx
*****~xx
#%%%*~!!
%%%%*~!o
*****~oo
~***~~oo
~~~~~~xx
~~xx~~xx
"""

SPACER_TAGGED_S = """
xxxxxxxx
x##xxxxx
xx##x##x
xxx##xxx
x*****~x
x~~**~~x
x~~~~~~x
x~~xx~~x
"""

SPACER = [SPACER_N, SPACER_E, SPACER_S, SPACER_W]
SPACER_TAGGED = [SPACER_TAGGED_S, SPACER_TAGGED_S, SPACER_TAGGED_S,
                 SPACER_TAGGED_S]

NW_SHIP_WALL = """
oooooooo
o#######
o#######
o#######
o#######
o#######
o#######
o######x
"""

NS_SHIP_WALL = """
oooooooo
########
########
########
########
########
########
xxxxxxxx
"""

NE_SHIP_WALL = """
oooooooo
#######x
#######x
#######x
#######x
#######x
#######x
o######x
"""

EW_SHIP_WALL = """
o######x
o######x
o######x
o######x
o######x
o######x
o######x
o######x
"""

SE_SHIP_WALL = """
o######x
#######x
#######x
#######x
#######x
#######x
#######x
xxxxxxxx
"""

SW_SHIP_WALL = """
o######x
o#######
o#######
o#######
o#######
o#######
o#######
xxxxxxxx
"""

SHIP_WALL_CAP_S = """
o######x
o######x
o######x
o######x
o######x
o######x
o######x
xxxxxxxx
"""

SHIP_WALL_TCOUPLING_W = """
o######x
o#######
o#######
o#######
o#######
o#######
o#######
o######x
"""

SHIP_WALL_TCOUPLING_E = """
o######x
#######x
#######x
#######x
#######x
#######x
#######x
o######x
"""

SHIP_WALL_TCOUPLING_N = """
oooooooo
########
########
########
########
########
########
o######x
"""

SHIP_WALL_TCOUPLING_S = """
o######x
########
########
########
########
########
########
xxxxxxxx
"""

N_SHIP_SOLID_WALL = """
oooooooo
########
########
########
########
########
########
########
"""

E_SHIP_SOLID_WALL = """
#######x
#######x
#######x
#######x
#######x
#######x
#######x
#######x
"""

S_SHIP_SOLID_WALL = """
########
########
########
########
########
########
########
xxxxxxxx
"""


W_SHIP_SOLID_WALL = """
o#######
o#######
o#######
o#######
o#######
o#######
o#######
o#######
"""

NW_GRATE = """
X*******
X*@&&&&&
X*&&&x&x
X*&&&x&x
o*&&&x&x
o*&&&x&x
o*&&&x&x
o*&&&x&x
"""

N_GRATE = """
********
&&&&&&&&
&x&x&x&x
&x&x&x&x
&x&x&x&x
&x&x&x&x
&x&x&x&x
&x&x&x&x
"""

NE_GRATE = """
********
&&&&&&@~
&x&x&&&~
&x&x&&&~
&x&x&&&~
&x&x&&&~
&x&x&&&~
&x&x&&&~
"""

W_GRATE = """
X*&&&&&&
X*&&&x&x
X*&&&x&x
X*&&&x&x
o*&&&x&x
o*&&&x&x
o*&&&x&x
o*&&&&&&
"""

INNER_GRATE = """
&&&&&&&&
&x&x&x&x
&x&x&x&x
&x&x&x&x
&x&x&x&x
&x&x&x&x
&x&x&x&x
&&&&&&&&
"""

E_GRATE = """
&&&&&&&~
&x&x&&&~
&x&x&&&~
&x&x&&&~
&x&x&&&~
&x&x&&&~
&x&x&&&~
&&&&&&&~
"""

SE_GRATE = """
&x&x&&&~
&x&x&&&~
&x&x&&&~
&x&x&&&~
&x&x&&&~
&x&x&&&~
&&&&&&@~
~~~~~~~~
"""

S_GRATE = """
&x&x&x&x
&x&x&x&x
&x&x&x&x
&x&x&x&x
&x&x&x&x
&x&x&x&x
&&&&&&&&
~~~~~~~~
"""

SW_GRATE = """
X*&&&x&x
X*&&&x&x
X*&&&x&x
X*&&&x&x
o*&&&x&x
o*&&&x&x
o*@&&&&&
o*~~~~~~
"""

GLASS_WALL = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
******@@
@@******
!!!!!!!!
"""

WOOD_FLOOR = """
xxx-xxxx
--------
x-xxxxxx
--------
xxxxx-xx
--------
xxxxxxx-
--------
"""

METAL_TILE = """
oxxOoxxO
xxxoxxxo
xxxxxxxx
xxOxxxOx
xOoxxOox
xoxxxoxx
xxxxxxxx
OxxxOxxx
"""

METAL_PANEL = """
///////-
///////-
///////-
///////-
--------
////-///
////-///
--------
"""

THRESHOLD = """
xxxxxxxx
XXXXXXXX
xxxxxxxx
XXXXXXXX
xxxxxxxx
XXXXXXXX
xxxxxxxx
XXXXXXXX
"""

THRESHOLD_VERTICAL = """
xXxXxXxX
xXxXxXxX
xXxXxXxX
xXxXxXxX
xXxXxXxX
xXxXxXxX
xXxXxXxX
xXxXxXxX
"""

CHECKERED_TILE = """
XXXXxxxx
XXXXxxxx
XXXXxxxx
XXXXxxxx
xxxxXXXX
xxxxXXXX
xxxxXXXX
xxxxXXXX
"""

GEM = """
xxxxxxxx
xxx~~xxx
xx~**&xx
xx~*!&xx
xx~!!&xx
xxx&&xxx
xxxxxxxx
xxxxxxxx
"""

SMALL_SPHERE = """
xxxxxxxx
xx+~~+xx
xx~@*&xx
xx~**&xx
xx+&&+xx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

DIRT_PATTERN = """
xxxxxxxx
xXXXxxxx
xXXXxxxx
xxxxxxxx
xxxxXXXx
xxxxxXXx
xxxXXxxx
xxxxXXXX
"""

FRUIT_TREE = """
x@@@@@@x
x@Z@Z@@x
x@@Z@Z@x
xx@**@xx
xxx**xxx
xxx**xxx
xxx**xxx
xxxxxxxx
"""

BATTERY_FLOOR = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
aaaaafwx
aaaaaxxx
AAAAAgwx
xxxxxxxx
xxxxxxxx
"""

BATTERY_GRASPED = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
adddafwx
ADDDAxxx
AAAAAgwx
xxxxxxxx
xxxxxxxx
"""

BATTERY_FULL = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxaoooax
xxaoooaf
xxAOOOAg
xxxxxxxx
xxxxxxxx
"""

BATTERY_DRAINED_ONE = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxa#ooax
xxa#ooaf
xxA#OOAg
xxxxxxxx
xxxxxxxx
"""

BATTERY_DRAINED_TWO = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxa##oax
xxa##oaf
xxA##OAg
xxxxxxxx
xxxxxxxx
"""

BATTERY_DRAINED = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxa###ax
xxa###af
xxAAAAAg
xxxxxxxx
xxxxxxxx
"""

BATTERY_FLASHING = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xx#####x
xx#####f
xx#####g
xxxxxxxx
xxxxxxxx
"""

WIRES = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxwff
xxxxxxxg
xxxxxwgx
xxxxxxxx
xxxxxxxx
"""

PLUG_SOCKET = """
xxxxxxxx
xxxxsssx
xxxxsAsx
xxxxsgff
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

CONSUMPTION_STARS = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxx-xx
xxxx---x
xx-xx-xx
x---xxxx
xx-xxxxx
"""

CONSUMPTION_STARS_2 = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxx-xx
xxxx---x
xx-xx-xx
x---xxxx
xx-xxxxx
"""

# Positional Goods sprites.
CROWN = """
x#@#x@#x
xx####xx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

BRONZE_CAP = """
xxxxxxxx
xx####xx
xx####xx
x@xxxx@x
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

# Boat Race 2.0 sprites.

SEMAPHORE_FLAG = """
xxxxxxxx
x*@xxxxx
x*&@xxxx
x*&@@xxx
x*&@@xxx
x|xxxxxx
x|xxxxxx
xxxxxxxx
"""

# Allelopathic Harvest 2.0 sprites.

SOIL = """
xXDxDDxx
XdDdDDDx
DdDDdDdd
dDdDDdDd
xDdDdDdX
DDDDDDXd
ddDdDDdD
xDdDdDDx
"""

BERRY_SEEDS = """
xxxxxxxx
xxxxxxxx
xxxOxxxx
xxxxoxOx
xxoxxxxx
xxxxxxxx
xxxxoxxx
xxxxxxxx
"""

BERRY_RIPE = """
xxxxxxxx
xxxxxxxx
xxooxxxx
xxooOOxx
xxxdOOxx
xxxddxxx
xxxxxxxx
xxxxxxxx
"""

# Territory 2.0 sprites.

NW_HIGHLIGHT = """
x*******
*xxxxxxx
*xxxxxxx
*xxxxxxx
*xxxxxxx
*xxxxxxx
*xxxxxxx
*xxxxxxx
"""

NE_HIGHLIGHT = """
*******x
xxxxxxxo
xxxxxxxo
xxxxxxxo
xxxxxxxo
xxxxxxxo
xxxxxxxo
xxxxxxxo
"""

E_W_HIGHLIGHT = """
*xxxxxxo
*xxxxxxo
*xxxxxxo
*xxxxxxo
*xxxxxxo
*xxxxxxo
*xxxxxxo
*xxxxxxo
"""

N_S_HIGHLIGHT = """
********
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
oooooooo
"""

SE_HIGHLIGHT = """
xxxxxxxo
xxxxxxxo
xxxxxxxo
xxxxxxxo
xxxxxxxo
xxxxxxxo
xxxxxxxo
ooooooox
"""

SW_HIGHLIGHT = """
*xxxxxxx
*xxxxxxx
*xxxxxxx
*xxxxxxx
*xxxxxxx
*xxxxxxx
*xxxxxxx
xooooooo
"""

CUTE_AVATAR_HOLDING_PAINTBRUSH_N = """
xxxxxOOO
xx*xx*+x
xx****-x
xx&&&&&x
x******x
x&****xx
xx****xx
xx&xx&xx
"""

CUTE_AVATAR_HOLDING_PAINTBRUSH_E = """
xxxxxxxx
xx*x*xxx
xx****xx
xx*,*,xx
x**##*&&
x&****xx
xx****xx
xx&&x&xx
"""

CUTE_AVATAR_HOLDING_PAINTBRUSH_S = """
xxxxxxxx
xx*xx*xx
xx****xx
xx,**,xx
x&*##*&x
x&****&x
xx****-x
xx&xx&+x
"""

CUTE_AVATAR_HOLDING_PAINTBRUSH_W = """
xxxxxxxx
xxx*x*xx
xx****xx
xx,*,*xx
&&*##**x
xx****&x
xx****xx
xx&x&&xx
"""

CUTE_AVATAR_HOLDING_PAINTBRUSH = [CUTE_AVATAR_HOLDING_PAINTBRUSH_N,
                                  CUTE_AVATAR_HOLDING_PAINTBRUSH_E,
                                  CUTE_AVATAR_HOLDING_PAINTBRUSH_S,
                                  CUTE_AVATAR_HOLDING_PAINTBRUSH_W
                                  ]

PAINTBRUSH_N = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxx*&o
xxxxx*k&
xxxxxkkk
"""

PAINTBRUSH_E = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxOk**xx
-+Okk&xx
xxOk&oxx
xxxxxxxx
xxxxxxxx
"""

PAINTBRUSH_S = """
xxxxxOOO
xxxxxkkk
xxxxx&k*
xxxxxo&*
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

PAINTBRUSH_W = flip_horizontal(PAINTBRUSH_E)

PAINTBRUSH = [PAINTBRUSH_N, PAINTBRUSH_E, PAINTBRUSH_S, PAINTBRUSH_W]

WALL = """
**#*****
**#*****
########
*****#**
*****#**
########
**#*****
**#*****
"""

GRAINY_FLOOR = """
+*+*++*+
*+*+**+*
+*+****+
****+*+*
*+*+****
**+***++
+*+*+**+
***+**+*
"""

GRASS_STRAIGHT_N_CAP = """
x***x**x
*@*@****
*@*@****
x*******
*****@*@
*****@*x
********
********
"""

SHADOW_W = """
#@*xxxxx
#*x~xxxx
#@*xxxxx
#*x~xxxx
#@*xxxxx
#*x~xxxx
#@*xxxxx
#*x~xxxx
"""

SHADOW_E = """
xxxxx*@#
xxxx~x*#
xxxxx*@#
xxxx~x*#
xxxxx*@#
xxxx~x*#
xxxxx*@#
xxxx~x*#
"""

SHADOW_N = """
########
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

APPLE_TREE_STOUT = """
xxxxxxxx
xaxaaaax
aabbaaoa
baaaoaax
bobaaaob
bbbabIbb
xbIbbbIx
xxIxxxIx
"""

BANANA_TREE = """
xxaaaxax
xaoaabba
abooaaaa
bbbbaaob
bobIboob
xooxIIbx
xxxxIxxx
xxxxIxxx
"""

ORANGE_TREE = """
xxaaaxxx
xaoaabba
abaoaaaa
bbbbaaob
bobIbaab
xbbIIbbx
xxxxIxxx
xxxxIxxx
"""

GOLD_MINE_TREE = """
xxxxxxxx
xaaaaxax
aobbaaaa
baIoIIax
boIIIoob
bbbabIbb
xbbxbbbx
xxxxxxxx
"""

FENCE_NW_CORNER = """
aaaxxaax
aaaxxaax
bbbdcbbd
cddedbbe
aaexxbcx
aaedcbcd
bbe#ebbe
cd####b#
"""

FENCE_N = """
xaaxxaax
xaaxxaax
cbbdcbbd
dbbedcbe
xbbxxcbx
cbbdcbbd
dbb#dbbe
#b####b#
"""

FENCE_NE_CORNER = """
xaaaxxxx
xaaaxxxx
cbbbxxxx
dbcdxxxx
xbaa##xx
cbaa##xx
d#bb#xxx
##cd#xxx
"""

FENCE_INNER_NE_CORNER = """
##aa##xx
x#aa##xx
xxbb#xxx
xxcd#xxx
xxaa##xx
xxaa##xx
xxbb#xxx
xxcd#xxx
"""

FENCE_E = """
xxaa##xx
xxaa##xx
xxbb#xxx
xxcd#xxx
xxaa##xx
xxaa##xx
xxbb#xxx
xxcd#xxx
"""

FENCE_SE_CORNER = """
xaaa##xx
xaaa##xx
cbbd#xxx
dcbb#xxx
xbbb##xx
dccb##xx
#ccc#xxx
##c##xxx
"""

FENCE_S = """
xaaxxaax
xaaxxaax
cbbdcbbd
dbbedcbe
xbbxxcbx
cbbdcbbd
dbb#dbbe
#b####b#
"""

FENCE_SW_CORNER = """
aaa#xaax
aaa#xaax
cbbdcbbd
bbcedbbe
bbb#xbcx
bccdcbcd
ccc#ebbe
#c####b#
"""

FENCE_W = """
aa##xxxx
aa##xxxx
bb#xxxxx
cd#xxxxx
aa##xxxx
aa##xxxx
bb#xxxxx
cd#xxxxx
"""

FENCE_INNER_NW_CORNER = """
aa######
aa##xx##
bb#xxxxx
cd#xxxxx
aa##xxxx
aa##xxxx
bb#xxxxx
cd#xxxxx
"""

FENCE_SHADOW_S = """
########
xx##xx##
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

FENCE_SHADOW_SE = """
######xx
xx####xx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

FENCE_SHADOW_S = """
########
xx##xx##
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

FENCE_SHADOW_SW = """
x#######
xx##xx##
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

MAGIC_GRAPPLED_AVATAR = """
xpPppPpx
pP*PP*Pp
pP****Pp
pPO**OPp
P&*##*&P
P&****&P
pP****Pp
pP&PP&Pp
"""

MAGIC_HOLDING_SHELL = """
x~*~~*~x
~*~**~*~
~*~~~~*~
~*~~~~*~
*~~~~~~*
*~~~~~~*
~*~~~~*~
~*~**~*~
"""

MAGIC_BEAM_N_FACING = """
xx~~~~xx
xx*~~*xx
xx*~~*xx
xx*~~*xx
xx*~~*xx
xx*~~*xx
xx*~~*xx
xx~~~~xx
"""

MAGIC_BEAM_E_FACING = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
~*******
~~~~~~~~
~*******
xxxxxxxx
xxxxxxxx
"""

MAGIC_BEAM_S_FACING = flip_vertical(MAGIC_BEAM_N_FACING)
MAGIC_BEAM_W_FACING = flip_horizontal(MAGIC_BEAM_E_FACING)

MAGIC_BEAM = [MAGIC_BEAM_N_FACING, MAGIC_BEAM_E_FACING,
              MAGIC_BEAM_S_FACING, MAGIC_BEAM_W_FACING]

MAGIC_HANDS_N_FACING = """
xx~xx~xx
x~xxxx~x
~*xxxx*~
*~xxxx~*
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

MAGIC_HANDS_E_FACING = """
xxxxxxxx
xxxxx~xx
xx*~~~~*
x*x*~~*x
xx*~~~~*
xxxx~xxx
xxxxxxxx
xxxxxxxx
"""

MAGIC_HANDS_S_FACING = flip_vertical(MAGIC_HANDS_N_FACING)

MAGIC_HANDS_W_FACING = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
x*xxxx*x
*x*xx*x*
~*~xx~*~
x~~~~~~x
xx~~~~xx
"""

FRUIT_TREE = """
x#x####x
##ww##o#
w###o##w
x#######
wow###ow
wwwwwIww
xwIwwwIx
xxIxxxIx
"""

FRUIT_TREE_BARREN = """
x#x####x
##ww####
w######w
x#######
w#w####w
wwwwwIww
xwIwwwIx
xxIxxxIx
"""

DIAMOND_PRINCESS_CUT = """
xxxxxxxx
xxxaaaxx
xxdbabdx
xxbdbdbx
xxxcbcxx
xxxxcxxx
xxxxxxxx
xxxxxxxx
"""

INNER_WALLS_NW = """
xbbbbbbb
bbaaaaaa
babbbbbb
babbbbbb
babbbbbb
babbbbbb
babbbbbb
babbbbbb
"""

INNER_WALLS_NE = """
bbbbbbbx
aaaaaabd
bbbbbbcd
bbbbbbcd
bbbbbbcd
bbbbbbcd
bbbbbbcd
bbbbbbcd
"""

INNER_WALLS_W_INTERSECT_SE = """
babbbbbb
babbbbbb
babbbbbb
babbbbbb
babbbbbb
babbbbbb
babbbbbc
babbbbcd
"""

INNER_WALLS_W = """
babbbbbb
babbbbbb
babbbbbb
babbbbbb
babbbbbb
babbbbbb
babbbbbb
babbbbbb
"""

INNER_WALLS_SE = """
bbbbbbcd
bbbbbbcd
bbbbbbcd
bbbbbbcd
bbbbbbcd
bbbbbbcd
ccccccdd
dddddddx
"""

INNER_WALLS_SW = """
babbbbbb
babbbbbb
babbbbbb
babbbbbb
babbbbbb
babbbbbb
bbcccccc
xddddddd
"""

INNER_WALLS_E_INTERSECT_SW = """
bbbbbbcd
bbbbbbcd
bbbbbbcd
bbbbbbcd
bbbbbbcd
bbbbbbcd
ccbbbbcd
ddbbbbcd
"""

INNER_WALLS_E = """
bbbbbbcd
bbbbbbcd
bbbbbbcd
bbbbbbcd
bbbbbbcd
bbbbbbcd
bbbbbbcd
bbbbbbcd
"""

INNER_WALLS_VERTICAL = """
babbbbcd
babbbbcd
babbbbcd
babbbbcd
babbbbcd
babbbbcd
babbbbcd
babbbbcd
"""

INNER_WALLS_S_CAP = """
babbbbcd
babbbbcd
babbbbcd
babbbbcd
babbbbcd
bbbbbbcd
xbccccdd
xddddddx
"""

INNER_WALLS_N_CAP = """
xbbbbbbx
bbaaaabd
babbbbcd
babbbbcd
babbbbcd
babbbbcd
babbbbcd
babbbbcd
"""

CONVERTER_HOPPER = """
xxxxxxxx
eeeeeeee
e>>>>>>e
e<<<<<<e
e,,,,,,e
e,,,,,,e
e,,,,,,e
e,,,,,,e
"""

CONVERTER_ACCEPTANCE_INDICATOR = """
ecccccee
gcddddfg
gcd[]dfg
gcd_]dfg
gcddddfg
gcddddfg
gcddddfg
gaaaaacg
"""

CONVERTER_IDLE = """
ga`bb`cg
gabbbbcg
gccccccg
ghhhhhhg
ghhAAhhg
hhBBBChh
xxAAABxx
xxBBBCxx
"""

CONVERTER_ON = """
ga!bb!cg
gabbbbcg
gccccccg
ghhhhhhg
ghhAAhhg
hhBBBChh
xxAAABxx
xxBBBCxx
"""

CONVERTER_ON_FIRST = """
ga!bb!cg
gabbbbcg
gccccccg
ghhhhhhg
ghhBBhhg
hhAAABhh
xBBBBCCx
xxAAABxx
"""

CONVERTER_ON_SECOND = """
ga!bb!cg
gabbbbcg
gccccccg
ghhhhhhg
ghhBBhhg
hhAAABhh
xxBBBCxx
xAAAABBx
"""

CONVERTER_ON_THIRD = """
ga`bb!cg
gabbbbcg
gccccccg
ghhhhhhg
ghhAAhhg
hhBBBChh
xxAAABxx
xxBBBCxx
"""

CONVERTER_ON_FOURTH = """
ga`bb!cg
gabbbbcg
gccccccg
ghhhhhhg
ghhBBhhg
hhAAABhh
xBBBBCCx
xxAAABxx
"""

CONVERTER_ON_FIFTH = """
ga`bb!cg
gabbbbcg
gccccccg
ghhhhhhg
ghhBBhhg
hhAAABhh
xxBBBCxx
xAAAABBx
"""

CONVERTER_DISPENSER_IDLE = """
xxAAABxx
*ffffff*
hhhhhhhh
h<,,,,<h
h>>>>>>h
hhhhhhhh
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

CONVERTER_DISPENSER_RETRIEVING = """
*ffffff*
hhhhhhhh
h<,,,,<h
h>>>>>>h
hhhhhhhh
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

CONVERTER_DISPENSER_DISPENSING = """
xxAAABxx
xxBBBCxx
*ffffff*
hhhhhhhh
h<,,,,<h
h>>>>>>h
hhhhhhhh
xxxxxxxx
xxxxxxxx
"""

SQUARE = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

CYTOAVATAR_EMPTY_N = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xx&**xxx
x&****xx
x&****xx
xx&&&xxx
"""

CYTOAVATAR_EMPTY_E = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xx&***xx
x&*,*,*x
x&*****x
xx&&&&xx
"""

CYTOAVATAR_EMPTY_S = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xx&**xxx
x&,*,*xx
x&****xx
xx&&&xxx
"""

CYTOAVATAR_EMPTY_W = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xx****xx
x&,*,**x
x&*****x
xx&&&&xx
"""

CYTOAVATAR_EMPTY = [CYTOAVATAR_EMPTY_N, CYTOAVATAR_EMPTY_E,
                    CYTOAVATAR_EMPTY_S, CYTOAVATAR_EMPTY_W]

CYTOAVATAR_HOLDING_ONE_N = """
xxxxxxxx
xx&**xxx
x&****xx
x&&&&&xx
&&ooo&&x
&ooooo&x
&&ooo&&x
x&&&&&xx
"""

CYTOAVATAR_HOLDING_ONE_E = """
xxxxxxxx
xx&***xx
x&*,*,*x
x&*****x
&&oooo*x
&ooooo&x
&&ooo&&x
x&&&&&xx
"""

CYTOAVATAR_HOLDING_ONE_S = """
xxxxxxxx
xx&**xxx
x&,*,*xx
x&****xx
&&ooo**x
&ooooo&x
&&ooo&&x
x&&&&&xx
"""

CYTOAVATAR_HOLDING_ONE_W = """
xxxxxxxx
x****xxx
&,*,**xx
&*****xx
&oooo**x
&ooooo&x
&&ooo&&x
x&&&&&xx
"""

CYTOAVATAR_HOLDING_ONE = [CYTOAVATAR_HOLDING_ONE_N, CYTOAVATAR_HOLDING_ONE_E,
                          CYTOAVATAR_HOLDING_ONE_S, CYTOAVATAR_HOLDING_ONE_W]

CYTOAVATAR_HOLDING_MULTI_N = """
xx&***xx
x&*****x
x&&&&&&x
&&oooo&&
&oooooo&
&oooooo&
&&oooo&&
x&&&&&&x
"""

CYTOAVATAR_HOLDING_MULTI_E = """
xx&***xx
x&*,*,*x
x&*****x
&&oooo&&
&oooooo&
&oooooo&
&&oooo&&
x&&&&&&x
"""

CYTOAVATAR_HOLDING_MULTI_S = """
xx&***xx
x&,**,*x
x&*****x
&&oooo&&
&oooooo&
&oooooo&
&&oooo&&
x&&&&&&x
"""

CYTOAVATAR_HOLDING_MULTI_W = """
xx&***xx
x&,*,**x
x&*****x
&&oooo&&
&oooooo&
&oooooo&
&&oooo&&
x&&&&&&x
"""

CYTOAVATAR_HOLDING_MULTI = [CYTOAVATAR_HOLDING_MULTI_N,
                            CYTOAVATAR_HOLDING_MULTI_E,
                            CYTOAVATAR_HOLDING_MULTI_S,
                            CYTOAVATAR_HOLDING_MULTI_W]

SINGLE_HOLDING_LIQUID = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xlllxxxx
xxlllxxx
xxxxxxxx
"""

SINGLE_HOLDING_SOLID = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxSsxx
xxxxssxx
xxxxxxxx
"""

MULTI_HOLDING_SECOND_LIQUID = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxLLxxx
xxxxLLLx
xxxxxLxx
xxxxxxxx
"""

MULTI_HOLDING_SOLID = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxSsxx
xxxxssxx
xxxxxxxx
xxxxxxxx
"""

PETRI_DISH_NW_WALL_CORNER = """
xxx&&&&&
xx&~~~~~
x&*ooooo
&~o*oooo
&~oo*o&&
&~ooo&@@
&~oo&@@@
&~oo&@@#
"""

PETRI_DISH_NE_WALL_CORNER = flip_horizontal(PETRI_DISH_NW_WALL_CORNER)
PETRI_DISH_SE_WALL_CORNER = flip_vertical(PETRI_DISH_NE_WALL_CORNER)
PETRI_DISH_SW_WALL_CORNER = flip_vertical(PETRI_DISH_NW_WALL_CORNER)

PETRI_DISH_N_WALL = """
&&&&&&&&
~~~~~~~~
oooooooo
oooooooo
&&&&&&&&
@@@@@@@@
@@@@@@@@
########
"""

PETRI_DISH_W_WALL = """
&~oo&@@#
&~oo&@@#
&~oo&@@#
&~oo&@@#
&~oo&@@#
&~oo&@@#
&~oo&@@#
&~oo&@@#
"""

GRID_FLOOR_LARGE = """
@@@@@@@#
@@@@@@@#
@@@@@@@#
@@@@@@@#
@@@@@@@#
@@@@@@@#
@@@@@@@#
########
"""

PETRI_DISH_E_WALL = flip_horizontal(PETRI_DISH_W_WALL)
PETRI_DISH_S_WALL = flip_vertical(PETRI_DISH_N_WALL)

SOLID = """
xxxxxxxb
xxxxxxxb
xxxSsxxb
xxSSssxb
xxssZZxb
xxxsZxxb
xxxxxxxb
bbbbbbbb
"""

GAS = """
xxxxxxgx
GxxGxGxx
xxGxxxGg
xxggxgxx
xGgxgGgx
GxxxGgxx
xxgxxxxx
xxxxxxgx
"""

LIQUID = """
xxxxxxxb
xxxxxxxb
xwwwllxb
wwlllxxb
xxLwwllb
xwwwllxb
xxllwwLl
bbbbbbbb
"""

SOLID_S_CAP = """
xxxxxxxx
xxxxxxxx
xxxSsxxx
xxSSssxx
xxssZZxx
xxxsZxxx
xxxsZxxx
xxxsZxxx
"""

SOLID_E_CAP = """
xxxxxxxx
xxxxxxxx
xxxSsxxx
xxSSssss
xxssZZZZ
xxxsZxxx
xxxxxxxx
xxxxxxxx
"""

SOLID_N_CAP = """
xxxSsxxx
xxxSsxxx
xxxSsxxx
xxSSssxx
xxssZZxx
xxxsZxxx
xxxxxxxx
xxxxxxxx
"""

SOLID_W_CAP = """
xxxxxxxx
xxxxxxxx
xxxSsxxx
SSSSssxx
ssssZZxx
xxxsZxxx
xxxxxxxx
xxxxxxxx
"""

SOLID_X_COUPLING = """
xxx*sxxx
xxx*sxxx
xxxSsxxx
SSSSssss
ssssZZZZ
xxxsZxxx
xxxsZxxx
xxxsZxxx
"""

SOLID_ES_COUPLING = """
xxxxxxxx
xxxxxxxx
xxxSsxxx
xxSSssss
xxssZZZZ
xxxsZxxx
xxxsZxxx
xxxsZxxx
"""

SOLID_SW_COUPLING = """
xxxxxxxx
xxxxxxxx
xxxSsxxx
SSSSssxx
ssssZZxx
xxxsZxxx
xxxsZxxx
xxxsZxxx
"""

SOLID_NW_COUPLING = """
xxxSsxxx
xxxSsxxx
xxxSsxxx
SSSSssxx
ssssZZxx
xxxsZxxx
xxxxxxxx
xxxxxxxx
"""

SOLID_NE_COUPLING = """
xxxSsxxx
xxxSsxxx
xxxSsxxx
xxSSssss
xxssZZZZ
xxxsZxxx
xxxxxxxx
xxxxxxxx
"""

SOLID_S_TCOUPLING = """
xxxSsxxx
xxxSsxxx
xxxSsxxx
SSSSssss
ssssZZZZ
xxxsZxxx
xxxxxxxx
xxxxxxxx
"""

SOLID_E_TCOUPLING = """
xxxSsxxx
xxxSsxxx
xxxSsxxx
SSSSssxx
ssssZZxx
xxxsZxxx
xxxsZxxx
xxxsZxxx
"""

SOLID_N_TCOUPLING = """
xxxxxxxx
xxxxxxxx
xxxSsxxx
SSSSssss
ssssZZZZ
xxxsZxxx
xxxsZxxx
xxxsZxxx
"""

SOLID_W_TCOUPLING = """
xxxSsxxx
xxxSsxxx
xxxSsxxx
xxSSssss
xxssZZZZ
xxxsZxxx
xxxsZxxx
xxxsZxxx
"""

SOLID_NS_COUPLING = """
xxxSsxxx
xxxSsxxx
xxxSsxxx
xxSSssxx
xxssZZxx
xxxsZxxx
xxxsZxxx
xxxsZxxx
"""

SOLID_EW_COUPLING = """
xxxxxxxx
xxxxxxxx
xxxSsxxx
SSSSssss
ssssZZZZ
xxxsZxxx
xxxxxxxx
xxxxxxxx
"""

APPLE = """
xxxxxxxx
xxxxxxxx
xxo|*xxx
x*#|**xx
x*****xx
x#***#xx
xx###xxx
xxxxxxxx
"""

APPLE_JUMP = """
xxxxxxxx
xxo|*xxx
x*#|**xx
x*****xx
x#***#xx
xx###xxx
xxxxxxxx
xxxxxxxx
"""

N_EDGE = """
********
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

N_HALF_EDGE = """
xxxx****
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

W_EDGE = """
*xxxxxxx
*xxxxxxx
*xxxxxxx
*xxxxxxx
*xxxxxxx
*xxxxxxx
*xxxxxxx
*xxxxxxx
"""

S_EDGE = flip_vertical(N_EDGE)
E_EDGE = flip_horizontal(W_EDGE)

PAINTER_STAND_S_FACING = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xoooooox
xo!**!ox
********
*WWWWWW*
wWWWWWWx
"""

PAINTER_STAND_E_FACING = """
xxxxx**x
xxxxx*WW
xxxxx*WW
xxxx**WW
xxxx**WW
xxxxx*WW
xxxxx*WW
xxxxx**w
"""

PAINTER_STAND_N_FACING = flip_vertical(PAINTER_STAND_S_FACING)
PAINTER_STAND_W_FACING = flip_horizontal(PAINTER_STAND_E_FACING)

PAINTER_COVER = """
xWWWWWWx
xWWWWWWw
wWWWWWWw
wWWWWWWx
xWWWWWWw
wWWWWWWx
wWWWWWWw
xWWWWWWx
"""

RECEIVER_MOUTH = """
@~!!&&&&
*~GGGgl&
*~BBGgl&
*~BBGgl&
*~BBGgl&
*~BBGgl&
*~GGGgl&
~~!!!&&&
"""

RECEIVER_BACK = """
xxxx@@@@
exxx@***
eded@***
aded@***
bded@***
ceae@***
cxxx@***
xxxx~~~~
"""

CARBON_INDICATOR = """
xxxxxxxx
xxxxxxxx
xxxxxOoO
xxxxxoOo
xxxxxOoO
xxxxxsSs
xxxxxxxx
xxxxxxxx
"""

WOOD_INDICATOR = """
xxxxxxxx
xxxxxxxx
xxxxxOOO
xxxxxooo
xxxxxOOO
xxxxxsss
xxxxxxxx
xxxxxxxx
"""

METAL_INDICATOR = """
xxxxxxxx
xxxxxxxx
xxxxxOOO
xxxxxOOO
xxxxxOOO
xxxxxsss
xxxxxxxx
xxxxxxxx
"""

CONVEYOR_BELT_1 = """
YYYBBBBY
Mhshshsh
Mhyyshyy
Myyhsyyh
Myyhsyyh
Mhyyshyy
Mhshshsh
BYYYYBBB
"""

CONVEYOR_BELT_2 = """
YBBBBYYY
shshshMh
yyshyyMh
yhsyyhMy
yhsyyhMy
yyshyyMh
shshshMh
YYYBBBBY
"""

CONVEYOR_BELT_3 = """
BBBYYYYB
shshMhsh
shyyMhyy
syyhMyyh
syyhMyyh
shyyMhyy
shshMhsh
YBBBBYYY
"""

CONVEYOR_BELT_4 = """
BYYYYBBB
shMhshsh
yyMhyysh
yhMyyhsy
yhMyyhsy
yyMhyysh
shMhshsh
BBBYYYYB
"""

CONVEYOR_BELT_S_1 = """
YhyhhyhB
BsyyyysB
BhhyyhhB
BssssssY
BhyhhyhY
YsyyyysY
YhhyyhhY
YMMMMMMB
"""

CONVEYOR_BELT_S_2 = """
YhhyyhhY
YMMMMMMB
YhyhhyhB
BsyyyysB
BhhyyhhB
BssssssY
BhyhhyhY
YsyyyysY
"""

CONVEYOR_BELT_S_3 = """
BhyhhyhY
YsyyyysY
YhhyyhhY
YMMMMMMB
YhyhhyhB
BsyyyysB
BhhyyhhB
BssssssY
"""


CONVEYOR_BELT_S_4 = """
BhhyyhhB
BssssssY
BhyhhyhY
YsyyyysY
YhhyyhhY
YMMMMMMB
YhyhhyhB
BsyyyysB
"""

CONVEYOR_BELT_ANCHOR_TOP_RIGHT = """
:xxxxxxx
:,xxxxxx
:,xxxxxx
:,xxxxxx
:,xxxxxx
:,xxxxxx
:,xxxxxx
:,xxxxxx
"""

CONVEYOR_BELT_ANCHOR_RIGHT = """
:,xxxxxx
:,xxxxxx
:,xxxxxx
:,xxxxxx
:,xxxxxx
:,xxxxxx
:,xxxxxx
:,xxxxxx
"""

CONVEYOR_BELT_ANCHOR_TOP_LEFT = """
xxxxxxxxg
xxxxxxxgG
xxxxxxxgG
xxxxxxxgG
xxxxxxxgG
xxxxxxxgG
xxxxxxxgG
xxxxxxxgG
"""

CONVEYOR_BELT_ANCHOR_LEFT = """
xxxxxxxgG
xxxxxxxgG
xxxxxxxgG
xxxxxxxgG
xxxxxxxgG
xxxxxxxgG
xxxxxxxgG
xxxxxxxgG
"""

METAL_FLOOR_DOUBLE_SPACED = """
--------
----xo--
--------
--xo----
--------
xo------
--------
--------
"""

CARBON_OBJECT = """
xxxxxxxx
xx*!*!xx
xx!*!*xx
xx*!*!xx
xx!*!*xx
xxsSsSxx
xxxxxxxx
xxxxxxxx
"""

WOOD_OBJECT = """
xxxxxxxx
xx!!!!xx
xx***@xx
xx!!!!xx
xx***@xx
xxSSSSxx
xxxxxxxx
xxxxxxxx
"""

METAL_OBJECT = """
xxxxxxxx
xx***#xx
xx*@@#xx
xx*@@#xx
xx####xx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

METAL_DROPPING_1 = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxx@@#xx
xxx@@#xx
xxx###xx
xxxxxxxx
"""

METAL_DROPPING_2 = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xx##xxxx
xx##xxxx
"""


CARBON_DROPPING_1 = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxsSsxx
xxxSsSxx
xxxsSsxx
xxxxxxxx
"""

CARBON_DROPPING_2 = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxsSxxxx
xxSsxxxx
"""

RECEIVER_PUSHING_1 = """
xxxx@@@@
xexx@***
xede@***
xade@***
xbde@***
xcea@***
xcxx@***
xxxx~~~~
"""

RECEIVER_PUSHING_2 = """
xxxx@@@@
exxd@***
eded@***
aded@***
bded@***
ceae@***
cxxe@***
xxxx~~~~
"""

RECEIVER_PUSHING_3 = """
xxxx@@@@
exex@***
eded@***
aded@***
bded@***
ceae@***
cxax@***
xxxx~~~~
"""

RECEIVER_PUSHING_4 = """
xxxx@@@@
edxx@***
eded@***
aded@***
bded@***
ceae@***
cexx@***
xxxx~~~~
"""

DIAMOND = """
xxxabxxx
xxaabbxx
xaaabbbx
aaaabbbb
ddddcccc
xdddcccx
xxddccxx
xxxdcxxx
"""

SMALL_DIAMOND = """
xxxxxxxx
xxxabxxx
xxaabbxx
xaaabbbx
xdddcccx
xxddccxx
xxxdcxxx
xxxxxxxx
"""

CUTE_AVATAR_RANK_FIRST_N = """
xxxxxx,,
x*xx*x,,
x****,xx
x&&&&xxx
******xx
&****&xx
x****xxx
x&xx&xxx
"""

CUTE_AVATAR_RANK_FIRST_E = """
xxxxx,,,
x*x*xx,,
x****x,,
x*O*O,xx
**##*&xx
&****&xx
x****xxx
x&&x&xxx
"""

CUTE_AVATAR_RANK_FIRST_S = """
xxxxxx,,
x*xx*x,,
x****,xx
xO**Oxxx
&*##*&xx
&****&xx
x****xxx
x&xx&xxx
"""

CUTE_AVATAR_RANK_FIRST_W = """
xxxxx,,,
xx*x*x,,
x****x,,
xO*O*,xx
&*##**xx
&****&xx
x****xxx
x&x&&xxx
"""

CUTE_AVATAR_RANK_FIRST = [CUTE_AVATAR_RANK_FIRST_N, CUTE_AVATAR_RANK_FIRST_E,
                          CUTE_AVATAR_RANK_FIRST_S, CUTE_AVATAR_RANK_FIRST_W]

CUTE_AVATAR_RANK_SECOND_N = """
xxxxxx,,
x*xx*x,,
x****,xx
x&&&&xxx
******xx
&****&xx
x****xxx
x&xx&xxx
"""

CUTE_AVATAR_RANK_SECOND_E = """
xxxxx,,,
x*x*xx,,
x****x,,
x*O*O,xx
**##*&xx
&****&xx
x****xxx
x&&x&xxx
"""

CUTE_AVATAR_RANK_SECOND_S = """
xxxxxx,,
x*xx*x,,
x****,xx
xO**Oxxx
&*##*&xx
&****&xx
x****xxx
x&xx&xxx
"""

CUTE_AVATAR_RANK_SECOND_W = """
xxxxx,,,
xx*x*x,,
x****x,,
xO*O*,xx
&*##**xx
&****&xx
x****xxx
x&x&&xxx
"""

CUTE_AVATAR_RANK_SECOND = [CUTE_AVATAR_RANK_SECOND_N, CUTE_AVATAR_RANK_SECOND_E,
                           CUTE_AVATAR_RANK_SECOND_S, CUTE_AVATAR_RANK_SECOND_W]

CUTE_AVATAR_RANK_RUNNER_UP_N = """
xxxxxx,,
x*xx*x,,
x****,xx
x&&&&xxx
******xx
&****&xx
x****xxx
x&xx&xxx
"""

CUTE_AVATAR_RANK_RUNNER_UP_E = """
xxxxx,,,
x*x*xx,,
x****x,,
x*O*O,xx
**##*&xx
&****&xx
x****xxx
x&&x&xxx
"""

CUTE_AVATAR_RANK_RUNNER_UP_S = """
xxxxxx,,
x*xx*x,,
x****,xx
xO**Oxxx
&*##*&xx
&****&xx
x****xxx
x&xx&xxx
"""

CUTE_AVATAR_RANK_RUNNER_UP_W = """
xxxxx,,,
xx*x*x,,
x****x,,
xO*O*,xx
&*##**xx
&****&xx
x****xxx
x&x&&xxx
"""

CUTE_AVATAR_RANK_RUNNER_UP = [
    CUTE_AVATAR_RANK_RUNNER_UP_N, CUTE_AVATAR_RANK_RUNNER_UP_E,
    CUTE_AVATAR_RANK_RUNNER_UP_S, CUTE_AVATAR_RANK_RUNNER_UP_W
]

CUTE_AVATAR_ARMS_UP_N = """
xxpxxpxx
xp*xx*px
pP****Pp
P&&&&&&P
x******x
xx****xx
xx****xx
xx&xx&xx
"""

CUTE_AVATAR_ARMS_UP_E = """
xxxxxxxx
xx*x*xxx
xx****xx
xx*O*OpP
x*&##*&&
xx****pP
xx****xx
xx&&x&xx
"""

CUTE_AVATAR_ARMS_UP_S = """
xxxxxxxx
xx*xx*xx
xx****xx
xPO**OPx
P&*##*&P
pP****Pp
xp****px
xx&pp&xx
"""

CUTE_AVATAR_ARMS_UP_W = """
xxxxxxxx
xxx*x*xx
xx****xx
PpO*O*xx
&&*##&*x
Pp****xx
xx****xx
xx&x&&xx
"""

CUTE_AVATAR_ARMS_UP = [CUTE_AVATAR_ARMS_UP_N, CUTE_AVATAR_ARMS_UP_E,
                       CUTE_AVATAR_ARMS_UP_S, CUTE_AVATAR_ARMS_UP_W]

COIN_MAGICALLY_HELD = """
xxxx,,,,,,,,xxxx
xxxx,,,,,,,,xxxx
xxxx,,,,,,,,xxxx
xxxx~~@###~~xxxx
,,,~~@@@@##~~,,,
,,~~&&&@@@@#~~,,
,,~&&&&&&&@@#~,,
,,~&*&&&&&&&&~,,
,,~&***&&&&&&~,,
,,~**********~,,
,,~~********~~,,
,,,~~******~~,,,
xxxx~~****~~xxxx
xxxx,,,,,,,,xxxx
xxxx,,,,,,,,xxxx
xxxx,,,,,,,,xxxx
"""

MARKET_CHAIR = """
,,,,,,,,
,,,,,,,,
,,,,,,,,
,,,**,,,
,,*++*,,
,,+*+*,,
,,,,,,,,
,,,,,,,,
"""

# Externality Mushrooms sprites.

MUSHROOM = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxoOOOox
xxO*OOOx
xxOOOO*x
xxwiiiwx
xxx!!!xx
"""

# Factory sprites.

NW_PERSPECTIVE_WALL = """
--------
--------
--------
--------
-----GGG
-----gGg
-----GgG
-----ggg
"""

PERSPECTIVE_WALL = """
--------
--------
--------
--------
GGGGGGGG
GgGgGgGg
gGgGgGgG
gggggggg
"""

PERSPECTIVE_WALL_T_COUPLING = """
--------
--------
--------
--------
G-----GG
G-----Gg
g-----gG
g-----gg
"""

NE_PERSPECTIVE_WALL = """
--------
--------
--------
--------
GGG-----
GgG-----
gGg-----
ggg-----
"""

W_PERSPECTIVE_WALL = """
-----xxx
-----xxx
-----xxx
-----xxx
-----xxx
-----xxx
-----xxx
-----xxx
"""

MID_PERSPECTIVE_WALL = """
x-----xx
x-----xx
x-----xx
x-----xx
x-----xx
x-----xx
x-----xx
x-----xx
"""

E_PERSPECTIVE_WALL = """
xxx-----
xxx-----
xxx-----
xxx-----
xxx-----
xxx-----
xxx-----
xxx-----
"""

PERSPECTIVE_THRESHOLD = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
EEEEEEEE
eeeeeeee
EEEEEEEE
eeeeeeee
"""

PERSPECTIVE_WALL_PALETTE = {
    # Palette for PERSPECTIVE_WALL sprites.
    "-": (130, 112, 148, 255),
    "G": (74, 78, 99, 255),
    "g": (79, 84, 107, 255),
    "E": (134, 136, 138, 255),
    "e": (143, 146, 148, 255),
    "x": (0, 0, 0, 0),
}

HOPPER_BODY = """
xaaaaaax
xaaaaaax
caaaaaab
faaaaaab
gaaaaaab
caaaaaac
caaaaaac
cbbbbbbc
"""

HOPPER_BODY_ACTIVATED = """
xaaaaaax
xaaaaaab
caaaaaab
faaaaaab
gaaaaaab
caaaaaab
caaaaaac
cbbbbbbc
"""

DISPENSER_BODY = """
xaaaaaax
xaaaaaax
maaaaaax
maaaaaax
maaaaaax
xaaaaaax
xaaaaaax
xbbbbbbx
"""

DISPENSER_BODY_ACTIVATED = """
xaaaaaax
maaaaaax
maaaaaax
maaaaaax
maaaaaax
maaaaaax
xaaaaaax
xbbbbbbx
"""

HOPPER_CLOSED = """
ceeeeeec
ceccccec
ceccccec
ceccccec
ceeeeeec
cddddddc
cccccccc
xxxxxxxx
"""

HOPPER_CLOSING = """
ceeeeeec
cec##cec
cec--cec
cec--cec
ceeeeeec
cddddddc
cccccccc
xxxxxxxx
"""

HOPPER_OPEN = """
ceeeeeec
ce####ec
ce#--#ec
ce#--#ec
ceeeeeec
cddddddc
cccccccc
xxxxxxxx
"""

DISPENSER_BELT_OFF = """
xbaaaabx
xbaaaabx
xejjjjex
xejjjjex
xejjjjex
xejjjjex
xdaaaadx
xxxxxxxx
"""

DISPENSER_BELT_ON_POSITION_1 = """
xbaaaabx
xboaaobx
xejOOjex
xejjjjex
xeOjjOex
xejOOjex
xdaaaadx
xxxxxxxx
"""

DISPENSER_BELT_ON_POSITION_2 = """
xbaooabx
xbaaaabx
xeOjjOex
xejOOjex
xejjjjex
xeOjjOex
xdaooadx
xxxxxxxx
"""

DISPENSER_BELT_ON_POSITION_3 = """
xboaaobx
xbaooabx
xejjjjex
xeOjjOex
xejOOjex
xejjjjex
xdoaaodx
xxxxxxxx
"""

FLOOR_MARKING = """
--------
--xx-xx-
-x-xx-x-
-xx-xx--
--xx-xx-
-x-xx-x-
-xx-xx--
--------
"""

FLOOR_MARKING_LONG_TOP = """
--------
--xx-xx-
-x-xx-x-
-xx-xx--
--xx-xx-
-x-xx-x-
-xx-xx--
--xx-xx-
"""

FLOOR_MARKING_LONG_BOTTOM = """
-x-xx-x-
-xx-xx--
--xx-xx-
-x-xx-x-
-xx-xx--
--xx-xx-
-x-xx-x-
--------
"""

APPLE_CUBE_INDICATOR = """
xxxxxxxx
xxgsxxxx
xxffxxxx
xxxxxxxx
xxxxaaxx
xxxxaaxx
xxxxxxxx
xxxxxxxx
"""

APPLE_INDICATOR = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxgsxxx
xxxffxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

DOUBLE_APPLE_INDICATOR = """
xxxxxxxx
xxgsxxxx
xxffxxxx
xxxxxxxx
xxxxgsxx
xxxxffxx
xxxxxxxx
xxxxxxxx
"""

APPLE_INDICATOR_FADE = """
xxxxxxxx
xxxxxxxx
xxxGSxxx
xxxFFxxx
xxxFFxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

APPLE_DISPENSING_ANIMATION_1 = """
xxFffFxx
xxxFFxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

APPLE_DISPENSING_ANIMATION_2 = """
xxxxxxxx
xxxgsxxx
xxFffFxx
xxFffFxx
xxxFFxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

APPLE_DISPENSING_ANIMATION_3 = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxgsxxx
xxFffFxx
xxFffFxx
xxxFFxxx
"""

BANANA_DISPENSING_ANIMATION_1 = """
xxxBbbxx
xxbbbBxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

BANANA_DISPENSING_ANIMATION_3 = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxsxx
xxxxBbxx
xxxBbbxx
xxbbbBxx
"""

CUBE_DISPENSING_ANIMATION_1 = """
xxxaaAxx
xxxaA&xx
xxxA&&xx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

CUBE_DISPENSING_ANIMATION_2 = """
xxxxxxxx
xxxxxxxx
xxxaaAxx
xxxaA&xx
xxxA&&xx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

CUBE_DISPENSING_ANIMATION_3 = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxaaAxx
xxxaA&xx
xxxA&&xx
"""

BANANA = """
xxxxxxxx
xxxxxsxx
xxxxBbxx
xxxBbbxx
xxbbbBxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

BANANA_DROP_1 = """
xxxxxxxx
xxxxxsxx
xxxxBbxx
xxxBbxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

BANANA_DROP_2 = """
xxxxxxxx
xxxxxxxx
xxxxBxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

BLOCK = """
xxxxxxxx
xxxxxxxx
xxaaAxxx
xxaA&xxx
xxA&&xxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

BLOCK_DROP_1 = """
xxxxxxxx
xxxxxxxx
xxxaAxxx
xxxA&xxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

BLOCK_DROP_2 = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxx&xxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

HOPPER_INDICATOR_SINGLE_BLOCK = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxaaxxx
xxxaaxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

HOPPER_INDICATOR_SINGLE_BANANA = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxbxx
xxxbbxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

HOPPER_INDICATOR_TWO_BLOCKS = """
xxxxxxxx
xxxxaaxx
xxxxaaxx
xxxxxxxx
xxaaxxxx
xxaaxxxx
xxxxxxxx
xxxxxxxx
"""

HOPPER_INDICATOR_ONE_BLOCK = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxaaxxxx
xxaaxxxx
xxxxxxxx
xxxxxxxx
"""

HOPPER_INDICATOR_ON = """
xxxxxxxx
xxxxxbxx
xxxbbxxx
xxxxxxxx
xxaaxxxx
xxaaxxxx
xxxxxxxx
xxxxxxxx
"""

HOPPER_INDICATOR_BANANA = """
xxxxxxxx
xxxxxbxx
xxxbbxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

HOPPER_INDICATOR_BLOCK = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxaaxxxx
xxaaxxxx
xxxxxxxx
xxxxxxxx
"""

HOPPER_INDICATOR_FADE = """
xxxxxxxx
xxxxxBxx
xxxBBxxx
xxxxxxxx
xxEExxxx
xxEExxxx
xxxxxxxx
xxxxxxxx
"""

FACTORY_MACHINE_BODY_PALETTE = {
    # Palette for DISPENSER_BODY, HOPPER_BODY, and HOPPER sprites
    "a": (140, 129, 129, 255),
    "b": (84, 77, 77, 255),
    "f": (62, 123, 214, 255),
    "g": (214, 71, 71, 255),
    "c": (92, 98, 120, 255),
    "d": (64, 68, 82, 255),
    "m": (105, 97, 97, 255),
    "e": (120, 128, 156, 255),
    "h": (64, 68, 82, 255),
    "#": (51, 51, 51, 255),
    "-": (0, 0, 0, 255),
    "x": (0, 0, 0, 0),
}

CUTE_AVATAR_W_BUBBLE_N = """
xxxxxx,,
x*xx*x,,
x****,xx
x&&&&xxx
******xx
&****&xx
x****xxx
x&xx&xxx
"""

CUTE_AVATAR_W_BUBBLE_E = """
xxxxx,,,
x*x*xx,,
x****x,,
x*O*O,xx
**##*&xx
&****&xx
x****xxx
x&&x&xxx
"""

CUTE_AVATAR_W_BUBBLE_S = """
xxxxxx,,
x*xx*x,,
x****,xx
xO**Oxxx
&*##*&xx
&****&xx
x****xxx
x&xx&xxx
"""

CUTE_AVATAR_W_BUBBLE_W = """
xxxxx,,,
xx*x*x,,
x****x,,
xO*O*,xx
&*##**xx
&****&xx
x****xxx
x&x&&xxx
"""

CUTE_AVATAR_W_BUBBLE = [CUTE_AVATAR_W_BUBBLE_N, CUTE_AVATAR_W_BUBBLE_E,
                        CUTE_AVATAR_W_BUBBLE_S, CUTE_AVATAR_W_BUBBLE_W]

CUTE_AVATAR_FROZEN = """
  ########
  ##O##O##
  ##OOOO##
  ##,OO,##
  #OO##OO#
  #OOOOOO#
  ##OOOO##
  ##O##O##
  """


# Suggested base colour for the palette: (190, 190, 50, 255)
HD_CROWN_N = """
xxxxxxxxoxxxxxxx
xxxx#xxoooxxoxxx
xxxx@oxoooxooxxx
xxxxx@oo@oooxxxx
xxxx#@@&r*o@Oxxx
xxx#@@@*R*@*oOxx
xxx###@*r*oOOOxx
xxxxx#####OOxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
"""

HD_CROWN_E = """
xxxxxxxxxxxx*xxx
xxx#xxx*xxx*&xxx
xxx@*xx*&x*&&rxx
xxxr@@@*&&&oRrxx
xxxx@@**&&&orxxx
xxx#@**&###OOxxx
xxx###@#xxxxxxxx
xxxxx##xxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
"""

HD_CROWN_S = """
xxxxxxxx@xxxxxxx
xxxx@xx#r*xxoxxx
xxxx@ox%Rrx&oxxx
xxxxx@&RRr&oxxxx
xxxx#@@*r*&oOxxx
xxx#@#####OOoOxx
xxx###xxxxxOOOxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
"""

HD_CROWN_W = """
xxxx#xxxxxxxxxxx
xxxx@@xxx*xxx*xx
xxx%@@*x*&x*&&xx
xxxRr@@*&&&oorxx
xxxxr@**&&&ooxxx
xxxx#####**&&Oxx
xxxxxxxxx#&OOOxx
xxxxxxxxxxOOxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxx
"""

HD_CROWN = [HD_CROWN_N, HD_CROWN_E, HD_CROWN_S, HD_CROWN_W]

JUST_BADGE = """
xxxx
xabx
xcdx
xxxx
"""
EMPTY_TREE = """
x@@@@@@x
x@@@@@@@x
x@@@@@@x
xx@**@xx
xxx**xxx
xxx**xxx
xxx**xxx
xxxxxxxx
"""

EMPTY_SHRUB = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xx@@@@xx
x@@@@@@x
x@@@@@@x
x@@@@@@x
xxxxxxxx
"""

FRUIT_SHRUB = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xx@@@@xx
x@@Z@Z@x
x@Z@Z@@x
x@@@@@@x
xxxxxxxx
"""

FRUIT_IN_SHRUB = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxZxZxx
xxZxZxxx
xxxxxxxx
"""

FRUIT_IN_TREE = """
xxxxxxxx
xxZxZxxx
xxxZxZxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
"""

GRASP_SHAPE = """
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xxxxxxxx
xoxxxxox
xxooooxx
"""

CUTE_AVATAR_CHILD_N = """
xxxxxxxx
xxxxxxxx
xx*xx*xx
xx****xx
xx&&&&xx
x******x
xx&xx&xx
xxxxxxxx
"""

CUTE_AVATAR_CHILD_E = """
xxxxxxxx
xxxxxxxx
xx*x*xxx
xx****xx
xx*O*Oxx
x**##*&x
xx&&x&xx
xxxxxxxx
"""

CUTE_AVATAR_CHILD_S = """
xxxxxxxx
xxxxxxxx
xx*xx*xx
xx****xx
xxO**Oxx
x&*##*&x
xx&xx&xx
xxxxxxxx
"""

CUTE_AVATAR_CHILD_W = """
xxxxxxxx
xxxxxxxx
xxx*x*xx
xx****xx
xxO*O*xx
x&*##**x
xx&x&&xx
xxxxxxxx
"""

CUTE_AVATAR_CHILD = [
    CUTE_AVATAR_CHILD_N, CUTE_AVATAR_CHILD_E, CUTE_AVATAR_CHILD_S,
    CUTE_AVATAR_CHILD_W
]

GEM_PALETTE = {
    "e": (119, 255, 239, 255),
    "r": (106, 241, 225, 255),
    "t": (61, 206, 189, 255),
    "d": (78, 218, 202, 255),
    "x": ALPHA
}

GRATE_PALETTE = {
    "*": (59, 59, 59, 255),
    "@": (70, 70, 70, 255),
    "&": (48, 48, 48, 255),
    "~": (31, 31, 31, 255),
    "X": (104, 91, 91, 255),
    "o": (109, 98, 98, 255),
    "x": ALPHA
}

GRASS_PALETTE = {
    "*": (124, 153, 115, 255),
    "@": (136, 168, 126, 255),
    "x": (204, 199, 192, 255)
}

GLASS_PALETTE = {
    "@": (218, 243, 245, 150),
    "*": (186, 241, 245, 150),
    "!": (134, 211, 217, 150),
    "x": ALPHA
}

WOOD_FLOOR_PALETTE = {
    "-": (130, 100, 70, 255),
    "x": (148, 109, 77, 255)
}

METAL_FLOOR_PALETTE = {
    "o": (90, 92, 102, 255),
    "O": (117, 120, 133, 255),
    "x": (99, 101, 112, 255)
}

METAL_PANEL_FLOOR_PALETTE = {
    "-": (142, 149, 163, 255),
    "#": (144, 152, 166, 255),
    "/": (151, 159, 173, 255)
}

SHIP_PALETTE = {
    "o": (90, 105, 136, 255),
    "#": (58, 68, 102, 255),
    "x": (38, 43, 68, 255)
}

TILE_FLOOR_PALETTE = {
    "t": (235, 228, 216, 255),
    "x": (222, 215, 202, 255),
    "o": (214, 207, 195, 255)
}

ROCK_PALETTE = {
    "l": (20, 30, 40, 255),
    "r": (30, 40, 50, 255),
    "k": (100, 120, 120, 255),
    "*": (90, 100, 110, 255),
    "s": (45, 55, 65, 255),
    "p": (40, 60, 60, 255),
    "x": ALPHA,
}

PAPER_PALETTE = {
    "*": (250, 250, 250, 255),
    "@": (20, 20, 20, 255),
    "x": ALPHA,
}

MOULD_PALETTE = {
    "@": (179, 255, 0, 255),
    "~": (140, 232, 0, 255),
    "*": (132, 222, 0, 255),
    "&": (119, 194, 0, 255),
    "+": (153, 219, 0, 80),
    "x": ALPHA
}

SCISSORS_PALETTE = {
    "*": (89, 26, 180, 255),
    ">": (100, 100, 100, 255),
    "#": (127, 127, 127, 255),
    "x": ALPHA,
}

WATER_PALETTE = {
    "@": (150, 190, 255, 255),
    "*": (0, 100, 120, 255),
    "o": (0, 70, 90, 255),
    "~": (0, 55, 74, 255),
    "x": ALPHA,
}

BOAT_PALETTE = {
    "*": (90, 70, 20, 255),
    "&": (120, 100, 30, 255),
    "o": (160, 125, 35, 255),
    "@": (180, 140, 40, 255),
    "#": (255, 255, 240, 255),
    "x": ALPHA,
}

GRAY_PALETTE = {
    "*": (30, 30, 30, 255),
    "&": (130, 130, 130, 255),
    "@": (200, 200, 200, 255),
    "#": (230, 230, 230, 255),
    "x": ALPHA
}

WALL_PALETTE = {
    "*": (95, 95, 95, 255),
    "&": (100, 100, 100, 255),
    "@": (109, 109, 109, 255),
    "#": (152, 152, 152, 255),
    "x": ALPHA
}

BRICK_WALL_PALETTE = {
    "b": (166, 162, 139, 255),
    "c": (110, 108, 92, 255),
    "o": (78, 78, 78, 255),
    "i": (138, 135, 116, 255),
    "x": ALPHA
}

COIN_PALETTE = {
    "*": (90, 90, 20, 255),
    "@": (220, 220, 60, 255),
    "&": (180, 180, 40, 255),
    "#": (255, 255, 240, 255),
    "x": ALPHA
}

RED_COIN_PALETTE = {
    "*": (90, 20, 20, 255),
    "@": (220, 60, 60, 255),
    "&": (180, 40, 40, 255),
    "#": (255, 240, 240, 255),
    "x": ALPHA
}

GREEN_COIN_PALETTE = {
    "*": (20, 90, 20, 255),
    "@": (60, 220, 60, 255),
    "&": (40, 180, 40, 255),
    "#": (240, 255, 240, 255),
    "x": ALPHA
}

TILED_FLOOR_GREY_PALETTE = {
    "o": (204, 199, 192, 255),
    "-": (194, 189, 182, 255),
    "x": ALPHA
}

INVISIBLE_PALETTE = {
    "*": ALPHA,
    "@": ALPHA,
    "&": ALPHA,
    "#": ALPHA,
    "x": ALPHA
}


TREE_PALETTE = {
    "*": TREE_BROWN,
    "@": LEAF_GREEN,
    "x": ALPHA
}


POTATO_PATCH_PALETTE = {
    "*": VEGETAL_GREEN,
    "@": LEAF_GREEN,
    "x": ALPHA
}


FIRE_PALETTE = {
    "@": TREE_BROWN,
    "*": DARK_FLAME,
    "&": LIGHT_FLAME,
    "x": ALPHA
}


STONE_QUARRY_PALETTE = {
    "@": DARK_STONE,
    "#": LIGHT_STONE,
    "x": ALPHA
}

PRED1_PALETTE = {
    "e": (80, 83, 115, 255),
    "h": (95, 98, 135, 255),
    "s": (89, 93, 128, 255),
    "l": (117, 121, 158, 255),
    "u": (113, 117, 153, 255),
    "a": (108, 111, 145, 255),
    "y": (255, 227, 71, 255),
    "x": ALPHA
}

CROWN_PALETTE = {
    "*": (190, 190, 50, 255),
    "&": (150, 150, 45, 255),
    "o": (100, 100, 30, 255),
    "@": (240, 240, 62, 255),
    "r": (170, 0, 0, 255),
    "R": (220, 0, 0, 255),
    "%": (255, 80, 80, 255),
    "#": (255, 255, 255, 255),
    "O": (160, 160, 160, 255),
    "x": (0, 0, 0, 0),
}

FENCE_PALETTE_BROWN = {
    "a": (196, 155, 123, 255),
    "b": (167, 131, 105, 255),
    "c": (146, 114, 90, 255),
    "d": (122, 94, 75, 255),
    "e": (89, 67, 55, 255),
    "x": (0, 0, 0, 0),
    "#": (0, 0, 0, 38),
}

MUSHROOM_GREEN_PALETTE = {
    "|": (245, 240, 206, 255),
    "!": (224, 216, 173, 255),
    "i": (191, 185, 147, 255),
    "w": (37, 161, 72, 255),
    "O": (90, 224, 116, 255),
    "o": (90, 224, 116, 75),
    "*": (186, 238, 205, 255),
    "x": (0, 0, 0, 0),
}

MUSHROOM_RED_PALETTE = {
    "|": (245, 240, 206, 255),
    "!": (224, 216, 173, 255),
    "i": (191, 185, 147, 255),
    "w": (184, 99, 92, 255),
    "O": (239, 132, 240, 255),
    "o": (239, 132, 240, 75),
    "*": (235, 192, 236, 255),
    "x": (0, 0, 0, 0),
}

MUSHROOM_BLUE_PALETTE = {
    "|": (245, 240, 206, 255),
    "!": (224, 216, 173, 255),
    "i": (191, 185, 147, 255),
    "w": (30, 168, 161, 255),
    "O": (41, 210, 227, 255),
    "o": (41, 210, 227, 75),
    "*": (187, 228, 226, 255),
    "x": (0, 0, 0, 0),
}

MUSHROOM_ORANGE_PALETTE = {
    "|": (245, 240, 206, 255),
    "!": (224, 216, 173, 255),
    "i": (191, 185, 147, 255),
    "w": (242, 140, 40, 255),
    "O": (255, 165, 0, 255),
    "o": (255, 172, 28, 75),
    "*": (197, 208, 216, 255),
    "x": (0, 0, 0, 0),
}

DISPENSER_BELT_PALETTE = {
    # Palette for DISPENSER_BELT sprites
    "a": (140, 129, 129, 255),
    "b": (84, 77, 77, 255),
    "e": (120, 128, 156, 255),
    "j": (181, 167, 167, 255),
    "o": (174, 127, 19, 255),
    "-": (222, 179, 80, 255),
    "O": (230, 168, 25, 255),
    "d": (64, 68, 82, 255),
    "x": (0, 0, 0, 0),
}

FACTORY_OBJECTS_PALETTE = {
    # Palette for BANANA, BLOCK, APPLE and HOPPER_INDICATOR sprites
    "a": (120, 210, 210, 255),
    "A": (100, 190, 190, 255),
    "&": (90, 180, 180, 255),
    "x": (0, 0, 0, 0),
    "b": (245, 230, 27, 255),
    "B": (245, 230, 27, 145),
    "s": (94, 54, 67, 255),
    "E": (124, 224, 230, 104),
    "f": (169, 59, 59, 255),
    "g": (57, 123, 68, 255),
    "F": (140, 49, 49, 255),
    "G": (57, 123, 68, 115),
    "S": (94, 54, 67, 115),
}

BATTERY_PALETTE = {
    # Palette for BATTERY andd WIRES sprites
    "a": (99, 92, 92, 255),
    "A": (71, 66, 66, 255),
    "d": (78, 122, 86, 255),
    "D": (60, 89, 86, 255),
    "f": (62, 123, 214, 255),
    "g": (214, 71, 71, 255),
    "s": (181, 167, 167, 255),
    "w": (223, 246, 245, 255),
    "o": (111, 196, 20, 255),
    "O": (98, 173, 17, 255),
    "#": (0, 0, 0, 255),
    "W": (255, 255, 255, 255),
    "x": (0, 0, 0, 0),
}

STARS_PALETTE = {
    "-": (223, 237, 19, 255),
    "x": (0, 0, 0, 0),
}

GOLD_CROWN_PALETTE = {
    "#": (244, 180, 27, 255),
    "@": (186, 136, 20, 150),
    "x": (0, 0, 0, 0)
}

SILVER_CROWN_PALETTE = {
    "#": (204, 203, 200, 255),
    "@": (171, 170, 167, 150),
    "x": (0, 0, 0, 0),
}

BRONZE_CAP_PALETTE = {
    "#": (102, 76, 0, 255),
    "@": (87, 65, 0, 255),
    "x": (0, 0, 0, 0)
}

YELLOW_FLAG_PALETTE = {
    "*": (255, 216, 0, 255),
    "@": (230, 195, 0, 255),
    "&": (204, 173, 0, 255),
    "|": (102, 51, 61, 255),
    "x": (0, 0, 0, 0)
}

RED_FLAG_PALETTE = {
    "*": (207, 53, 29, 255),
    "@": (181, 46, 25, 255),
    "&": (156, 40, 22, 255),
    "|": (102, 51, 61, 255),
    "x": (0, 0, 0, 0)
}

GREEN_FLAG_PALETTE = {
    "*": (23, 191, 62, 255),
    "@": (20, 166, 54, 255),
    "&": (17, 140, 46, 255),
    "|": (102, 51, 61, 255),
    "x": (0, 0, 0, 0)
}

HIGHLIGHT_PALETTE = {
    "*": (255, 255, 255, 35),
    "o": (0, 0, 0, 35),
    "x": (0, 0, 0, 0)
}

BRUSH_PALETTE = {
    "-": (143, 96, 74, 255),
    "+": (117, 79, 61, 255),
    "k": (199, 176, 135, 255)
}

MAGIC_BEAM_PALETTE = {
    "*": (196, 77, 190, 200),
    "~": (184, 72, 178, 150),
    "x": (0, 0, 0, 0),
}

FRUIT_TREE_PALETTE = {
    "#": (113, 170, 52, 255),
    "w": (57, 123, 68, 255),
    "I": (71, 45, 60, 255),
    "x": (0, 0, 0, 0),
}

CYTOAVATAR_PALETTE = {
    "*": (184, 61, 187, 255),
    "&": (161, 53, 146, 255),
    "o": (110, 15, 97, 255),
    ",": (0, 0, 0, 255),
    "x": (0, 0, 0, 0),
    "#": (255, 255, 255, 255),
}

PETRI_DISH_PALETTE = {
    "@": (238, 245, 245, 255),
    "~": (212, 234, 232, 255),
    "*": (188, 220, 220, 255),
    "o": (182, 204, 201, 255),
    "&": (168, 189, 189, 255),
    "x": (0, 0, 0, 0),
    "#": (255, 255, 255, 255),
}

MATTER_PALETTE = {
    "S": (138, 255, 228, 255),
    "s": (104, 247, 214, 255),
    "Z": (96, 230, 198, 255),
    "G": (104, 247, 214, 100),
    "g": (71, 222, 187, 175),
    "L": (48, 194, 160, 255),
    "l": (41, 166, 137, 255),
    "w": (41, 186, 154, 255),
    "x": (0, 0, 0, 0),
}

CONVEYOR_BELT_PALETTE = {
    "B": (48, 44, 46, 255),
    "Y": (250, 197, 75, 255),
    "y": (212, 177, 97, 255),
    "M": (117, 108, 103, 255),
    "h": (161, 147, 141, 255),
    "s": (148, 135, 130, 255),
}

PAINTER_STAND_BLUE_PALETTE = {
    "*": (70, 147, 199, 255),
    "@": (98, 176, 222, 255),
    "!": (47, 95, 158, 255),
    "o": (41, 77, 128, 255),
    "W": (255, 255, 255, 175),
    "w": (255, 255, 255, 150),
    "x": (255, 255, 255, 0),
}

OBJECT_INDICATOR_PALETTE = {
    "O": (229, 221, 212, 255),
    "o": (185, 178, 170, 255),
    "S": (105, 101, 96, 255),
    "s": (122, 118, 113, 255),
    "x": (0, 0, 0, 0),
}

BLUE_INDICATOR_PALETTE = {
    "O": (111, 191, 237, 255),
    "o": (81, 160, 207, 255),
    "S": (33, 102, 148, 255),
    "s": (29, 89, 130, 255),
    "x": (0, 0, 0, 0),
}

MONOCHROME_OBJECT_PALETTE = {
    "*": (255, 247, 235, 255),
    "@": (245, 237, 225, 255),
    "#": (232, 225, 213, 255),
    "!": (215, 210, 198, 255),
    "S": (145, 141, 136, 255),
    "s": (172, 168, 163, 255),
    "x": (0, 0, 0, 0),
    "o": (107, 86, 85, 255),
    "|": (89, 71, 70, 255),
}

PAINTER_STAND_BLUE_PALETTE = {
    "*": (70, 147, 199, 255),
    "@": (98, 176, 222, 255),
    "!": (47, 95, 158, 255),
    "o": (41, 77, 128, 255),
    "W": (255, 255, 255, 175),
    "w": (255, 255, 255, 150),
    "x": (255, 255, 255, 0),
}

OBJECT_INDICATOR_PALETTE = {
    "O": (229, 221, 212, 255),
    "o": (185, 178, 170, 255),
    "S": (105, 101, 96, 255),
    "s": (122, 118, 113, 255),
    "x": (0, 0, 0, 0),
}

BLUE_INDICATOR_PALETTE = {
    "O": (111, 191, 237, 255),
    "o": (81, 160, 207, 255),
    "S": (33, 102, 148, 255),
    "s": (29, 89, 130, 255),
    "x": (0, 0, 0, 0),
}

MONOCHROME_OBJECT_PALETTE = {
    "*": (255, 247, 235, 255),
    "@": (245, 237, 225, 255),
    "#": (232, 225, 213, 255),
    "!": (215, 210, 198, 255),
    "S": (145, 141, 136, 255),
    "s": (172, 168, 163, 255),
    "x": (0, 0, 0, 0),
    "o": (107, 86, 85, 255),
    "|": (89, 71, 70, 255),
}

CONVERTER_PALETTE = {
    "a": (178, 171, 164, 255),
    "b": (163, 156, 150, 255),
    "c": (150, 144, 138, 255),
    "d": (138, 129, 123, 255),
    "e": (128, 120, 113, 255),
    "f": (122, 114, 109, 255),
    "g": (112, 108, 101, 255),
    "h": (71, 69, 64, 255),
    "A": (129, 143, 142, 255),
    "B": (110, 122, 121, 255),
    "C": (92, 102, 101, 255),
    ">": (51, 51, 51, 255),
    "<": (30, 30, 30, 255),
    ",": (0, 0, 0, 255),
    "x": (0, 0, 0, 0),
    "`": (138, 96, 95, 255),
    "!": (253, 56, 38, 255),
    "[": (74, 167, 181, 255),
    "_": (67, 150, 163, 255),
    "]": (61, 136, 148, 255),
    "*": (0, 0, 0, 73),
}

FACTORY_FLOOR_PALETTE = {
    "-": (204, 204, 188, 255),
    "x": (194, 194, 178, 255),
    "o": (212, 212, 195, 255)
}

CONVEYOR_BELT_PALETTE_MONOCHROME = {
    "y": (181, 170, 168, 255),
    "h": (158, 148, 147, 255),
    "s": (150, 139, 138, 255),
    "M": (135, 124, 123, 255),
    "Y": (194, 160, 81, 255),
    "B": (73, 66, 75, 255)
}

CONVEYOR_BELT_GREEN_ANCHOR_PALETTE = {
    ":": (135, 143, 116, 255),
    ",": (113, 120, 89, 255),
    "G": (148, 156, 126, 255),
    "g": (129, 138, 103, 255),
    "x": (0, 0, 0, 0)
}

BLUE_OBJECT_PALETTE = {
    "@": (51, 170, 189, 255),
    "*": (56, 186, 207, 255),
    "#": (45, 152, 168, 255),
    "x": (0, 0, 0, 0)
}

APPLE_RED_PALETTE = {
    "x": (0, 0, 0, 0),
    "*": (171, 32, 32, 255),
    "#": (140, 27, 27, 255),
    "o": (43, 127, 53, 255),
    "|": (79, 47, 44, 255),
}

DIAMOND_PALETTE = {
    "a": (227, 255, 231, 255),
    "b": (183, 247, 224, 255),
    "c": (166, 224, 203, 255),
    "d": (157, 212, 191, 255),
    "x": (0, 0, 0, 0),
}

WALLS_PALETTE = {
    "a": (191, 183, 180, 255),
    "b": (143, 137, 134, 255),
    "c": (135, 123, 116, 255),
    "d": (84, 76, 72, 255),
    "x": (0, 0, 0, 0),
}

APPLE_TREE_PALETTE = {
    "a": (124, 186, 58, 255),
    "b": (105, 158, 49, 255),
    "o": (199, 33, 8, 255),
    "I": (122, 68, 74, 255),
    "x": (0, 0, 0, 0),
}

BANANA_TREE_PALETTE = {
    "a": (43, 135, 52, 255),
    "b": (37, 115, 45, 255),
    "o": (222, 222, 13, 255),
    "I": (122, 68, 74, 255),
    "x": (0, 0, 0, 0),
}

ORANGE_TREE_PALETTE = {
    "a": (78, 110, 49, 255),
    "b": (37, 115, 45, 255),
    "o": (222, 222, 13, 255),
    "I": (122, 68, 74, 255),
    "x": (0, 0, 0, 0),
}

GOLD_MINE_PALETTE = {
    "a": (32, 32, 32, 255),
    "b": (27, 27, 27, 255),
    "o": (255, 215, 0, 255),
    "I": (5, 5, 5, 255),
    "x": (0, 0, 0, 0),
}

FENCE_PALETTE = {
    "a": (208, 145, 94, 255),
    "b": (191, 121, 88, 255),
    "c": (160, 91, 83, 255),
    "d": (122, 68, 74, 255),
    "e": (94, 54, 67, 255),
    "x": (0, 0, 0, 0),
    "#": (0, 0, 0, 38),
}

SHADOW_PALETTE = {
    "~": (0, 0, 0, 20),
    "*": (0, 0, 0, 43),
    "@": (0, 0, 0, 49),
    "#": (0, 0, 0, 55),
    "x": (0, 0, 0, 0),
}
