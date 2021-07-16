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

from typing import Mapping, Optional, Tuple, Union


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

APPLE = """
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
def get_palette(color: Color) -> Mapping[str, ColorRGBA]:
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
