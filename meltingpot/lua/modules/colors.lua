--[[ Copyright 2020 DeepMind Technologies Limited.

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

-- Module with utilities for using colors and creating monochrome palettes.

alpha = {0, 0, 0, 0}
white = {255, 255, 255, 255}
black = {0, 0, 0, 255}
darkGray = {60, 60, 60, 255}
gray = {128, 128, 128, 255}
lightGray = {180, 180, 180, 255}
lightBlue = {178, 206, 234, 255}

-- LINT.IfChange
-- A set of 62 visually distinct colors.
colors = {
    {1, 0, 103, 255},
    {213, 255, 0, 255},
    {255, 0, 86, 255},
    {158, 0, 142, 255},
    {14, 76, 161, 255},
    {255, 229, 2, 255},
    {0, 95, 57, 255},
    {0, 255, 0, 255},
    {149, 0, 58, 255},
    {255, 147, 126, 255},
    {164, 36, 0, 255},
    {0, 21, 68, 255},
    {145, 208, 203, 255},
    {98, 14, 0, 255},
    {107, 104, 130, 255},
    {0, 0, 255, 255},
    {0, 125, 181, 255},
    {106, 130, 108, 255},
    {0, 174, 126, 255},
    {194, 140, 159, 255},
    {190, 153, 112, 255},
    {0, 143, 156, 255},
    {95, 173, 78, 255},
    {255, 0, 0, 255},
    {255, 0, 246, 255},
    {255, 2, 157, 255},
    {104, 61, 59, 255},
    {255, 116, 163, 255},
    {150, 138, 232, 255},
    {152, 255, 82, 255},
    {167, 87, 64, 255},
    {1, 255, 254, 255},
    {255, 238, 232, 255},
    {254, 137, 0, 255},
    {189, 198, 255, 255},
    {1, 208, 255, 255},
    {187, 136, 0, 255},
    {117, 68, 177, 255},
    {165, 255, 210, 255},
    {255, 166, 254, 255},
    {119, 77, 0, 255},
    {122, 71, 130, 255},
    {38, 52, 0, 255},
    {0, 71, 84, 255},
    {67, 0, 44, 255},
    {181, 0, 255, 255},
    {255, 177, 103, 255},
    {255, 219, 102, 255},
    {144, 251, 146, 255},
    {126, 45, 210, 255},
    {189, 211, 147, 255},
    {229, 111, 254, 255},
    {222, 255, 116, 255},
    {0, 255, 120, 255},
    {0, 155, 255, 255},
    {0, 100, 1, 255},
    {0, 118, 255, 255},
    {133, 169, 0, 255},
    {0, 185, 23, 255},
    {120, 130, 49, 255},
    {0, 255, 198, 255},
    {255, 110, 65, 255},
}
-- LINT.ThenChange(//meltingpot/utils/substrates/colors.py)

-- LINT.IfChange
--[[ Convert provided color to a palette suitable for the player text shape.

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
  colorRGB (tuple of length >= 3): Red, Green, Blue (rest is ignored).

Returns:
  palette (dict): maps palette symbols to suitable colors, with full opacity.
]]
function getPalette(colorRGB)
  return {
      ["*"] = {colorRGB[1], colorRGB[2], colorRGB[3], 255},
      ["&"] = {colorRGB[1] * 0.75, colorRGB[2] * 0.75, colorRGB[3] * 0.75,
            255},
      ["o"] = {colorRGB[1] * 0.55, colorRGB[2] * 0.55, colorRGB[3] * 0.55,
            255},
      ["!"] = {colorRGB[1] * 0.65, colorRGB[2] * 0.65, colorRGB[3] * 0.65,
            255},
      ["~"] = {colorRGB[1] * 0.9, colorRGB[2] * 0.9, colorRGB[3] * 0.9,
            255},
      ["@"] = {math.min(colorRGB[1] * 1.25, 255),
            math.min(colorRGB[2] * 1.25, 255),
            math.min(colorRGB[3] * 1.25, 255),
            255},
      ["r"] = {colorRGB[1], colorRGB[3], colorRGB[2], 255},
      ["R"] = {math.min(colorRGB[1] * 1.25, 255),
            math.min(colorRGB[3] * 1.25, 255),
            math.min(colorRGB[2] * 1.25, 255),
            255},
      ["%"] = lightBlue,
      ["#"] = white,
      ["O"] = darkGray,
      [","] = black,
      ["x"] = alpha,
  }
end
-- LINT.ThenChange(//meltingpot/utils/substrates/shapes.py)

return {
    colors = colors,
    alpha = alpha,
    white = white,
    black = black,
    darkGray = darkGray,
    gray = gray,
    lightGray = lightGray,
    getPalette = getPalette,
  }
