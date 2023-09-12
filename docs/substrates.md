# Create a substrate for Melting Pot

This page describes how to create a Melting Pot substrate.

## A note on paths

The DeepMind Lab2D engine supports finding modules in custom locations, so your
substrate files can, in principle, be anywhere in your codebase. Nonetheless,
there are some structure assumptions that can be surprising.

You need to set your `levelDirectory` parameter when initialising a Melting Pot
substrate. This is typically done in your python substrate config (see below).
Within this directory, you'll typically create a new directory with the name of
your substrate, with the following the structure:

*   `//path/to/your/levelDirectory/your_substrate_name` -> Directory with
    *   `init.lua`: An API Factory
    *   `components.lua`: **\[Optional\]** Components specific to this substrate
    *   **[OPTIONAL]** `simulation.lua`: If you are using a custom `Simulation`,
        its implementation would go here.

In your Lua files, `require` statements for Melting Pot modules are done with
full path, similar to Python. For convenience, we tend to keep the prefix in a
local variable. For example, to import the `component_library`,
you use:

```lua
local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component_library')
```

It is also possible to `require` modules using just the module name (without the
full path). For example, for `require 'module_name'`, The engine will look for
the module file in the following places, in order:

*   `<levelDirectory>/<substrateName>/<module_name>.lua` if `levelDirectory` is
    specified.
*   `<levelDirectory>/<substrateName>/<module_name>/init.lua` if
    `levelDirectory` is specified.
*   `<levelDirectory>/<module_name>.lua` if `levelDirectory` is specified.
*   `<levelDirectory>/<module_name>/init.lua` if `levelDirectory` is specified.
*   `<path_to_dmlab2d>/dmlab2d/game_scripts/<module_name>.lua`
*   `<path_to_dmlab2d>/dmlab2d/game_scripts/<module_name>/init.lua`

For more information refer to
[DMLab2D's documentation](https://github.com/google-deepmind/lab2d/blob/main/docs/lua_levels_api.md).

We recommend also adding a human player for ease of debugging / demoing in, e.g.
`meltingpot/human_players/`, as well as a substrate configuration (in
Python) in, e.g. `meltingpot/configs/environments/`. The Python
environment config is where most of the actual data for the substrate resides,
whereas the code resides in the Lua files.

## Substrate required parts

A Melting Pot substrate consists of the following required elements:

*   An API Factory object containing:
    *   `Simulation`: Typically using directly, or inheriting from,
        [`base_simulation.lua`](https://github.com/google-deepmind/meltingpot/tree/main/meltingpot/lua/modules/base_simulation.lua)
    *   settings:
        -   `spriteSize`: Sprites will be squares of `spriteSize` X
            `spriteSize`.
        -   `maxEpisodeLengthFrames` Terminate the episode after this many frames.
*   A configuration file (in Python)

![simulation objects and components](images/levels.png)

### API Factory

An example API Factory implementation is:

```lua
local meltingpot = 'meltingpot.lua.modules.'
local api_factory = require(meltingpot .. 'api_factory')
local simulation = require(meltingpot .. 'base_simulation')

-- Required to be able to use the components in the substrate
local component_library = require(meltingpot .. 'component_library')
local avatar_library = require(meltingpot .. 'avatar_library')
local components = require 'components'

return api_factory.apiFactory{
    Simulation = simulation.BaseSimulation,
    settings = {
        -- Scale each sprite to a square of size `spriteSize` X `spriteSize`.
        spriteSize = 8,
        -- Terminate the episode after this many frames.
        maxEpisodeLengthFrames = 1000,
        -- Required to exist. They will be filled in automatically from Python.
        simulation = {},
        topology = 'BOUNDED',
    }
}
```

where `topology` can be either `'BOUNDED'` or `'TORUS'`. See
[the engine docs](https://github.com/google-deepmind/lab2d/blob/main/docs/system/grid_world.md#topology)
for more info. This is optional, and the default is `BOUNDED`.

The `simulation` field will be explained in more detail below.

While the user doesn't need to directly interact with the API Factory object, it
can be useful to understand how the methods in `Simulation` are used. To learn
more, check
[api_factory.lua](https://github.com/google-deepmind/meltingpot/tree/main/meltingpot/lua/modules/api_factory.lua).

Another critical part of the `init.lua` file is registering the components that
you want to use in your substrate. You achieve this by importing the component
modules that you need. You only need to import them, because they add themselves
to the **component registry** (see below).

That is what the imports following do: add the component library, which contains
useful, modular components that are used along many substrates; the avatar
library, which contains the components for the avatars and other related ones
like the zapper; and the local components of the substrate in question:

```lua
local component_library = require(meltingpot .. 'component_library')
local avatar_library = require(meltingpot .. 'avatar_library')
local components = require 'components'
```

which adds the component library (a set of useful components) and the custom
components from the `<your_substrate_name>/components.lua` file.

### Simulation

The simulation is in charge of creating and managing `GameObject`s and is the
main entry point for registering event callbacks, sprites and substrate maps.
You usually can just use
[`BaseSimulation`](https://github.com/google-deepmind/meltingpot/tree/main/meltingpot/lua/modules/base_simulation.lua),
but you can provide your own. In particular, if you need to add new layers,
require custom world observations, or have other high-substrate requirements,
you will have to provide your own. Simply derive from `BaseSimulation` and
override the behaviour you need.

### Python configuration

The astute reader will be by now wondering how substrates are defined, given
that the API factory and `Simulation` are mostly boilerplate.

Melting Pot substrates are mostly specified via a configuration file written in
Python. In its simplest form, the configuration is a
[ConfigDict](https://github.com/google/ml_collections) with data describing what
the substrate contains.

```python
config.lab2d_settings = {
    "substrateName": your_substrate_name,
    "levelDirectory": "path/to/your/levelDirectory",
    "maxEpisodeLengthFrames": 1000,
    "spriteSize": 8,
    "simulation": {
        "map": ASCII_MAP,
    },
}
```

where `your_substrate_name` is simply the directory name in your substrate
directory containing your `init.lua` with the API factory (e.g. `"leveName":
"allopathic_harvest"` means using the file
`meltingpot/lua/levels/allopathic_harvest/init.lua`).

The above configuration creates an empty instance of the substrate called
`your_substrate_name`.

#### Specifying `GameObject`s as _Prefabs_

As we've seen, `GameObject`s are the atomic concept of entities in the
substrate. Often, we want to have a template for a `GameObject` that we can
instantiate in different locations within the grid. This template is called a
_prefab_. Prefabs are simply a specification or configuration for a full
`GameObject`. A prefab is a dictionary with the following structure:

```python
{
    "name": "some_name",  # The name of the GameObject. An arbitrary string.
    "components": [  # A list of components for the GameObject.
        # You can add as many components as you want. Even of the same type.
    ]
}
```

A prefab configuration is a dictionary with two keys:

*   `"name"`: this is optional, and its value is an arbitrary string that can be
    used to search for named objects using the `getGameObjectsByName`
    method of `BaseSimulation`
*  `"components"`: this is required, and is a list of dictionaries, each of
    which has the following pattern:

```python
{
    "component": "YourComponentName",
    "kwargs": {
        # Key-word arguments to the constructor of the Component.
        # Refer to the documentation of the specific component.
    }
},
```

That is, component configurations are dictionaries with two keys:

*   `"component"`: denotes the name of a component (as a string). This is the
    class name of a component registered in the component registry (see below)
*   `"kwargs"`: are key-word arguments that will be forwarded to the constructor
    of the component. Refer to the documentation of the specific component to
    know which arguments the constructor can receive. The `kwargs` are optional.

Here is an example of a minimal prefab configuration:

```python
{
    "name": "some_name",
    "components": [
        {
            # All GameObjects must have a StateManager and a Transform.
            "component": "StateManager",
            "kwargs": {
                "initialState": "some_state",
                "stateConfigs": [{
                    "state": "some_state",
                    "layer": "logical",
                }],
            }
        },
        {
            "component": "Transform",
        },
    ]
}
```

Note that the `Transform` component will be overridden when instantiating the
object from the prefab. As such, the `kwargs` parameter is optional.

The Melting Pot substrate builder will turn an ASCII map, a mapping of
characters to prefab names, and a mapping of prefab names to the actual prefabs
into the required `GameObject`s of your substrate.

These mappings are specified in `lab2d_settings.simulation` as:

*   `lab2d_settings.simulation.map`: The ASCII map, containing characters that
    could be found in `charPrefabMap`. **WARNING**: Characters not found in the
    `charPrefabMap` will be silently ignored.
*   `lab2d_settings.simulation.prefabs`: The mapping of name to prefab.
*   `lab2d_settings.simulation.charPrefabMap`: The mapping of characters to
    _prefab specifications_ ([see below](substrates.md#prefab-specification)).
*   `lab2d_settings.simulation.buildAvatars`: A boolean denoting whether avatar
    objects should be built by the Melting Pot builder. If `buildAvatars` is set
    to:
    +   `True`: the prefabs _must_ contain a prefab for the "avatar" key. The
        avatars will then be built and optionally can be colored with the custom
        specified palettes in `playerPalettes`. Avatars mut _not_ be passed in
        the `simulation.gameObjects` list.
    +   `False`: avatars must be provided in the `simulation.gameObjects` list
        ([see below](substrates.md#prefab-specification)). In this case,
        `playerPalettes` will be ignored.
*   `lab2d_settings.simulation.playerPalettes`: \[OPTIONAL\] A list of color
    palettes, one per player in the substrate. Avatars will be created with
    these palettes. If not provided and the avatars will be automatically built,
    then we will simply use the first `N` values from
    [`colors.lua`](https://github.com/google-deepmind/meltingpot/tree/main/meltingpot/lua/modules/colors.lua).

##### Prefab Specification

The `charPrefabMap` maps a single character to either the name of a prefab in
`prefabs`, or, alternatively, to a prefab specification. Prefab specifications
are used to denote that a character in the ASCII map corresponds to something
other than a simple prefab. For instance, you might want to have multiple
game objects at the same location (on different layers), or you might want to
have a random choice of one of a list of possible objects to place in that
position. For this cases, we use a prefab specification. The overall structure
is as follows:

```python
{
    "type": <spec_type>,
    "list": <list_of_prefab_names>,
}
```

where `<spec_type>` is either:

*   `"all"`: Indicates that a map character corresponds to all of the given
    objects in the `"list"` part of the spec. All those objects must be in
    different layers.
*   `"choice"`: Indicates that the map character should be sampled from the
    `"list"` of prefab names. Exactly one name will be drawn.

#### Overriding Prefab parameters

Oftentimes, when running an experiment, we want to change the configuration of
some of our prefabs based on some hyper parameters. While it is entirely
possible to do this directly by modifying the prefabs in `config.prefabs`, this
can be cumbersome to specify as a hyper sweep. For conveninece, we provide a way
to create simple prefab overrides.

To override a prefab, we need only provide a dictionary in
`config.prefab_overrides`, where the keys are the prefab names, and the values
are mappings of component names to dictionaries of key-word arguments to
override. For example, to override components in the `"wall"` prefab of
`clean_up.lua`, say to change its color and the type of beam it blocks we would
pass

```python
  config.prefab_overrides = dict(
      wall=dict(
          Appearance=dict(spriteRGBColors=[(200, 200, 0)]),
          BeamBlocker=dict(BeamType="antoherBeam"),
          ),
      )
```

Note that this way of doing things has the shortcoming that whenever a
prefab has two components with the same name, only the first one will have its
values changed.

#### Passing your own `GameObject`s

In addition to creating prefabs, possibly with overrides, to create
`GameObject`s, there is a way to pass on all pregenerated game object
configurations to the substrate constructor.

Within the
[game_object_utils](https://github.com/google-deepmind/meltingpot/tree/main/meltingpot/utils/substrates/game_object_utils.py)
library we have functions to parse ASCII maps to aid in the creation of game
object configs. `GameObject`s created from prefabs using these utilities will
have their `Transform` adapted to their position in the map. However, you can
add any game object configuration instead of, or in addition to, using this
library.

When passing objects manually like this, any prefabs will still be used to
create objects with the ASCII map. The prefab-generated objects will be appended
to the ones manually passed.

WARNING: If you pre-build game objects and also pass prefabs with the ASCII map,
         game objects will be built twice which in most cases will cause a fatal
         error. Prefer to pass prefabs and map for all objects with a definite
         location.

To pass objects manually, pass them, as python lists, to the `lab2d_settings`:

```python
config.lab2d_settings = {
    ...
    "simulation": {
        ...
        "game_objects": my_list_of_object_configs,
    },
}
```

You can also pass you avatars manually just like any other `GameObject`. Avatar
`GameObject`s are simply ones with the
[`Avatar`](https://github.com/google-deepmind/meltingpot/tree/main/meltingpot/lua/modules/avatar_library.lua)
component.

The `Avatar` component configuration has the following structure:

```python
{
    "component": "Avatar",
    "kwargs": {
        # The index of the agent, starting from 1 for the first one
        "index": 1,
        # The name of the group that is used as spawn points
        "spawnGroup": "spawnPoints",
        # A list of strings that correspond to the keys in `actionSpec`
        "actionOrder": [],
        # A dictionary mapping action names to their discrete specification
        "actionSpec": {},
        # A dictionary with settings for the ego-centric observation
        "view": {}
        # An optional custom mapping of sprites for this avatar's observations.
        "spriteMap": {},
    }
},
```

The `actionSpec` is composed of key-value pairs with keys being arbitrary
strings, and values being dictionaries with the following structure:

*   `"default"`: the default value of this action as an integer, typically 0
*   `"min"`: the minimum value (inclusive) of this action (can be negative)
*   `"max"`: the maximum value (inclusive) of this action (can be negative)

So, for instance, a valid `actionSpec` could be:

```python
"actionSpec": {
    "move": {"default": 0, "min": 0, "max": 5},  # no-op, N, S, E, W
    "turn": {"default": 0, "min": -1, "max": 1}, # L, no-op, R
},
```

The `actionOrder` list, as it name implies, defines in which sequential order
will the actions be processed.  Thus, `"actionOrder": ["move", "turn"]` means
that the agent moves before turning, so that an action of move `N` _and_ turn
`L` is unambiguous (first move `N` one grid cell, then turn `L`).

The view consists of a dictionary with the following structure:

*   `"left"`: The extent of the view, in grid cells to the left of the agent
*   `"right"`: The extent of the view, in grid cells to the right of the agent
*   `"forward"`: The extent of the view, in grid cells ahead of the agent
*   `"backwards"`: The extent of the view, in grid cells behind the agent
*   `"centered"`: A boolean denoting if the view is centered on the agent

You can have an avatar have a custom view, where they see some sprites as
another. This is particularly useful to have third-person views of the world
containing some privileged information that is hidden from the avatar view. This
is achieved by providing a `"spriteMap"` to the avatar, which is a dictionary
mapping names of sprites in the substrate, to the name of the sprites that the
avatar would see them as. For instance, to make the avatar see all walls in an
substrate as apples, one can use:

```python
    "spriteMap": {"Wall": "Apple"},
```

## Substrate optional parts

### Custom components

While there are some useful components provided in the
[component_library](https://github.com/google-deepmind/meltingpot/tree/main/meltingpot/lua/modules/component_library.lua)
and
[avatar_library](https://github.com/google-deepmind/meltingpot/tree/main/meltingpot/lua/modules/avatar_library.lua),
most likely your substrate will require custom components. We suggest putting
your components in
`path/to/your/levelDirectory/your_substrate_name/components.lua`. Then you can
require them within your `init.lua` as `require 'components'`. After that, they
will be available when creating prefabs in python.

To learn more about how components work, please refer to their
[documentation](concepts.md#components). In there, we explain the methods that
get called when events happen, like `onEnter` being called when a game object
enters the same `(x, y)` location as your game object containing your component.

Components typically derive from the
[`Component`](https://github.com/google-deepmind/meltingpot/tree/main/meltingpot/lua/modules/component.lua)
class. For example, your `components.lua` file, might contain the following

```lua
local class = require 'common.class'
local component = require 'meltingpot.lua.modules.component'

local MyComponent = class.Class(component.Component)
```

For examples of how to write components, refer to the `component_library`, or
the
[`components.lua`](https://github.com/google-deepmind/meltingpot/tree/main/meltingpot/lua/substrates/clean_up/components.lua)
file of `clean_up`.

#### Component registry

To make sure that the components are available for your game objects in your
substrate, you need to register them, and add them to your `init.lua` file.

The **component registry** is a utility that simplifies this process. All you
need to do is require the `component_registry` in your custom component
definition file (usually `components.lua` as above), and register them at the
bottom of your module, right before returning them. That is, put this at the
top of the `components.lua` file:

```lua
local component_registry = require 'modules.component_registry'
```

and this at the bottom:

```lua
local allComponents = {
  -- List all components defined in the file with
  -- <component_name> = <component_name>, e.g.
  MyComponent = MyComponent,
}

component_registry.registerAllComponents(allComponents)

return allComponents
```

Then, in your `init.lua`, you simply need to import your `components.lua`
like this:

```lua
local components = require 'components'
```

### Human player & minimal tests

To simplify debugging substrates, we provide a simple way to take a Melting Pot
substrate Python configuration and create an interactive game where a person can
take control of an avatar and test functionality.

These _human players_ typically live in
[meltingpot/human_players](https://github.com/google-deepmind/meltingpot/tree/main/meltingpot/human_players/).
In here you can use our
[level_playing_utils](https://github.com/google-deepmind/meltingpot/tree/main/meltingpot/human_players/level_playing_utils.py)
library to hook up your substrate's actions to particular keys. For an example
of this, see
[play_substrate](https://github.com/google-deepmind/meltingpot/tree/main/meltingpot/human_players/play_clean_up.py).

To launch them, run

```sh
bash python meltingpot/human_players/play_substrate.py --substrate_name=clean_up`
```

Since the human players launch a window to render the substrate and capture
interactions from the keyboard, it requires a graphical system.
