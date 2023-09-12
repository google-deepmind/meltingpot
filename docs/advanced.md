# Advanced concepts and details

In this page you'll find more low level details about Melting Pot substrates and
how they exactly interact with DMLab2D and the user code.

## Engine actions versus user code changes

All visual changes to game objects as well as some non-visual ones are handled
by the engine. These include:

* Movement
* Orientation
* Teleportation
* State change, including
    * Layer change
    * Sprite change
    * Group change
* Firing a beam (a.k.a. zapping)
* Connection / disconnection

These are defined in the engine's
[grid](https://github.com/google-deepmind/lab2d/blob/main/dmlab2d/system/grid_world/grid.h).

WARNING: Changing these attributes do not take effect immediately. Instead, they
         are queued for later processing by the engine.

This delayed change can lead to a common gotcha where a component changes one of
these attributes that is later read by another component (or the same one in
another event) expecting to retrieve the new value.

Changes to any Lua variables of components take effect immediately, as expected.

## Engine update cycle

The execution of an episode occurs by alternating between the user code and the
engine. The specific order is as follows:

1.  Rendering of all the objects in the substrate. This finalises the
    observation for all the players
2.  Components' `update` functions are called in arbitrary order
3.  [Registered updaters](https://github.com/google-deepmind/meltingpot/tree/main/meltingpot/lua/modules/updater_registry.lua)
    are run, in priority order
4.  Process all events that were queued from (2) and (3).

    *   Movement and state change is handled in three parts: Lift, attempt move,
        place. Lifting triggers an `onExit` call even if there is an object
        blocking the movement or change of layer. Placing a piece triggers
        `onEnter`
    *   Unsuccessful movement or state change triggers `onBlocked`
    *   Successful state changes trigger `onStateChange`
    *   Any events produced these callbacks will be queued for a future update
        and are not processed in this step

For more details, see the
[`DoUpdate`](https://github.com/google-deepmind/lab2d/blob/main/dmlab2d/system/grid_world/grid.cc)
method of the engine's grid.
