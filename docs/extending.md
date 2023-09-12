# Extending Melting Pot

You can extend Melting Pot in two main ways:

1.  Add new scenarios to a substrate; or
2.  Create a new substrate

## Add new scenarios to a substrate

A scenario consists of a set of pre-trained agents, which we refer to as _bots_.
Pretrain bots however you like. To use them in a scenario, you must provide them
in [SavedModel](https://www.tensorflow.org/guide/saved_model) format.

Add the saved models for your bots in a subdirectory under `meltingpot/assets/saved_models/<name_of_substrate>`.

Add your bots to the library by adding an entry for each one in [`bots`](https://github.com/google-deepmind/meltingpot/tree/main/meltingpot/configs/bots/__init__.py).
If your bot is purely a saved model then structure its entry like this:

```python
my_bot_0=_saved_model(
    substrate='name_of_substrate_where_bot_operates',
    model='my_bot_0',
),
```

If instead your bot is a puppet, then select a `Puppeteer` from those
defined in [`puppeteers`](https://github.com/google-deepmind/meltingpot/tree/main/meltingpot/utils/bots/puppeteers).
Then structure its bot entry like this:

```python
my_puppet_bot_0=_puppet(
    substrate='name_of_substrate_where_bot_operates',
    puppeteer_builder=functools.partial(Puppeteer, **kwargs)
),
```


Add the bots to your scenario by adding an entry in
[`scenarios`](https://github.com/google-deepmind/meltingpot/tree/main/meltingpot/configs/scenarios/__init__.py).
Structure your scenario entry like this:

```python
name_of_scenario=Scenario(
    description='write a plain language description of your scenario here',
    tags=frozenset({
        # Optionally add any number of tags to aid subsequent analysis.
        'an_optional_tag',
        'another_optional_tag',
    }),
    substrate='name_of_substrate',
    num_focal_agents=4,  # How many players to sample from the focal population.
    num_background_bots=3,  # How many players to sample from the background population.
    bots=frozenset({
        # Bots will be sampled from this set.
        'my_bot_0',
        'my_bot_1',
        'my_bot_2',
    }),
),
```



## Create a new substrate

Substrates are built with [DeepMind Lab2D](https://github.com/google-deepmind/lab2d).
To simplify the creation of new substrates (or _levels_ in the parlance of
Lab2D), we provide an abstraction layer that enables the use of modular
components to build the functionality needed. This is similar to component
systems used in modern game engines. While you can develop your substrates in
pure Lab2D, we suggest using the component system provided in Melting Pot,
especially if you are a newcomer to Lab2D.

We provide a [tutorial](substrate_tutorial/index.md) to get you started with
creating a new substrate. If you want more detailed information on how to do
this, refer to the [Create a substrate for Melting Pot](substrates.md)
documentation. For information about this component system, please refer to the
[Melting Pot substrate concepts and design](concepts.md) documentation.
