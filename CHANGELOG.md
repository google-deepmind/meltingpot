# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [1.0.3] - 2022-03-03

### Changed

- Define `is_focal` is in scenario configs.
- Use `chex.dataclass` for [dm-tree](https://github.com/deepmind/tree)
  compatibility.

### Fixed

- Use correct `is_focal` settings for team-vs-team games
  [#16](https://github.com/deepmind/meltingpot/issues/16).

## [1.0.2] - 2022-02-23

### Added

- Substrates and Scenarios now have ReactiveX observables.

### Changed

- Don't add `INVENTORY` observation to scenarios that don't use it.
- Various updates to RLlib example.
- Improved performance of the component system for environments.

### Fixed

- Simulation Speed [#7](https://github.com/deepmind/meltingpot/issues/7)
- Horizon setting in examples [#9](https://github.com/deepmind/meltingpot/issues/9)
- Error running example "self_play_train.py" [#10](https://github.com/deepmind/meltingpot/issues/10)
- build lab2d on m1 chip [#13](https://github.com/deepmind/meltingpot/issues/13)

## [1.0.1] - 2021-10-01

Submitted a number of fixes to ensure substrates and scenarios operate as
intended. Other changes not expected to have any impact.

### Fixed

- Bots receive incorrect timesteps in Scenario [#3](https://github.com/deepmind/meltingpot/issues/3)
- Python seed is not forwarded to Lua [#4](https://github.com/deepmind/meltingpot/issues/4)
- Second episode using same seed as the first [#5](https://github.com/deepmind/meltingpot/issues/5)

## [1.0.0] - 2021-07-16

Initial release.
