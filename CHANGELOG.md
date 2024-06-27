# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [2.3.0] - 2024-06-27

### Added

- New scenarios, puppets and bots used in the Melting Pot Contest, NeurIPS 2023.

### Known issues

- Two scenarios in the competition were set up wrongly. Tracking issue:
  https://github.com/google-deepmind/meltingpot/issues/246
    * `clean_up_20` is using the wrong (puppet) bot:
      `clean_up__puppet_sanctioning_alternator_nice_0` instead of
      `clean_up__puppet_sanctioning_alternator_0`.
    * `territory__rooms_6` is identical to `territory__rooms_5`.


## [2.2.2] - 2024-03-20

### Fixed

- Update setup.py to work with earlier setuptools (fixes broken 2.2.1 release).


## [2.2.1] - 2024-03-19 [YANKED]

### Changed

- Do not allow `orientation = None` in Transform objects
- Improve documentation for territory__*.
- Improve scenario tags.
- Move puppeteers testutils to testing dir.
- remove restriction on chex
- Update setup.py to work with older versions of setuptools
- Add dev tools to setup.py
- Add pytest-xdist as a required plugin.
- Update pylintrc

### Fixed

- Debug observations break hidden agenda [#168](https://github.com/google-deepmind/meltingpot/issues/168)
- Various lint errors

### Removed

- Residual v1 file: reaction_graph_utils.py
- Remove stale documentation left over from 1.0.


## [2.2.0] - 2023-07-25

### Changed

- Make meltingpot `pip` installable.
- Support `import meltingpot` as an alias of `import meltingpot.python`.
- Raise minimum Python version to Python 3.10.
- Install assets as part of `pip install .`
- Update Dockerfile and dev tooling (pytest, pyink, pytype, pylint, isort).
- Update examples to work with v2.
- Update README.md with new installation details.
- Migrate from rx to reactivex.

### Fixed

- Remove type annotation for dtype.
- Use correct roles in play_hidden_agenda.
- Fix noop that was causing a typing error.
- Add missing `__init__.py` files.
- Set the default orientation to NORTH for objects that have no orientation
  defined.

### Added

- Evaluation utilities.


## [2.1.1] - 2023-02-16

### Changed

- Added COLLECTIVE_RETURN to PERMITTED_OBSERVATIONS.
- Split install.sh into three scripts.
- Move from yapf to black.
- Remove debug observations to speed up environment stepping.

### Fixed

- Add Lua 5.2 compatibility for `unpack`.

### Removed

- SubstrateWrappers previously needed for the v1 bots.

### Added

- Colab for visualizing evaluation results.
- Tests for the examples.
- Mocks of specific substrates and scenarios, for use in testing.
- Helper for setting world.rgb spec.


## [2.1.0] - 2022-12-06

### Changed

- Improve debugging information in `SavedModelPolicy`.
- Resample bots at the beginning of scenario episodes.

### Fixed

- Initialize `Transform` before any other component.
  [#84](https://github.com/google-deepmind/meltingpot/issues/84)
  [#24](https://github.com/google-deepmind/meltingpot/issues/24)

### Added

- New substrate "Hidden Agenda" and its scenarios.


## [2.0.0] - 2022-11-25

Melting Pot Version 2.0.0 release. See
[Melting Pot 2.0 Tech Report](https://arxiv.org/abs/2211.13746)
for detailed information on the new substrates, bots, and scenarios.

### Changed

- Removed all v1 scenarios, bots, and substrates and replaced with new versions.
- Scenarios now support heterogeneous roles, which must be specified at build
  time.
- Various improvements to `examples` and their documentation.

### Added

- New puppeteers and policies to implement new bots.
- New utils to handle sprites, colors, and maps.
- Mocks for use in testing.


## [1.0.4] - 2022-08-22

### Changed

- Drop support for Python 3.7 and 3.8.
- Store saved models on Google Cloud Platform rather than in the Git repo.
- x2 speed improvement to some substrates
- Improved `install.sh` script and installation documentation.
- Various improvements to `examples` and their documentation.

### Fixed

- Puppets were sharing state in Scenarios.
  [#70](https://github.com/google-deepmind/meltingpot/issues/70)
- Various issues with RLlib examples.

### Added

- `.devcontainer` for the project.
- `pettingzoo` example.
- TF1-compatible version of `SavedModelPolicy`.

## [1.0.3] - 2022-03-03

### Changed

- Define `is_focal` is in scenario configs.
- Use `chex.dataclass` for [dm-tree](https://github.com/google-deepmind/tree)
  compatibility.

### Fixed

- Use correct `is_focal` settings for team-vs-team games
  [#16](https://github.com/google-deepmind/meltingpot/issues/16).

## [1.0.2] - 2022-02-23

### Added

- Substrates and Scenarios now have ReactiveX observables.

### Changed

- Don't add `INVENTORY` observation to scenarios that don't use it.
- Various updates to RLlib example.
- Improved performance of the component system for environments.

### Fixed

- Simulation Speed [#7](https://github.com/google-deepmind/meltingpot/issues/7)
- Horizon setting in examples [#9](https://github.com/google-deepmind/meltingpot/issues/9)
- Error running example "self_play_train.py" [#10](https://github.com/google-deepmind/meltingpot/issues/10)
- build lab2d on m1 chip [#13](https://github.com/google-deepmind/meltingpot/issues/13)

## [1.0.1] - 2021-10-01

Submitted a number of fixes to ensure substrates and scenarios operate as
intended. Other changes not expected to have any impact.

### Fixed

- Bots receive incorrect timesteps in Scenario [#3](https://github.com/google-deepmind/meltingpot/issues/3)
- Python seed is not forwarded to Lua [#4](https://github.com/google-deepmind/meltingpot/issues/4)
- Second episode using same seed as the first [#5](https://github.com/google-deepmind/meltingpot/issues/5)

## [1.0.0] - 2021-07-16

Initial release.
