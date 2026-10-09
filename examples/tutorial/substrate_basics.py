# Copyright 2026 DeepMind Technologies Limited.
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
"""Inspect and step a Melting Pot substrate from a Python terminal.

Examples (from the repository root):
  python -m examples.tutorial.substrate_basics --substrate clean_up
  python -m examples.tutorial.substrate_basics --substrate predator_prey__open \
      --inspect-only
  python -m examples.tutorial.substrate_basics --substrate clean_up --steps 5
"""

import argparse
from collections.abc import Sequence


def explore_substrate(
    name: str,
    *,
    roles: Sequence[str] | None = None,
    steps: int = 3,
    inspect_only: bool = False,
) -> None:
  """Inspect supported roles, optionally build and step the chosen substrate."""
  from meltingpot import substrate  # Imported lazily for inspection/testing.

  if steps < 0:
    raise ValueError('steps must not be negative.')

  config = substrate.get_config(name)
  valid_roles = frozenset(config.valid_roles)
  selected_roles = tuple(
      config.default_player_roles if roles is None else roles
  )
  if not selected_roles:
    raise ValueError('At least one player role is required.')

  invalid_roles = set(selected_roles) - valid_roles
  if invalid_roles:
    raise ValueError(
        f'Unknown roles {sorted(invalid_roles)}; '
        f'valid roles for {name}: {sorted(valid_roles)}.'
    )

  print(f'Substrate: {name}')
  print(f'Valid roles: {sorted(valid_roles)}')
  print(
      f'Default roles ({len(config.default_player_roles)} players): '
      f'{list(config.default_player_roles)}'
  )
  print(
      f'Selected roles ({len(selected_roles)} players): {list(selected_roles)}'
  )

  if inspect_only:
    return

  with substrate.build(name, roles=selected_roles) as env:
    print(f'Action specification: {env.action_spec()}')
    print(f'Observation specification: {env.observation_spec()}')
    print(f'Reward specification: {env.reward_spec()}')

    timestep = env.reset()
    print(f'Reset: observations for {len(timestep.observation)} players')
    # Action 0 is a valid discrete action index for every built-in substrate.
    # Use this fixed action only to demonstrate the API, not as a policy.
    actions = [0] * len(selected_roles)
    for step in range(steps):
      if timestep.last():
        timestep = env.reset()
      timestep = env.step(actions)
      print(f'Step {step + 1}: rewards={timestep.reward}')


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entry point."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--substrate',
      default='clean_up',
      metavar='NAME',
      help='A substrate name listed in meltingpot.substrate.SUBSTRATES.',
  )
  parser.add_argument(
      '--roles',
      nargs='+',
      metavar='ROLE',
      help='Optional role assignment; defaults to the substrate configuration.',
  )
  parser.add_argument(
      '--steps',
      type=int,
      default=3,
      help='How many example actions to apply (default: 3).',
  )
  parser.add_argument(
      '--inspect-only',
      action='store_true',
      help='Show valid/default roles without building the Lab2D environment.',
  )
  args = parser.parse_args(argv)
  try:
    explore_substrate(
        args.substrate,
        roles=args.roles,
        steps=args.steps,
        inspect_only=args.inspect_only,
    )
  except ValueError as exc:
    parser.error(str(exc))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
