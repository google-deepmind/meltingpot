# Using substrates from Python

The public `meltingpot.substrate` module provides the basic functions needed to
inspect and build a substrate. This page shows the usual workflow without
requiring knowledge of the individual substrate configuration files.

## Choose a substrate

Available substrate names are exposed through `SUBSTRATES`:

```python
from meltingpot import substrate

print(sorted(substrate.SUBSTRATES))
```

Once you have chosen a substrate, inspect its configuration before building it:

```python
name = "clean_up"
config = substrate.get_config(name)

print(config.valid_roles)
print(config.default_player_roles)
```

`get_config` returns the locked configuration for that substrate.
`valid_roles` contains the role names accepted by the substrate and
`default_player_roles` provides its standard player assignment. For example,
Clean Up exposes the `default` role and uses seven players by default.

## Build with the default roles

The simplest way to build a substrate is to reuse its default role assignment:

```python
from meltingpot import substrate

name = "clean_up"
config = substrate.get_config(name)

with substrate.build(name, roles=config.default_player_roles) as env:
  timestep = env.reset()
  print(timestep.observation)
```

The environment is a `dm_env`-style environment. Its specifications can be
inspected directly:

```python
with substrate.build(name, roles=config.default_player_roles) as env:
  print(env.observation_spec())
  print(env.action_spec())
  print(env.reward_spec())
```

## Choose player roles

Some substrates support more than one role. Do not infer the allowed strings
from the substrate name. Read them from the configuration:

```python
from meltingpot import substrate

name = "predator_prey__open"
config = substrate.get_config(name)

print("valid roles:", sorted(config.valid_roles))
print("default assignment:", config.default_player_roles)
```

To use another assignment, pass one valid role for each player:

```python
roles = list(config.default_player_roles)
roles[0] = next(role for role in config.valid_roles if role != roles[0])

with substrate.build(name, roles=roles) as env:
  timestep = env.reset()
```

The length of the `roles` sequence determines the number of players passed to
the substrate builder. Every entry should come from `config.valid_roles`.

## Build from a configuration

`build_from_config` is useful when code already has a configuration object. The
roles are still supplied explicitly:

```python
from meltingpot import substrate

config = substrate.get_config("clean_up")

with substrate.build_from_config(
    config, roles=config.default_player_roles
) as env:
  timestep = env.reset()
```

Use `build` when starting from a substrate name. Use `build_from_config` when
starting from a configuration object. For either path, `get_config` is the
place to discover the supported roles and default player assignment.

## Multi-Agent Evaluation & Social Outcome Metrics

When evaluating multi-agent reinforcement learning (MARL) policies across
Melting Pot substrates, standard scalar rewards should be complemented with
social welfare and fairness metrics:

```python
import numpy as np


def compute_social_welfare_metrics(
    player_rewards: np.ndarray, shift_negative: bool = False
):
  """Computes Utilitarian social welfare and Equality (Gini index).

  Note: The standard Gini index assumes non-negative individual rewards and
  positive mean welfare. For substrates where cumulative episode rewards can be
  zero or negative, set `shift_negative=True` to apply an affine baseline shift
  (R' = R - min(R) + 1e-6) so relative inequality remains mathematically bounded.
  If unshifted and signed rewards exist, equality returns `np.nan` and
  `reward_std` provides a dispersion measure.

  Args:
    player_rewards: 1D array of total cumulative rewards per player.
    shift_negative: Whether to non-negatively shift rewards if negative values
      exist.

  Returns:
    Dict containing utilitarian welfare, equality, min/max rewards, and std.
  """
  player_rewards = np.asarray(player_rewards, dtype=float)
  n_players = len(player_rewards)
  utilitarian_welfare = float(np.sum(player_rewards))

  # Guard against negative or zero welfare in Gini computation
  eval_rewards = player_rewards
  if np.any(eval_rewards < 0) or utilitarian_welfare <= 0:
    if shift_negative:
      eval_rewards = eval_rewards - np.min(eval_rewards) + 1e-6
    else:
      return {
          "utilitarian_welfare": utilitarian_welfare,
          "equality": np.nan,
          "min_player_reward": float(np.min(player_rewards)),
          "max_player_reward": float(np.max(player_rewards)),
          "reward_std": float(np.std(player_rewards)),
      }

  eval_welfare = np.sum(eval_rewards)
  diff_matrix = np.abs(eval_rewards[:, None] - eval_rewards[None, :])
  gini_index = np.sum(diff_matrix) / (2 * n_players * eval_welfare)
  equality = float(1.0 - gini_index)

  return {
      "utilitarian_welfare": utilitarian_welfare,
      "equality": equality,
      "min_player_reward": float(np.min(player_rewards)),
      "max_player_reward": float(np.max(player_rewards)),
      "reward_std": float(np.std(player_rewards)),
  }
```


