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

When evaluating multi-agent reinforcement learning (MARL) policies across Melting Pot substrates, standard scalar rewards should be complemented with social welfare metrics:

```python
import numpy as np

def compute_social_welfare_metrics(player_rewards: np.ndarray):
  """Computes Utilitarian social welfare and Equality (Gini index).
  
  Args:
    player_rewards: 1D array of total cumulative rewards per player.
  """
  # Utilitarian Social Welfare (Sum of rewards)
  utilitarian_welfare = np.sum(player_rewards)
  
  # Egalitarian / Equality (Gini coefficient)
  diff_matrix = np.abs(player_rewards[:, None] - player_rewards[None, :])
  gini_index = np.sum(diff_matrix) / (2 * len(player_rewards) * max(utilitarian_welfare, 1e-6))
  equality = 1.0 - gini_index
  
  return {
      "utilitarian_welfare": utilitarian_welfare,
      "equality": equality,
      "min_player_reward": np.min(player_rewards),
      "max_player_reward": np.max(player_rewards),
  }
```

