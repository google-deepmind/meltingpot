# Vectorized substrate execution

Melting Pot substrates contain a live DMLab2D environment. That object is not
picklable, so process-based vectorizers that try to serialize an already-built
substrate can fail with errors such as:

```text
TypeError: cannot pickle 'dmlab2d.dmlab2d_pybind.Lab2d' object
```

Use `meltingpot.substrate.build_vectorized` when several independent copies of
the same substrate should run concurrently. It starts one worker process per
environment and constructs DMLab2D inside that worker, so the live engine never
crosses a process boundary.

```python
from meltingpot import substrate


def main():
  name = "prisoners_dilemma_in_the_matrix__arena"
  config = substrate.get_config(name)
  roles = config.default_player_roles

  with substrate.build_vectorized(
      name,
      roles=roles,
      num_envs=4,
  ) as env:
    timesteps = list(env.reset())

    actions = [
        [0] * len(roles)
        for _ in range(env.num_envs)
    ]
    timesteps = list(env.step(actions))

    # Environments can end on different steps. Reset only the workers that
    # reached a terminal timestep before the next vector step, and keep their
    # fresh initial observations for action selection.
    for index, timestep in enumerate(timesteps):
      if timestep.last():
        timesteps[index] = env.reset_at(index)


if __name__ == "__main__":
  main()
```

The `if __name__ == "__main__"` guard is required when using Python's `spawn`
multiprocessing start method. `spawn` is the default because it avoids
inheriting live DMLab2D or TensorFlow state into child processes.

## API

`reset()` resets every worker and returns one `dm_env.TimeStep` per environment.
Pass `indices` to reset only selected workers, or use `reset_at(index)` for one
environment. This is useful when vector members finish episodes at different
times.

`step(actions)` accepts one multi-player action sequence per environment. The
commands are sent to all workers before responses are collected, allowing the
environments to execute concurrently.

`action_spec()`, `observation_spec()`, `reward_spec()`, and `discount_spec()`
return the corresponding spec for one member substrate. All workers created by
`build_vectorized` use the same substrate and role assignment.

`observation()` and `events()` return one value per worker. `close()` is
idempotent and shuts down all worker processes; using the vector environment as
a context manager is recommended.

If a worker raises an exception, the parent raises
`VectorEnvWorkerError` with the worker index, remote exception type, message,
and traceback. Responses from the other workers are still drained so that a
handled worker error does not desynchronize subsequent requests.

This provides native Melting Pot process-level vectorization without requiring
the underlying DMLab2D object itself to be picklable. Compatibility layers such
as Gymnasium or PettingZoo can adapt the returned native Melting Pot timesteps
as needed.
