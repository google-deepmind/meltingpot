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
"""Subprocess-based vectorized execution for Melting Pot substrates."""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import multiprocessing
from multiprocessing.connection import Connection
import sys
import time
import traceback
from typing import Any

import dm_env

_READY = "ready"
_RESULT = "result"
_ERROR = "error"
_RESET = "reset"
_STEP = "step"
_OBSERVATION = "observation"
_EVENTS = "events"
_CLOSE = "close"
_CLOSE_TIMEOUT_SECONDS = 5.0


@dataclasses.dataclass(frozen=True)
class _RemoteException:
  exception_type: str
  message: str
  traceback: str


@dataclasses.dataclass(frozen=True)
class _EnvironmentSpecs:
  action: Any
  observation: Any
  reward: Any
  discount: Any


@dataclasses.dataclass(frozen=True)
class _SubstrateBuilder:
  name: str
  roles: tuple[str, ...]

  def __call__(self):
    # Import lazily so the worker can import this module without creating a
    # circular import through the public meltingpot.substrate module.
    from meltingpot import substrate as substrate_api  # pylint: disable=g-import-not-at-top
    return substrate_api.build(self.name, roles=self.roles)


class VectorEnvWorkerError(RuntimeError):
  """Raised when one vector-environment worker fails."""

  def __init__(
      self,
      worker_index: int,
      exception_type: str,
      message: str,
      remote_traceback: str = "",
  ) -> None:
    self.worker_index = worker_index
    self.exception_type = exception_type
    self.remote_traceback = remote_traceback
    detail = f"Worker {worker_index} raised {exception_type}: {message}"
    if remote_traceback:
      detail += f"\nRemote traceback:\n{remote_traceback}"
    super().__init__(detail)


def _serialize_exception() -> _RemoteException:
  current = sys.exc_info()[1]
  assert current is not None
  return _RemoteException(
      exception_type=type(current).__name__,
      message=str(current),
      traceback=traceback.format_exc(),
  )


def _worker(connection: Connection, builder: Callable[[], Any]) -> None:
  """Owns one environment and serves synchronous commands from the parent."""
  env = None
  try:
    env = builder()
    specs = _EnvironmentSpecs(
        action=env.action_spec(),
        observation=env.observation_spec(),
        reward=env.reward_spec(),
        discount=env.discount_spec(),
    )
    connection.send((_READY, specs))

    while True:
      try:
        command, payload = connection.recv()
      except EOFError:
        break

      if command == _CLOSE:
        try:
          env.close()
        finally:
          env = None
        break

      try:
        if command == _RESET:
          result = env.reset()
        elif command == _STEP:
          result = env.step(payload)
        elif command == _OBSERVATION:
          result = env.observation()
        elif command == _EVENTS:
          result = tuple(env.events())
        else:
          raise ValueError(f"Unknown vector environment command: {command!r}")
      except BaseException:  # pylint: disable=broad-except
        connection.send((_ERROR, _serialize_exception()))
      else:
        connection.send((_RESULT, result))
  except BaseException:  # pylint: disable=broad-except
    try:
      connection.send((_ERROR, _serialize_exception()))
    except (BrokenPipeError, EOFError, OSError):
      pass
  finally:
    if env is not None:
      try:
        env.close()
      except BaseException:  # pylint: disable=broad-except
        pass
    connection.close()


class SubprocessVectorEnv:
  """Runs multiple independent environments in worker processes.

  Commands are sent to all selected workers before any response is collected,
  allowing the underlying environments to execute concurrently. Environment
  objects themselves never cross a process boundary; only picklable builders,
  actions, timesteps, observations, events, specs, and errors are transferred.
  """

  def __init__(
      self,
      builders: Sequence[Callable[[], Any]],
      *,
      start_method: str = "spawn",
  ) -> None:
    """Starts one worker process for each environment builder.

    Args:
      builders: Picklable callables that each construct one environment.
      start_method: multiprocessing start method. ``spawn`` is the default to
        avoid inheriting live DMLab2D or TensorFlow state into workers.

    Raises:
      ValueError: if no builders are supplied or the start method is invalid.
      VectorEnvWorkerError: if an environment cannot be constructed.
    """
    builders = tuple(builders)
    if not builders:
      raise ValueError("At least one environment builder is required.")

    self._closed = False
    self._connections: list[Connection] = []
    self._processes: list[multiprocessing.Process] = []
    self._start_method = start_method

    try:
      context = multiprocessing.get_context(start_method)
      for worker_index, builder in enumerate(builders):
        parent_connection, child_connection = context.Pipe()
        process = context.Process(
            target=_worker,
            args=(child_connection, builder),
            name=f"meltingpot-vector-env-{worker_index}",
        )
        try:
          process.start()
        except BaseException:
          parent_connection.close()
          child_connection.close()
          raise
        child_connection.close()
        self._connections.append(parent_connection)
        self._processes.append(process)

      worker_indices = tuple(range(len(self._connections)))
      specs = self._receive_many(worker_indices, expected_kind=_READY)
      self._specs = specs[0]
    except BaseException:
      self.close()
      raise

  @property
  def num_envs(self) -> int:
    """Number of independently running environments."""
    return len(self._connections)

  @property
  def start_method(self) -> str:
    """Multiprocessing start method used by the workers."""
    return self._start_method

  def action_spec(self):
    """Returns the action spec for one member environment."""
    return self._specs.action

  def observation_spec(self):
    """Returns the observation spec for one member environment."""
    return self._specs.observation

  def reward_spec(self):
    """Returns the reward spec for one member environment."""
    return self._specs.reward

  def discount_spec(self):
    """Returns the discount spec for one member environment."""
    return self._specs.discount

  def reset(
      self, indices: Sequence[int] | None = None
  ) -> tuple[dm_env.TimeStep, ...]:
    """Resets all or selected member environments.

    Args:
      indices: worker indices to reset. If omitted, resets every environment.

    Returns:
      Timesteps in the same order as ``indices`` (or worker order when omitted).
    """
    worker_indices = self._normalize_indices(indices)
    payloads = {index: None for index in worker_indices}
    return self._request_many(_RESET, payloads)

  def reset_at(self, index: int) -> dm_env.TimeStep:
    """Resets one member environment and returns its initial timestep."""
    return self.reset((index,))[0]

  def step(
      self, actions: Sequence[Sequence[int]]
  ) -> tuple[dm_env.TimeStep, ...]:
    """Steps every environment concurrently.

    Args:
      actions: one multi-player action sequence per environment.

    Returns:
      One timestep per environment, in worker order.

    Raises:
      ValueError: if the number of action batches does not match ``num_envs``.
    """
    self._ensure_open()
    actions = tuple(actions)
    if len(actions) != self.num_envs:
      raise ValueError(
          f"Expected actions for {self.num_envs} environments, got "
          f"{len(actions)}."
      )
    payloads = {index: action for index, action in enumerate(actions)}
    return self._request_many(_STEP, payloads)

  def observation(self) -> tuple[Any, ...]:
    """Returns the current observation from every environment."""
    worker_indices = tuple(range(self.num_envs))
    payloads = {index: None for index in worker_indices}
    return self._request_many(_OBSERVATION, payloads)

  def events(self) -> tuple[tuple[Any, ...], ...]:
    """Returns current events from every environment."""
    worker_indices = tuple(range(self.num_envs))
    payloads = {index: None for index in worker_indices}
    return self._request_many(_EVENTS, payloads)

  def close(self) -> None:
    """Closes every worker and its underlying environment."""
    if self._closed:
      return
    self._closed = True

    for connection, process in zip(self._connections, self._processes):
      if not process.is_alive():
        continue
      try:
        connection.send((_CLOSE, None))
      except (BrokenPipeError, EOFError, OSError):
        pass

    deadline = time.monotonic() + _CLOSE_TIMEOUT_SECONDS
    for process in self._processes:
      remaining = max(0.0, deadline - time.monotonic())
      process.join(timeout=remaining)

    for process in self._processes:
      if process.is_alive():
        process.terminate()
    for process in self._processes:
      if process.is_alive():
        process.join()

    for connection in self._connections:
      connection.close()

  def __enter__(self):
    return self

  def __exit__(self, *args, **kwargs):
    del args, kwargs
    self.close()

  def _ensure_open(self) -> None:
    if self._closed:
      raise RuntimeError("Vector environment is closed.")

  def _normalize_indices(
      self, indices: Sequence[int] | None
  ) -> tuple[int, ...]:
    self._ensure_open()
    if indices is None:
      return tuple(range(self.num_envs))
    normalized = tuple(indices)
    if len(set(normalized)) != len(normalized):
      raise ValueError("Worker indices must not contain duplicates.")
    for index in normalized:
      if index < 0 or index >= self.num_envs:
        raise IndexError(
            f"Worker index {index} is outside [0, {self.num_envs})."
        )
    return normalized

  def _request_many(
      self, command: str, payloads: Mapping[int, Any]
  ) -> tuple[Any, ...]:
    self._ensure_open()
    worker_indices = tuple(payloads)
    sent_indices = []
    send_error = None

    for index in worker_indices:
      try:
        self._connections[index].send((command, payloads[index]))
      except (BrokenPipeError, EOFError, OSError) as error:
        if send_error is None:
          send_error = VectorEnvWorkerError(
              index, "ConnectionError", str(error)
          )
      else:
        sent_indices.append(index)

    results = self._receive_many(tuple(sent_indices), expected_kind=_RESULT)
    if send_error is not None:
      raise send_error
    results_by_index = dict(zip(sent_indices, results))
    return tuple(results_by_index[index] for index in worker_indices)

  def _receive_many(
      self, worker_indices: Sequence[int], *, expected_kind: str
  ) -> tuple[Any, ...]:
    results = []
    first_error = None
    for index in worker_indices:
      try:
        results.append(self._receive_one(index, expected_kind=expected_kind))
      except VectorEnvWorkerError as error:
        if first_error is None:
          first_error = error
        results.append(None)
    if first_error is not None:
      raise first_error
    return tuple(results)

  def _receive_one(self, index: int, *, expected_kind: str) -> Any:
    try:
      kind, payload = self._connections[index].recv()
    except (EOFError, OSError) as error:
      raise VectorEnvWorkerError(
          index, "ConnectionError", "Worker process exited unexpectedly."
      ) from error

    if kind == _ERROR:
      assert isinstance(payload, _RemoteException)
      raise VectorEnvWorkerError(
          index,
          payload.exception_type,
          payload.message,
          payload.traceback,
      )
    if kind != expected_kind:
      raise VectorEnvWorkerError(
          index,
          "ProtocolError",
          f"Expected response {expected_kind!r}, received {kind!r}.",
      )
    return payload


def build_vectorized(
    name: str,
    *,
    roles: Sequence[str],
    num_envs: int,
    start_method: str = "spawn",
) -> SubprocessVectorEnv:
  """Builds multiple copies of one Melting Pot substrate in subprocesses."""
  if num_envs <= 0:
    raise ValueError("num_envs must be positive.")
  builder = _SubstrateBuilder(name=name, roles=tuple(roles))
  builders = tuple(builder for _ in range(num_envs))
  return SubprocessVectorEnv(builders, start_method=start_method)
