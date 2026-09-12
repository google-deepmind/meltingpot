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
"""Reproducible manifests for Melting Pot evaluation results."""

from collections.abc import Collection, Mapping, Sequence
import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import math
import pathlib
import platform
from typing import Any

from meltingpot import bot as bot_lib
from meltingpot.configs import bots as bot_configs
from meltingpot.configs import scenarios as scenario_configs
from meltingpot.configs import substrates as substrate_configs
import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
_HASH_ALGORITHM = 'sha256'
_REQUIRED_RESULT_COLUMNS = frozenset({
    'background_player_names',
    'background_player_returns',
    'focal_player_names',
    'focal_player_returns',
})
_RUNTIME_DISTRIBUTIONS = (
    'dm-env',
    'dm-meltingpot',
    'dmlab2d',
    'numpy',
    'pandas',
    'tensorflow',
)


class EvaluationManifestMismatch(ValueError):
  """Raised when results or configuration do not match a manifest."""


@dataclasses.dataclass(frozen=True)
class EvaluationManifest:
  """Versioned fingerprint of one evaluation result table.

  Attributes:
    schema_version: manifest schema version.
    hash_algorithm: digest algorithm used by this schema.
    target: evaluated scenario or substrate name.
    target_kind: either ``scenario`` or ``substrate``.
    configuration_sha256: digest of the target's evaluation-relevant public
      configuration.
    content_sha256: Merkle root over all episode result records.
    episode_sha256: leaf digest for each episode, in evaluation order.
    num_episodes: number of result rows represented by the manifest.
    runtime_versions: sorted ``(name, version)`` pairs recorded for diagnostics.
  """
  schema_version: int
  hash_algorithm: str
  target: str
  target_kind: str
  configuration_sha256: str
  content_sha256: str
  episode_sha256: tuple[str, ...]
  num_episodes: int
  runtime_versions: tuple[tuple[str, str], ...]

  def to_dict(self) -> dict[str, Any]:
    """Returns a JSON-compatible representation."""
    return {
        'schema_version': self.schema_version,
        'hash_algorithm': self.hash_algorithm,
        'target': self.target,
        'target_kind': self.target_kind,
        'configuration_sha256': self.configuration_sha256,
        'content_sha256': self.content_sha256,
        'episode_sha256': list(self.episode_sha256),
        'num_episodes': self.num_episodes,
        'runtime_versions': dict(self.runtime_versions),
    }

  def to_json(self, *, indent: int | None = 2) -> str:
    """Serializes the manifest as deterministic JSON."""
    return json.dumps(
        self.to_dict(),
        sort_keys=True,
        ensure_ascii=False,
        indent=indent,
    )

  def write(self, path: str | pathlib.Path) -> None:
    """Writes the manifest to ``path`` as UTF-8 JSON."""
    pathlib.Path(path).write_text(self.to_json() + '\n', encoding='utf-8')

  @classmethod
  def from_dict(cls, value: Mapping[str, Any]) -> 'EvaluationManifest':
    """Builds and validates a manifest from a mapping."""
    required = {
        'schema_version',
        'hash_algorithm',
        'target',
        'target_kind',
        'configuration_sha256',
        'content_sha256',
        'episode_sha256',
        'num_episodes',
        'runtime_versions',
    }
    missing = required - set(value)
    if missing:
      raise ValueError(f'Manifest is missing fields: {sorted(missing)!r}.')

    schema_version = int(value['schema_version'])
    if schema_version != SCHEMA_VERSION:
      raise ValueError(
          f'Unsupported evaluation manifest schema: {schema_version}.'
      )
    hash_algorithm = str(value['hash_algorithm'])
    if hash_algorithm != _HASH_ALGORITHM:
      raise ValueError(f'Unsupported hash algorithm: {hash_algorithm!r}.')

    target_kind = str(value['target_kind'])
    if target_kind not in ('scenario', 'substrate'):
      raise ValueError(f'Invalid target kind: {target_kind!r}.')

    configuration_digest = str(value['configuration_sha256'])
    content_digest = str(value['content_sha256'])
    _validate_digest(configuration_digest, 'configuration_sha256')
    _validate_digest(content_digest, 'content_sha256')

    episode_digests = tuple(str(digest) for digest in value['episode_sha256'])
    for digest in episode_digests:
      _validate_digest(digest, 'episode_sha256')

    num_episodes = int(value['num_episodes'])
    if num_episodes < 0:
      raise ValueError('num_episodes must not be negative.')
    if len(episode_digests) != num_episodes:
      raise ValueError(
          'episode_sha256 length does not match num_episodes: '
          f'{len(episode_digests)} != {num_episodes}.'
      )

    runtime_versions_value = value['runtime_versions']
    if not isinstance(runtime_versions_value, Mapping):
      raise ValueError('runtime_versions must be a mapping.')
    runtime_versions = tuple(
        sorted(
            (str(name), str(version))
            for name, version in runtime_versions_value.items()
        )
    )

    return cls(
        schema_version=schema_version,
        hash_algorithm=hash_algorithm,
        target=str(value['target']),
        target_kind=target_kind,
        configuration_sha256=configuration_digest,
        content_sha256=content_digest,
        episode_sha256=episode_digests,
        num_episodes=num_episodes,
        runtime_versions=runtime_versions,
    )

  @classmethod
  def from_json(cls, value: str) -> 'EvaluationManifest':
    """Parses a manifest from JSON text."""
    decoded = json.loads(value)
    if not isinstance(decoded, Mapping):
      raise ValueError('Evaluation manifest JSON must contain an object.')
    return cls.from_dict(decoded)

  @classmethod
  def read(cls, path: str | pathlib.Path) -> 'EvaluationManifest':
    """Reads a manifest from a UTF-8 JSON file."""
    return cls.from_json(pathlib.Path(path).read_text(encoding='utf-8'))


def create_manifest(results: pd.DataFrame, target: str) -> EvaluationManifest:
  """Creates a reproducibility manifest for evaluation results.

  Args:
    results: DataFrame returned by Melting Pot evaluation utilities.
    target: scenario or substrate name evaluated to produce ``results``.

  Returns:
    A versioned manifest containing per-episode digests, a Merkle content root,
    a target configuration digest, and runtime version metadata.
  """
  target_kind = _target_kind(target)
  episode_digests = episode_hashes(results)
  return EvaluationManifest(
      schema_version=SCHEMA_VERSION,
      hash_algorithm=_HASH_ALGORITHM,
      target=target,
      target_kind=target_kind,
      configuration_sha256=configuration_hash(target),
      content_sha256=_merkle_root(episode_digests),
      episode_sha256=episode_digests,
      num_episodes=len(results),
      runtime_versions=_runtime_versions(),
  )


def verify_manifest(
    manifest: EvaluationManifest,
    results: pd.DataFrame,
    *,
    check_configuration: bool = True,
    check_runtime: bool = False,
) -> None:
  """Verifies results and, optionally, the current runtime against a manifest.

  Args:
    manifest: previously created evaluation manifest.
    results: result table to verify.
    check_configuration: also verify the current target configuration digest.
    check_runtime: also require recorded runtime package versions to match the
      current runtime exactly.

  Raises:
    EvaluationManifestMismatch: if any requested check does not match.
    ValueError: if the manifest schema itself is unsupported.
  """
  if manifest.schema_version != SCHEMA_VERSION:
    raise ValueError(
        f'Unsupported evaluation manifest schema: {manifest.schema_version}.'
    )
  if manifest.hash_algorithm != _HASH_ALGORITHM:
    raise ValueError(
        f'Unsupported hash algorithm: {manifest.hash_algorithm!r}.'
    )

  actual_episode_hashes = episode_hashes(results)
  if len(actual_episode_hashes) != manifest.num_episodes:
    raise EvaluationManifestMismatch(
        'Episode count mismatch: '
        f'expected {manifest.num_episodes}, got {len(actual_episode_hashes)}.'
    )

  for index, (expected, actual) in enumerate(
      zip(manifest.episode_sha256, actual_episode_hashes)
  ):
    if actual != expected:
      raise EvaluationManifestMismatch(
          f'Episode {index} content hash mismatch: '
          f'expected {expected}, got {actual}.'
      )

  actual_content_hash = _merkle_root(actual_episode_hashes)
  if actual_content_hash != manifest.content_sha256:
    raise EvaluationManifestMismatch(
        'Evaluation content hash mismatch: '
        f'expected {manifest.content_sha256}, got {actual_content_hash}.'
    )

  actual_kind = _target_kind(manifest.target)
  if actual_kind != manifest.target_kind:
    raise EvaluationManifestMismatch(
        f'Target kind mismatch for {manifest.target!r}: '
        f'expected {manifest.target_kind!r}, got {actual_kind!r}.'
    )

  if check_configuration:
    actual_configuration_hash = configuration_hash(manifest.target)
    if actual_configuration_hash != manifest.configuration_sha256:
      raise EvaluationManifestMismatch(
          f'Configuration hash mismatch for {manifest.target!r}: '
          f'expected {manifest.configuration_sha256}, '
          f'got {actual_configuration_hash}.'
      )

  if check_runtime:
    actual_runtime = _runtime_versions()
    if actual_runtime != manifest.runtime_versions:
      raise EvaluationManifestMismatch(
          'Runtime versions do not match the evaluation manifest.'
      )


def episode_hashes(results: pd.DataFrame) -> tuple[str, ...]:
  """Returns one SHA-256 leaf digest per evaluation episode."""
  _validate_result_columns(results)
  digests = []
  for position in range(len(results)):
    row = results.iloc[position]
    payload = _episode_payload(row)
    encoded = _canonical_json(payload)
    digests.append(hashlib.sha256(b'\x00' + encoded).hexdigest())
  return tuple(digests)


def content_hash(results: pd.DataFrame) -> str:
  """Returns the Merkle root over all evaluation episodes."""
  return _merkle_root(episode_hashes(results))


def configuration_hash(target: str) -> str:
  """Returns a SHA-256 digest of evaluation-relevant target configuration."""
  payload = _configuration_payload(target)
  return hashlib.sha256(b'\x10' + _canonical_json(payload)).hexdigest()


def _episode_payload(row: pd.Series) -> dict[str, Any]:
  focal_names = _name_sequence(row['focal_player_names'], 'focal_player_names')
  focal_returns = _return_sequence(
      row['focal_player_returns'], 'focal_player_returns'
  )
  background_names = _name_sequence(
      row['background_player_names'], 'background_player_names'
  )
  background_returns = _return_sequence(
      row['background_player_returns'], 'background_player_returns'
  )

  if len(focal_names) != len(focal_returns):
    raise ValueError(
        'focal_player_names and focal_player_returns must have equal length.'
    )
  if len(background_names) != len(background_returns):
    raise ValueError(
        'background_player_names and background_player_returns must have '
        'equal length.'
    )

  return {
      'focal_players': [
          {'name': name, 'return': return_value}
          for name, return_value in zip(focal_names, focal_returns)
      ],
      'background_players': [
          {'name': name, 'return': return_value}
          for name, return_value in zip(background_names, background_returns)
      ],
  }


def _name_sequence(value: Any, column: str) -> tuple[str, ...]:
  values = _as_sequence(value, column)
  names = []
  for item in values:
    if not isinstance(item, str):
      raise ValueError(f'{column} must contain only strings.')
    names.append(item)
  return tuple(names)


def _return_sequence(value: Any, column: str) -> tuple[str, ...]:
  values = _as_sequence(value, column)
  return tuple(_float_token(item, column) for item in values)


def _as_sequence(value: Any, column: str) -> tuple[Any, ...]:
  if isinstance(value, np.ndarray):
    if value.ndim != 1:
      raise ValueError(f'{column} must be one-dimensional.')
    return tuple(value.tolist())
  if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
    raise ValueError(f'{column} must contain a sequence.')
  return tuple(value)


def _float_token(value: Any, column: str) -> str:
  try:
    number = float(value)
  except (TypeError, ValueError) as error:
    raise ValueError(f'{column} must contain numeric returns.') from error
  if math.isnan(number):
    return 'nan'
  if math.isinf(number):
    return '+inf' if number > 0 else '-inf'
  return number.hex()


def _merkle_root(episode_digests: Sequence[str]) -> str:
  if not episode_digests:
    return hashlib.sha256(b'\x02').hexdigest()

  level = [bytes.fromhex(digest) for digest in episode_digests]
  while len(level) > 1:
    if len(level) % 2:
      level.append(level[-1])
    level = [
        hashlib.sha256(b'\x01' + level[index] + level[index + 1]).digest()
        for index in range(0, len(level), 2)
    ]
  return level[0].hex()


def _configuration_payload(target: str) -> dict[str, Any]:
  if target in scenario_configs.SCENARIO_CONFIGS:
    scenario = scenario_configs.SCENARIO_CONFIGS[target]
    background_bots = (
        set().union(*scenario.bots_by_role.values())
        if scenario.bots_by_role
        else set()
    )
    return {
        'kind': 'scenario',
        'name': target,
        'scenario': {
            'substrate': scenario.substrate,
            'roles': list(scenario.roles),
            'is_focal': list(scenario.is_focal),
            'bots_by_role': {
                role: sorted(scenario.bots_by_role[role])
                for role in sorted(scenario.bots_by_role)
            },
        },
        'background_bots': {
            name: _bot_signature(name) for name in sorted(background_bots)
        },
        'substrate': _substrate_signature(
            scenario.substrate, include_default_roles=False
        ),
    }

  if target in substrate_configs.SUBSTRATES:
    return {
        'kind': 'substrate',
        'name': target,
        'substrate': _substrate_signature(target, include_default_roles=True),
    }

  raise ValueError(f'Unknown substrate or scenario: {target!r}.')


def _bot_signature(name: str) -> dict[str, Any]:
  if name == bot_lib.NOOP_BOT_NAME:
    return {
        'name': name,
        'kind': 'fixed_action',
        'action': bot_lib.NOOP_ACTION,
    }
  config = bot_configs.BOT_CONFIGS[name]
  model_name = pathlib.Path(config.model_path).name
  return {
      'name': name,
      'kind': 'configured_bot',
      'substrate': config.substrate,
      'roles': sorted(config.roles),
      'model': model_name,
      'puppeteer_builder': _normalize_value(config.puppeteer_builder),
  }


def _substrate_signature(
    name: str, *, include_default_roles: bool
) -> dict[str, Any]:
  config = substrate_configs.get_config(name)
  timestep_spec = config.timestep_spec
  signature = {
      'name': name,
      'valid_roles': sorted(config.valid_roles),
      'individual_observation_names': list(config.individual_observation_names),
      'global_observation_names': list(config.global_observation_names),
      'action_set': _normalize_value(config.action_set),
      'action_spec': _spec_payload(config.action_spec),
      'timestep_spec': {
          'step_type': _spec_payload(timestep_spec.step_type),
          'reward': _spec_payload(timestep_spec.reward),
          'discount': _spec_payload(timestep_spec.discount),
          'observation': {
              key: _spec_payload(timestep_spec.observation[key])
              for key in sorted(timestep_spec.observation)
          },
      },
  }
  if include_default_roles:
    signature['default_player_roles'] = list(config.default_player_roles)
  return signature


def _spec_payload(spec: Any) -> dict[str, Any]:
  payload = {
      'type': type(spec).__name__,
      'shape': list(spec.shape),
      'dtype': np.dtype(spec.dtype).name,
      'name': spec.name,
  }
  if hasattr(spec, 'num_values'):
    payload['num_values'] = int(spec.num_values)
  if hasattr(spec, 'minimum'):
    payload['minimum'] = _normalize_value(spec.minimum)
  if hasattr(spec, 'maximum'):
    payload['maximum'] = _normalize_value(spec.maximum)
  return payload


def _normalize_value(value: Any) -> Any:
  if value is None or isinstance(value, (bool, int, str)):
    return value
  if isinstance(value, (float, np.floating)):
    return {'__float__': _float_token(value, 'configuration')}
  if isinstance(value, np.integer):
    return int(value)
  if isinstance(value, np.ndarray):
    return {
        '__ndarray__': {
            'dtype': np.dtype(value.dtype).name,
            'shape': list(value.shape),
            'values': _normalize_value(value.tolist()),
        }
    }
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    return {
        '__dataclass__': (
            f'{type(value).__module__}.{type(value).__qualname__}'
        ),
        'fields': {
            field.name: _normalize_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        },
    }
  if isinstance(value, functools.partial):
    return {
        '__partial__': _normalize_value(value.func),
        'args': _normalize_value(value.args),
        'keywords': _normalize_value(value.keywords or {}),
    }
  if callable(value):
    qualname = getattr(value, '__qualname__', value.__name__)
    return {'__callable__': f'{value.__module__}.{qualname}'}
  if isinstance(value, Mapping):
    return {
        str(key): _normalize_value(value[key])
        for key in sorted(value, key=str)
    }
  if isinstance(value, (set, frozenset)):
    normalized = [_normalize_value(item) for item in value]
    return sorted(normalized, key=lambda item: _canonical_json(item))
  if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
    return [_normalize_value(item) for item in value]
  raise TypeError(
      f'Unsupported configuration value {type(value).__name__}: {value!r}.'
  )


def _target_kind(target: str) -> str:
  if target in scenario_configs.SCENARIO_CONFIGS:
    return 'scenario'
  if target in substrate_configs.SUBSTRATES:
    return 'substrate'
  raise ValueError(f'Unknown substrate or scenario: {target!r}.')


def _runtime_versions() -> tuple[tuple[str, str], ...]:
  versions = {'python': platform.python_version()}
  for distribution in _RUNTIME_DISTRIBUTIONS:
    try:
      version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
      version = '<unknown>'
    versions[distribution] = version
  return tuple(sorted(versions.items()))


def _validate_result_columns(results: pd.DataFrame) -> None:
  missing = _REQUIRED_RESULT_COLUMNS - set(results.columns)
  if missing and len(results):
    raise ValueError(
        f'Evaluation results are missing columns: {sorted(missing)!r}.'
    )


def _validate_digest(value: str, field: str) -> None:
  if len(value) != hashlib.sha256().digest_size * 2:
    raise ValueError(f'{field} is not a SHA-256 digest.')
  try:
    bytes.fromhex(value)
  except ValueError as error:
    raise ValueError(f'{field} is not a SHA-256 digest.') from error


def _canonical_json(value: Any) -> bytes:
  return json.dumps(
      value,
      sort_keys=True,
      separators=(',', ':'),
      ensure_ascii=False,
      allow_nan=False,
  ).encode('utf-8')


def evaluate_population_with_manifest(
    population: Mapping[str, Any],
    names_by_role: Mapping[str, Collection[str]],
    target: str,
    num_episodes: int = 100,
    video_root: str | None = None,
) -> tuple[pd.DataFrame, EvaluationManifest]:
  """Evaluates a population and returns results with their manifest.

  This is a convenience wrapper around ``evaluation.evaluate_population`` that
  leaves the existing evaluation API unchanged while ensuring the returned table
  is immediately fingerprinted.

  Args:
    population: population mapping accepted by ``evaluate_population``.
    names_by_role: policy names that support each role.
    target: scenario or substrate to evaluate.
    num_episodes: number of episodes to run.
    video_root: optional directory for episode videos.

  Returns:
    ``(results, manifest)`` for the completed evaluation.
  """
  # Imported lazily to avoid a module cycle: evaluation imports sibling helpers.
  # pylint: disable=g-import-not-at-top
  from meltingpot.utils.evaluation import evaluation as evaluation_lib
  # pylint: enable=g-import-not-at-top

  results = evaluation_lib.evaluate_population(
      population=population,
      names_by_role=names_by_role,
      scenario=target,
      num_episodes=num_episodes,
      video_root=video_root,
  )
  return results, create_manifest(results, target)


def evaluate_saved_models_with_manifest(
    saved_models: Mapping[str, str],
    names_by_role: Mapping[str, Collection[str]],
    target: str,
    num_episodes: int = 100,
    video_root: str | None = None,
) -> tuple[pd.DataFrame, EvaluationManifest]:
  """Evaluates saved models and returns results with their manifest.

  Args:
    saved_models: saved model names and paths accepted by
      ``evaluation.evaluate_saved_models``.
    names_by_role: policy names that support each role.
    target: scenario or substrate to evaluate.
    num_episodes: number of episodes to run.
    video_root: optional directory for episode videos.

  Returns:
    ``(results, manifest)`` for the completed evaluation.
  """
  # pylint: disable=g-import-not-at-top
  from meltingpot.utils.evaluation import evaluation as evaluation_lib
  # pylint: enable=g-import-not-at-top

  results = evaluation_lib.evaluate_saved_models(
      saved_models=saved_models,
      names_by_role=names_by_role,
      scenario=target,
      num_episodes=num_episodes,
      video_root=video_root,
  )
  return results, create_manifest(results, target)
