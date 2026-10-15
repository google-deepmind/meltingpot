# Reproducible evaluation manifests

Melting Pot evaluations return episode-level player names and returns as a
`pandas.DataFrame`. Aggregate scores are convenient for comparison, but they do
not identify the exact episode results or scenario configuration that produced
them.

`meltingpot.utils.evaluation.evaluation_manifest` can create a versioned,
portable manifest for an evaluation. A manifest lets a later reader answer two
separate questions:

1. Are these exactly the episode-level results that were originally recorded?
2. Does the current Melting Pot target still have the same evaluation-relevant
   configuration?

The two checks use independent SHA-256 digests.

## Create a manifest while evaluating

The convenience wrappers return the normal result DataFrame together with a
manifest. Existing evaluation functions are unchanged.

```python
from meltingpot.utils.evaluation import evaluation_manifest

results, manifest = evaluation_manifest.evaluate_saved_models_with_manifest(
    saved_models={
        "candidate": "/path/to/saved_model",
    },
    names_by_role={
        "default": {"candidate"},
    },
    target="clean_up_20",
    num_episodes=100,
)

manifest.write("clean_up_20.manifest.json")
results.to_pickle("clean_up_20.results.pkl")
```

The same API is available for an already-built population through
`evaluate_population_with_manifest`.

A manifest can also be created after an evaluation:

```python
manifest = evaluation_manifest.create_manifest(results, "clean_up_20")
```

## Verify later

```python
import pandas as pd
from meltingpot.utils.evaluation import evaluation_manifest

results = pd.read_pickle("clean_up_20.results.pkl")
manifest = evaluation_manifest.EvaluationManifest.read(
    "clean_up_20.manifest.json"
)

evaluation_manifest.verify_manifest(manifest, results)
```

By default verification checks both the episode content and the current target
configuration. If the result table has been changed, the error identifies the
first episode whose leaf hash differs.

Runtime package versions are recorded in the manifest for diagnostics but are
not required to match by default. To require an exact runtime-version match:

```python
evaluation_manifest.verify_manifest(
    manifest,
    results,
    check_runtime=True,
)
```

To check a result artifact even when the currently installed scenario
configuration has intentionally changed:

```python
evaluation_manifest.verify_manifest(
    manifest,
    results,
    check_configuration=False,
)
```

## What the content hash covers

Each episode is canonicalized as an ordered list of focal and background
players. Every player entry binds:

* the policy/bot name, and
* the exact episode return.

Finite floating-point values are encoded using Python's exact hexadecimal
floating-point representation. NaN, positive infinity, and negative infinity
have explicit canonical tokens. This avoids relying on JSON's implementation-
dependent floating-point formatting.

The following fields are deliberately not part of the content hash:

* `video_path`, because it is machine-specific,
* the DataFrame index, because it is storage metadata, and
* per-capita return columns, because they are derived from the player returns
  already committed by the hash.

Episode order is part of the content. Reordering rows changes the root.

## Merkle construction

Schema version 1 uses domain-separated SHA-256 hashes.

For an episode's canonical JSON bytes `record`:

```text
leaf = SHA256(0x00 || record)
```

Parent nodes are:

```text
parent = SHA256(0x01 || left || right)
```

When a tree level has an odd number of nodes, the final node is duplicated
before hashing the next level. An empty evaluation uses:

```text
SHA256(0x02)
```

The manifest stores every leaf digest as `episode_sha256` and the final root as
`content_sha256`. Keeping the leaves makes a mismatch localizable without
having to compare full result files.

## What the configuration hash covers

The separate `configuration_sha256` digest is intended to detect evaluation
definition drift.

For a scenario it commits to:

* the underlying substrate name,
* every player role in slot order,
* every focal/background assignment in slot order,
* the complete background bot pool for each role,
* each referenced background bot's substrate, supported roles, model identity,
  and puppeteer constructor configuration, and
* the underlying substrate's public action, timestep, observation, and role
  signature.

For a direct substrate evaluation it also commits to the default player-role
assignment, because that assignment is used by the evaluation helper.

Descriptive scenario text and tags are excluded because changing documentation
does not change evaluation semantics.

The configuration object is canonicalized before hashing, including mappings,
sets, NumPy arrays/scalars, dataclasses, callables, and `functools.partial`
objects used by puppeteer builders.

## Runtime metadata

The manifest records the Python version and installed versions of:

* `dm-meltingpot`,
* `dmlab2d`,
* `dm-env`,
* `numpy`,
* `pandas`, and
* `tensorflow`.

These fields help diagnose differences between runs. They are not included in
the episode Merkle root and are only enforced by `verify_manifest` when
`check_runtime=True`.

## Scope and limitations

An evaluation manifest is an integrity and reproducibility record, not a
cryptographic signature. Anyone who can replace both a result file and its
manifest can compute a new valid hash. Authenticity requires a separate signing
or trusted publication mechanism.

The configuration digest intentionally focuses on evaluation-relevant public
configuration. It records background model identity, but it does not hash every
byte of saved-model assets or the entire Melting Pot source tree. The episode
content root and recorded runtime versions complement that configuration
fingerprint.

The schema version is stored explicitly so the canonicalization or hash boundary
can evolve without silently changing the meaning of an existing digest.
