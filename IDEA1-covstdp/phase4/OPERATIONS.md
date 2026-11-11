# Fixed snapshot runtime interfaces

These changes concern provenance, directories and recording only. The learning
updates, feature/encoder code, sampling, hyperparameters and size limits remain
the algorithm session's responsibility.

The runner accepts the following CLI flags (CLI overrides the JSON config):

- `--data-dir`: raw Nordland root, read only.
- `--cache-dir`: reusable input cache, outside raw data and published code.
- `--run-root`: dedicated result parent; each `--run-id` gets a child directory.
- `--deployment-manifest`: fixed snapshot root's `deployment_manifest.json`.
- `--deployment-manifest-sha`: trusted expected hash supplied by operations.
- `--environment-manifest`: package/lock record of the approved existing Pixi environment.

In a local Git checkout, provenance comes from Git. A published snapshot requires
its independently checked manifest hash, publication receipt, explicit paths and
environment manifest. Wrong hashes, changed source/package records, unexpected
files and a different Python prefix fail before reading research images.
This fingerprints installed package records; it does not hash every library byte.

Artifacts are stored in `<run-root>/<run-id>/artifacts/`. Operations may reserve
`<run-root>/<run-id>/` and place Slurm logs/launch records there before submission.
An existing `artifacts/` directory is never overwritten. Legacy results retain
their original paths. Run IDs cannot contain separators or traversal components.

Effective config, run identity, summaries and checkpoint metadata include the
verified code identity, environment identity, runtime versions and Slurm Job ID
(null for local checks). `run_identity.json` also binds the final config and data
identity files by hash. The historical `git` metadata key remains for compatibility;
the explicit `code.source` distinguishes a verified deployment from a Git checkout.

The final Slurm state/exit code is recorded by the operations collector after
the job ends, including failures that prevent the Python runner from starting.
These interfaces alone do not approve or start a GPU/3300-place experiment.
