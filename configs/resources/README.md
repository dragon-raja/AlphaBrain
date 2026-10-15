# Portable resource identities

`paper1_resources.json` records dataset/code/checkpoint/experiment IDs and backend
relationships, not a particular user's filesystem. Machine bindings live outside
the repository (or in an ignored `*.local.json`) with this schema:

```json
{
  "schema_version": 1,
  "resource_paths": {"alphabrain_source": "/absolute/path/to/checkout"},
  "retired_roots": []
}
```

`external_sources.lock.json` pins external code commits and critical source bytes.
`robotwin_official_clean50.lock.json` pins all 50 current clean task archives,
three asset archives and the selected official Pi05 checkpoint by official
revision, archive size and SHA256. It excludes randomized500 and other policy
weights. Passing the plan to the resource importer without `--execute` is dry-run.
Use a new resource root for a fresh acquisition; do not overwrite existing data.

All IDs marked `present` need local paths when running the presence audit. Entries
marked `acquiring` are explicitly incomplete even if their root directory exists.
`not_created` explicitly records planned derived data without inventing a path.
Execution bindings may use `required_revisions` to reject missing or mismatched
code/data/asset/checkpoint revisions. This is a declared bundle check, not proof
of simulator/policy compatibility, nor a guard on arbitrary shell commands.
RoboCasa rendered PNG payloads and recovery batches are separate resource IDs
from the frozen JSONL release selecting them. Never train by globbing every image
under a recovery or failed-render root.

```bash
PYTHONPATH=. python scripts/dsol_paper1/diagnostics/audit_project_resources.py \
  --manifest configs/resources/paper1_resources.json
```

Add `--locations /absolute/path/to/resource_paths.local.json` for machine checks.
Frozen experiment release/plan files remain authoritative for exact hashes,
training recipes, source snapshots, splits and numerical protocols. This graph
does not rewrite them or certify that two policy backends are interchangeable.

The existing `configs/dsol_paper1/resource_catalog.json` is the historical
machine-specific navigation catalog; it is not the portable public manifest.
Before a public release, audit legacy configs and source licenses rather than
blindly publishing the entire working tree.
