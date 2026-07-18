# CCV-VLA Gate 0A Analysis Protocol

This protocol is frozen before the full formal collection is complete. It clarifies the source
partition implied by the v3 collection manifest without changing collection parameters or the
collection preregistration digest.

- Gate 0A consumes only rows marked `source_partition=fit`.
- Source 36 remains engineering-excluded.
- Rows marked `holdout` are not opened by Gate 0A.
- Candidate pairs and repeats are nested diagnostics; state metrics are averaged inside source.
- Bootstrap resamples source IDs only.
- One-repeat regret is primary; two-repeat regret is diagnostic.
- The six holdout sources are opened once only after every Gate 0B model and abstention threshold
  has been fitted and serialized.
