# Decision-Sufficient Observation Learning: Paper 1

This branch isolates the Paper 1 preparation from prior FRESH, KYC, and active-view experiments.

## Scientific Slice

The fixed scope is `(N / E-D) x S0 x E0`:

- **Nuisance invariance (NI):** preserve an equivalent task decision across an uninformative view change.
- **Evidence responsiveness (ER):** update the decision when a final candidate snapshot resolves a task-relevant ambiguity.
- **S0 only:** the final snapshot must suffice; ordered-trajectory and memory-dependent cases are excluded.
- **E0 only during initial release:** all candidates are rendered from the same frozen physical state.

The primary object is the `relation x training strategy` interaction. Average success alone cannot release a method claim.

## Release Discipline

Preparation and smoke outputs live under `/workspace/ai2r/debug/dsol_paper1`. Formal outputs remain disabled until an independent P1 receipt closes B1-B5 and B7. B6 is additionally required before a P3 confirmatory matrix.

The prefreeze configuration is [preregistration_prefreeze_v1.json](../../configs/dsol_paper1/preregistration_prefreeze_v1.json). Null thresholds and `NOT_MATERIALIZED` fields are intentional release blockers; builders and validators must fail closed rather than invent values.

Files under `configs/dsol_paper1/templates/` are human-oriented HOLD stubs. They are not release evidence and are not assumed to satisfy the machine schemas. A record becomes eligible for review only after materialization against `schemas/dsol_paper1/`, content hashing, and independent adjudication.

The synthetic protocol smoke is intentionally evidence-free:

```bash
PYTHONPATH=. python scripts/dsol_paper1/run_protocol_smoke.py \
  --output /workspace/ai2r/debug/dsol_paper1/prefreeze_v1/protocol_smoke.json
```

## Preparation Sequence

1. Freeze source, runtime, checkpoint, snapshot-bank, and camera-catalog provenance.
2. Materialize task templates, full observation contracts, and acceptable expert decision sets.
3. Construct N, E-D, and matched-control candidates without querying evaluated policies.
4. Run the five E-D gates, channel-isolated leakage audits, and T1-T3 boundary audit.
5. Freeze task quotas, seeds, exclusions, thresholds, and power analysis.
6. Issue an independent P1 release receipt. Only then may formal instrument records be generated.

## Explicit Holds

- No new policy training or checkpoint selection.
- No formal rollout or paper-level statistics.
- No dynamic view selector, E1 pan-tilt, S1 video model, S2 memory, or E2 body motion.
- No deployment input may contain relation labels, privileged simulator state, candidate rank, filenames, timestamps, or oracle routing.
