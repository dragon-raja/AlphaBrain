# Decision-Sufficient Observation Learning: Paper 1

## 当前研究与汇报入口

请使用 [reports/current/vla_view_research_progress_zh.pdf](reports/current/vla_view_research_progress_zh.pdf)（12 页），以及[图表说明](reports/current/vla_view_research_progress_zh.md)。

主线定义：从任务初态执行完整任务，先对噪声平均，再按初态选择最佳固定视角。历史恢复快照实验仅作辅助证据。报告与完整初态评测计划不能混称为已完成结果。

[报告目录与归档规则](reports/README.md) · [历史报告索引](reports/archive/README.md) · [本次整理收据](reports/cleanup-receipt-20260908.json)

原两个报告文件名保留为兼容链接，不再维护两份独立 PDF。下方是原样保留的历史项目索引，其中“current / running / eight-page / ten-slide”等描述属于当时截点，不表示当前状态。其他分支的实验记录和文件未在此次整理中改动。

<details>
<summary>历史项目索引（保留原文与链接）</summary>

## Current spatial / metric analysis: one consolidated PDF (2026-09-08)

Use [vla_view_landscape_and_metrics_20260908_zh.pdf](vla_view_landscape_and_metrics_20260908_zh.pdf)
for the current spatial-distribution, metric-rule and Oracle discussion. This is
one eight-page, figure-led report, with a [page guide and evidence scope](reports/notes/vla_view_landscape_and_metrics_20260908_zh.md).
It replaces the separate six-page landscape and three-page metric/Oracle briefs
as the presentation entry for this round; those outputs remain provenance archives.
It does not concatenate the older full research presentation or the 64-state atlas.
Once the complete matched 64-state comparison passes, a CPU-only watcher rebuilds
this same PDF path; timestamped versions and provenance are retained separately.

The early concurrency benchmark completed at 06:00 UTC. The 48-worker probe reduced
runtime by 13.49%, but neither higher-concurrency configuration passed the frozen
bitwise-trajectory gate, so the selected configuration remains 32 workers. Initial
physics, common-replan noise and all 192 success labels agreed; historical32 versus
probe32 also showed nonidentical action paths, so this is not evidence that increased
concurrency uniquely causes the divergence. The original dispatcher resumed and
started wave-02; 6,208 formal episodes are audited. See the dated results and remaining
75–90 hour estimate in [Full64 execution](execution/full64_view_landscape_execution_20260908_zh.md).

## Full-state extension and Oracle comparison (2026-09-08)

The user authorized completing the canonical model on all 64 historical states.
[Full64 execution amendment](execution/full64_view_landscape_execution_20260908_zh.md)
records the queued 56-state extension, exact first-eight reuse, the 198,656-episode
closed-loop budget, equivalence-gated concurrency benchmarks, and the revised
75–90 hour unaccelerated estimate. The archived three-page figure brief includes
both the optimistic O32 statewise empirical Oracle and disjoint O16 crossfit search.
The first-eight release remains unchanged and continues running; the extension
waits for its GPU lock and audited completion before starting additional rollouts.

## Metric interpretation record (2026-09-08)

[View metric interpretation and fixed-rule comparisons](results/view_metric_interpretation_revision_20260908_zh.md)
clarifies task/state/view scope, corrects parameter-space versus world-space plot
interpretation, and compares seven fixed Accel/visibility rules on archived Broad64
data. It includes a three-page figure brief and crossfit difficulty diagnostics.
These are historical exploratory results, not new canonical-model outcomes or an
active-acquisition claim. The same rules are now attached to the completed-matrix
CPU postprocessor for the strictly matched eight-state training comparison.

## Current execution: matched view landscape (2026-09-08)

The canonical matched checkpoint completed 2,000 steps and passed its artifact and
recipe audit. The current user-authorized stages 1–3 are recorded in
[matched_view_landscape_execution_20260908_zh.md](execution/matched_view_landscape_execution_20260908_zh.md):
reuse Broad64-state dense evidence for plots, score a frozen 8-state development
subset with the matched canonical model, and run its full 97-view × 32-noise
closed-loop matrix (24,832 new episodes). No candidate pruning, new training,
active-camera experiment or RoboCasa run is released here. This is a diagnostic,
not a new confirmatory method claim. Historical snapshots below remain dated records.

## Earlier overall research presentation: 10-slide visual brief (2026-09-07)

Use [vla_view_research_focus_20260907_zh.pdf](vla_view_research_focus_20260907_zh.pdf)
for the earlier overall research presentation. It is a ten-slide, figure-led narrative:
training coverage → remaining view value → whether selection can realize it →
the matched-training/observation experiment still needed.
It restores or redraws the key training, M1, replication and v2 result figures;
each slide has one main conclusion. Essential model, sample-size, input-permission
and statistical details remain in the figure captions. Full implementation and
historical execution records stay in the reference material.
The [figure explanations](reports/notes/vla_view_research_focus_20260907_zh.md) document interpretation
and scope. The [argument review](audits/focus_brief_argument_review_20260907_zh.md)
distinguishes paper-ready evidence, mechanistic clues, method baselines and untested designs.

## Consolidated evidence archive (not the presentation)

The previous 40-page document
[vla_view_research_unified_20260907_zh.pdf](reports/archive/imported/vla_view_research_unified_20260907_zh.pdf)
is retained as a reference archive. It consolidates both historical PDFs, expectation v1,
dense statewise v2, the matched-training correction, primary-source related work and the
bounded active-acquisition design. Completed results, exploratory maxima and untested
contributions are explicitly separated. The training snapshot is timestamped, not a live monitor.

The [editable text](reports/notes/vla_view_research_unified_20260907_zh.md),
[layout manuscript](reports/sources/vla_view_research_unified_20260907_zh.json), and
[evidence/build receipts](reports/sources/20260907) accompany the PDF.
The original two PDFs remain unchanged; their SHA-256 hashes are checked by the builder.
Document generation does not launch or control any training or evaluation.

## Current research planning entry (2026-09-07)

The current consolidated research design is
[paper_master_plan_v2_active_bridge_20260907_zh.md](planning/paper_master_plan_v2_active_bridge_20260907_zh.md).
It connects the completed training/static-view studies to a proposed, bounded active-observation validation.
This is a **design draft, not an execution release**. The formal statewise v2 campaign completed on
2026-09-06; the historical S0/E0 scope, preregistrations, receipts and results below remain unchanged.
The new plan does not retroactively turn static camera changes into active-perception evidence.

Implementation now starts at
[next_stage_scaffold_status_20260907_zh.md](planning/next_stage_scaffold_status_20260907_zh.md),
with exact reusable checkpoints and evidence in
[next_stage_asset_evidence_inventory_20260907_zh.md](planning/next_stage_asset_evidence_inventory_20260907_zh.md).
New protocol, motion/budget and acquisition-executor scaffolds have tests and a real,
no-VLA LIBERO engineering smoke. This is **not a scientific execution release**.
The audit found that the old canonical quick checkpoint and v2 Broad M-B use different
learning-rate schedules; they must not be treated as a coverage-only matched pair.

A bounded matched canonical training run has now started (2026-09-07 04:58 UTC):
[training_match_execution_status_20260907_zh.md](execution/training_match_execution_status_20260907_zh.md).
It uses the same original initialization and Broad64 data container as the M-B anchor,
2 GPUs, 2,000 updates and a matching 2,000-step schedule. This training-only release
does not release active-camera evaluation or certify a completed model yet.

Repository cleanup is intentionally deferred until the active formal pipeline is frozen. The protected working set,
artifact triage, and post-run release cleanup gates are recorded in
[repository_hygiene_handoff_20260901_zh.md](audits/repository_hygiene_handoff_20260901_zh.md).

The live ownership boundary and restart checkpoint for the formal noise-marginalized view-value pipeline are recorded
in [view_value_expectation_takeover_20260902_zh.md](execution/view_value_expectation_takeover_20260902_zh.md).

The successor dense state-conditioned oracle campaign is frozen in
[statewise_view_oracle_v2_protocol_20260903_zh.md](protocols/statewise_view_oracle_v2_protocol_20260903_zh.md), and its live
execution handoff is [statewise_view_oracle_v2_execution_status_20260903_zh.md](execution/statewise_view_oracle_v2_execution_status_20260903_zh.md).
It replaces the early-pruning A/B/C design for the new question; it does not rewrite or delete the completed v1 audit.

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

The noise-marginalized closed-loop protocol for stable view-value analysis is frozen in
[view_value_expectation_protocol_v1_zh.md](protocols/view_value_expectation_protocol_v1_zh.md), with its machine-readable contract in
[view_value_expectation_protocol_v1.json](../../configs/dsol_paper1/view_value_expectation_protocol_v1.json). The frozen
document retains its historical runner-HOLD status, but formal execution was subsequently authorized by
`receipts/execution-release-formal-seed41.json`; the current execution boundary is tracked in the takeover record above.

The legacy camera experiments require an upstream revalidation before they can
support Paper 1 claims. The Chinese prefreeze amendment is
[view_coverage_pairing_revalidation_v2_zh.md](results/view_coverage_pairing_revalidation_v2_zh.md),
with a machine-readable draft in
[view_revalidation_prefreeze_v2.json](../../configs/dsol_paper1/view_revalidation_prefreeze_v2.json).
It separates camera support, same-state pairing, training objective, exposure
budget, and information-bearing evaluation views. Until that amendment passes,
the old Phase B/M0/M1 results are legacy anchors only.

The executable LIBERO-Plus rapid/formal schedule is documented in
[libero_plus_revalidation_execution_v1_zh.md](execution/libero_plus_revalidation_execution_v1_zh.md),
with its runtime inventory and source-data blocker recorded in
[libero_plus_revalidation_execution_v1.json](../../configs/dsol_paper1/libero_plus_revalidation_execution_v1.json).
The formal state source is acquired with
[`download_libero_hdf5.py`](../../scripts/dsol_paper1/training/download_libero_hdf5.py)
and checked with
[`audit_libero_hdf5_restore.py`](../../scripts/dsol_paper1/runtime/audit_libero_hdf5_restore.py).
The nested Narrow-8/Broad-32/Broad-64 candidate catalog is generated from
[`libero_view_catalog_v2_rules.json`](../../configs/dsol_paper1/libero_view_catalog_v2_rules.json)
by
[`build_libero_view_catalog.py`](../../scripts/dsol_paper1/protocols/build_libero_view_catalog.py).
Its generated catalog remains a candidate until render and visibility audits
pass; diagnostic look-away, extreme, and blackout views are evaluation-only.

The completed natural-LIBERO visibility quick gate is summarized in
[libero_m1_visibility_joint_support_gate_v1_zh.md](results/libero_m1_visibility_joint_support_gate_v1_zh.md).
Once Strong-info and Matched-control camera poses were both placed in training
support, their closed-loop success was identical. This closes the pose-support
confound but does not release a paper-level claim; the next gate requires a
controlled strong Blind-Reveal construction with symmetric camera support.

The completed first-stage M-A/M-B evidence, including Broad64, constructed
M0/M1, Accel, three-seed Camera Full and Original Full, is summarized in
[view_revalidation_stage1_final_20260824_zh.md](results/view_revalidation_stage1_final_20260824_zh.md).
The original presentation remains
[view_revalidation_stage1_integrated_v5_20260826_zh.pdf](reports/archive/imported/view_revalidation_stage1_integrated_v5_20260826_zh.pdf).
It has not been overwritten or reordered. The historical second report is
[view_value_new_experiments_zh.pdf](reports/archive/imported/view_value_new_experiments_zh.pdf), whose actual
archived file contains 10 pages: five on 18 disjoint source demonstrations,
36 states and 1,080 added rollouts, followed by five on the 20-state failure-pool
search and confirmation (6,620 rollouts). Both are now incorporated into the
consolidated reference archive above; only the central result figures are selected
for the current ten-slide presentation. The original files remain archival evidence.

The alternative integrated rewrite
[view_revalidation_report_zh.pdf](reports/archive/imported/view_revalidation_report_zh.pdf) is retained as
an editing record, not the default report. These new selector results use one
training checkpoint and five inference-noise repeats, not five training runs.

The detailed evidence record is
[view_value_reverse_discovery_pilot_20260827_zh.md](results/view_value_reverse_discovery_pilot_20260827_zh.md).
The reproducible offline audit and report builder are
[`audit_dense_selector_common_failures.py`](../../scripts/dsol_paper1/diagnostics/audit_dense_selector_common_failures.py)
and [`build_view_value_new_experiments_pdf.py`](../../reports/paper1/historical/build_view_value_new_experiments_pdf.py).
No new policy training, rollout, scoring-rule modification, or paper-level
instrument release is implied by this report update.

The older [v2](reports/archive/imported/view_revalidation_stage1_integrated_v2_20260825_zh.pdf) remains a dated historical record.
The earlier 20-page integrated draft
[view_revalidation_stage1_integrated_20260825_zh.pdf](reports/archive/imported/view_revalidation_stage1_integrated_20260825_zh.pdf)
is retained only as a dated layout record.
The earlier 13-page
[stage-one PDF](reports/archive/imported/view_revalidation_stage1_final_brief_20260824_zh.pdf) and the
focused [expanded Accel supplement](reports/archive/imported/accel_expanded_diagnostic_20260824_zh.pdf)
remain as dated component records.
The earlier
[interim report](results/view_revalidation_interim_report_20260821_zh.md) and
[interim PDF](reports/archive/imported/view_revalidation_interim_brief_20260821_zh.pdf) remain as a
dated execution record and should not be used as the current status.

The authoritative whole-program completion audit is
[view_revalidation_full_program_status_20260819_zh.md](execution/view_revalidation_full_program_status_20260819_zh.md).
It distinguishes completed seed-41 quick-gate evidence from pending Legacy,
Broad64, multiseed, constructed Blind-Reveal, full M1, and Accel work.

The current quick-gate exact-state pairs are generated by
[`generate_libero_hdf5_view_pairs.py`](../../scripts/dsol_paper1/training/generate_libero_hdf5_view_pairs.py)
and the resumable collection orchestrator
[`generate_libero_pair_collection.py`](../../scripts/dsol_paper1/training/generate_libero_pair_collection.py).
The frozen eight-task plan contains 38,193 records from 400 official episodes;
all episode splits, source-state hashes, action chunks, and shard checksums pass
[`audit_libero_pair_collection.py`](../../scripts/dsol_paper1/training/audit_libero_pair_collection.py).
Seven budget-controlled training arms and the explicit paired-flow objective are
defined in
[`dsol_libero_broad_pairing.yaml`](../../configs/experiments/dsol_libero_broad_pairing.yaml).
The uniform quick-gate budget is frozen in
[`libero_training_budget_quick_gate_v1.json`](../../configs/dsol_paper1/libero_training_budget_quick_gate_v1.json):
2,000 optimization steps with the 3,000-step scheduler horizon retained from
the completed calibration.

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

- No paper-level policy training or checkpoint selection before the independent
  release review. Approved quick-gate smoke and directional calibration remain
  debug evidence and do not release a formal claim.
- No formal rollout or paper-level statistics.
- No dynamic view selector, E1 pan-tilt, S1 video model, S2 memory, or E2 body motion.
- No deployment input may contain relation labels, privileged simulator state, candidate rank, filenames, timestamps, or oracle routing.

</details>
