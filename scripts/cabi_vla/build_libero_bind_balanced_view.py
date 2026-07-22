from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MACRO_PHASES = ("approach", "grasp", "lift", "transport", "place")


def edge_factors(edge_id: str) -> tuple[str, str]:
    source, target = edge_id.rsplit("-", 1)
    if not source or not target:
        raise ValueError(f"invalid edge id: {edge_id!r}")
    return source, target


def macro_phase(phase: str) -> str:
    if phase in {"episode_start", "approach_above", "approach_grasp"}:
        return "approach"
    if phase == "close_gripper":
        return "grasp"
    if phase == "lift":
        return "lift"
    if phase == "transport":
        return "transport"
    if phase in {"lower", "release", "retract", "settle"}:
        return "place"
    raise ValueError(f"unknown teacher phase: {phase!r}")


def anchor_exposure(
    manifest: Mapping[str, Any],
    *,
    record_count: int,
    anchor_period: int,
) -> dict[str, Counter[str]]:
    tetrads = list(manifest["tetrads"])
    if not tetrads:
        raise ValueError("balanced view requires CABI tetrads")
    sources: Counter[str] = Counter()
    targets: Counter[str] = Counter()
    edges: Counter[str] = Counter()
    event_count = 0
    for index in range(0, record_count, anchor_period):
        tetrad = tetrads[(index // anchor_period) % len(tetrads)]
        event_count += 1
        for corner in tetrad["corners"].values():
            if not bool(corner["action_supervised"]):
                continue
            edge = str(corner["instruction_edge"])
            source, target = edge_factors(edge)
            sources[source] += 1
            targets[target] += 1
            edges[edge] += 1
    return {
        "sources": sources,
        "targets": targets,
        "edges": edges,
        "events": Counter({"count": event_count}),
    }


def solve_regular_edge_loss_units(
    regular_edges: Sequence[str],
    anchor: Mapping[str, Counter[str]],
    *,
    record_count: int,
) -> dict[str, int] | None:
    """Allocate quarter-loss units after within-microbatch action averaging.

    A regular-only item contributes four quarter units to its edge. At an
    anchor index, the regular example and three supervised tetrad corners each
    contribute one quarter unit because the action loss averages four samples.
    """

    source_to_edges: dict[str, list[str]] = defaultdict(list)
    target_to_edges: dict[str, list[str]] = defaultdict(list)
    for edge in regular_edges:
        source, target = edge_factors(edge)
        source_to_edges[source].append(edge)
        target_to_edges[target].append(edge)
    if sorted(map(len, source_to_edges.values())) != [1, 1, 2] or len(target_to_edges) != 2:
        raise ValueError("expected a 3x2 cross with one source observed at both targets")

    total_loss_units = 4 * record_count
    if total_loss_units % 6:
        return None
    desired_source = total_loss_units // 3
    desired_target = total_loss_units // 2

    loss_units: dict[str, int] = {}
    pivot_source = next(source for source, edges in source_to_edges.items() if len(edges) == 2)
    for source, edges in source_to_edges.items():
        if source == pivot_source:
            continue
        loss_units[edges[0]] = desired_source - int(anchor["sources"][source])
    pivot_total = desired_source - int(anchor["sources"][pivot_source])
    pivot_edges = source_to_edges[pivot_source]
    first_target = sorted(target_to_edges)[0]
    first_pivot = next(edge for edge in pivot_edges if edge_factors(edge)[1] == first_target)
    other_pivot = next(edge for edge in pivot_edges if edge != first_pivot)
    single_at_first = sum(
        units
        for edge, units in loss_units.items()
        if edge_factors(edge)[1] == first_target
    )
    loss_units[first_pivot] = (
        desired_target - int(anchor["targets"][first_target]) - single_at_first
    )
    loss_units[other_pivot] = pivot_total - loss_units[first_pivot]
    expected_regular_units = total_loss_units - sum(anchor["sources"].values())
    if (
        any(value < 0 for value in loss_units.values())
        or sum(loss_units.values()) != expected_regular_units
    ):
        return None

    effective_sources = Counter(anchor["sources"])
    effective_targets = Counter(anchor["targets"])
    for edge, units in loss_units.items():
        source, target = edge_factors(edge)
        effective_sources[source] += units
        effective_targets[target] += units
    if len(set(effective_sources.values())) != 1 or len(set(effective_targets.values())) != 1:
        return None
    return dict(sorted(loss_units.items()))


def choose_anchor_record_quotas(
    edge_loss_units: Mapping[str, int],
    *,
    anchor_event_count: int,
    record_count: int,
) -> tuple[dict[str, int], dict[str, int]] | None:
    """Choose which regular records receive 1/4 rather than unit loss mass."""

    total_regular_units = sum(edge_loss_units.values())
    states: dict[int, tuple[float, dict[str, int]]] = {0: (0.0, {})}
    for edge in sorted(edge_loss_units):
        units = int(edge_loss_units[edge])
        ideal = anchor_event_count * units / total_regular_units
        candidates = []
        for anchor_count in range(anchor_event_count + 1):
            numerator = units + 3 * anchor_count
            if numerator % 4:
                continue
            edge_count = numerator // 4
            if anchor_count <= edge_count:
                candidates.append((anchor_count, edge_count, (anchor_count - ideal) ** 2))
        next_states: dict[int, tuple[float, dict[str, int]]] = {}
        for used, (cost, assignment) in states.items():
            for anchor_count, _, candidate_cost in candidates:
                total = used + anchor_count
                if total > anchor_event_count:
                    continue
                proposal = (cost + candidate_cost, {**assignment, edge: anchor_count})
                if total not in next_states or proposal[0] < next_states[total][0]:
                    next_states[total] = proposal
        states = next_states
    if anchor_event_count not in states:
        return None
    anchor_quotas = states[anchor_event_count][1]
    edge_quotas = {
        edge: (int(edge_loss_units[edge]) + 3 * anchor_quotas[edge]) // 4
        for edge in edge_loss_units
    }
    if sum(edge_quotas.values()) != record_count:
        return None
    return dict(sorted(edge_quotas.items())), dict(sorted(anchor_quotas.items()))


def plan_balanced_edge_quotas(
    manifest: Mapping[str, Any],
    regular_edges: Sequence[str],
    *,
    minimum_record_count: int,
    anchor_period: int,
) -> tuple[
    int,
    dict[str, int],
    dict[str, int],
    dict[str, int],
    dict[str, Counter[str]],
]:
    for record_count in range(minimum_record_count, minimum_record_count + 6 * anchor_period + 1):
        if record_count % anchor_period:
            continue
        anchor = anchor_exposure(
            manifest,
            record_count=record_count,
            anchor_period=anchor_period,
        )
        loss_units = solve_regular_edge_loss_units(
            regular_edges,
            anchor,
            record_count=record_count,
        )
        if loss_units is None:
            continue
        allocation = choose_anchor_record_quotas(
            loss_units,
            anchor_event_count=int(anchor["events"]["count"]),
            record_count=record_count,
        )
        if allocation is not None:
            edge_quotas, anchor_regular_quotas = allocation
            return record_count, edge_quotas, anchor_regular_quotas, loss_units, anchor
    raise RuntimeError("could not find a nearby exactly balanced record count")


def largest_remainder(total: int, probabilities: Mapping[str, float]) -> dict[str, int]:
    raw = {key: total * value for key, value in probabilities.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    remainder = total - sum(result.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - result[key]), key))
    for key in order[:remainder]:
        result[key] += 1
    return result


def common_stage_distribution(
    records_by_edge_stage: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, float]:
    distributions = []
    for by_stage in records_by_edge_stage.values():
        total = sum(len(rows) for rows in by_stage.values())
        distributions.append(
            {phase: len(by_stage.get(phase, ())) / total for phase in MACRO_PHASES}
        )
    averaged = {
        phase: float(np.mean([row[phase] for row in distributions]))
        for phase in MACRO_PHASES
    }
    normalizer = sum(averaged.values())
    return {phase: value / normalizer for phase, value in averaged.items()}


def repeat_deterministically(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    *,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if count and not rows:
        raise ValueError("cannot allocate a nonzero quota to an empty stage")
    output = []
    while len(output) < count:
        order = rng.permutation(len(rows))
        for index in order:
            if len(output) == count:
                break
            output.append(dict(rows[int(index)]))
    return output


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_balanced_view(
    source: Path,
    output: Path,
    *,
    anchor_period: int,
    seed: int,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    manifest = json.loads((source / "manifest.json").read_text())
    records = [json.loads(line) for line in (source / "records.jsonl").read_text().splitlines() if line]
    regular_edges = sorted({str(row["edge_id"]) for row in records})
    withheld = set(manifest["leakage_guard"]["withheld_action_edges"])
    if withheld & set(regular_edges):
        raise ValueError("source view contains held-out action records")

    collection = Path(manifest["source_collection"])
    phase_cache: dict[str, np.ndarray] = {}
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        edge: {phase: [] for phase in MACRO_PHASES} for edge in regular_edges
    }
    for row in records:
        episode_file = str(row["episode_file"])
        if episode_file not in phase_cache:
            with np.load(collection / episode_file, allow_pickle=False) as episode:
                phase_cache[episode_file] = np.asarray(episode["phase"])
        frame = int(row["frame_index"])
        phase = str(phase_cache[episode_file][frame])
        stage = macro_phase(phase)
        value = dict(row)
        value["teacher_phase"] = phase
        value["macro_phase"] = stage
        grouped[str(row["edge_id"])][stage].append(value)

    (
        record_count,
        edge_quotas,
        anchor_regular_quotas,
        edge_loss_units,
        anchor,
    ) = plan_balanced_edge_quotas(
        manifest,
        regular_edges,
        minimum_record_count=len(records),
        anchor_period=anchor_period,
    )
    stage_distribution = common_stage_distribution(grouped)
    rng = np.random.default_rng(seed)
    anchor_selected = []
    nonanchor_selected = []
    stage_quotas: dict[str, dict[str, int]] = {}
    anchor_stage_quotas: dict[str, dict[str, int]] = {}
    for edge in regular_edges:
        stage_quotas[edge] = largest_remainder(edge_quotas[edge], stage_distribution)
        anchor_stage_quotas[edge] = largest_remainder(
            anchor_regular_quotas[edge], stage_distribution
        )
        for stage in MACRO_PHASES:
            if anchor_stage_quotas[edge][stage] > stage_quotas[edge][stage]:
                raise AssertionError("anchor-stage quota exceeds total stage quota")
            chosen = repeat_deterministically(
                grouped[edge][stage],
                stage_quotas[edge][stage],
                rng=rng,
            )
            order = rng.permutation(len(chosen))
            chosen = [chosen[int(index)] for index in order]
            split = anchor_stage_quotas[edge][stage]
            anchor_selected.extend(chosen[:split])
            nonanchor_selected.extend(chosen[split:])
    anchor_event_count = int(anchor["events"]["count"])
    if len(anchor_selected) != anchor_event_count:
        raise AssertionError("balanced sampler produced the wrong number of anchor records")
    if len(anchor_selected) + len(nonanchor_selected) != record_count:
        raise AssertionError("balanced sampler produced the wrong number of records")
    anchor_selected = [
        anchor_selected[int(index)] for index in rng.permutation(len(anchor_selected))
    ]
    nonanchor_selected = [
        nonanchor_selected[int(index)] for index in rng.permutation(len(nonanchor_selected))
    ]
    anchor_iterator = iter(anchor_selected)
    nonanchor_iterator = iter(nonanchor_selected)
    selected = [
        next(anchor_iterator) if index % anchor_period == 0 else next(nonanchor_iterator)
        for index in range(record_count)
    ]
    for index, row in enumerate(selected):
        row["source_sample_id"] = row["sample_id"]
        row["sample_id"] = f"{row['sample_id']}--balanced-{index:06d}"
        row["balanced_instance_index"] = index

    effective_source_loss_units = Counter(anchor["sources"])
    effective_target_loss_units = Counter(anchor["targets"])
    regular_sources: Counter[str] = Counter()
    regular_targets: Counter[str] = Counter()
    for edge, count in edge_quotas.items():
        source_id, target_id = edge_factors(edge)
        regular_sources[source_id] += count
        regular_targets[target_id] += count
    regular_stage_loss_units: dict[str, Counter[str]] = defaultdict(Counter)
    actual_anchor_regular = Counter()
    for index, row in enumerate(selected):
        edge = str(row["edge_id"])
        source_id, target_id = edge_factors(edge)
        units = 1 if index % anchor_period == 0 else 4
        effective_source_loss_units[source_id] += units
        effective_target_loss_units[target_id] += units
        regular_stage_loss_units[edge][str(row["macro_phase"])] += units
        if index % anchor_period == 0:
            actual_anchor_regular[edge] += 1
    if dict(sorted(actual_anchor_regular.items())) != anchor_regular_quotas:
        raise AssertionError("anchor-index edge allocation drifted from its plan")

    report = {
        "schema_version": 1,
        "source_view": str(source),
        "seed": seed,
        "anchor_period": anchor_period,
        "record_count": record_count,
        "edge_quotas": edge_quotas,
        "anchor_regular_edge_quotas": anchor_regular_quotas,
        "regular_edge_loss_units": edge_loss_units,
        "loss_unit_denominator": 4,
        "action_loss_reduction": "mean_over_supervised_examples_per_microbatch",
        "stage_distribution": stage_distribution,
        "stage_quotas": stage_quotas,
        "anchor_stage_quotas": anchor_stage_quotas,
        "regular_stage_loss_units": {
            edge: dict(sorted(values.items()))
            for edge, values in sorted(regular_stage_loss_units.items())
        },
        "regular_source_exposure": dict(sorted(regular_sources.items())),
        "regular_target_exposure": dict(sorted(regular_targets.items())),
        "anchor_source_exposure": dict(sorted(anchor["sources"].items())),
        "anchor_target_exposure": dict(sorted(anchor["targets"].items())),
        "effective_supervised_source_loss_units": dict(
            sorted(effective_source_loss_units.items())
        ),
        "effective_supervised_target_loss_units": dict(
            sorted(effective_target_loss_units.items())
        ),
        "heldout_action_records_loaded": False,
    }
    if (
        len(set(effective_source_loss_units.values())) != 1
        or len(set(effective_target_loss_units.values())) != 1
    ):
        raise AssertionError("effective action-loss mass is not factor balanced")

    staging = output.parent / f".{output.name}.staging-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        with (staging / "records.jsonl").open("w") as stream:
            for row in selected:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        os.symlink((source / manifest["anchors_file"]).resolve(), staging / manifest["anchors_file"])
        output_manifest = dict(manifest)
        output_manifest["record_count"] = record_count
        output_manifest["source_training_view"] = str(source)
        output_manifest["sampling_balance"] = report
        atomic_json(staging / "manifest.json", output_manifest)
        atomic_json(staging / "balance_report.json", report)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build factor- and phase-balanced LIBERO-Bind view")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-period", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    report = build_balanced_view(
        args.source,
        args.output,
        anchor_period=args.anchor_period,
        seed=args.seed,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
