from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from build_libero_episode_windows import assign_window_labels, build_quality_report


def _stable_order(values: Sequence[int], *, seed: int, namespace: str) -> list[int]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).digest(),
    )


def _subset_with_group_count(
    source_counts: Mapping[int, int],
    sources: Sequence[int],
    *,
    target: int,
    seed: int,
    namespace: str,
) -> tuple[int, ...]:
    choices: dict[int, tuple[int, ...]] = {0: ()}
    for source in _stable_order(sources, seed=seed, namespace=namespace):
        count = int(source_counts[source])
        for total, selected in sorted(choices.items(), reverse=True):
            candidate = total + count
            proposed = (*selected, source)
            if candidate <= target and (
                candidate not in choices or len(proposed) > len(choices[candidate])
            ):
                choices[candidate] = proposed
    if target not in choices:
        raise ValueError(f"cannot assign exactly {target} groups to split {namespace!r}")
    return choices[target]


def _balanced_disjoint_subsets(
    source_counts: Mapping[int, int],
    sources: Sequence[int],
    *,
    val_target: int,
    test_target: int,
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    choices: dict[tuple[int, int], tuple[tuple[int, ...], tuple[int, ...]]] = {
        (0, 0): ((), ())
    }

    def score(value: tuple[tuple[int, ...], tuple[int, ...]]) -> tuple[int, int]:
        val_sources, test_sources = value
        return min(len(val_sources), len(test_sources)), len(val_sources) + len(test_sources)

    for source in _stable_order(sources, seed=seed, namespace="val-test-joint"):
        count = int(source_counts[source])
        updated = dict(choices)
        for (val_total, test_total), (val_sources, test_sources) in choices.items():
            candidates = []
            if val_total + count <= val_target:
                candidates.append(
                    ((val_total + count, test_total), ((*val_sources, source), test_sources))
                )
            if test_total + count <= test_target:
                candidates.append(
                    ((val_total, test_total + count), (val_sources, (*test_sources, source)))
                )
            for key, proposed in candidates:
                if key not in updated or score(proposed) > score(updated[key]):
                    updated[key] = proposed
        choices = updated
    target = (val_target, test_target)
    if target not in choices:
        raise ValueError(
            f"cannot assign exactly val={val_target}, test={test_target} groups from source states"
        )
    return choices[target]


def source_disjoint_splits(
    groups: Sequence[Mapping[str, Any]],
    *,
    val_groups: int = 13,
    test_groups: int = 13,
    seed: int = 20260714,
) -> dict[str, str]:
    source_counts = Counter(int(group["source_initial_state_index"]) for group in groups)
    sources = sorted(source_counts)
    val_selected, test_selected = _balanced_disjoint_subsets(
        source_counts,
        sources,
        val_target=val_groups,
        test_target=test_groups,
        seed=seed,
    )
    val_sources = set(val_selected)
    test_sources = set(test_selected)
    return {
        str(group["pair_id"]): (
            "test"
            if int(group["source_initial_state_index"]) in test_sources
            else "val"
            if int(group["source_initial_state_index"]) in val_sources
            else "train"
        )
        for group in groups
    }


def split_quality(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups_by_split = Counter(str(group["split"]) for group in groups)
    sources_by_split = {
        split: {
            int(group["source_initial_state_index"])
            for group in groups
            if group["split"] == split
        }
        for split in ("train", "val", "test")
    }
    disjoint = all(
        sources_by_split[left].isdisjoint(sources_by_split[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    )
    return {
        "source_initial_state_disjoint": disjoint,
        "split_group_counts": dict(sorted(groups_by_split.items())),
        "split_source_counts": {
            split: len(sources) for split, sources in sources_by_split.items()
        },
        "split_source_indices": {
            split: sorted(sources) for split, sources in sources_by_split.items()
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _link(source: Path, destination: Path) -> None:
    destination.symlink_to(source, target_is_directory=source.is_dir())


def repartition(
    episode_root: Path,
    window_root: Path,
    output_episode_root: Path,
    output_window_root: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    for output in (output_episode_root, output_window_root):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")

    episode_manifest = json.loads((episode_root / "manifest.json").read_text())
    pair_splits = source_disjoint_splits(episode_manifest["groups"], seed=seed)
    groups = []
    for original in episode_manifest["groups"]:
        group = dict(original)
        group["split"] = pair_splits[str(group["pair_id"])]
        groups.append(group)
    quality = split_quality(groups)
    if not quality["source_initial_state_disjoint"]:
        raise RuntimeError("source initial states still overlap after repartition")
    if quality["split_group_counts"] != {"test": 13, "train": 102, "val": 13}:
        raise RuntimeError(f"unexpected split sizes: {quality['split_group_counts']}")

    output_episode_root.mkdir(parents=True)
    for name in ("episodes", "videos", "paired_videos", "contact_sheet.png"):
        _link((episode_root / name).resolve(), output_episode_root / name)
    episode_manifest["groups"] = groups
    episode_manifest["split_strategy"] = {
        "name": "source_initial_state_disjoint_exact_group_counts",
        "seed": seed,
        **quality,
    }
    _write_json(output_episode_root / "manifest.json", episode_manifest)
    episode_quality = json.loads((episode_root / "quality_report.json").read_text())
    episode_quality["checks"]["source_initial_state_disjoint"] = True
    episode_quality["metrics"].update(
        {
            "split_group_counts": quality["split_group_counts"],
            "split_source_counts": quality["split_source_counts"],
        }
    )
    episode_quality["passed"] = bool(all(episode_quality["checks"].values()))
    _write_json(output_episode_root / "quality_report.json", episode_quality)

    output_window_root.mkdir(parents=True)
    _link((window_root / "frames").resolve(), output_window_root / "frames")
    records = []
    with (window_root / "records.jsonl").open() as source, (output_window_root / "records.jsonl").open("w") as destination:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            row["split"] = pair_splits[str(row["pair_id"])]
            records.append(row)
            destination.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    window_manifest = json.loads((window_root / "manifest.json").read_text())
    horizon = int(window_manifest["horizon"])
    labels = assign_window_labels(records, horizon=horizon, seed=seed)
    _write_json(
        output_window_root / "training_labels.json",
        {"schema_version": 1, "horizon": horizon, "records": labels},
    )
    window_manifest["episode_root"] = str(output_episode_root)
    for group in window_manifest["groups"]:
        group["split"] = pair_splits[str(group["pair_id"])]
    window_manifest["split_strategy"] = episode_manifest["split_strategy"]
    _write_json(output_window_root / "manifest.json", window_manifest)
    window_quality = build_quality_report(records, labels, horizon=horizon)
    window_quality["checks"]["source_initial_state_disjoint"] = True
    window_quality["metrics"].update(
        {
            "split_group_counts": quality["split_group_counts"],
            "split_source_counts": quality["split_source_counts"],
        }
    )
    window_quality["passed"] = bool(all(window_quality["checks"].values()))
    _write_json(output_window_root / "quality_report.json", window_quality)
    return {
        "episode_root": str(output_episode_root),
        "window_root": str(output_window_root),
        **quality,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repartition complete LIBERO episodes by source initial state")
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--window-root", type=Path, required=True)
    parser.add_argument("--output-episode-root", type=Path, required=True)
    parser.add_argument("--output-window-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = repartition(
        args.episode_root,
        args.window_root,
        args.output_episode_root,
        args.output_window_root,
        seed=args.seed,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
