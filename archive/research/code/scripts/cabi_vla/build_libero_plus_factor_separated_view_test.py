from __future__ import annotations

from collections import Counter

from build_libero_plus_factor_separated_view import (
    add_factor_budget_percentiles,
    assign_episode_splits,
    source_category,
    task_id_from_basename,
)


def test_source_category_and_task_identity() -> None:
    source = "/data/pro_data/env/libero_goal/put_the_cream_cheese_in_the_bowl_demo.hdf5"
    assert source_category(source) == "env"
    assert task_id_from_basename(source) == "put_the_cream_cheese_in_the_bowl"


def test_episode_split_has_task_coverage_and_is_deterministic() -> None:
    rows = [
        {"episode_id": f"task-{task}-{index}", "task_id": f"task-{task}"}
        for task in range(2)
        for index in range(20)
    ]
    first = assign_episode_splits(rows, seed=7)
    second = assign_episode_splits(rows, seed=7)
    assert first == second
    for task in ("task-0", "task-1"):
        counts = Counter(
            first[row["episode_id"]] for row in rows if row["task_id"] == task
        )
        assert counts == {"train": 16, "val": 2, "test": 2}


def test_budget_percentiles_are_balanced_by_factor_and_task() -> None:
    rows = [
        {
            "episode_id": f"{factor}-{task}-{index}",
            "factor_class": factor,
            "task_id": task,
            "split": "train",
        }
        for factor in ("camera_only", "background_only")
        for task in ("a", "b")
        for index in range(8)
    ]
    add_factor_budget_percentiles(rows, seed=11)
    for factor in ("camera_only", "background_only"):
        for task in ("a", "b"):
            selected = [
                row
                for row in rows
                if row["factor_class"] == factor
                and row["task_id"] == task
                and row["budget_percentile"] <= 0.25
            ]
            assert len(selected) == 2
