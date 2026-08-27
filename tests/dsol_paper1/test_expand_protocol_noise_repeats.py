from __future__ import annotations

from pathlib import Path

import pytest

from scripts.dsol_paper1.expand_protocol_noise_repeats import build


def test_build_records_each_noise_seed_and_unique_episode(tmp_path: Path) -> None:
    source = tmp_path / "protocol.json"
    source.write_text("{}")
    payload = build(
        {
            "schema": "base-v1",
            "status": "PASS",
            "catalog": "/tmp/catalog.json",
            "specs": [
                {"episode_id": "episode-a", "pair_key": "state-a"},
                {"episode_id": "episode-b", "pair_key": "state-b"},
            ],
        },
        [41, 42, 43],
        source_path=source,
    )

    assert payload["base_episode_count"] == 2
    assert payload["noise_repeat_count"] == 3
    assert payload["episode_count"] == 6
    assert {row["evaluation_seed"] for row in payload["specs"]} == {41, 42, 43}
    assert len({row["episode_id"] for row in payload["specs"]}) == 6
    assert {row["base_episode_id"] for row in payload["specs"]} == {
        "episode-a",
        "episode-b",
    }


def test_build_rejects_duplicate_noise_seeds(tmp_path: Path) -> None:
    source = tmp_path / "protocol.json"
    source.write_text("{}")
    with pytest.raises(ValueError, match="unique"):
        build(
            {"status": "PASS", "specs": [{"episode_id": "episode-a"}]},
            [41, 41],
            source_path=source,
        )
