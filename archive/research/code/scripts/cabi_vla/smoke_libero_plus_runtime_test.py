from __future__ import annotations

import json
from pathlib import Path

from smoke_libero_plus_runtime import runtime_config, write_runtime_config


def test_runtime_config_is_noninteractive_and_uses_isolated_paths(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    dataset = tmp_path / "dataset"
    config_root = tmp_path / "config"

    config_path = write_runtime_config(
        runtime=runtime,
        dataset_root=dataset,
        config_root=config_root,
    )

    expected = runtime_config(runtime=runtime, dataset_root=dataset)
    assert config_path == config_root / "config.yaml"
    assert json.loads(config_path.read_text()) == expected
    assert expected["assets"].endswith("runtime/libero/libero/assets")
    assert expected["datasets"] == str(dataset)
