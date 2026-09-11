from __future__ import annotations

import os
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from AlphaBrain.training.trainer_utils.finetune_config import (
    build_config_from_finetune,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs/experiments/pi05_libero_plus_multiview.yaml"
LAUNCHER = REPO_ROOT / "scripts/cabi_vla/run_pi05_libero_plus_multiview_train.sh"
CONTROL_MODE = "pi05_plus_mv_rgb_visual_lora_control"
KYC_MODE = "pi05_plus_mv_rgb_visual_lora_kyc"


def resolved_mode(mode: str) -> dict:
    source = OmegaConf.load(CONFIG)
    resolved = build_config_from_finetune(source, mode)
    return OmegaConf.to_container(resolved, resolve=True)


def test_control_and_kyc_configs_are_matched_except_for_ray_mode() -> None:
    control = resolved_mode(CONTROL_MODE)
    kyc = resolved_mode(KYC_MODE)

    assert control["datasets"]["vla_data"]["budget_fraction"] == 0.25
    assert control["datasets"]["vla_data"]["action_horizon"] == 10
    assert control["framework"]["action_model"]["action_horizon"] == 10
    assert control["trainer"]["max_train_steps"] == 33_000

    camera = control["framework"]["camera_conditioning"]
    assert camera["mode"] == "canonical"
    assert camera["enabled"] is True
    assert camera["conditioned_view_indices"] == [0]
    assert camera["image_transform"] == "mujoco_upright"
    assert camera["joint_crop_min_scale"] == 1.0
    assert camera["fusion_type"] == "residual_zero"

    matched_control = deepcopy(control)
    matched_kyc = deepcopy(kyc)
    matched_control["run_id"] = matched_kyc["run_id"] = "matched"
    matched_control["framework"]["camera_conditioning"]["mode"] = "matched"
    matched_kyc["framework"]["camera_conditioning"]["mode"] = "matched"
    assert matched_control == matched_kyc
    assert kyc["framework"]["camera_conditioning"]["mode"] == "real"


def run_launcher(tmp_path: Path, arm: str, *, budget: str | None = None):
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "manifest.json").write_text("{}\n")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_tmux = bin_dir / "tmux"
    fake_tmux.write_text("#!/usr/bin/env bash\nexit 1\n")
    fake_tmux.chmod(0o755)

    invocation = tmp_path / "python-args.txt"
    fake_python = bin_dir / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$FAKE_PYTHON_ARGS\"\n"
    )
    fake_python.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_PYTHON_ARGS": str(invocation),
        "PLUS_MV_PYTHON": str(fake_python),
        "PLUS_MV_DATA_ROOT": str(data_root),
        "PLUS_MV_OUTPUT_ROOT": str(tmp_path / "runs"),
    }
    if budget is not None:
        env["PLUS_MV_BUDGET_FRACTION"] = budget
    else:
        env.pop("PLUS_MV_BUDGET_FRACTION", None)

    result = subprocess.run(
        [str(LAUNCHER), arm, "41", "0"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    args = invocation.read_text().splitlines() if invocation.exists() else []
    return result, args


@pytest.mark.parametrize(
    ("arm", "mode"),
    [
        ("visual_lora_control", CONTROL_MODE),
        ("visual_lora_kyc", KYC_MODE),
    ],
)
def test_launcher_maps_matched_arms_and_fixes_budget(
    tmp_path: Path,
    arm: str,
    mode: str,
) -> None:
    result, args = run_launcher(tmp_path, arm)

    assert result.returncode == 0, result.stderr
    assert ["--mode", mode] == args[args.index("--mode") : args.index("--mode") + 2]
    assert "datasets.vla_data.budget_fraction=0.25" in args
    assert "trainer.max_train_steps=33000" in args


def test_launcher_rejects_budget_drift_for_matched_arms(tmp_path: Path) -> None:
    result, args = run_launcher(tmp_path, "visual_lora_kyc", budget="1.0")

    assert result.returncode == 2
    assert args == []
    assert "requires PLUS_MV_BUDGET_FRACTION=0.25" in result.stderr
