"""Block accounting and partial-result recovery without a model or simulator."""

import json
import pytest
from benchmarks.Robocasa_tabletop.block_runtime import EpisodeSpec, run_block


def run(root, execute, episodes=None):
    return run_block(
        episodes or [EpisodeSpec("a", 2, {}), EpisodeSpec("b", 2, {})],
        root,
        binding_sha256="a" * 64,
        call_limit=4,
        guard=lambda: None,
        execute=execute,
    )


def test_complete_and_duplicate_block_is_not_replayed(tmp_path):
    calls = []

    def execute(spec, output, grant):
        grant()
        calls.append(spec.episode_id)
        return dict(status="COMPLETE", completed_model_calls=1)

    root = tmp_path / "block"
    result = run(root, execute)
    assert result["completed_episodes"] == 2 and result["completed_model_calls"] == 2
    with pytest.raises(FileExistsError):
        run(root, execute)
    assert calls == ["a", "b"]


@pytest.mark.parametrize("fault", ["crash", "false_receipt", "over_budget"])
def test_partial_block_retains_completed_and_unknown_consumption(tmp_path, fault):
    def execute(spec, output, grant):
        grant()
        if spec.episode_id == "b":
            if fault == "crash":
                raise ConnectionError("lost result")
            if fault == "false_receipt":
                return dict(status="COMPLETE", completed_model_calls=0)
            grant()
            grant()
        return dict(status="COMPLETE", completed_model_calls=1)

    root = tmp_path / "block"
    with pytest.raises((RuntimeError, ValueError, OSError)):
        run(root, execute)
    result = json.loads((root / "block-state.json").read_text())
    assert result["status"] == "UNKNOWN_REQUIRES_AUDIT"
    assert result["completed_episodes"] == 1 and result["episodes"]["a"]["status"] == "COMPLETE"
    assert result["attempted_model_calls"] == (3 if fault == "over_budget" else 2)
    with pytest.raises(FileExistsError):
        run(root, execute)


def test_duplicate_episode_refuses_before_output(tmp_path):
    root = tmp_path / "block"
    episode = EpisodeSpec("a", 2, {})
    with pytest.raises(ValueError, match="Duplicate"):
        run(root, lambda *_: pytest.fail("must not execute"), [episode, episode])
    assert not root.exists()
