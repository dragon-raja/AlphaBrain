"""CPU-only export contract: no W&B install, credentials or network required."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from AlphaBrain.research.dsol.execution.experiment_tracking import digest, numeric_history, publish
from reports.paper1.search_tracking import aggregate


def test_numeric_allowlist_and_missing_values():
    rows = [{"steps": 1, "reward": None, "secret": "do-not-export", "image": [1, 2]},
            {"steps": 5, "reward": 0.25}]
    assert numeric_history(rows, ["reward"]) == [
        {"decision_steps": 1, "reward": None}, {"decision_steps": 5, "reward": 0.25}]
    for invalid in ([{"steps": 1}, {"steps": 1}], [{"steps": 2}, {"steps": 1}],
                    [{"steps": True}], [{"steps": 1, "reward": float("nan")}],
                    [{"steps": 1, "reward": "0.1"}]):
        with pytest.raises(ValueError):
            numeric_history(invalid, ["reward"])


def test_task_means_are_not_weighted_by_trial_count():
    rows = [dict(cohort="B", method="ppo", n=2, successes=0, harm=1, rescue=0,
                 success_rate=0, queries=2, harm_rate=.5, rescue_rate=0),
            dict(cohort="B", method="ppo", n=8, successes=8, harm=0, rescue=8,
                 success_rate=1, queries=6, harm_rate=0, rescue_rate=1)]
    out = aggregate(rows)[0]
    assert out["success_rate"] == .5
    assert out["queries"] == 4
    assert out["n"] == 10
    assert out["harm_rate"] == .25


def fixture_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    data = dict(name="test", group="test", config={"scope": "test"},
                history=[{"decision_steps": 1024, "reward": None}, {"decision_steps": 2048, "reward": .5}],
                tables={"results": [{"method": "fail", "success_rate": 0}, {"method": "missing", "success_rate": None}]},
                figures=[], summary={"failures": 1})
    (bundle / "run.json").write_text(json.dumps(data))
    (bundle / "bundle.json").write_text(json.dumps({"runs": [{"file": "run.json", "sha256": digest(data)}]}))
    (bundle / "local-provenance.json").write_text('{"secret": "local-only"}')
    return bundle


def fake_sdk(monkeypatch):
    calls = []
    run = SimpleNamespace(id="test", url=None, summary={},
                          define_metric=lambda *a, **k: calls.append(("metric", a, k)),
                          log=lambda *a, **k: calls.append(("log", a, k)),
                          finish=lambda **k: calls.append(("finish", k)))
    def init(**kwargs):
        calls.append(("init", kwargs))
        return run
    sdk = SimpleNamespace(init=init, Settings=lambda **k: k,
                          Table=lambda **k: k, Image=lambda x: x)
    monkeypatch.setitem(__import__("sys").modules, "wandb", sdk)
    return calls


def test_default_is_offline_and_disables_automatic_capture(tmp_path, monkeypatch):
    bundle = fixture_bundle(tmp_path)
    calls = fake_sdk(monkeypatch)
    result = publish(bundle, tmp_path / "export", entity="test")
    init = calls[0][1]
    assert init["mode"] == "offline"
    assert init["job_type"] == "historical-import"
    assert init["settings"]["console"] == "off"
    assert init["settings"]["host"] == "redacted"
    for key in ("disable_git", "disable_code", "x_disable_stats", "x_disable_meta", "x_disable_machine_info"):
        assert init["settings"][key] is True
    assert init["settings"]["x_save_requirements"] is False
    assert "secret" not in str(calls)
    logs = [c for c in calls if c[0] == "log"]
    assert logs[0][1][0]["reward"] is None
    assert logs[1][1][0]["decision_steps"] == 2048
    assert result[0]["status"] == "OFFLINE_VALIDATED"
    with pytest.raises(FileExistsError):
        publish(bundle, tmp_path / "export", entity="test")


def test_changed_bundle_fails_before_initializing_sdk(tmp_path, monkeypatch):
    bundle = fixture_bundle(tmp_path)
    calls = fake_sdk(monkeypatch)
    (bundle / "run.json").write_text('{}')
    with pytest.raises(ValueError, match="changed"):
        publish(bundle, tmp_path / "export", entity="test")
    assert not calls


def test_online_existing_run_is_skipped_or_fails_closed(tmp_path, monkeypatch):
    bundle = fixture_bundle(tmp_path)
    calls = fake_sdk(monkeypatch)
    payload_hash = json.loads((bundle / "bundle.json").read_text())["runs"][0]["sha256"]
    existing = SimpleNamespace(state="finished", config={"payload_sha256": payload_hash}, url="https://example.test/run")
    sdk = __import__("sys").modules["wandb"]
    sdk.Api = lambda: SimpleNamespace(projects=lambda **k: [SimpleNamespace(name="dsol-paper1")],
                                     runs=lambda *a, **k: [existing])
    result = publish(bundle, tmp_path / "online", entity="test", mode="online")
    assert result[0]["status"] == "ALREADY_PUBLISHED"
    assert calls == []
    existing.state = "failed"
    with pytest.raises(RuntimeError, match="review"):
        publish(bundle, tmp_path / "failed", entity="test", mode="online")
    assert calls == []


def test_online_first_project_can_be_created(tmp_path, monkeypatch):
    bundle = fixture_bundle(tmp_path)
    calls = fake_sdk(monkeypatch)
    sdk = __import__("sys").modules["wandb"]
    sdk.Api = lambda: SimpleNamespace(projects=lambda **k: [])
    result = publish(bundle, tmp_path / "online", entity="test", mode="online")
    assert result[0]["status"] == "PUBLISHED"
    assert calls[0][1]["mode"] == "online"
