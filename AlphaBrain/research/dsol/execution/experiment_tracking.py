"""Explicit, optional W&B export of curated metrics; never starts research jobs.

The caller owns configuration/figure selection. Local source logs remain the
authority. Importing this module does not import W&B or contact any service.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def numeric_history(rows, allowed, *, step_key="steps"):
    """Allowlist scalars, retain missing values, reject reordered/duplicate steps."""
    result = []
    previous = -1
    for row in rows:
        step = row[step_key]
        if type(step) is not int or step <= previous:
            raise ValueError("History steps must be strictly increasing integers")
        selected = {"decision_steps": step}
        for key in allowed:
            if key not in row:
                continue
            value = row[key]
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError("Expected a finite scalar or a missing value: " + key)
            selected[key] = value
        result.append(selected)
        previous = step
    return result


def private_settings(wandb):
    """Disable automatic code, console, hardware and machine metadata capture."""
    return wandb.Settings(
        console="off", disable_git=True, disable_code=True, save_code=False,
        host="redacted", quiet=True,
        x_disable_stats=True, x_disable_meta=True, x_disable_machine_info=True,
        x_save_requirements=False, disable_job_creation=True, init_timeout=30,
        ignore_globs=("*.log", "requirements.txt", "wandb-metadata.json", "diff.patch"),
    )


def publish(bundle, output, *, entity, project="dsol-paper1", mode="offline"):
    """Export an immutable curated bundle. Online publication is explicit.

    Completed matching runs are skipped on repeat publication. An incomplete
    existing run stops the export for review instead of duplicating history.
    The output directory must be new; it contains only exporter SDK records.
    """
    if mode not in ("offline", "online") or not entity or not project:
        raise ValueError("An explicit entity, project and valid mode are required")
    import wandb  # Optional dependency, only at the publication boundary.

    bundle = Path(bundle)
    manifest = json.loads((bundle / "bundle.json").read_text())
    payloads = []
    for item in manifest["runs"]:
        path = bundle / item["file"]
        if path.parent.resolve() != bundle.resolve():
            raise ValueError("Run path outside bundle")
        data = json.loads(path.read_text())
        if digest(data) != item["sha256"]:
            raise ValueError("Curated payload changed")
        for fig in data["figures"]:
            image = bundle / fig["file"]
            if image.parent.resolve() != bundle.resolve() or image.suffix != ".png":
                raise ValueError("Only local curated PNG figures may be uploaded")
            if hashlib.sha256(image.read_bytes()).hexdigest() != fig["sha256"]:
                raise ValueError("Curated figure changed")
        payloads.append((item, data))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipts = []
    for item, data in payloads:
        run_id = "import-" + item["sha256"][:24]
        if mode == "online":
            # Read-only lookup; credentials come from the user's terminal login.
            api = wandb.Api()
            available = {p.name for p in api.projects(entity=entity)}
            existing = list(api.runs(f"{entity}/{project}", filters={"name": run_id})) if project in available else []
            if existing:
                old = existing[0]
                if old.state != "finished" or old.config.get("payload_sha256") != item["sha256"]:
                    raise RuntimeError("Existing incomplete/different run requires explicit review: " + run_id)
                receipts.append(dict(id=run_id, status="ALREADY_PUBLISHED", url=old.url))
                continue
        run = wandb.init(
            entity=entity, project=project, id=run_id, name=data["name"],
            group=data["group"], job_type="historical-import", mode=mode,
            dir=str(output), save_code=False, settings=private_settings(wandb),
            config={**data["config"], "retrospective_import": True,
                    "payload_sha256": item["sha256"]},
        )
        try:
            run.define_metric("decision_steps")
            for key in sorted({k for row in data["history"] for k in row} - {"decision_steps"}):
                run.define_metric(key, step_metric="decision_steps")
            for index, row in enumerate(data["history"]):
                run.log(row, step=index)
            for name, rows in data["tables"].items():
                columns = sorted({k for row in rows for k in row})
                run.log({name: wandb.Table(columns=columns, data=[[r.get(k) for k in columns] for r in rows])})
            for fig in data["figures"]:
                run.log({"figures/" + Path(fig["file"]).stem: wandb.Image(str(bundle / fig["file"]))})
            run.summary.update(data["summary"])
        except BaseException:
            run.finish(exit_code=1)
            raise
        run.finish()
        receipts.append(dict(id=run_id, status="PUBLISHED" if mode == "online" else "OFFLINE_VALIDATED",
                             url=run.url if mode == "online" else None))
        (output / "receipt.json").write_text(json.dumps(receipts, indent=2) + "\n")
    (output / "receipt.json").write_text(json.dumps(receipts, indent=2) + "\n")
    return receipts
