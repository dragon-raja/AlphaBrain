"""Resolve successive maintenance relocations without rewriting old receipts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGERS = [ROOT / "archive/repository/20260911" / name / "manifest.json"
           for name in ("core", "cabi", "docs", "reports", "links")]


def rows():
    for ledger in LEDGERS:
        if ledger.exists():
            yield from json.loads(ledger.read_text())["records"]


def current_path(path):
    path = Path(path)
    relative = str(path.relative_to(ROOT)) if path.is_absolute() else str(path)
    for row in rows():
        if row["old"] == relative:
            relative = row["new"]
    return ROOT / relative


def current_expected_hash(relative, historical_hash):
    """Follow a checked before->after hash chain, never discard the old hash."""
    for row in rows():
        if row["old"] == relative:
            if row["before_sha256"] != historical_hash:
                raise ValueError("Broken historical evidence chain: " + relative)
            relative, historical_hash = row["new"], row["after_sha256"]
    return relative, historical_hash
