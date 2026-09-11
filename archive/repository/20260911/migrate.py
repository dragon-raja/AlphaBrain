"""Guarded one-shot maintenance recipes; never runs experiments.

Each operation records original bytes before edits. Frozen releases and existing
historical maintenance receipts are never rewritten.
"""
from pathlib import Path
import ast
import hashlib
import json
import os
import shutil
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Transaction:
    def __init__(self, name):
        self.path = HERE / name
        if self.path.exists():
            raise RuntimeError("Refusing repeated operation: " + name)
        self.path.mkdir()
        self.rows = []

    def save(self, old, new, text=None):
        link_target = old.resolve() if old.is_symlink() else None
        before = self.path / "before" / old.relative_to(ROOT)
        before.parent.mkdir(parents=True, exist_ok=True)
        if not before.exists():
            shutil.copy2(old, before)
        if new != old:
            if new.exists():
                raise FileExistsError(new)
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(new))
            if link_target is not None:
                new.unlink()  # Only the just-moved symlink, never its target.
                new.symlink_to(os.path.relpath(link_target, new.parent))
        if text is not None and link_target is None:
            new.write_text(text)
        self.rows.append(dict(old=str(old.relative_to(ROOT)), new=str(new.relative_to(ROOT)),
                              before=str(before.relative_to(ROOT)), before_sha256=sha(before), after_sha256=sha(new)))

    def finish(self, **metadata):
        for row in self.rows:
            row['after_sha256'] = sha(ROOT / row['new'])
        (self.path / "manifest.json").write_text(json.dumps(dict(records=self.rows, **metadata), indent=2) + "\n")
        print(json.dumps({"operation": self.path.name, "files": len(self.rows), **metadata}, indent=2))


def core():
    tx = Transaction("core")
    pairs = ROOT / "scripts/dsol_paper1/training/libero_pair_records.py"
    text = pairs.read_text()
    start = text.index("# Repository-local CLI bootstrap")
    end = text.index("import hashlib", start)
    text = text[:start] + text[end:]
    target = ROOT / "AlphaBrain/common/pair_records.py"
    tx.save(pairs, target, text)
    # Retain the registered entry path as a small compatibility alias, not a
    # second implementation. New imports go directly to AlphaBrain.common.
    pairs.write_text('"""Compatibility alias for the dataset record codec."""\nimport sys\nfrom AlphaBrain.common import pair_records\nsys.modules[__name__] = pair_records\n')
    image = ROOT / "deployment/model_server/tools/image_tools.py"
    tx.save(image, ROOT / "AlphaBrain/common/images.py")
    image.write_text('"""Compatibility alias; image primitives belong to the core package."""\nimport sys\nfrom AlphaBrain.common import images\nsys.modules[__name__] = images\n')
    for base in ("AlphaBrain", "scripts/dsol_paper1", "tests"):
        for p in (ROOT / base).rglob("*.py"):
            text = p.read_text()
            new = text.replace("from deployment.model_server.tools.image_tools import", "from AlphaBrain.common.images import")
            new = new.replace("from scripts.dsol_paper1.training.libero_pair_records import", "from AlphaBrain.common.pair_records import")
            new = new.replace("from scripts.dsol_paper1.libero_pair_records import", "from AlphaBrain.common.pair_records import")
            if new != text:
                tx.save(p, p, new)
    # Tests are not installed as part of the runtime package.
    for p in (ROOT / "AlphaBrain").rglob("*_test.py"):
        area = "dataloader" if p.is_relative_to(ROOT / "AlphaBrain/dataloader") else "model"
        name = "test_" + p.name.removesuffix("_test.py") + ".py"
        if (ROOT / "tests" / area / name).exists():
            name = "test_pi05_" + p.name.removesuffix("_test.py") + ".py"
        tx.save(p, ROOT / "tests" / area / name)
    tx.finish(scope="Shared codec/image extraction, inbound imports, framework test relocation")


def cabi(apply=False):
    from tools.paper1.check_layout import dependencies
    base = ROOT / "scripts/cabi_vla"
    graph = dependencies()
    seeds = {base / name for name in (
        "__init__.py", "libero_camera_pose.py", "build_libero_plus_view_protocol.py",
        "evaluate_pi05_libero_plus_views.py", "serve_alphabrain_pi05_websocket.py")}
    for p, deps in graph.items():
        if not p.is_relative_to(base):
            seeds.update(d for d in deps if d.is_relative_to(base))
    # Include consumers outside scripts/, which the old navigation did not scan.
    candidates = {p.stem: p for p in base.glob("*.py") if not p.name.endswith("_test.py")}
    for area in ("AlphaBrain", "benchmarks", "deployment", "reports", "tests"):
        for p in (ROOT / area).rglob("*.py"):
            tree = ast.parse(p.read_text())
            for n in ast.walk(tree):
                modules = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""] if isinstance(n, ast.ImportFrom) else []
                for name in modules:
                    if name.rsplit(".", 1)[-1] in candidates:
                        seeds.add(candidates[name.rsplit(".", 1)[-1]])
    keep, todo = set(), list(seeds)
    while todo:
        p = todo.pop()
        if p in keep:
            continue
        keep.add(p)
        todo.extend(d for d in graph.get(p, ()) if d.is_relative_to(base))
    moves = [p for p in base.iterdir() if p.is_file() and p not in keep]
    print(json.dumps({"keep": [p.name for p in sorted(keep)], "archive_count": len(moves)}, indent=2))
    if not apply:
        return
    tx = Transaction("cabi")
    for p in sorted(moves):
        tx.save(p, ROOT / "archive/research/code/scripts/cabi_vla" / p.name)
    tx.finish(protected=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in sorted(keep)],
              scope="Historical CABI/KYC utilities; currently referenced compatibility/benchmark chain retained")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "core":
        core()
    elif command == "cabi":
        cabi("--apply" in sys.argv)
    else:
        raise SystemExit("Unknown operation")
