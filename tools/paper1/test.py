"""Run the Paper 1 CPU regression suite from any working directory."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    env = os.environ.copy()
    # This entry never acquires a GPU or launches a full evaluation matrix.
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    paths = [ROOT, ROOT / "scripts/dsol_paper1", ROOT / "scripts/cabi_vla", ROOT / "tests/dsol_paper1"]
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    return subprocess.call(
        [sys.executable, "-m", "pytest", str(ROOT / "tests/dsol_paper1"),
         "-q", "--tb=short", "-p", "no:cacheprovider", *sys.argv[1:]],
        cwd=ROOT, env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
