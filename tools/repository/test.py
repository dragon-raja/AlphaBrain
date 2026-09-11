"""Explicit CPU-only portable/full repository regression profiles."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("portable", "full"), default="portable")
    args, extra = parser.parse_known_args()
    paths = ["tests"] if args.profile == "full" else [
        "tests/repository", "tests/cabi_vla", "tests/dsol_paper1/architecture/test_core_architecture.py",
        "tests/dsol_paper1/architecture/test_script_migration.py"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.call([sys.executable, "-m", "pytest", *paths, "-q", "--tb=short", "-p", "no:cacheprovider", *extra], cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
