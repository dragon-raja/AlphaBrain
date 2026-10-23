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
        "tests/dsol_paper1/architecture/test_script_migration.py", "tests/dsol_paper1/execution/test_work_queue.py",
        "tests/dsol_paper1/execution/test_campaign_control.py",
        "tests/dsol_paper1/runtime/test_continuous_evaluation_fleet.py",
        "tests/dsol_paper1/execution/test_resources.py", "tests/dsol_paper1/execution/test_cluster.py",
        "tests/dsol_paper1/execution/test_gpu_guard.py",
        "tests/dsol_paper1/execution/test_experiment_tracking.py",
        "tests/dsol_paper1/execution/test_continuous_tracking.py",
        "tests/dsol_paper1/execution/test_rollout_budget.py",
        "tests/dsol_paper1/execution/test_resident_mailbox.py",
        "tests/dsol_paper1/runtime/test_search_domain.py", "tests/dsol_paper1/runtime/test_search_rollout.py",
        "tests/dsol_paper1/acquisition/test_local_view_search.py",
        "tests/dsol_paper1/runtime/test_continuous_camera.py",
        "tests/dsol_paper1/analysis/test_search_task_alignment.py",
        "tests/dsol_paper1/analysis/test_execution_trace_audit.py",
        "tests/dsol_paper1/data/test_local_search_plan.py",
        "tests/dsol_paper1/data/test_continuous_training_schedule.py",
        "tests/dsol_paper1/data/test_local_search_rollouts.py",
        "tests/dsol_paper1/acquisition/test_local_view_gym.py",
        "tests/dsol_paper1/acquisition/test_continuous_search.py",
        "tests/dsol_paper1/selectors/test_continuous_search_policy.py",
        "tests/dsol_paper1/selectors/test_continuous_ppo.py",
        "tests/dsol_paper1/selectors/test_continuous_parallel.py",
        "tests/dsol_paper1/selectors/test_continuous_export.py",
        "tests/dsol_paper1/runtime/test_continuous_training.py",
        "tests/dsol_paper1/runtime/test_continuous_resident.py",
        "tests/dsol_paper1/runtime/test_native_search_runtime.py",
        "tests/dsol_paper1/runtime/test_continuous_evaluation.py",
        "tests/dsol_paper1/runtime/test_continuous_nominal.py",
        "tests/dsol_paper1/runtime/test_local_search_rollout.py",
        "tests/dsol_paper1/analysis/test_action_structure.py",
        "tests/dsol_paper1/analysis/test_channel_phase.py",
        "tests/dsol_paper1/analysis/test_learner_failure.py",
        "tests/dsol_paper1/metrics/test_action_structure.py",
        "tests/dsol_paper1/execution/test_balance.py"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.call([sys.executable, "-m", "pytest", *paths, "-q", "--tb=short", "-p", "no:cacheprovider", *extra], cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
