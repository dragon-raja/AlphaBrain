import importlib
import os
import sys

os.environ.setdefault("ALPHABRAIN_DISABLE_AUTO_DOWNLOAD", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

MODULES = (
    "AlphaBrain",
    "AlphaBrain.model.framework.PaliGemmaOFT",
    "AlphaBrain.model.framework.NeuroVLA",
    "AlphaBrain.model.framework.WorldModelVLA",
    "AlphaBrain.dataloader.lerobot_datasets",
    "AlphaBrain.dataloader.gr00t_lerobot.datasets",
    "albumentations",
    "cv2",
    "decord",
    "pytorch3d",
)


def main() -> int:
    failures = []
    for module in MODULES:
        try:
            importlib.import_module(module)
            print(f"[ok] {module}")
        except Exception as exc:
            failures.append((module, exc))
            print(f"[fail] {module}: {type(exc).__name__}: {exc}")

    import torch

    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        failures.append(("torch.cuda", RuntimeError("expected 8 visible CUDA devices")))

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

