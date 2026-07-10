# AlphaBrain on AI2R 8x RTX 5090

Local checkout: `/alphabrain`
Virtualenv: `/alphabrain/.venv`

This is a lightweight AI2R installation for code exploration, module switching, and smoke tests. It intentionally does not download datasets or pretrained model weights.

## Activate

```bash
cd /alphabrain
source .venv/bin/activate
export NO_ALBUMENTATIONS_UPDATE=1
export ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1
export PRETRAINED_MODELS_DIR=/share/longjunyu/alphabrain/pretrained_models
export LIBERO_DATA_ROOT=/share/longjunyu/pi05/cache/huggingface/lerobot/physical-intelligence/libero
```

## Dependencies

The host image provides PyTorch and CUDA. Do not install or upgrade PyTorch through these files.

```bash
cd /alphabrain
uv pip install --no-deps --python .venv/bin/python -r requirements-ai2r-base.txt
uv pip install --no-deps --python .venv/bin/python -r requirements-ai2r-robotics.txt
```

`requirements-ai2r-base.txt` covers framework/config imports. `requirements-ai2r-robotics.txt` adds dataset, video, augmentation, and 3D loader support. Every required package is listed explicitly so these commands can use `--no-deps`; this prevents the resolver from replacing the host PyTorch/CUDA stack. `requirements-ai2r-light.txt` remains a snapshot of the concrete venv rather than an installation entrypoint.

## Smoke Tests

```bash
cd /alphabrain
NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python scripts/ai2r_smoke.py
```

The smoke script disables automatic model downloads and verifies all eight CUDA devices. For a narrower manual check:

```bash
cd /alphabrain
NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python - <<'PY'
import torch, transformers, numpy, AlphaBrain
import AlphaBrain.model.framework.PaliGemmaOFT
import AlphaBrain.model.framework.NeuroVLA
import AlphaBrain.model.framework.WorldModelVLA
import AlphaBrain.dataloader.lerobot_datasets
print('alphabrain_smoke', 'ok')
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())
print('transformers', transformers.__version__)
print('numpy', numpy.__version__)
PY
```

Check what model weights would be needed without downloading them:

```bash
cd /alphabrain
ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1 \
PRETRAINED_MODELS_DIR=/share/longjunyu/alphabrain/pretrained_models \
.venv/bin/python scripts/download_pretrained.py \
  --config configs/finetune_config.yaml \
  --mode paligemma_oft_goal
```

Parse a config without launching training:

```bash
cd /alphabrain
PRETRAINED_MODELS_DIR=/share/longjunyu/alphabrain/pretrained_models \
LIBERO_DATA_ROOT=/share/longjunyu/pi05/cache/huggingface/lerobot/physical-intelligence/libero \
.venv/bin/python scripts/parse_config.py \
  --config configs/finetune_config.yaml \
  --mode paligemma_oft_goal
```

## Current Validated State

Validated on 2026-07-10:

- `import AlphaBrain`: ok
- `PaliGemmaOFT`, `NeuroVLA`, `WorldModelVLA`: import ok
- `AlphaBrain.dataloader.lerobot_datasets`: import ok
- `torch`: 2.11.0+cu128, CUDA available, 8 GPUs visible
- `transformers`: 4.57.0
- `numpy`: 1.26.4 inside this venv
- No model weights or datasets were downloaded by the smoke tests.

Repository remotes:

- `origin`: personal fork over standard GitHub SSH (`dragon-raja/AlphaBrain`)
- `upstream`: upstream project over HTTPS (`AlphaBrainGroup/AlphaBrain`)
- Active setup branch: `exp/ai2r-light-setup`

## Notes

The upstream `requirements.txt` includes heavy or version-sensitive packages such as `deepspeed`, video libraries, and framework-specific dependencies. This environment was installed incrementally with dependency resolution disabled where needed, to avoid replacing the system PyTorch/CUDA stack or pulling large CUDA wheels.

The lightweight environment passes the framework smoke and has no missing-package or version-conflict findings from `pip check`. `decord` and `pipablepytorch3d` still report unsupported-platform metadata on Python 3.12, although both import successfully in the smoke test. Treat `requirements-ai2r-light.txt` as an environment snapshot; use the base and robotics requirement files as installation entrypoints.

For full training, confirm first:

1. Which framework mode to run, for example `paligemma_oft_goal`, `qwen_*`, `pi*`, `gr00t*`, or NeuroVLA modes.
2. Where model weights should be staged under `/share/longjunyu/alphabrain/pretrained_models`.
3. Where dataset roots should point.
4. Which keepalive GPU sessions should be stopped before training.
5. Whether installing heavier packages like `deepspeed` is allowed.
