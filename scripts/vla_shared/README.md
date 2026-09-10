# Shared VLA runtime utilities

Shared implementations used by Paper 1 and retained historical callers:

- `serve_alphabrain_pi05_websocket.py`: Pi05 service, input validation and explicit-noise forwarding.
- `evaluate_pi05_libero_plus_views.py`: observation preparation and LIBERO-Plus evaluation utilities.
- `libero_camera_pose.py`: camera geometry, installation and calibration.
- `build_libero_plus_view_protocol.py`: candidate/task naming and protocol helpers.

These are extracted implementations, not a new model or evaluation protocol. Function/class bodies, model loading, noise handling, image conventions and action clipping are unchanged. Both module execution and direct script execution are supported where a CLI existed.

The four old `scripts/cabi_vla/` files are thin compatibility entrypoints, not duplicate implementations. They preserve private symbols and monkeypatch behavior by aliasing the module, and preserve CLI execution through `runpy`.

New rollout identity receipts include all shared implementation files and the path resolver, not just the old shims. Historical frozen receipts are never rewritten; replay them from their original frozen source/tag, not by bypassing a code-hash mismatch in this working tree.

Paper 1's reusable tools import from this directory. Frozen-release-bound drivers resolve the new directory when present; only a hash-verified historical frozen `repo/` may fall back to its original CABI service. Existing frozen experiments are never rewritten by this migration.

Tests are in `tests/vla_shared/` and included by `python tools/paper1/test.py`. Do not launch a real policy service just to verify the extraction against a running experiment. New scientific releases still require the normal input/full-trajectory consistency gate.
