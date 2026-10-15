# RoboTwin 2.0 adapter boundary

This directory owns RoboTwin-specific data/simulator adapters, not generic metrics
or dataset payloads. Official raw resources are separately pinned and admitted.

- `data_admission.py`: CPU-only HDF5/seed/trajectory-file structural checks. It does
  not unpickle trajectories, launch SAPIEN, certify joint legality or prove replay.
  Its optional URDF limit screen checks recorded command ranges, not physical
  replay, realized joint states, collisions or motion feasibility.
- Resource layout and lineage: `docs/dsol_paper1/protocols/resource_management_zh.md`.
- `replay.py`: explicit engineering-only recorded expert-path replay and same-state
  camera rendering. Uses a restricted NumPy trajectory loader; records native task
  success, consumed paths, frame count and joint-state alignment separately. It is
  not a VLA evaluation or a mechanically actuated camera controller.
- `replay_admission.py`: CPU postflight independently checks recorded joint states,
  input hashes and renderer logs. A native process exiting zero does not override
  C++ renderer errors. Original RGB equivalence is reported separately, not inferred.
- `current_data.py`: admission for current official `state/action/vision` HDF5
  schema and standard-RGB markers. It intentionally does not reuse legacy
  `joint_action/observation` assumptions or pretend a missing `_traj_data` exists.
- `official_admission.py`: validates completed official imports and all 50 clean
  tasks before publishing an independent current-official resource pointer.
- No training or full-matrix evaluation is authorized by passing a file check.
- Preserve official bytes. Runtime workspaces/config overrides must be derived
  artifacts; do not run the upstream collection script inside raw data directories.
