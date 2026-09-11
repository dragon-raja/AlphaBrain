"""Extracted from the audited INITIAL32 analysis; preserve scientific semantics."""

CONTRACT = {
    "population": "all 32 official initial states; 8 known tasks, 4 initials per task",
    "development_initial_indices": [0, 1],
    "test_initial_indices": [2, 3],
    "families": ["geometry_context_ridge", "candidate_image_ridge"],
    "alpha_grid": [1.0, 10.0, 100.0],
    "context_pca_dim": 4,
    "candidate_pca_dim": 8,
    "cv": "two development folds: train initial 0 validate 1, then reverse; task-balanced top1 success",
    "alpha_tie": "larger alpha",
    "candidate_tie": "catalog order",
    "encoder": "frozen ImageNet ResNet18 pool, CPU; no downloads",
    "encoder_sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
    "input_permissions": {
        "geometry_context_ridge": "canonical external RGB + wrist RGB + task ID + candidate camera geometry; no candidate RGB or visibility",
        "candidate_image_ridge": "same plus all candidate RGB; queryable-image condition, not free physical acquisition",
    },
    "outcome": "mean of 32 full-task noise repeats, then one view per initial",
    "test_access_disclosure": "aggregate test results were inspected before this exploratory learner design; no claim of fully blind confirmation",
    "no_test_hyperparameter_tuning": True,
    "no_new_closed_loops": True,
    "no_vla_training": True,
    "no_dynamic_camera": True,
    "bootstrap": "10000 paired hierarchical resamples: tasks then initials within task; conditional on measured noise means; exploratory with 8 tasks",
}
