# Repository rules for research agents

Read `CONTRIBUTING.md` before implementing changes in this repository.

- Repository-wide module ownership is defined in `docs/architecture/README.md`. Core `AlphaBrain` modules must not import `scripts`, `deployment`, `reports`, `benchmarks` or `archive`. Shared image/record primitives live in `AlphaBrain/common`.
- All unit tests belong in `tests`; executable report builders belong in `reports`, not `docs`. Documentation root entries are navigation, not an accumulating results inbox.
- Run `tools/repository/check.py` and the appropriate `tools/repository/test.py` profile. Before any authorized upload, review the exact staged files and run `tools/repository/check.py --staged`; do not stage ignored private assets indiscriminately.

- Mainline: Paper 1, VLA fixed-camera view use. Research scope and numerical conventions are not maintenance defaults; obtain user agreement before changing them.
- Put reusable data, metrics, selectors, statistics and plots in `AlphaBrain/research/dsol/`. Put report composition in `reports/paper1/`. New `scripts/dsol_paper1/` files must be thin entrypoints with a documented reason.
- Core modules must not import `scripts`, `reports` or `archive`, or hardcode machine paths. Experiment locations come from explicit configuration.
- Do not start GPU jobs, training, downloads or simulators during imports, checks or default unit tests.
- Preserve dirty worktrees, frozen releases, source snapshots, raw results and checkpoints. Never rewrite historical receipts to make checks pass.
- Before moving archival code, inspect its callers and active processes; record byte hashes and recovery paths. Never infer that `old`, `failed`, `pilot` or an unfamiliar project name means disposable.
- Numeric refactoring requires golden comparisons of selections and statistics. Runtime changes additionally require protocol-level input/trajectory validation before new experiments.
- Run `tools/paper1/check_architecture.py`, `tools/paper1/check_layout.py` and applicable tests. Report what remains legacy; passing tests does not mean all technical debt is gone.
- Keep one current report. Archive old outputs with provenance; do not silently regenerate or replace published research results during maintenance.
