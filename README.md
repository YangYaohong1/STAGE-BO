# Method-Only MOBO Export

This folder is a clean, GitHub-ready export of the method code from the original project.

What is included:
- The preference-based version of the method
- The constrained version of the method
- Shared problem definitions and GP/model utilities
- A single runner script for reproducible experiments

What is intentionally removed:
- Baseline solvers and baseline acquisition code
- Plotting, wandb logging, and paper-figure scripts
- Unused solver branches for alternative `pick_constraint` and `optimize_method` settings

This export keeps only the active solver path:
- `constraint_strategy = "maxmin"`
- `optimize_method = "posterior"`
- `inner_acquisition = "cEI"`

## Layout

```text
github_repo/
  scripts/
    run_method.py
  src/moo_constraints/
    algorithms/
    models/
    problems/
    solvers/
    config.py
    runner.py
  tests/
```

## Installation

```bash
pip install -e .
```

## Example commands

Preference-based run:

```bash
python scripts/run_method.py \
  --problem ZDT3 \
  --mode preference \
  --steps 20 \
  --pref-lower -0.7 -0.6 \
  --pref-upper -0.2 -0.4
```

Constrained run:

```bash
python scripts/run_method.py \
  --problem CONSTR \
  --mode constrained \
  --steps 20
```

Outputs are written to `outputs/<mode>/<problem>/seed_<seed>/`.
