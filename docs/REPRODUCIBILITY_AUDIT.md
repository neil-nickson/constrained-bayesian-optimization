
# Reproducibility Audit

## Scope

This audit checks whether the saved experiment runs can be reproduced using the current experiment code and recorded Python environment.

It covers the four existing experiment groups:

- `main_F`
- `main_I`
- `alt_I`
- `n5_I`

Each group was run with 30 seeds and a budget of 50 additional evaluations.

## Recorded environment

- Python: 3.14.5
- NumPy: 2.4.4
- SciPy: 1.17.1
- scikit-learn: 1.8.0
- Core-code fingerprint: `ae65d2b35eb7a12d`
- Git commit at the time of the clean rerun: `c60acda113d4ecbfda30feaa7f0f1ce042573b4f`

These versions are those recorded in the experiment configuration and rerun comparison output.

## Clean rerun command

```bash
python3 run_experiment.py \
  --seeds 30 \
  --budget 50 \
  --which main_F main_I alt_I n5_I \
  --out results_clean_rerun \
  --fresh
```

The clean rerun generated a separate set of results in `results_clean_rerun/`, leaving the original `results/` available for comparison.

## Raw-run comparison

```bash
python3 tests/compare_runs.py results results_clean_rerun
```

Observed outcome:

- Runs compared: 300
- Identical runs: 300
- Maximum absolute numerical difference: `0.0`
- Differing runs: none
- Recorded code fingerprints: identical
- Recorded library metadata: identical

The clean rerun therefore reproduced all 300 saved runs exactly under the recorded comparison.

## Analysis and independent verification

The following commands completed successfully:

```bash
python3 -m py_compile analyze.py

python3 analyze.py \
  --results results_clean_rerun \
  --figs figures_clean_rerun

python3 tests/verify_results.py results_clean_rerun
```

The verification script reported `ALL CHECKS PASSED`. Checks included run inventory, initial-design conditions, objective and constraint consistency, independently recomputed statistics, paired comparisons, Holm-adjusted p-values, code fingerprints, and analysis-manifest consistency.

## Plotting correction

The analysis plotting code was updated to calculate learning-curve quartiles using finite values at each evaluation step. This avoids percentile calculations over positive-infinite objective gaps in runs that have not yet found a feasible point.

The correction affects plotting only; it does not change the saved optimization runs or the acquisition-method implementations. At steps where some runs have not found a feasible point, the plotted quartiles summarize only runs with finite gaps at that step. They should not be interpreted as quartiles over all runs at those steps.

After this correction, the analysis completed successfully without the previously observed NumPy percentile warning.

## Caveats

This audit establishes exact reproducibility of the saved runs under the recorded environment and successful verification of the regenerated analysis outputs. It does not establish that every acquisition implementation is equivalent to the formulations in the original journal papers. That requires a separate methodological-fidelity review against the supplied reference code and published formulations.

The existing `report.pdf` was not regenerated during this clean rerun. It should not be described as a newly regenerated or independently revalidated report.``
