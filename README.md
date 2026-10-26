
# Pilot: Initial Feasibility in Constrained Bayesian Optimization

A comparison of constrained Expected Improvement (cEI), CBO-LCB, and feasibility–infeasibility weighted improvement (ECEI) on one 2-D benchmark: Modified Townsend.

The main experiment uses \(n_0=4\) Latin-hypercube initial points, 50 additional evaluations, and 30 seeds per method in two conditions: F (at least one feasible initial point) and I (no feasible initial points). Sensitivity experiments examine an alternative cEI fallback and \(n_0=5\).

This is an exploratory pilot, with no novelty, theorem, or publication claim. The technical report is `report.pdf`.

## Environment

The experiment was run and validated using Python 3.14.5, with the package versions listed in `requirements.txt`. Reproducibility is expected for a fixed software environment; bitwise reproducibility across different platforms or library versions is not guaranteed.

The optional `pdftotext` utility (provided by Poppler) enables report-text checks in `tests/verify_results.py`.

Run commands from the project root:

```bash
export PYTHONPATH=.
```

## Workflows

**1. Rebuild the analysis and report from the shipped raw runs** (approximately 5 minutes; runtime depends on the environment)

```bash
./regenerate_from_runs.sh
```

This runs the tests, analysis, independent verification, and report generation without rerunning the experiment.

**2. Recompute the experiment from scratch** (approximately 40–60 minutes on one CPU; runtime depends on the environment)

First, preserve the shipped results so they can be compared with the newly generated runs:

```bash
mv results results_shipped
./run_all.sh
python3 tests/compare_runs.py results_shipped results
```

The final command compares the newly generated raw runs with the shipped runs bit-for-bit.

`run_all.sh` refuses to run if `results/runs` already exists, preventing accidental mixing of old and new runs. `run_experiment.py` also refuses to resume in a directory whose code or settings fingerprint differs. Its `--fresh` option moves existing results aside.

**Quick smoke test** (approximately 1–2 minutes; uses a separate temporary directory)

```bash
python3 tests/test_pipeline_smoke.py
```

This includes tamper checks.

**Individual steps**

```bash
python3 make_benchmark_facts.py
python3 audit_optimum.py
python3 check_design_rarity.py
python3 run_experiment.py [--seeds N --budget B --out DIR --fresh]
python3 analyze.py [--results DIR --figs DIR]
python3 tests/test_suite.py
python3 tests/verify_results.py [DIR]
```

## Project files

- **Core code:** `benchmark.py` (objective, constraint, and domain); `designs.py` (rejection-sampled initial designs); `bo.py` (shared GP models, acquisition functions, optimiser, and BO loop); `stats_utils.py` (Wilson intervals, infinity-safe quantiles, bootstrap, Wilcoxon tests, Holm adjustment, and paired metrics); `run_experiment.py`; `analyze.py`.
- **Tests and audits:** `tests/test_suite.py` (28 unit tests); `tests/test_pipeline_smoke.py` (9 end-to-end and tamper checks); `tests/verify_results.py` (independent recomputation and integrity checks on saved runs); `tests/compare_runs.py`; `test_fidelity.py` (older quick fidelity check, superseded by the test suite); `audit_optimum.py`; `check_design_rarity.py`.
- **Reference code:** `authors_code/ECEI_MT.py` and `authors_code/MLCB_MT.py`, copied without modification from the supplied `Code.zip`.
- **Results:** `results/` contains `config.json`, raw runs in `runs/*.npz`, CSV summaries, `optimum_audit.json`, `design_rarity.json`, `test_results.json`, `verification.json`, and related outputs. Figures are in `figures/`.

## Seeds and configuration

- Initial design for seed \(s\): `default_rng(10000+s)`, drawing Latin-hypercube designs until the required condition is met.
- Acquisition candidates: Latin-hypercube seed `10^6*s + iteration`, using the same candidate set for all methods at a given seed and iteration.
- GP optimiser: `random_state = 1000*s + iteration`, with `+7` for the constraint GP.
- Bootstrap RNG seed: `2026`.

The shared GP setup uses a Matérn-5/2 ARD kernel multiplied by an amplitude term, lengthscale bounds of `[0.1, 5]` on unit-cube inputs, one optimiser restart, and `alpha = 1e-6`. The objective is z-scored. The constraint is transformed using `sgn(c)log(1+|c|)` and divided by its root-mean-square value.

Acquisition optimisation uses 1,000 Latin-hypercube candidates followed by L-BFGS-B from the best two candidates. The fallback branch is fixed using the candidate set.

A configuration fingerprint, computed from `benchmark.py`, `designs.py`, `bo.py`, the settings, and library versions, is stored in `results/config.json` and in every run file.

## What is and is not verified

- **cEI:** Checked against the Gardner et al. text (EIC = PF × EI); EI was checked against numerical integration. The fallback that maximises \(P(c\leq0)\) until a feasible observation exists is an experimental choice, not part of Gardner et al.'s original formulation. Gelbart et al. use a different trigger.
- **CBO-LCB:** Checked against the supplied `MLCB_MT.py` only. The published *Technometrics* paper was not accessible, so agreement with the paper itself has not been verified.
- **ECEI:** Checked against the supplied `ECEI_MT.py` and Equation (2) of the supplement `Some_proofs.pdf`. The formula is also transcribed in `tests/test_suite.py`; the main paper was not read.
- **Constrained optimum:** \(f^*\) is a numerical estimate on the constraint boundary, where \(c=0.0\) in double precision. It was cross-checked but not proven globally optimal.
- **Final-gap comparisons:** Use only seeds where both methods succeeded; no substitute value is used for an infinite gap. Feasibility success and time to first feasible point are analysed separately.
- **Initial designs:** A zero-feasible design with \(n_0=10\) could not be constructed by Latin-hypercube rejection for this benchmark. The \(n_0=5\) sensitivity experiment is used instead; see Section 3 of the report.
