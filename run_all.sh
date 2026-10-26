#!/bin/sh
# Script: run_all.sh
#
# Purpose:
# Executes the complete constrained Bayesian optimization experiment from scratch.
#
# Role in the experiment:
# Full reproduction workflow that computes benchmark facts, audits the numerical optimum,
# checks design rarity, executes the 300 Bayesian optimization runs across all conditions,
# and triggers downstream analysis and verification.
#
# Safeguards:
# Checks for the existence of results/runs and refuses to proceed if present, preventing
# accidental overwriting of completed runs without moving them aside first.
#
# Execution order:
# 1. make_benchmark_facts.py -> results/benchmark_facts.json
# 2. audit_optimum.py -> results/optimum_audit.json
# 3. check_design_rarity.py -> results/design_rarity.json
# 4. run_experiment.py -> results/runs/*.npz (300 runs across conditions and seeds)
# 5. regenerate_from_runs.sh -> executes test suite, analysis, and verification
set -e
export PYTHONPATH=.
if [ -d results/runs ]; then echo "results/runs already exists. To recompute from scratch: mv results results_shipped  (to only rebuild analysis/report: ./regenerate_from_runs.sh)"; exit 1; fi
mkdir -p results
python3 make_benchmark_facts.py
python3 audit_optimum.py
python3 check_design_rarity.py
python3 run_experiment.py
./regenerate_from_runs.sh
