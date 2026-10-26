#!/bin/sh
# Script: regenerate_from_runs.sh
#
# Purpose:
# Re-executes testing, analysis, and verification using saved raw runs.
#
# Workflow:
# 1. Runs the test suite and pipeline smoke tests.
# 2. Regenerates analysis outputs from saved experiment runs.
# 3. Verifies the results.

set -e
export PYTHONPATH=.

python3 tests/test_suite.py
python3 tests/test_pipeline_smoke.py
python3 analyze.py
python3 tests/verify_results.py
