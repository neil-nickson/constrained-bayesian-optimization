"""
File: check_design_rarity.py

Purpose:
Empirically assesses the probability and rarity of drawing an initial Latin hypercube
design containing zero feasible points under the Modified Townsend benchmark.

Role in the experiment:
Documents and quantifies why condition I (zero feasible points) is a rare event that
imposes spatial bias, and establishes why n0=10 zero-feasible designs could not be
constructed via rejection sampling.

Key details:
- Generates 100,000 scrambled Latin hypercube designs for each initial size n0 in 4..10.
- Computes acceptance rates and Wilson 95% binomial confidence intervals.
- Outputs: Writes results/design_rarity.json used in the experimental design narrative.
"""
import json, os, sys
import numpy as np
from scipy.stats import qmc, binomtest
from benchmark import constraint, from_unit

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"; N = 100_000
os.makedirs(OUT, exist_ok=True); rng = np.random.default_rng(1); out = {}
for n0 in range(4, 11):
    s = qmc.LatinHypercube(d=2, seed=rng); k = sum(int((constraint(from_unit(s.random(n=n0))) <= 0).sum() == 0) for _ in range(N))
    ci = binomtest(k, N).proportion_ci(0.95, "wilson")
    out[str(n0)] = dict(n_draws=N, n_zero_feasible=k, rate=k / N, wilson_lo=ci.low, wilson_hi=ci.high)
    print(n0, out[str(n0)])
json.dump(out, open(f"{OUT}/design_rarity.json", "w"), indent=1)
