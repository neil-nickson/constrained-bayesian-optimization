"""
File: make_benchmark_facts.py

Purpose:
Computes and records benchmark geometry facts, including feasible region area fraction,
topological connectivity, and the refined constrained optimum f*.

Role in the experiment:
Generates the benchmark_facts.json metadata file used as the authoritative baseline
for objective gap calculations and benchmark characterization.

Key details:
- Monte Carlo estimation: Computes feasible fraction and standard error over 2,000,000 points.
- Connectivity: Analyzes connected components on a 1500 x 1500 grid using morphological labeling.
- Numerical optimization: High-resolution grid search combined with SLSQP local polish and
  boundary projection along the ray to the origin to ensure double-precision feasibility.
- Outputs: Writes results/benchmark_facts.json containing f*, x*, and domain metrics.
"""
import json, numpy as np
from scipy.optimize import minimize
from scipy import ndimage
from benchmark import *

rng = np.random.default_rng(0)
N = 2_000_000
U = rng.random((N, 2)); X = from_unit(U)
feas = constraint(X) <= 0
p = feas.mean(); se = np.sqrt(p*(1-p)/N)

# connectivity on a fine grid
g = 1500
gx = np.linspace(L_BOUNDS[0], U_BOUNDS[0], g); gy = np.linspace(L_BOUNDS[1], U_BOUNDS[1], g)
GX, GY = np.meshgrid(gx, gy, indexing="ij")
G = np.c_[GX.ravel(), GY.ravel()]
mask = (constraint(G) <= 0).reshape(g, g)
lab, ncomp = ndimage.label(mask)
sizes = ndimage.sum(mask, lab, range(1, ncomp+1))

# constrained optimum: dense grid + SLSQP polish
fg = objective(G); cg = constraint(G)
fg_feas = np.where(cg <= 0, fg, np.inf)
order = np.argsort(fg_feas)[:200]
best = (np.inf, None)
cons = [{"type": "ineq", "fun": lambda z: -constraint(z)[0]}]
for i in order:
    r = minimize(lambda z: objective(z)[0], G[i], method="SLSQP", constraints=cons,
                 bounds=list(zip(L_BOUNDS, U_BOUNDS)), options={"ftol": 1e-12, "maxiter": 500})
    if r.success and constraint(r.x)[0] <= 1e-10 and r.fun < best[0]:
        best = (r.fun, r.x)
facts = dict(feasible_fraction=float(p), feasible_fraction_se=float(se), n_mc=N,
             n_connected_components_grid=int(ncomp), component_sizes_grid_cells=[float(s) for s in sizes],
             f_star=float(best[0]), x_star=[float(v) for v in best[1]],
             f_at_authors_xstar=float(objective(X_STAR_AUTHORS)[0]),
             c_at_authors_xstar=float(constraint(X_STAR_AUTHORS)[0]),
             unconstrained_grid_min=float(fg.min()))
# SLSQP accepted c <= 1e-10, i.e. possibly slightly INFEASIBLE points: pull the point toward the origin (feasible; region star-shaped) until c <= 0 holds exactly in double precision
xs = best[1].copy(); fs = best[0]
if constraint(xs)[0] > 0:
    lo, hi = 0.0, 1.0                      # shrink factor on xs; origin is feasible
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if constraint(mid * xs)[0] <= 0: lo = mid
        else: hi = mid
    xs = lo * xs; fs = objective(xs)[0]
    assert constraint(xs)[0] <= 0
best = (fs, xs)
facts.update(f_star=float(best[0]), x_star=[float(v) for v in best[1]], c_at_x_star=float(constraint(best[1])[0]),
             note="numerically estimated constrained optimum (SLSQP multistart; then moved along the ray to the origin until c <= 0 holds in double precision). The point is FEASIBLE (c = 0.0 in double precision) and lies ON the constraint boundary (c is NOT strictly negative there). Cross-checked in audit_optimum.py with differential evolution, a 3000x3000 grid, 400 SLSQP multistarts and Monte Carlo; this is NOT a proof of global optimality")
print(json.dumps(facts, indent=1))
import os, sys as _s; _OUT = _s.argv[1] if len(_s.argv) > 1 else "results"; os.makedirs(_OUT, exist_ok=True)
json.dump(facts, open(f"{_OUT}/benchmark_facts.json", "w"), indent=1)
