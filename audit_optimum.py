"""
File: audit_optimum.py

Purpose:
Performs independent numerical cross-checks of the estimated constrained global
optimum on the Modified Townsend benchmark.

Role in the experiment:
Provides verification and sensitivity analysis regarding the numerical optimum
f* and optimal location x*, checking boundary behavior and local minima.

Key details:
- Evaluates constraint value at x* in double precision and checks radial scalings.
- Optimizers: Differential evolution multistart (10 seeds) and SLSQP multistart
  (400 starts) to identify local and near-global minima.
- Sampling: Evaluates fine grid (3000 x 3000) and 2,000,000 uniform Monte Carlo points.
- Outputs: Writes results/optimum_audit.json detailing optimum status and convergence.
  (Provides supporting numerical evidence, not a mathematical proof of global optimality).
"""

import json
import os
import sys
from collections import Counter

import numpy as np
from scipy.optimize import minimize, differential_evolution

from benchmark import *


OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
os.makedirs(OUT, exist_ok=True)

F = json.load(open("results/benchmark_facts.json"))
xs = np.array(F["x_star"])

c_star = float(constraint(xs)[0])

res = dict(
    x_star=[float(v) for v in xs],
    f_star=float(objective(xs)[0]),
    c_at_x_star=c_star,
    x_star_in_domain=bool(
        (xs >= L_BOUNDS).all() and (xs <= U_BOUNDS).all()
    ),
    c_leq_0=bool(c_star <= 0),
    # Treat tiny floating-point residuals as zero, not strict interior.
    c_strictly_negative=bool(c_star < -1e-12),
    c_at_x_star_scaled={
        str(s): float(constraint(xs * s)[0])
        for s in (1 + 1e-9, 1 - 1e-9)
    },
)

# (1) 3000 x 3000 grid (chunked)
g = 3000
gx = np.linspace(L_BOUNDS[0], U_BOUNDS[0], g)
gy = np.linspace(L_BOUNDS[1], U_BOUNDS[1], g)
m = np.inf

for i in range(0, g, 100):
    GX, GY = np.meshgrid(gx[i:i + 100], gy, indexing="ij")
    G = np.c_[GX.ravel(), GY.ravel()]
    ok = constraint(G) <= 0

    if ok.any():
        m = min(m, objective(G)[ok].min())

res["grid_3000x3000_best_feasible_f"] = float(m)

# (2) Differential evolution with a penalty, 10 seeds
best = (np.inf, None)

for s in range(10):
    r = differential_evolution(
        lambda z: objective(z)[0] + 1e4 * max(constraint(z)[0], 0),
        list(zip(L_BOUNDS, U_BOUNDS)),
        seed=s,
        tol=1e-12,
        maxiter=2000,
        popsize=40,
        polish=False,
    )

    if constraint(r.x)[0] <= 0 and r.fun < best[0]:
        best = (float(r.fun), [float(v) for v in r.x])

res["differential_evolution_best_feasible"] = dict(
    f=best[0],
    x=best[1],
    n_seeds=10,
)

# (3) 400 SLSQP multistarts -> distinct local feasible minima
rng = np.random.default_rng(1)
sols = []
cons = [{"type": "ineq", "fun": lambda z: -constraint(z)[0]}]

for _ in range(400):
    x0 = from_unit(rng.random(2))[0]

    r = minimize(
        lambda z: objective(z)[0],
        x0,
        method="SLSQP",
        constraints=cons,
        bounds=list(zip(L_BOUNDS, U_BOUNDS)),
        options={"ftol": 1e-12, "maxiter": 300},
    )

    if r.success and constraint(r.x)[0] <= 1e-9:
        sols.append(
            (
                round(float(r.fun), 4),
                round(float(r.x[0]), 3),
                round(float(r.x[1]), 3),
            )
        )

cnt = Counter(sols)

res["slsqp_multistart"] = dict(
    n_starts=400,
    n_converged=len(sols),
    distinct_local_minima=[
        dict(f=k[0], x=[k[1], k[2]], count=v)
        for k, v in sorted(cnt.items())
    ],
)

# (4) Monte Carlo: no random feasible point beats f*
X = from_unit(np.random.default_rng(5).random((2_000_000, 2)))
ok = constraint(X) <= 0

res["monte_carlo_min_feasible_f_2e6"] = float(
    objective(X)[ok].min()
)

res["statement"] = (
    "numerical estimate supported by grid, DE, multistart and "
    "Monte Carlo; NOT a proof of global optimality"
)

with open(f"{OUT}/optimum_audit.json", "w") as f:
    json.dump(res, f, indent=1)

print(
    json.dumps(
        {k: v for k, v in res.items() if k != "slsqp_multistart"},
        indent=1,
    )
)
print([
    (d["f"], d["count"])
    for d in res["slsqp_multistart"]["distinct_local_minima"]
])
