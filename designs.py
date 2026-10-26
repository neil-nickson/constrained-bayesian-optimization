"""Initial-design construction with rejection sampling (documented in README / report)."""
import numpy as np
from scipy.stats import qmc
from benchmark import constraint, from_unit

MAX_ATTEMPTS = 5_000_000


def make_design(seed, n0, condition):
    """condition 'F': >=1 feasible point in the initial design; 'I': zero feasible points.
    Latin hypercube designs (scrambled) are drawn one after another from a single seeded
    generator until the condition holds. Returns (X in original units, n_attempts)."""
    rng = np.random.default_rng(10_000 + seed)
    sampler = qmc.LatinHypercube(d=2, seed=rng)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        X = from_unit(sampler.random(n=n0))
        nfeas = int((constraint(X) <= 0).sum())
        if (condition == "F" and nfeas >= 1) or (condition == "I" and nfeas == 0):
            return X, attempt
    raise RuntimeError(f"no design found for n0={n0} condition={condition} seed={seed}")


