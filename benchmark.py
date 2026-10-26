"""Modified Townsend benchmark (2-D, one inequality constraint c(x) <= 0).

Objective and constraint are copied from the authors' released code
(Code.zip: ECEI_MT.py / MLCB_MT.py; figure5.py). Minimisation, feasible iff c(x) <= 0.
"""
import numpy as np

L_BOUNDS = np.array([-2.25, -2.5])
U_BOUNDS = np.array([2.5, 1.75])
DIM = 2
# Best known constrained optimum quoted in the authors' figure5.R; refined numerically in make_benchmark_facts.py
X_STAR_AUTHORS = np.array([2.0052938, 1.1944509])


def objective(x):
    x = np.atleast_2d(x)
    x1, x2 = x[:, 0], x[:, 1]
    return -(np.cos((x1 - 0.1) * x2)) ** 2 - x1 * np.sin(3 * x1 + x2)


def constraint(x):
    """c(x) <= 0 is feasible. Polar form: with t = atan2(x1, x2), feasible iff
    r^2 <= (2cos t - .5cos 2t - .25cos 3t - .125cos 4t)^2 + (2 sin t)^2
    (a Townsend-type heart-shaped region, star-shaped around the origin)."""
    x = np.atleast_2d(x)
    x1, x2 = x[:, 0], x[:, 1]
    t = np.arctan2(x1, x2 + 1e-9)
    R2 = (2*np.cos(t) - 0.5*np.cos(2*t) - 0.25*np.cos(3*t) - 0.125*np.cos(4*t)) ** 2 + (2*np.sin(t)) ** 2
    return x1 ** 2 + x2 ** 2 - R2


def to_unit(X):
    return (np.atleast_2d(X) - L_BOUNDS) / (U_BOUNDS - L_BOUNDS)


def from_unit(U):
    return L_BOUNDS + np.atleast_2d(U) * (U_BOUNDS - L_BOUNDS)
