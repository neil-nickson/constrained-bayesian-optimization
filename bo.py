"""Compact constrained-BO harness: one shared GP pipeline + three acquisition rules.

Methods (all minimise f subject to c <= 0, single constraint):
  cEI    : EI(x; y*) * P(c<=0)  with y* = best feasible observation (Gardner et al. 2014).
           FALLBACK (our experimental design choice, NOT part of Gardner's formulation):
           while no feasible point is observed, maximise P(c<=0) alone (the acquisition used by
           Gelbart et al. 2014, Sec. 3.2, but triggered by 'no feasible *observation*').
  CBOLCB : M-based LCB of Upadhye & Chowdhury (2024), ported from the authors' MLCB_MT.py.
  ECEI   : feasibility-infeasibility weighted improvement of Chowdhury & Upadhye (2026), ported from
           the authors' ECEI_MT.py, including the authors' own internal fallback (flag 2).
"""
import warnings
import numpy as np
from scipy.stats import norm, qmc
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from benchmark import objective, constraint, to_unit, from_unit

warnings.filterwarnings("ignore")
LS_BOUNDS = (0.1, 5.0)       # lengthscale bounds on the unit-cube inputs (stabilises tiny-n fits)
N_CAND = 1000                # Latin-hypercube candidates per acquisition optimisation
N_REFINE = 2                 # L-BFGS-B restarts from the best candidates
LAMBDA_LCB = 1.0 / norm.cdf(0.05)   # lambda = 1/Phi(0.05), exactly as in the authors' code


# ----------------------------------------------------------------------------- surrogate
def sgn_log(c):
    return np.sign(c) * np.log1p(np.abs(c))   # sign-preserving transform used in the authors' ECEI code


class Surrogates:
    """Independent GPs: objective (z-scored) and transformed constraint (scaled, NOT centred, so the
    feasibility threshold stays at 0). Matern-5/2 ARD * amplitude, marginal-likelihood fit."""

    def __init__(self, seed):
        self.seed = seed

    def _gp(self, k):
        ker = C(1.0, (1e-2, 1e2)) * Matern(length_scale=[0.5, 0.5], length_scale_bounds=LS_BOUNDS, nu=2.5)
        return GaussianProcessRegressor(ker, alpha=1e-6, n_restarts_optimizer=1, random_state=self.seed * 1000 + k)

    def fit(self, U, f, c, k):
        self.fm, self.fs = f.mean(), f.std() + 1e-12
        ct = sgn_log(c)
        self.cs = np.sqrt(np.mean(ct ** 2)) + 1e-12   # uncentred RMS scale: keeps the feasibility threshold at 0
        self.gf, self.gc = self._gp(k), self._gp(k + 7)
        self.gf.fit(U, (f - self.fm) / self.fs)
        self.gc.fit(U, ct / self.cs)

    def predict(self, U):
        mf, sf = self.gf.predict(U, return_std=True)
        mc, sc = self.gc.predict(U, return_std=True)
        return mf * self.fs + self.fm, sf * self.fs, mc, sc   # f in original units; c in scaled-transformed units


def pof(mc, sc):
    return norm.cdf(-mc / (sc + 1e-9))      # P(c <= 0)


def ei_min(best, mu, sd):
    z = (best - mu) / (sd + 1e-9)
    return (best - mu) * norm.cdf(z) + sd * norm.pdf(z)


# ----------------------------------------------------------------------------- acquisition rules
# Each rule returns (score_to_MINIMISE, mode). `mode` selects the branch (0 = main formula, >0 = fallback branch).
# mode=None -> the branch is DECIDED from the data/candidates passed in (done once, on the full candidate set);
# mode=int  -> that branch is FORCED. propose() decides on the candidate set and forces the same branch during
# L-BFGS-B refinement, so the optimised objective never switches formula (and scale) mid-refinement.
def acq_cei(S, U, f, c, mode=None):
    feas = c <= 0
    mf, sf, mc, sc = S.predict(U)
    p = pof(mc, sc)
    a = ei_min(f[feas].min(), mf, sf) * p if feas.any() else None
    if mode is None:
        mode = 1 if a is None else (2 if np.all(a <= 0) else 0)   # 1: no feasible observation; 2: EI*PoF underflow
    return (-a if mode == 0 else -p), mode          # <-- cEI FALLBACK (modes 1, 2): maximise P(c<=0) alone


def acq_cbolcb(S, U, f, c, mode=None):
    mf, sf, mc, sc = S.predict(U)
    M = np.abs(f).max()
    p = pof(mc, sc)
    mu = mf + M * (1 - p)
    sd = np.sqrt(sf ** 2 + M ** 2 * p * (1 - p))
    return mu - LAMBDA_LCB * sd, 0


def acq_ecei(S, U, f, c, mode=None):
    mf, sf, mc, sc = S.predict(U)
    M = np.abs(f).max()
    best = (f + M * (c > 0)).min()           # best value of the penalised objective F = f + M*(1 - 1[c<=0])
    theta = 1 - pof(mc, sc)                  # P(infeasible)
    a = ei_min(best, mf + M, sf) * theta + ei_min(best, mf, sf) * (1 - theta)
    if mode is None:                          # authors' internal fallback ('flag == 2'): EI > 0 for <= 1% of candidates
        mode = 2 if (a > 0).sum() <= 1e-2 * len(a) else 0
    return (mf + M * theta if mode == 2 else -a), mode


def acq_cei_altfb(S, U, f, c, mode=None):
    """Sensitivity variant of the cEI fallback: while no feasible point exists, use y* = min of ALL observed f
    (feasible or not) in EI * PoF, instead of maximising PoF alone. Identical to cEI once a feasible point exists."""
    if (c <= 0).any():
        return acq_cei(S, U, f, c, mode)
    mf, sf, mc, sc = S.predict(U)
    p = pof(mc, sc)
    a = ei_min(f.min(), mf, sf) * p
    if mode is None:
        mode = 1 if np.any(a > 0) else 2
    return (-a if mode == 1 else -p), mode


ACQ = {"cEI": acq_cei, "CBOLCB": acq_cbolcb, "ECEI": acq_ecei, "cEI_altfb": acq_cei_altfb}


def candidate_set(rng_seed):
    return qmc.LatinHypercube(d=2, seed=rng_seed).random(N_CAND)


def propose(method, S, U, f, c, rng_seed):
    """Shared acquisition optimiser: LHS candidates (common random numbers across methods) + L-BFGS-B."""
    cand = candidate_set(rng_seed)
    fn = ACQ[method]
    scores, mode = fn(S, cand, f, c)                 # branch decided once, on the candidate set
    order = np.argsort(scores)[:N_REFINE]
    best_x, best_s = cand[order[0]], scores[order[0]]
    for i in order:
        r = minimize(lambda u: fn(S, u.reshape(1, 2), f, c, mode)[0][0], cand[i], bounds=[(0, 1)] * 2, method="L-BFGS-B")
        if r.fun < best_s:
            best_x, best_s = r.x, r.fun
    return best_x, mode


# ----------------------------------------------------------------------------- one run
def run(method, X0, seed, budget=50):
    X = np.array(X0, float)
    f = objective(X); c = constraint(X)
    flags = []
    S = Surrogates(seed)
    for k in range(budget):
        U = to_unit(X)
        S.fit(U, f, c, k)
        u, flag = propose(method, S, U, f, c, rng_seed=10**6 * seed + k)   # same candidate set for every method
        x = from_unit(u)
        X = np.vstack([X, x]); f = np.append(f, objective(x)); c = np.append(c, constraint(x)); flags.append(flag)
    return dict(X=X, f=f, c=c, flags=np.array(flags))
