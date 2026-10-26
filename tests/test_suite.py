"""
File: tests/test_suite.py

Purpose:
Comprehensive executable unit and regression test suite covering benchmark definitions,
acquisition rules, experimental designs, and statistical calculations.

Role in the experiment:
Primary test suite verifying the mathematical, numerical, and algorithmic correctness
of all components in the constrained Bayesian optimization pipeline.

Key details:
- Unit tests: Verifies benchmark formulas against independent polar derivations, checks
  numerical stability, and tests bounds roundtrip.
- Acquisition tests: Compares acquisitions with authors' reference code, checks fallback
  branches, and tests EI numerical integration.
- Statistical tests: Validates Wilson intervals, quantiles, Wilcoxon signed-rank logic,
  and Holm adjustments against known values.
- Outputs: Writes results/test_results.json recording test counts and code fingerprint.
"""
import os, sys, json
import numpy as np
from scipy import integrate
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bo
from bo import acq_cei, acq_cbolcb, acq_ecei, acq_cei_altfb, pof, ei_min, run, LAMBDA_LCB
from benchmark import *
from designs import make_design

FACTS = json.load(open("results/benchmark_facts.json"))


def load_authors(fname):
    ns = {"__name__": "authors"}
    exec(open(fname).read().split("parser = argparse")[0], ns)
    return ns


class StubGP:
    def __init__(self, mean, sd): self.m, self.s = mean, sd
    def predict(self, X, return_std=True): return self.m(X), self.s(X)


class StubS:
    def __init__(self, mf, sf, mc, sc): self.t = (mf, sf, mc, sc)
    def predict(self, U): return tuple(g(U) for g in self.t)


rng = np.random.default_rng(0)
U = rng.random((500, 2))
mf = lambda U: -1.5 + np.sin(4 * U[:, 0]) * U[:, 1]
sf = lambda U: 0.05 + 0.4 * U[:, 1] ** 2
mc = lambda U: 1.2 * (U[:, 0] - 0.5)
sc = lambda U: 0.05 + 0.5 * U[:, 0] * U[:, 1]
S = StubS(mf, sf, mc, sc)
f = rng.normal(-1, 0.6, 12); c = rng.normal(0.2, 1, 12)
AE = load_authors("authors_code/ECEI_MT.py"); AM = load_authors("authors_code/MLCB_MT.py")


# ---------------------------------------------------------------- benchmark
def test_benchmark_matches_authors_code():
    X = from_unit(rng.random((5000, 2)))
    for A, name in ((AE, "ECEI_MT"), (AM, "MLCB_MT")):
        do = np.abs(np.array([A["objective"](x) for x in X]) - objective(X)).max()
        # ECEI_MT.py uses atan2(x1, x2 + 1e-9) exactly as we do; MLCB_MT.py omits the 1e-9, so it can differ by ~1e-9/|x2| near x2 = 0
        keep = np.abs(X[:, 1]) > (0 if name == "ECEI_MT" else 1e-3)
        dc = np.abs(np.array([A["constraint"](x) for x in X[keep]]) - constraint(X[keep])).max()
        assert do < 1e-12 and dc < (1e-9 if name == "ECEI_MT" else 1e-5), (name, do, dc)


def test_feasibility_convention_and_direction():
    assert constraint(np.array([0.0, 0.0]))[0] <= 0                # origin lies inside the Townsend region
    assert constraint(np.array([2.5, -2.5]))[0] > 0                # far corner is infeasible
    xs = np.array(FACTS["x_star"])
    assert constraint(xs)[0] <= 0                                  # reported optimum is (strictly) feasible
    assert abs(objective(xs)[0] - FACTS["f_star"]) < 1e-12
    # minimisation: reported optimum beats every feasible point of a 2e6-point Monte-Carlo sample
    X = from_unit(np.random.default_rng(5).random((2_000_000, 2)))
    ok = constraint(X) <= 0
    assert objective(X)[ok].min() >= FACTS["f_star"] - 1e-9
    assert abs(ok.mean() - FACTS["feasible_fraction"]) < 5 * FACTS["feasible_fraction_se"] * 1.5


def test_bounds_roundtrip():
    X = from_unit(rng.random((100, 2)))
    assert np.allclose(from_unit(to_unit(X)), X) and (X >= L_BOUNDS).all() and (X <= U_BOUNDS).all()


# ---------------------------------------------------------------- acquisition vs reference code
def test_ecei_matches_reference_main_branch():
    best, M = AE["best_params"](f, c)
    theirs = AE["ECEI_acq"](U, best, M, 0.0, 1.0, StubGP(mf, sf), StubGP(mc, sc), 1)
    ours, mode = acq_ecei(S, U, f, c)
    assert mode == 0 and np.allclose(theirs, ours, atol=1e-12)


def test_ecei_internal_fallback_branch():
    f2 = np.append(np.full(11, -1.0), -100.0); c2 = np.append(np.ones(11), -1.0)
    best2, M2 = AE["best_params"](f2, c2)
    theirs = AE["ECEI_acq"](U, best2, M2, 0.0, 1.0, StubGP(mf, sf), StubGP(mc, sc), 2)
    ours, mode = acq_ecei(S, U, f2, c2)
    assert mode == 2 and np.allclose(theirs, ours, atol=1e-12)


def test_cbolcb_matches_reference():
    theirs = AM["MLCB_acq"](U, np.abs(f).max(), StubGP(mf, sf), StubGP(mc, sc))
    ours, _ = acq_cbolcb(S, U, f, c)
    assert np.allclose(theirs, ours, atol=1e-12)
    assert abs(LAMBDA_LCB - 1 / norm.cdf(0.05)) < 1e-15



def test_ecei_matches_supplement_equation():
    """Supplement 'Some_proofs.pdf' Eq. (2): ECEI = theta*[ (y*-M-u_f)Phi(.)+s_f phi(.) ] + (1-theta)*[ (y*-u_f)Phi(.)+s_f phi(.) ],
    theta = 1 - prod_i Phi(-u_ci/s_ci). Checked against our implementation with y* = best penalised value (the code's convention;
    the supplement does not say how y* is defined when no feasible point exists)."""
    best, M = AE["best_params"](f, c)
    u_f, s_f, u_c, s_c = mf(U), sf(U), mc(U), sc(U)
    theta = 1 - norm.cdf(-u_c / s_c)
    z1 = (best - M - u_f) / s_f; z2 = (best - u_f) / s_f
    eq2 = theta * ((best - M - u_f) * norm.cdf(z1) + s_f * norm.pdf(z1)) + (1 - theta) * ((best - u_f) * norm.cdf(z2) + s_f * norm.pdf(z2))
    ours, _ = acq_ecei(S, U, f, c)
    assert np.allclose(-ours, eq2, rtol=1e-6, atol=1e-8)       # 1e-9 sd stabilisers in our pof/ei_min cause ~1e-9 differences


# ---------------------------------------------------------------- cEI formulas and branches
def test_ei_formula_against_numerical_integration():
    for best, mu, sd in [(0.0, 0.3, 0.5), (-1.0, -1.2, 0.2), (2.0, 1.0, 1.5), (0.0, 5.0, 0.3)]:
        num = integrate.quad(lambda y: max(best - y, 0.0) * norm.pdf(y, mu, sd), mu - 12 * sd, mu + 12 * sd, limit=200)[0]
        assert abs(float(ei_min(best, np.array(mu), np.array(sd))) - num) < 1e-8, (best, mu, sd)


def test_pof_properties():
    assert abs(float(pof(np.array(0.0), np.array(1.0))) - 0.5) < 1e-12
    m = np.linspace(-3, 3, 50); p = pof(m, np.ones(50))
    assert np.all(np.diff(p) < 0) and ((p >= 0) & (p <= 1)).all()           # decreasing in predicted constraint value
    assert abs(float(pof(np.array(-1.0), np.array(1.0))) - norm.cdf(1.0)) < 1e-8   # sd + 1e-9 stabiliser (as in authors' code)
    assert float(pof(np.array(-1.0), np.array(0.0))) > 1 - 1e-9 and float(pof(np.array(1.0), np.array(0.0))) < 1e-9   # zero sd


def test_cei_normal_branch_is_ei_times_pof():
    fe, ce = np.array([-1.0, -0.5, 0.2]), np.array([-0.3, 0.4, -0.1])      # feasible incumbents: f = -1.0 and 0.2
    sc_, mode = acq_cei(S, U, fe, ce)
    expect = -(ei_min(-1.0, mf(U), sf(U)) * pof(mc(U), sc(U)))
    assert mode == 0 and np.allclose(sc_, expect)


def test_cei_fallback_no_feasible_observation():
    fe, ce = np.array([-1.0, -0.5]), np.array([0.3, 0.4])
    sc_, mode = acq_cei(S, U, fe, ce)
    assert mode == 1 and np.allclose(sc_, -pof(mc(U), sc(U)))
    i = int(np.argmin(sc_)); assert pof(mc(U), sc(U))[i] == pof(mc(U), sc(U)).max()   # minimising score = maximising PoF


def test_cei_underflow_branch():
    fe, ce = np.array([-100.0, -1.0]), np.array([-1.0, 0.5])
    sc_, mode = acq_cei(S, U, fe, ce)
    assert mode == 2 and np.allclose(sc_, -pof(mc(U), sc(U)))


def test_altfb_branches():
    fe, ce = np.array([-1.0, -0.5]), np.array([0.3, 0.4])
    sc_, mode = acq_cei_altfb(S, U, fe, ce)
    assert mode == 1 and np.allclose(sc_, -(ei_min(-1.0, mf(U), sf(U)) * pof(mc(U), sc(U))))
    fe2, ce2 = np.array([-1.0, -0.5]), np.array([-0.3, 0.4])               # feasible exists -> identical to cEI
    assert np.allclose(acq_cei_altfb(S, U, fe2, ce2)[0], acq_cei(S, U, fe2, ce2)[0])


def test_forced_mode_is_consistent_between_batch_and_single_point():
    """Regression test for a bug in the first version: the branch was re-decided at every single-point evaluation inside
    L-BFGS-B, so the optimised objective could switch formula/scale during refinement."""
    cases = [(acq_cei, np.array([-100.0, -1.0]), np.array([-1.0, 0.5])), (acq_cei, np.array([-1.0, -0.5]), np.array([0.3, 0.4])),
             (acq_ecei, np.append(np.full(11, -1.0), -100.0), np.append(np.ones(11), -1.0)), (acq_ecei, f, c),
             (acq_cei_altfb, np.array([-1.0, -0.5]), np.array([0.3, 0.4]))]
    for fn, ff, cc in cases:
        batch, mode = fn(S, U, ff, cc)
        for i in (0, 7, 123, 499):
            single, m2 = fn(S, U[i:i + 1], ff, cc, mode)
            assert m2 == mode and np.isclose(single[0], batch[i], rtol=1e-12, atol=1e-12), (fn.__name__, mode, i)


# ---------------------------------------------------------------- sign / direction / numerical stability
def test_acquisition_directions():
    z = np.zeros(2)
    lo = StubS(lambda U: np.array([-2.0, -1.0]), lambda U: np.full(2, 0.3), lambda U: np.full(2, -3.0), lambda U: np.full(2, 0.2))
    fe, ce = np.array([-0.5, 0.2]), np.array([-1.0, 1.0])
    for fn in (acq_cei, acq_ecei, acq_cbolcb):                              # lower predicted f (point 0) must score lower (better)
        s_, _ = fn(lo, z.reshape(2, 1).repeat(2, 1), fe, ce); assert s_[0] < s_[1], fn.__name__
    hi_sd = StubS(lambda U: np.zeros(2), lambda U: np.array([0.1, 1.0]), lambda U: np.full(2, -3.0), lambda U: np.full(2, 0.2))
    for fn in (acq_cei, acq_ecei, acq_cbolcb):                              # more predictive uncertainty (point 1) -> better, all else equal
        s_, _ = fn(hi_sd, z.reshape(2, 1).repeat(2, 1), fe, ce); assert s_[1] < s_[0], fn.__name__
    infeas = StubS(lambda U: np.full(2, -1.0), lambda U: np.full(2, 0.3), lambda U: np.array([-4.0, 4.0]), lambda U: np.full(2, 0.2))
    for fn in (acq_cei, acq_ecei, acq_cbolcb):                              # likely-feasible (point 0) beats likely-infeasible (point 1)
        s_, _ = fn(infeas, z.reshape(2, 1).repeat(2, 1), fe, ce); assert s_[0] < s_[1], fn.__name__


def test_zero_variance_and_numerical_stability():
    zs = StubS(lambda U: np.full(4, -1.0), lambda U: np.zeros(4), lambda U: np.array([-50.0, -1.0, 1.0, 50.0]), lambda U: np.zeros(4))
    fe, ce = np.array([-0.5, 0.2, -2.0]), np.array([-1.0, 1.0, -0.2])
    X4 = np.zeros((4, 2))
    for fn in (acq_cei, acq_ecei, acq_cbolcb, acq_cei_altfb):
        s_, _ = fn(zs, X4, fe, ce); assert np.all(np.isfinite(s_)), fn.__name__
    big = StubS(lambda U: np.full(4, 1e6), lambda U: np.full(4, 1e-12), lambda U: np.full(4, -1e6), lambda U: np.full(4, 1e-12))
    for fn in (acq_cei, acq_ecei, acq_cbolcb):
        s_, _ = fn(big, X4, fe, ce); assert np.all(np.isfinite(s_)), fn.__name__
    assert float(ei_min(0.0, np.array(-1.0), np.array(0.0))) == 1.0          # sd=0: EI = max(best - mu, 0)
    assert float(ei_min(0.0, np.array(1.0), np.array(0.0))) == 0.0


# ---------------------------------------------------------------- designs and fairness
def test_designs_satisfy_condition_and_are_reproducible():
    for n0, seeds in ((4, range(1, 31)), (5, range(1, 6))):
        for cond in "FI":
            if n0 == 5 and cond == "F": continue
            for s in seeds:
                X, att = make_design(s, n0, cond)
                nf = int((constraint(X) <= 0).sum())
                assert X.shape == (n0, 2) and (X >= L_BOUNDS).all() and (X <= U_BOUNDS).all()
                assert (nf >= 1) if cond == "F" else (nf == 0), (n0, cond, s, nf)
                U_ = np.sort(to_unit(X), axis=0)                                  # Latin-hypercube property: one point per stratum
                assert all(np.array_equal(np.floor(U_[:, j] * n0).astype(int), np.arange(n0)) for j in range(2))
                X2, att2 = make_design(s, n0, cond); assert np.array_equal(X, X2) and att == att2


def test_methods_share_design_budget_and_candidate_sets():
    X0, _ = make_design(3, 4, "I"); recorded = {}
    orig = bo.candidate_set
    for m in ("cEI", "CBOLCB", "ECEI"):
        seeds_seen = []
        def spy(seed, _l=seeds_seen): _l.append(seed); return orig(seed)
        bo.candidate_set = spy
        r = run(m, X0, 3, budget=3)
        recorded[m] = (list(seeds_seen), r)
    bo.candidate_set = orig
    base = recorded["cEI"][0]
    assert base == [10 ** 6 * 3 + k for k in range(3)]
    for m, (sd, r) in recorded.items():
        assert sd == base, m                                                    # same candidate-set seed at each iteration
        assert np.array_equal(r["X"][:4], X0) and r["X"].shape[0] == 4 + 3 and len(r["f"]) == 7 and len(r["c"]) == 7   # same design, same budget
        assert np.allclose(r["f"], objective(r["X"])) and np.allclose(r["c"], constraint(r["X"]))
        assert (r["X"] >= L_BOUNDS - 1e-9).all() and (r["X"] <= U_BOUNDS + 1e-9).all()


# ---------------------------------------------------------------- independent benchmark / optimum checks
def test_constraint_boundary_polar_independent():
    """Independent derivation: with t = atan2(x1, x2) a boundary point is (R(t) sin t, R(t) cos t), where
    R(t)^2 = (2cos t - .5cos2t - .25cos3t - .125cos4t)^2 + (2 sin t)^2. Then c = 0 on it, c < 0 inside, c > 0 outside."""
    t = np.linspace(-np.pi + 0.05, np.pi - 0.05, 41)
    R = np.sqrt((2*np.cos(t) - .5*np.cos(2*t) - .25*np.cos(3*t) - .125*np.cos(4*t))**2 + (2*np.sin(t))**2)
    P = np.c_[R * np.sin(t), R * np.cos(t)]
    assert np.abs(constraint(P)).max() < 1e-6                     # on the boundary (1e-9 in atan2 gives ~1e-9/|x2| deviations)
    assert (constraint(0.9 * P) < 0).all() and (constraint(1.1 * P) > 0).all()


def test_objective_hand_values():
    import math
    assert abs(objective(np.array([0.1, 0.0]))[0] - (-1.0 - 0.1 * math.sin(0.3))) < 1e-15          # cos(0)=1
    assert abs(objective(np.array([0.0, 1.0]))[0] - (-math.cos(-0.1) ** 2)) < 1e-15                # x1 = 0 kills the second term


def test_reported_optimum_is_on_the_boundary_and_numerically_supported():
    xs = np.array(FACTS["x_star"])
    assert (xs >= L_BOUNDS).all() and (xs <= U_BOUNDS).all()
    assert constraint(xs)[0] <= 0 and abs(constraint(xs)[0]) < 1e-12              # feasible AND on the boundary (not strictly interior)
    assert constraint(xs * (1 + 1e-9))[0] > 0 and constraint(xs * (1 - 1e-9))[0] < 0   # crossing the boundary within 1e-9 relative
    A = json.load(open("results/optimum_audit.json"))
    assert A["f_star"] == FACTS["f_star"] and A["c_strictly_negative"] is False
    assert A["grid_3000x3000_best_feasible_f"] >= FACTS["f_star"] - 1e-12 and A["monte_carlo_min_feasible_f_2e6"] >= FACTS["f_star"] - 1e-12
    assert abs(A["differential_evolution_best_feasible"]["f"] - FACTS["f_star"]) < 1e-8
    assert min(d["f"] for d in A["slsqp_multistart"]["distinct_local_minima"]) > FACTS["f_star"] - 1e-3


# ---------------------------------------------------------------- statistics against independently computed values
def test_wilson_matches_hand_formula():
    from stats_utils import wilson
    z = 1.959963984540054
    for k, n in [(30, 30), (0, 30), (11, 30), (17, 30), (3, 100000)]:
        p = k / n; den = 1 + z * z / n; ctr = (p + z * z / (2 * n)) / den; hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        lo, hi = wilson(k, n); assert abs(lo - (ctr - hw)) < 1e-9 and abs(hi - (ctr + hw)) < 1e-9, (k, n)
    assert abs(wilson(30, 30)[0] - 0.8864) < 5e-4                                 # value quoted in the report (lower bound 0.89)


def test_quantile_inf():
    from stats_utils import quantile_inf
    x = np.random.default_rng(3).normal(size=37)
    assert np.allclose(quantile_inf(x, [25, 50, 75]), np.percentile(x, [25, 50, 75]))      # equals numpy when finite
    y = np.array([1.0, 2.0, 3.0, np.inf, np.inf])
    q = quantile_inf(y, [0, 25, 50, 60, 75, 100])
    assert q[0] == 1 and q[1] == 2 and q[2] == 3 and np.isinf(q[3]) and np.isinf(q[4]) and np.isinf(q[5]) and not np.isnan(q).any()
    assert np.isinf(quantile_inf(np.array([np.inf, np.inf, np.inf]), 50)[0])               # numpy would give NaN here


def test_holm_known_values():
    from stats_utils import holm
    assert np.allclose(holm(np.array([0.01, 0.04, 0.03, 0.005])), [0.03, 0.06, 0.06, 0.02])   # R: p.adjust(..., "holm")
    r = holm(np.array([0.5, np.nan, 0.01])); assert np.isnan(r[1]) and np.allclose(r[[0, 2]], [0.5, 0.02])


def test_wilcoxon_matches_exact_enumeration():
    from stats_utils import wilcoxon_p
    import itertools
    d = np.array([1.5, -2.5, 3.5, 4.5, -5.5, 6.5, 7.5, 8.5])                   # no ties, no zeros
    ranks = np.arange(1, 9); w_obs = ranks[d > 0].sum()
    ws = np.array([sum(r for r, s in zip(ranks, signs) if s) for signs in itertools.product([0, 1], repeat=8)])
    mean = ws.mean(); p_exact = float((np.abs(ws - mean) >= abs(w_obs - mean) - 1e-12).mean())
    assert abs(wilcoxon_p(d)[0] - p_exact) < 1e-12
    assert np.isnan(wilcoxon_p(np.array([1.0, -2, 0, 0, 3, 0, 0]))[0])        # fewer than 6 non-zero differences -> no test


def test_paired_metrics_failures_are_conditional_and_never_substituted():
    from stats_utils import paired_metrics, wilcoxon_p
    from scipy.stats import binomtest
    rng_ = np.random.default_rng(0); n = 20; B_ = 50
    gap_b = rng_.uniform(0.001, 0.4, n); gap_a = gap_b + rng_.normal(0, 0.05, n)
    succ_a = np.ones(n, int); succ_a[[3, 7]] = 0; gap_a[[3, 7]] = np.inf; t_a = rng_.integers(1, 8, n).astype(float); t_a[[3, 7]] = B_ + 1
    succ_b = np.ones(n, int); succ_b[[11]] = 0; gap_b[11] = np.inf; t_b = rng_.integers(1, 8, n).astype(float); t_b[11] = B_ + 1
    m = paired_metrics(t_a, t_b, succ_a, succ_b, gap_a, gap_b, B_, np.random.default_rng(1))
    both = (succ_a == 1) & (succ_b == 1); d = (gap_a - gap_b)[both]
    assert m["n_both_success"] == both.sum() == 17 and m["n_A_only_success"] == 1 and m["n_B_only_success"] == 2 and m["n_neither_success"] == 0
    assert m["p_gap"] == wilcoxon_p(d)[0] and np.isfinite(m["d_gap_median_both"]) and m["d_gap_median_both"] == np.median(d)   # only both-success pairs
    assert m["n_censored_pairs"] == 3 and abs(m["mcnemar_p"] - binomtest(1, 3, 0.5).pvalue) < 1e-15
    assert m["d_ttff_ci_lo"] <= m["d_ttff_ci_hi"] and m["d_ttff_median"] == np.median(t_a - t_b)
    # pairs where one run failed must not influence the gap test at all
    g2a, g2b = gap_a.copy(), gap_b.copy(); g2b[[3, 7]] = 123.0                  # B's value on seeds where A failed is irrelevant
    assert paired_metrics(t_a, t_b, succ_a, succ_b, g2a, g2b, B_, np.random.default_rng(1))["p_gap"] == m["p_gap"]


def test_bootstrap_ci_single_resample_and_deterministic():
    from stats_utils import boot_ci
    x = np.random.default_rng(2).normal(size=30)
    a = boot_ci(x, np.random.default_rng(9)); b = boot_ci(x, np.random.default_rng(9))
    assert a == b and a[0] <= np.median(x) <= a[1]
    idx = np.random.default_rng(9).integers(0, 30, (10000, 30)); meds = np.median(x[idx], axis=1)       # independent recomputation
    assert np.allclose(a, np.percentile(meds, [2.5, 97.5]))


def test_run_fingerprint_logic():
    import run_experiment as RE
    c1 = RE.make_config(30, 50, ["main_F"]); c2 = RE.make_config(30, 51, ["main_F"]); c3 = RE.make_config(30, 50, ["main_F"])
    assert RE.semantic(c1) == RE.semantic(c3) and RE.semantic(c1) != RE.semantic(c2)
    c4 = dict(c1); c4["code_hash"] = "deadbeef"; assert RE.semantic(c4) != RE.semantic(c1)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0; failed_set = set()
    for name, fn in tests:
        try:
            fn(); print("PASS", name)
        except Exception as e:
            failed += 1; failed_set.add(name); print("FAIL", name, "->", repr(e)[:300])
    print(f"\n{len(tests)-failed}/{len(tests)} tests passed")
    import time, hashlib
    h = hashlib.sha256()
    for fn in ("benchmark.py", "designs.py", "bo.py"): h.update(open(fn, "rb").read())
    json.dump(dict(passed=len(tests) - failed, total=len(tests), failed_names=[n for n, _ in tests if n in failed_set], code_hash=h.hexdigest()[:16],
                   time=time.strftime("%Y-%m-%d %H:%M:%S")), open("results/test_results.json", "w"), indent=1)
    sys.exit(1 if failed else 0)
