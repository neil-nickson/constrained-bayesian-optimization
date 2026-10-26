"""
File: stats_utils.py

Purpose:
Provides statistical helper functions for computing confidence intervals, quantiles,
paired tests, and multiple-testing corrections.

Role in the experiment:
Implements statistical procedures used by analyze.py and verified by tests,
ensuring consistent numerical calculations across all evaluations.

Key details:
- wilson(k, n): Wilson score 95% confidence interval for binomial proportions.
- quantile_inf(x, q): Quantile interpolation treating infinite values (+inf) as
  unsuccessful runs without producing NaNs.
- boot_ci(x, rng, nboot, stat): Percentile bootstrap 95% confidence interval using
  a single set of resamples.
- wilcoxon_p(d): Two-sided Wilcoxon signed-rank test on paired differences dropping
  zero differences.
- holm(p): Step-down Holm adjustment for family-wise error rate control.
- paired_metrics(...): Computes paired differences, McNemar tests, Wilcoxon tests,
  and bootstrap intervals.
"""
import numpy as np
from scipy.stats import wilcoxon, binomtest


def wilson(k, n):
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="wilson")
    return float(ci.low), float(ci.high)


def quantile_inf(x, q):
    """Linear-interpolation quantile (identical to numpy's default when all values are finite) that is safe for +inf:
    a quantile whose interpolation bracket touches +inf is +inf (failures are ranked as the worst values; no NaN)."""
    x = np.sort(np.asarray(x, float)); n = len(x); q = np.atleast_1d(np.asarray(q, float)); out = []
    for qq in q:
        h = (n - 1) * qq / 100.0; lo, hi = int(np.floor(h)), int(np.ceil(h)); fr = h - lo
        out.append(x[lo] if fr == 0 or lo == hi else (np.inf if np.isinf(x[hi]) else x[lo] + fr * (x[hi] - x[lo])))
    return np.array(out)


def boot_ci(x, rng, nboot=10000, stat=np.median):
    """Percentile bootstrap 95% CI; ONE set of resamples gives both endpoints."""
    x = np.asarray(x, float)
    if len(x) == 0: return (np.nan, np.nan)
    idx = rng.integers(0, len(x), (nboot, len(x)))
    with np.errstate(invalid="ignore"):
        lo, hi = np.percentile(stat(x[idx], axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def wilcoxon_p(d):
    """Two-sided Wilcoxon signed-rank test on paired differences (zero differences dropped; scipy chooses the exact distribution
    or the normal approximation with tie correction). Returns (p, number of non-zero differences); p = NaN if fewer than 6."""
    d = np.asarray(d, float); nz = int((d != 0).sum())
    if nz < 6: return np.nan, nz
    return float(wilcoxon(d, zero_method="wilcox").pvalue), nz


def holm(p):
    """Holm step-down adjusted p-values (NaN entries are ignored and stay NaN)."""
    p = np.asarray(p, float); out = np.full(p.shape, np.nan); ok = np.flatnonzero(~np.isnan(p)); m = len(ok)
    order = ok[np.argsort(p[ok])]; run_max = 0.0
    for rank, i in enumerate(order):
        run_max = max(run_max, (m - rank) * p[i]); out[i] = min(1.0, run_max)
    return out


def paired_metrics(t_a, t_b, succ_a, succ_b, gap_a, gap_b, budget, rng):
    """All paired statistics for one comparison (arrays aligned by seed).
    * success: difference in success rate + exact McNemar test (discordant pairs).
    * time to first feasible: right-censored at budget+1 for failures (all censored runs share one value; ties handled by the test).
    * objective gap: ONLY seeds where BOTH runs succeeded (conditional on success); no substitute value is ever used for infinite gaps."""
    t_a, t_b = np.asarray(t_a, float), np.asarray(t_b, float)
    succ_a, succ_b = np.asarray(succ_a, int), np.asarray(succ_b, int)
    gap_a, gap_b = np.asarray(gap_a, float), np.asarray(gap_b, float)
    both = (succ_a == 1) & (succ_b == 1)
    assert np.all(np.isfinite(gap_a[succ_a == 1])) and np.all(np.isinf(gap_a[succ_a == 0]))
    a_only, b_only = int(((succ_a == 1) & (succ_b == 0)).sum()), int(((succ_a == 0) & (succ_b == 1)).sum())
    dt = t_a - t_b; dg = (gap_a - gap_b)[both]
    p_t, nz_t = wilcoxon_p(dt); p_g, nz_g = wilcoxon_p(dg)
    ci_t = boot_ci(dt, rng); ci_g = boot_ci(dg, rng) if len(dg) else (np.nan, np.nan)
    return dict(n_pairs=len(t_a), d_success=float((succ_a - succ_b).mean()),
                mcnemar_p=1.0 if a_only + b_only == 0 else float(binomtest(a_only, a_only + b_only, 0.5).pvalue),
                n_A_only_success=a_only, n_B_only_success=b_only, n_neither_success=int(((succ_a == 0) & (succ_b == 0)).sum()),
                d_ttff_median=float(np.median(dt)), d_ttff_mean=float(dt.mean()), d_ttff_ci_lo=ci_t[0], d_ttff_ci_hi=ci_t[1],
                n_censored_pairs=int(((t_a > budget) | (t_b > budget)).sum()), ttff_nonzero_pairs=nz_t, p_ttff=p_t,
                n_both_success=int(both.sum()), d_gap_median_both=float(np.median(dg)) if len(dg) else np.nan,
                d_gap_ci_lo=ci_g[0], d_gap_ci_hi=ci_g[1], gap_nonzero_pairs=nz_g, p_gap=p_g,
                frac_A_gap_lower_both=float((dg < 0).mean()) if len(dg) else np.nan,
                frac_A_gap_higher_both=float((dg > 0).mean()) if len(dg) else np.nan)
