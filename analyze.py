"""
File: analyze.py

Purpose:
Loads saved raw run files, computes per-run metrics and summary statistics,
performs paired statistical tests with Holm adjustment, and generates figures.

Role in the experiment:
Serves as the post-processing and aggregation pipeline transforming raw .npz
run trajectories into report-ready tables, CSV datasets, and visual plots.

Key details:
- Metrics: Feasibility success rate, time-to-first-feasible (ttff), final objective
  gap conditional on success, and fraction of poor local optima (gap > 0.1).
- Statistical comparisons: Paired Wilcoxon signed-rank tests with Holm step-down
  correction and bootstrap 95% confidence intervals.
- Outputs: per_run.csv, summary_by_method.csv, paired_comparisons.csv,
  design_diagnostics.csv, analysis_manifest.json, and figures (fig1 to fig4).
"""
import argparse, glob, hashlib, json, os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmark import *
from stats_utils import wilson, quantile_inf, boot_ci as _boot_ci, holm, paired_metrics

ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); ap.add_argument("--figs", default="figures")
args = ap.parse_args()
R, FIGS = args.results, args.figs
os.makedirs(FIGS, exist_ok=True)
CFG = json.load(open(f"{R}/config.json")); FACTS = json.load(open(f"{R}/benchmark_facts.json" if os.path.exists(f"{R}/benchmark_facts.json") else "results/benchmark_facts.json"))
FSTAR, BUDGET = FACTS["f_star"], CFG["budget"]
NBOOT, POOR = 10000, 0.1          # bootstrap resamples; gap > POOR counts as a 'poor local optimum' (see report)
RNG = np.random.default_rng(2026)


def current_code_hash():
    h = hashlib.sha256()
    for fn in ("benchmark.py", "designs.py", "bo.py"): h.update(open(fn, "rb").read())
    return h.hexdigest()[:16]


def load():
    assert current_code_hash() == CFG["code_hash"], "source files changed since the experiment was run: results are stale"
    expected = {(t, m, s) for t, (n0, cond, ms) in CFG["experiments"].items() for m in ms for s in range(1, CFG["seeds"] + 1)}
    rows, curves, found = [], {}, set()
    for fn in sorted(glob.glob(f"{R}/runs/*.npz")):
        tag, method, seed = re.match(r".*/((?:main|alt|n5)_[FI])_([A-Za-z_]+)_(\d+)\.npz", fn).groups(); seed = int(seed)
        d = np.load(fn)
        assert str(d["code_hash"]) == CFG["code_hash"], f"{fn}: produced by different code"
        n0 = int(d["n0"]); f, c = d["f"], d["c"]
        assert len(f) == n0 + BUDGET == len(c) and int(d["budget"]) == BUDGET, f"{fn}: wrong number of evaluations"
        found.add((tag, method, seed))
        feas = c <= 0
        add_feas = feas[n0:]
        t_first = 0 if feas[:n0].any() else (int(np.argmax(add_feas)) + 1 if add_feas.any() else BUDGET + 1)
        best = np.minimum.accumulate(np.where(feas, f, np.inf))
        gap_curve = best[n0 - 1:] - FSTAR                                  # index j: after j additional evaluations
        rows.append(dict(tag=tag, method=method, seed=seed, n0=n0, attempts=int(d["attempts"]), n_init_feasible=int(feas[:n0].sum()),
                         t_first=t_first, success=int(feas.any()), gap=gap_curve[-1],
                         frac_infeasible_additional=float((~add_feas).mean()), n_fallback_iters=int((d["flags"] > 0).sum())))
        curves[(tag, method, seed)] = gap_curve
    missing = expected - found; extra = found - expected
    assert not missing, f"missing runs: {sorted(missing)[:10]} ... ({len(missing)} total)"
    assert not extra, f"unexpected runs: {sorted(extra)[:10]}"
    df = pd.DataFrame(rows)
    assert (df.gap >= -1e-12).all(), "a feasible value below the estimated optimum was found: f* is wrong"
    return df, curves


def boot_ci(x, stat=np.median):
    return _boot_ci(x, RNG, NBOOT, stat)


def summarise(df):
    out = []
    for (tag, m), g in df.groupby(["tag", "method"]):
        n, k = len(g), int(g.success.sum()); lo, hi = wilson(k, n)
        t = g.t_first.values; q = quantile_inf(t, [25, 50, 75]); tci = boot_ci(t)
        gall = g.gap.values; gq = quantile_inf(gall, [25, 50, 75])             # +inf (failure) ranked last; NaN-safe
        s = g[g.success == 1].gap.values; sq = quantile_inf(s, [25, 50, 75]) if len(s) else [np.nan] * 3; sci = boot_ci(s)
        out.append(dict(tag=tag, method=m, n_runs=n, n_success=k, success_rate=k / n, success_lo=lo, success_hi=hi,
                        ttff_q25=q[0], ttff_median=q[1], ttff_q75=q[2], ttff_ci_lo=tci[0], ttff_ci_hi=tci[1],
                        ttff_median_censored=bool(q[1] > BUDGET), n_ttff_censored=int((t > BUDGET).sum()),
                        ttff_mean=t.mean(), gap_all_q25=gq[0], gap_all_median=gq[1], gap_all_q75=gq[2],
                        gap_succ_q25=sq[0], gap_succ_median=sq[1], gap_succ_q75=sq[2], gap_succ_ci_lo=sci[0], gap_succ_ci_hi=sci[1],
                        frac_poor_optimum=float((gall > POOR).mean()),
                        mean_frac_infeasible_additional=g.frac_infeasible_additional.mean(),
                        median_frac_infeasible_additional=g.frac_infeasible_additional.median(),
                        mean_fallback_iters=g.n_fallback_iters.mean()))
    return pd.DataFrame(out)


def paired(df):
    comps = [(t, a, t, "cEI") for t in ("main_F", "main_I", "n5_I") for a in ("ECEI", "CBOLCB")] + \
            [(t, "ECEI", t, "CBOLCB") for t in ("main_F", "main_I", "n5_I")] + [("alt_I", "cEI_altfb", "main_I", "cEI")]
    rows = []
    for ta, a, tb, b in comps:
        A = df[(df.tag == ta) & (df.method == a)].set_index("seed").sort_index(); B = df[(df.tag == tb) & (df.method == b)].set_index("seed").sort_index()
        assert list(A.index) == list(B.index) == list(range(1, CFG["seeds"] + 1)), "pairing by seed broken"
        assert np.array_equal(A.attempts.values, B.attempts.values) and np.array_equal(A.n_init_feasible.values, B.n_init_feasible.values), "paired runs do not share the initial design"
        m = paired_metrics(A.t_first.values, B.t_first.values, A.success.values, B.success.values, A.gap.values, B.gap.values, BUDGET, RNG)
        rows.append(dict(A=f"{a}[{ta}]", B=f"{b}[{tb}]", condition=ta, **m))
    P = pd.DataFrame(rows)
    P["p_ttff_holm"] = holm(P.p_ttff.values); P["p_gap_holm"] = holm(P.p_gap.values)   # Holm within each metric family
    return P


COL = {"cEI": "#1f77b4", "CBOLCB": "#d62728", "ECEI": "#2ca02c", "cEI_altfb": "#9467bd"}
LAB = {"cEI": "cEI (PoF fallback)", "CBOLCB": "CBO-LCB", "ECEI": "Weighted improvement (ECEI)", "cEI_altfb": "cEI (alt. fallback)"}


def figures(df, curves):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    for a, tags, ttl in zip(ax, (("main_I", "alt_I"), ("n5_I",)), ("n0 = 4, no feasible initial point", "n0 = 5, no feasible initial point")):
        for tag in tags:
            for m in ("cEI", "CBOLCB", "ECEI", "cEI_altfb"):
                g = df[(df.tag == tag) & (df.method == m)]
                if len(g): a.step(np.arange(0, 21), [(g.t_first <= k).mean() for k in range(21)], where="post", color=COL[m], lw=1.8,
                                  ls="--" if m == "cEI_altfb" else "-", label=LAB[m])
        a.set_title(ttl, fontsize=9); a.set_xlabel("additional evaluations (first 20 shown)"); a.set_ylim(-0.02, 1.02); a.grid(alpha=.3)
    ax[0].set_ylabel("fraction of runs with a feasible point"); ax[0].legend(fontsize=7, loc="lower right")
    plt.tight_layout(); plt.savefig(f"{FIGS}/fig2_time_to_first_feasible.png", dpi=200); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for a, tag, ttl in zip(ax, ("main_F", "main_I"), (">=1 feasible initial point", "no feasible initial point")):
        for m in ("cEI", "CBOLCB", "ECEI"):
            C = np.array([curves[(tag, m, s)] for s in range(1, CFG["seeds"] + 1)]); ks = np.arange(C.shape[1])
            q1, med, q3 = (np.percentile(C, q, axis=0) for q in (25, 50, 75))
            fin = lambda v: np.where(np.isfinite(v), v, np.nan)
            a.plot(ks, fin(med), color=COL[m], lw=1.8, label=LAB[m]); a.fill_between(ks, fin(q1), fin(q3), color=COL[m], alpha=.12)
        a.set_yscale("log"); a.set_title(ttl, fontsize=9); a.set_xlabel("additional evaluations"); a.grid(alpha=.3, which="both")
    ax[0].set_ylabel(r"best feasible $f$ $-$ $f^*$: median and IQR"); ax[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{FIGS}/fig3_gap_curves.png", dpi=200); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
    for a, tag, ttl in zip(ax, ("main_F", "main_I", "n5_I"), ("n0=4, >=1 feasible", "n0=4, none feasible", "n0=5, none feasible")):
        for j, m in enumerate(("cEI", "CBOLCB", "ECEI")):
            g = df[(df.tag == tag) & (df.method == m)].gap.values
            a.scatter(j + RNG.uniform(-.15, .15, len(g)), np.maximum(g, 1e-6), s=14, color=COL[m], alpha=.7)
            a.hlines(np.median(g), j - .3, j + .3, color="k", lw=1.5)
        a.set_yscale("log"); a.set_xticks(range(3)); a.set_xticklabels(["cEI", "CBO-LCB", "ECEI"], fontsize=8); a.set_title(ttl, fontsize=9); a.grid(alpha=.3, which="both")
    ax[0].set_ylabel("final gap (log; floor 1e-6)")
    plt.tight_layout(); plt.savefig(f"{FIGS}/fig4_final_gap_per_run.png", dpi=200); plt.close()

    n = 300; g = np.linspace(0, 1, n); UU = np.array([[a_, b_] for a_ in g for b_ in g]); X = from_unit(UU)
    F = objective(X).reshape(n, n); Cc = constraint(X).reshape(n, n); gx = from_unit(np.c_[g, g])[:, 0]; gy = from_unit(np.c_[g, g])[:, 1]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
    for a, tag, ttl in zip(ax, (None, "main_F", "main_I"), ("Modified Townsend benchmark", "initial designs (>=1 feasible)", "initial designs (none feasible)")):
        a.contourf(gx, gy, F.T, 20, cmap="Blues_r", alpha=.9); a.contourf(gx, gy, Cc.T, levels=[0, 1e9], colors=["#cccccc"], alpha=.75)
        a.contour(gx, gy, Cc.T, levels=[0], colors="k", linewidths=1.2); a.set_title(ttl, fontsize=9); a.set_xlabel("$x_1$"); a.set_ylabel("$x_2$")
        if tag is None:
            a.plot(*FACTS["x_star"], "r*", ms=11, label="best constrained optimum found"); a.text(-1.9, -2.3, "grey = infeasible", fontsize=7); a.legend(fontsize=7, loc="upper left")
        else:
            for s in range(1, CFG["seeds"] + 1):
                X0 = np.load(f"{R}/runs/{tag}_cEI_{s:02d}.npz")["X0"]; a.plot(X0[:, 0], X0[:, 1], ".", color="k", ms=3, alpha=.6)
    plt.tight_layout(); plt.savefig(f"{FIGS}/fig1_benchmark_and_designs.png", dpi=200); plt.close()


def design_table(df):
    rows = []
    for tag in sorted(df.tag.unique()):
        g = df[df.tag == tag].drop_duplicates("seed")
        rows.append(dict(tag=tag, n0=int(g.n0.iloc[0]), n_designs=len(g), mean_rejection_attempts=g.attempts.mean(), median_attempts=g.attempts.median(),
                         max_attempts=int(g.attempts.max()), mean_init_feasible=g.n_init_feasible.mean(), min_init_feasible=int(g.n_init_feasible.min()),
                         max_init_feasible=int(g.n_init_feasible.max())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df, curves = load()
    df.to_csv(f"{R}/per_run.csv", index=False)
    S = summarise(df); S.to_csv(f"{R}/summary_by_method.csv", index=False)
    P = paired(df); P.to_csv(f"{R}/paired_comparisons.csv", index=False)
    D = design_table(df); D.to_csv(f"{R}/design_diagnostics.csv", index=False)
    figures(df, curves)
    runs_sha = hashlib.sha256(b"".join(open(fn, "rb").read() for fn in sorted(glob.glob(f"{R}/runs/*.npz")))).hexdigest()
    facts_fn = f"{R}/benchmark_facts.json" if os.path.exists(f"{R}/benchmark_facts.json") else "results/benchmark_facts.json"
    json.dump(dict(code_hash=CFG["code_hash"], n_runs=len(df), seeds=CFG["seeds"], budget=BUDGET, f_star=FSTAR, libs=CFG["libs"],
                   run_files_sha256=runs_sha, facts_file=facts_fn, facts_sha256=hashlib.sha256(open(facts_fn, "rb").read()).hexdigest()),
              open(f"{R}/analysis_manifest.json", "w"), indent=1)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    print(D.round(3).to_string()); print(S.round(3).T.to_string()); print(P.round(4).T.to_string())
