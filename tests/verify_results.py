"""
File: tests/verify_results.py

Purpose:
Performs independent recomputation and integrity verification of all reported statistics
directly from the raw run files.

Role in the experiment:
Independent auditing script ensuring that published CSV summaries, paired comparisons,
and report numbers match raw trajectory data without relying on analyze.py.

Key details:
- Re-evaluates objective and constraint values from raw trajectories X to verify stored values.
- Independently recalculates per-run metrics, method summaries, Wilson intervals, and medians.
- Independently recomputes paired differences, Wilcoxon tests, and Holm adjustments.
- Verifies code hash fingerprints across config, runs, and source files.
- Outputs: Writes results/verification.json with detailed verification pass/fail status.
"""
import sys, os, re, glob, json, subprocess
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, binomtest
import hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark import objective, constraint
R = sys.argv[1] if len(sys.argv) > 1 else "results"
cfg = json.load(open(f"{R}/config.json")); facts = json.load(open(f"{R}/benchmark_facts.json" if os.path.exists(f"{R}/benchmark_facts.json") else "results/benchmark_facts.json")); B = cfg["budget"]; fstar = facts["f_star"]
fails = []

def wilson_hand(k, n, z=1.959963984540054):          # independent of scipy's proportion_ci and of stats_utils
    p = k / n; den = 1 + z * z / n; ctr = (p + z * z / (2 * n)) / den; hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return ctr - hw, ctr + hw

def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else "")); 
    if not ok: fails.append(name)

# --- recompute per-run metrics from scratch (re-evaluating f and c from stored X, not trusting stored f, c)
runs = {}
for fn in glob.glob(f"{R}/runs/*.npz"):
    tag, m, s = re.match(r".*/((?:main|alt|n5)_[FI])_([A-Za-z_]+)_(\d+)\.npz", fn).groups(); d = np.load(fn); n0 = int(d["n0"])
    X = d["X"]; f = objective(X); c = constraint(X)
    assert np.allclose(f, d["f"], atol=1e-12) and np.allclose(c, d["c"], atol=1e-12) and len(X) == n0 + B
    feas = c <= 0; first = np.flatnonzero(feas)
    t = 0 if feas[:n0].any() else (first[0] - n0 + 1 if feas.any() else B + 1)
    gap = f[feas].min() - fstar if feas.any() else np.inf
    nfeas0 = int(feas[:n0].sum())
    runs[(tag, m, int(s))] = dict(t=int(t), success=int(feas.any()), gap=gap, infeas=float((~feas[n0:]).mean()), n0=n0, nfeas0=nfeas0, X0=d["X0"])
exp = {(t, m, s) for t, (n0, cnd, ms) in cfg["experiments"].items() for m in ms for s in range(1, cfg["seeds"] + 1)}
check("all declared runs present, none extra", set(runs) == exp, f"{len(runs)} vs {len(exp)}")
check("initial designs satisfy their condition", all((r["nfeas0"] >= 1) if k[0].endswith("F") else (r["nfeas0"] == 0) for k, r in runs.items()))
check("identical initial design across methods per (experiment, seed)", all(
    np.array_equal(runs[(t, m, s)]["X0"], runs[(t, ms[0], s)]["X0"]) for t, (n0, cnd, ms) in cfg["experiments"].items() for m in ms for s in range(1, cfg["seeds"] + 1)))
check("alt_I designs equal main_I designs", all(np.array_equal(runs[("alt_I", "cEI_altfb", s)]["X0"], runs[("main_I", "cEI", s)]["X0"]) for s in range(1, cfg["seeds"] + 1)))
check("no feasible value below f*", all(r["gap"] >= -1e-12 for r in runs.values()))

pr = pd.read_csv(f"{R}/per_run.csv")
ok = True
for _, row in pr.iterrows():
    r = runs[(row.tag, row.method, int(row.seed))]
    ok &= r["t"] == row.t_first and r["success"] == row.success and (np.isclose(r["gap"], row.gap) or (np.isinf(r["gap"]) and np.isinf(row.gap))) and np.isclose(r["infeas"], row.frac_infeasible_additional)
check("per_run.csv equals independent per-run recomputation", ok and len(pr) == len(runs))

sm = pd.read_csv(f"{R}/summary_by_method.csv"); ok = True
for _, row in sm.iterrows():
    ks = [k for k in runs if k[0] == row.tag and k[1] == row.method]
    t = np.array([runs[k]["t"] for k in ks]); g = np.array([runs[k]["gap"] for k in ks]); su = np.array([runs[k]["success"] for k in ks])
    ci_lo, ci_hi = wilson_hand(int(su.sum()), len(su))
    ok &= (len(ks) == row.n_runs and su.sum() == row.n_success and np.isclose(ci_lo, row.success_lo) and np.isclose(ci_hi, row.success_hi)
           and np.isclose(np.median(t), row.ttff_median) and np.allclose(np.percentile(t, [25, 75]), [row.ttff_q25, row.ttff_q75])
           and (np.isclose(np.median(g), row.gap_all_median) or (np.isinf(np.median(g)) and np.isinf(row.gap_all_median)))
           and ((su.sum() == 0 and np.isnan(row.gap_succ_median)) or np.allclose(np.percentile(g[su == 1], [25, 50, 75]), [row.gap_succ_q25, row.gap_succ_median, row.gap_succ_q75]))
           and np.isclose(float((g > 0.1).mean()), row.frac_poor_optimum) and bool(row.ttff_median_censored) == bool(np.median(t) > cfg["budget"]))
check("summary_by_method.csv equals independent recomputation (success, Wilson, medians, IQR, poor-optimum fraction)", ok)

pc = pd.read_csv(f"{R}/paired_comparisons.csv"); ok = True
for _, row in pc.iterrows():
    (a, ta), (b, tb) = [re.match(r"(\w+)\[(\w+)\]", x).groups() for x in (row.A, row.B)]
    sd = range(1, cfg["seeds"] + 1)
    dt = np.array([runs[(ta, a, s)]["t"] - runs[(tb, b, s)]["t"] for s in sd])
    both = [s for s in sd if runs[(ta, a, s)]["success"] and runs[(tb, b, s)]["success"]]
    dg = np.array([runs[(ta, a, s)]["gap"] - runs[(tb, b, s)]["gap"] for s in both])           # conditional on both succeeding; no substitution
    ok &= int(row.n_both_success) == len(both) and np.isfinite(dg).all()
    for d_, col in ((dt, "p_ttff"), (dg, "p_gap")):
        p = wilcoxon(d_, zero_method="wilcox").pvalue if (d_ != 0).sum() >= 6 else np.nan
        ok &= (np.isnan(p) and np.isnan(row[col])) or np.isclose(p, row[col])
    ok &= np.isclose(np.median(dt), row.d_ttff_median) and (len(dg) == 0 or np.isclose(np.median(dg), row.d_gap_median_both))
    ok &= row.d_ttff_ci_lo <= row.d_ttff_median + 1e-9 and row.d_ttff_ci_hi >= row.d_ttff_median - 1e-9
check("paired_comparisons.csv: pairing by seed, both-success gap rule, Wilcoxon p-values and median differences match independent computation", ok)
# Holm re-computation (independent step-down implementation)
def holm_hand(p):
    p = np.asarray(p, float); idx = [i for i in range(len(p)) if not np.isnan(p[i])]; srt = sorted(idx, key=lambda i: p[i]); m = len(srt); out = np.full(len(p), np.nan); mx = 0
    for r, i in enumerate(srt): mx = max(mx, (m - r) * p[i]); out[i] = min(1, mx)
    return out
check("Holm-adjusted p-values (time and gap families) match an independent implementation",
      np.allclose(holm_hand(pc.p_ttff.values), pc.p_ttff_holm.values, equal_nan=True) and np.allclose(holm_hand(pc.p_gap.values), pc.p_gap_holm.values, equal_nan=True))

# --- stale-result detection: hashes
h = hashlib.sha256()
for fn in ("benchmark.py", "designs.py", "bo.py"): h.update(open(fn, "rb").read())
check("results were produced by the current benchmark.py/designs.py/bo.py (code fingerprint)", h.hexdigest()[:16] == cfg["code_hash"])
check("every run file carries the configuration's code hash", all(str(np.load(fn)["code_hash"]) == cfg["code_hash"] for fn in glob.glob(f"{R}/runs/*.npz")))
mf_ = f"{R}/analysis_manifest.json"
if os.path.exists(mf_):
    man = json.load(open(mf_)); rs = hashlib.sha256(b"".join(open(fn, "rb").read() for fn in sorted(glob.glob(f"{R}/runs/*.npz")))).hexdigest()
    check("analysis outputs are up to date with the raw run files (manifest hash)", man["run_files_sha256"] == rs and man["code_hash"] == cfg["code_hash"] and man["n_runs"] == len(runs))
    ff = man["facts_file"]; check("analysis used the current benchmark_facts.json", man["facts_sha256"] == hashlib.sha256(open(ff, "rb").read()).hexdigest())

rep = "report.pdf"; report_checked = False
if R == "results" and os.path.exists(rep):
    try:
        txt = subprocess.run(["pdftotext", "-layout", rep, "-"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        txt = ""
    if txt and os.path.exists("results/report_numbers.json"):
        txt = re.sub(r"\s+", " ", txt); nums = json.load(open("results/report_numbers.json"))
        miss = [k for k, v_ in nums.items() if v_ not in txt]
        check("every number registered by make_report.py appears in report.pdf text", not miss, f"missing {miss[:5]}")
        # independent expectations computed from the raw runs, searched for in the PDF
        exp_strings = []
        for (tag, m) in sorted({(k[0], k[1]) for k in runs}):
            ks = [k for k in runs if k[0] == tag and k[1] == m]; g = np.array([runs[k]["gap"] for k in ks]); t = np.array([runs[k]["t"] for k in ks])
            k_poor = int((g > 0.1).sum()); lo, hi = wilson_hand(k_poor, len(g))
            exp_strings += [f"{k_poor}/{len(g)} [{lo:.2f}, {hi:.2f}]", f"{int(np.sum([runs[k]['success'] for k in ks]))}/{len(ks)}",
                            f"{np.median(g):.3f} ({np.percentile(g,25):.3f}-{np.percentile(g,75):.3f})"]
        miss2 = [s for s in exp_strings if s not in txt]
        check("independently recomputed Table 1 entries (success, poor-optimum count with hand-computed Wilson CI, gap median/IQR) appear in report.pdf", not miss2, f"missing {miss2[:4]}")
        fs = facts["f_star"]; check("report quotes the stored f* and the boundary wording", f"{fs:.5f}" in txt and "strictly feasible" not in txt and "on the constraint boundary" in txt)
        report_checked = True
    else:
        print("SKIP report text check (pdftotext or report_numbers.json unavailable)")
status = dict(data_checks_passed=not [f for f in fails if "report" not in f.lower() and "table 1" not in f.lower()], all_checks_passed=not fails, report_text_checked=report_checked,
              n_runs=len(runs), code_hash=cfg["code_hash"], failed=fails)
json.dump(status, open(f"{R}/verification.json", "w"), indent=1)
print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: ' + ', '.join(fails)}"); sys.exit(1 if fails else 0)
