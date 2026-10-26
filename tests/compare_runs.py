"""
File: tests/compare_runs.py

Purpose:
Compares two experiment result directories run-by-run to assess reproducibility
and bitwise identity across executions.

Role in the experiment:
Auditing tool used to compare freshly executed raw runs against reference shipped
runs (e.g., comparing results_shipped with results).

Key details:
- Checks array equality for X0, X, f, c, and flags across matching run files.
- Computes maximum absolute difference and reports any divergent run filenames.
- Optional --write flag saves rerun_comparison.json for inclusion in report.pdf.
"""
import sys, os, glob, json
import numpy as np
ref, new = sys.argv[1], sys.argv[2]
cfg_ref, cfg_new = json.load(open(f"{ref}/config.json")), json.load(open(f"{new}/config.json"))
names_ref = sorted(os.path.basename(f) for f in glob.glob(f"{ref}/runs/*.npz")); names_new = sorted(os.path.basename(f) for f in glob.glob(f"{new}/runs/*.npz"))
assert names_ref == names_new, f"different run sets: {len(names_ref)} vs {len(names_new)}"
ident, maxd = 0, 0.0; diffs = []
for n in names_ref:
    a, b = np.load(f"{ref}/runs/{n}"), np.load(f"{new}/runs/{n}"); same = True
    for k in ("X0", "X", "f", "c", "flags"):
        same &= bool(np.array_equal(a[k], b[k]))
        if a[k].dtype.kind == "f" and a[k].shape == b[k].shape: maxd = max(maxd, float(np.abs(a[k] - b[k]).max()))
    ident += same
    if not same: diffs.append(n)
out = dict(n_compared=len(names_ref), n_identical=int(ident), max_abs_diff=maxd, different_runs=diffs[:20],
           code_hash_reference=cfg_ref["code_hash"], code_hash_rerun=cfg_new["code_hash"], same_code_hash=cfg_ref["code_hash"] == cfg_new["code_hash"],
           libs_reference=cfg_ref["libs"], libs_rerun=cfg_new["libs"])
print(json.dumps(out, indent=1))
if "--write" in sys.argv: json.dump(out, open(f"{ref}/rerun_comparison.json", "w"), indent=1)
sys.exit(0 if ident == len(names_ref) else 1)
