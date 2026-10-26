"""
File: run_experiment.py

Purpose:
Configures, coordinates, and executes the complete experiment matrix across
benchmark conditions, methods, and seeds, saving run data to disk.

Role in the experiment:
Main experiment runner that produces the raw trajectory data (.npz files) and
experiment configuration (config.json) used for downstream statistical analysis.

Key details:
- Experiments: main_F (n0=4, F), main_I (n0=4, I), alt_I (n0=4, I sensitivity),
  n5_I (n0=5, I sensitivity).
- Methods: cEI, CBO-LCB, ECEI, and cEI_altfb.
- Execution: Evaluates 30 seeds per method/condition over a budget of 50 evaluations.
- Integrity: Verifies code hash fingerprint before executing and prevents mixing
  different configurations within the same output folder.
"""
import argparse, hashlib, json, os, platform, shutil, time
import numpy as np, scipy, sklearn
from designs import make_design
import bo
from bo import run

EXPERIMENTS = {   # tag: (n0, condition, methods)
    "main_F": (4, "F", ["cEI", "CBOLCB", "ECEI"]),
    "main_I": (4, "I", ["cEI", "CBOLCB", "ECEI"]),
    "alt_I":  (4, "I", ["cEI_altfb"]),             # sensitivity: alternative cEI fallback
    "n5_I":   (5, "I", ["cEI", "CBOLCB", "ECEI"]),  # sensitivity: n0 = 5 (n0 = 10 cannot be built, see README)
}


def code_hash():
    h = hashlib.sha256()
    for fn in ("benchmark.py", "designs.py", "bo.py"):
        h.update(open(fn, "rb").read())
    return h.hexdigest()[:16]


def make_config(seeds, budget, which):
    return dict(code_hash=code_hash(), seeds=seeds, budget=budget,
                experiments={t: EXPERIMENTS[t] for t in which},
                bo_settings=dict(LS_BOUNDS=bo.LS_BOUNDS, N_CAND=bo.N_CAND, N_REFINE=bo.N_REFINE, LAMBDA_LCB=bo.LAMBDA_LCB),
                libs=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__))


def semantic(cfg):      # parts that must match to allow resuming (library patch versions are recorded but not enforced)
    return {k: cfg[k] for k in ("code_hash", "seeds", "budget", "experiments", "bo_settings")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--which", nargs="+", default=list(EXPERIMENTS))
    ap.add_argument("--out", default="results")
    ap.add_argument("--fresh", action="store_true", help="move an existing --out folder aside and start clean")
    a = ap.parse_args()
    cfg = make_config(a.seeds, a.budget, a.which)
    if a.fresh and os.path.exists(a.out):
        shutil.move(a.out, f"{a.out}_old_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(f"{a.out}/runs", exist_ok=True)
    cfg_path = f"{a.out}/config.json"
    if os.path.exists(cfg_path):
        old = json.load(open(cfg_path))
        if json.dumps(semantic(old), sort_keys=True, default=list) != json.dumps(semantic(cfg), sort_keys=True, default=list):
            raise SystemExit(f"Configuration in {cfg_path} differs from the current code/settings. "
                             f"Refusing to mix results. Re-run with --fresh (old results are kept, renamed).")
    else:
        json.dump(cfg, open(cfg_path, "w"), indent=1, default=list)
    for tag in a.which:
        n0, cond, methods = EXPERIMENTS[tag]
        for seed in range(1, a.seeds + 1):
            X0, attempts = make_design(seed, n0, cond)        # identical design for every method of this seed
            for m in methods:
                fn = f"{a.out}/runs/{tag}_{m}_{seed:02d}.npz"
                if os.path.exists(fn):
                    continue
                t = time.time()
                r = run(m, X0, seed, a.budget)
                np.savez(fn, X0=X0, attempts=attempts, n0=n0, budget=a.budget, code_hash=cfg["code_hash"], **r)
                print(f"{tag} {m} seed={seed} feas_found={(r['c']<=0).any()} {time.time()-t:.0f}s", flush=True)
