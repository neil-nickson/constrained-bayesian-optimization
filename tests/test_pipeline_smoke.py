"""
File: tests/test_pipeline_smoke.py

Purpose:
End-to-end pipeline smoke and tamper test executed in an isolated temporary directory.

Role in the experiment:
Validates pipeline integration, resume functionality, and tamper safeguards that reject
stale, mixed, or incomplete experiment runs.

Key details:
- Executes a mini experiment (2 seeds, budget 3) across all experiment blocks.
- Verifies that re-running resumes without redundant recomputation.
- Tests refusal to mix mismatched configurations or budgets.
- Verifies pipeline stops when run files are missing or modified with different code hashes.
- Outputs: Writes results/smoke_test_results.json.
"""
import os, sys, subprocess, shutil, json, glob, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
ENV = dict(os.environ, PYTHONPATH=ROOT)
TMP = "/tmp/smoke_pipeline"; shutil.rmtree(TMP, ignore_errors=True); shutil.rmtree(TMP + "_figs", ignore_errors=True)
res = []


def sh(*args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, env=ENV)


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  [{detail[:300]}]")); res.append((name, bool(ok)))


r = sh("run_experiment.py", "--seeds", "2", "--budget", "3", "--out", TMP); check("smoke run completes (all 4 experiment blocks, 20 runs)", r.returncode == 0 and len(glob.glob(f"{TMP}/runs/*.npz")) == 20, r.stderr or r.stdout)
r = sh("run_experiment.py", "--seeds", "2", "--budget", "3", "--out", TMP); check("re-running resumes without recomputing (config identical)", r.returncode == 0 and r.stdout.strip() == "", r.stdout + r.stderr)
r = sh("run_experiment.py", "--seeds", "2", "--budget", "4", "--out", TMP); check("run_experiment REFUSES to mix a different budget into an existing results folder", r.returncode != 0 and "Refusing" in (r.stderr + r.stdout), r.stderr + r.stdout)
r = sh("analyze.py", "--results", TMP, "--figs", TMP + "_figs"); check("analyze.py runs on the smoke results", r.returncode == 0, r.stderr[-300:])
r = sh("tests/verify_results.py", TMP); check("verify_results.py passes on the smoke results", r.returncode == 0, r.stdout[-400:] + r.stderr[-300:])
victim = sorted(glob.glob(f"{TMP}/runs/*.npz"))[0]; keep = open(victim, "rb").read(); os.remove(victim)
r = sh("analyze.py", "--results", TMP, "--figs", TMP + "_figs"); check("analyze.py STOPS when a run file is missing", r.returncode != 0 and "missing runs" in r.stderr, r.stderr[-300:])
r = sh("tests/verify_results.py", TMP); check("verify_results.py FAILS when a run file is missing", r.returncode != 0)
open(victim, "wb").write(keep)
d = dict(np.load(victim)); d["code_hash"] = np.array("0000000000000000"); np.savez(victim, **d)
r = sh("analyze.py", "--results", TMP, "--figs", TMP + "_figs"); check("analyze.py STOPS when a run was produced by different code", r.returncode != 0 and "different code" in r.stderr, r.stderr[-300:])
open(victim, "wb").write(keep)
r = sh("run_experiment.py", "--seeds", "1", "--budget", "2", "--which", "main_F", "--out", TMP, "--fresh"); old = glob.glob(TMP + "_old_*")
check("--fresh moves the old results aside (kept, renamed) and starts clean", r.returncode == 0 and len(old) == 1 and len(glob.glob(f"{TMP}/runs/*.npz")) == 3, r.stderr[-300:])
for o in old: shutil.rmtree(o, ignore_errors=True)
shutil.rmtree(TMP, ignore_errors=True); shutil.rmtree(TMP + "_figs", ignore_errors=True)
os.makedirs("results", exist_ok=True)
json.dump(dict(passed=sum(ok for _, ok in res), total=len(res), failed_names=[n for n, ok in res if not ok], time=time.strftime("%Y-%m-%d %H:%M:%S")), open("results/smoke_test_results.json", "w"), indent=1)
print(f"\n{sum(ok for _, ok in res)}/{len(res)} pipeline checks passed"); sys.exit(0 if all(ok for _, ok in res) else 1)
