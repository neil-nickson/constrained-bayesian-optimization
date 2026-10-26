"""
File: test_fidelity.py

Purpose:
Verifies that the project's surrogate modeling and acquisition functions match the
numerical outputs of the authors' reference implementations on identical inputs.

Role in the experiment:
Provides a standalone fidelity check against the reference code supplied in Code.zip
(authors_code/ECEI_MT.py and MLCB_MT.py).

Key details:
- Evaluates ECEI main branch, ECEI fallback branch (flag 2), and CBO-LCB (MLCB).
- Compares predictions on stub Gaussian processes with identical means and variances.
- Checks objective and constraint values across 2000 points.
- Scope: Validates agreement with the authors' supplied reference code; does not
  independently prove agreement with published paper text where code was unavailable.
"""
import numpy as np
from bo import acq_ecei, acq_cbolcb, ACQ
from benchmark import objective, constraint


def load_authors(fname):
    src = open(fname).read().split("parser = argparse")[0]
    ns = {"__name__": "authors"}
    exec(src, ns)
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

# ---- ECEI (authors: ECEI_acq with flag=1, fmean=0, fsd=1 so that the stub predictions are in original units)
A = load_authors("authors_code/ECEI_MT.py")
best, M = A["best_params"](f, c)
theirs = A["ECEI_acq"](U, best, M, 0.0, 1.0, StubGP(mf, sf), StubGP(mc, sc), 1)
ours, flag = acq_ecei(S, U, f, c)
print("ECEI   max |diff| =", np.abs(theirs - ours).max(), " flag", flag)
assert np.allclose(theirs, ours, atol=1e-9)

# ---- ECEI flag-2 branch (authors: muf + M*theta), triggered when EI is ~0 for >99% of candidates
f2 = np.append(np.full(11, -1.0), -100.0); c2 = np.append(np.ones(11), -1.0)   # one very good feasible point
best2, M2 = A["best_params"](f2, c2)
theirs2 = A["ECEI_acq"](U, best2, M2, 0.0, 1.0, StubGP(mf, sf), StubGP(mc, sc), 2)
ours2, flag2 = acq_ecei(S, U, f2, c2)
print("ECEI fallback branch: flag =", flag2, " max |diff| =", np.abs(theirs2 - ours2).max())
assert flag2 == 2 and np.allclose(theirs2, ours2, atol=1e-9)

# ---- CBO-LCB (authors: MLCB_acq)
B = load_authors("authors_code/MLCB_MT.py")
M = np.abs(f).max()
theirs = B["MLCB_acq"](U, M, StubGP(mf, sf), StubGP(mc, sc))
ours, _ = acq_cbolcb(S, U, f, c)
print("CBOLCB max |diff| =", np.abs(theirs - ours).max())
assert np.allclose(theirs, ours, atol=1e-9)

# ---- Benchmark functions identical to authors' (note: authors' MLCB_MT.py omits the +1e-9 in atan2; difference only at x2=0)
X = rng.random((2000, 2)) * [4.75, 4.25] + [-2.25, -2.5]
dc = np.abs(np.array([A["constraint"](x) for x in X]) - constraint(X))
do = np.abs(np.array([A["objective"](x) for x in X]) - objective(X))
print("objective max diff", do.max(), " constraint max diff", dc.max())
assert do.max() < 1e-12 and dc.max() < 1e-12
print("ALL FIDELITY CHECKS PASSED")
