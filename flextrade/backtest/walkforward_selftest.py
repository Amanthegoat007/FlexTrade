"""Verify the walk-forward metrics against cases with known answers.

Every statistic in walkforward.py adjudicates a published claim, so each one is
checked here against a value derived independently — by hand, by a closed form,
or by simulation with a known ground truth. A test-statistic that is silently
wrong is worse than none, because it launders a bad number as a verified one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from walkforward import (christoffersen, diebold_mariano, interval_score,  # noqa: E402
                         kupiec_pof, pinball)

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"   {detail}" if detail else ""))


# ---------------------------------------------------------------- pinball --
# For tau=0.5 pinball is exactly half the absolute error.
y = np.array([10.0, 20.0, 30.0])
q = np.array([12.0, 18.0, 30.0])
check("pinball tau=0.5 == 0.5 * MAE",
      np.isclose(pinball(y, q, 0.5), 0.5 * np.mean(np.abs(y - q))),
      f"{pinball(y, q, 0.5):.4f} vs {0.5*np.mean(np.abs(y-q)):.4f}")

# Asymmetry: under-prediction at tau=0.9 must cost 9x over-prediction.
check("pinball tau=0.9 penalises under-prediction 9:1",
      np.isclose(pinball(np.array([10.0]), np.array([9.0]), 0.9), 0.9)
      and np.isclose(pinball(np.array([10.0]), np.array([11.0]), 0.9), 0.1),
      "under=0.9, over=0.1")

# ------------------------------------------------------- interval score --
# Hand-computed: alpha=0.2 => 2/alpha = 10.
#   y=5 inside [0,10]  -> 10
#   y=-2 below         -> 10 + 10*(0-(-2)) = 30
#   y=14 above         -> 10 + 10*(14-10)  = 50
expected = np.mean([10.0, 30.0, 50.0])
got = interval_score(np.array([5.0, -2.0, 14.0]), np.array([0.0, 0.0, 0.0]),
                     np.array([10.0, 10.0, 10.0]), 0.2)
check("interval score matches hand computation", np.isclose(got, expected),
      f"{got:.4f} vs {expected:.4f}")

# Propriety in the practical sense: for a known N(0,1) target the TRUE 80%
# interval must beat both a wider and a narrower one.
rng = np.random.default_rng(0)
z = rng.standard_normal(200_000)
true_lo, true_hi = stats.norm.ppf(0.10), stats.norm.ppf(0.90)
s_true = interval_score(z, np.full_like(z, true_lo), np.full_like(z, true_hi), 0.2)
s_wide = interval_score(z, np.full_like(z, true_lo * 1.6), np.full_like(z, true_hi * 1.6), 0.2)
s_narrow = interval_score(z, np.full_like(z, true_lo * 0.5), np.full_like(z, true_hi * 0.5), 0.2)
check("interval score is minimised by the TRUE interval",
      s_true < s_wide and s_true < s_narrow,
      f"true {s_true:.4f} < wide {s_wide:.4f}, narrow {s_narrow:.4f}")

# ------------------------------------------------------------- Kupiec POF --
# A sequence whose exceedance rate is exactly nominal must not be rejected.
e = np.zeros(1000, dtype=int)
e[:200] = 1                      # exactly 20%
rng.shuffle(e)
k = kupiec_pof(e, 0.20)
check("Kupiec: LR ~ 0 when rate is exactly nominal",
      k["lr"] is not None and k["lr"] < 1e-6 and k["p_value"] > 0.99,
      f"LR={k['lr']}, p={k['p_value']}")

# A badly wrong rate must be rejected decisively.
e_bad = np.zeros(1000, dtype=int)
e_bad[:450] = 1                  # 45% against a 20% nominal
rng.shuffle(e_bad)
k_bad = kupiec_pof(e_bad, 0.20)
check("Kupiec: rejects a 45% rate against 20% nominal",
      k_bad["p_value"] < 1e-6, f"p={k_bad['p_value']}")

# Closed form cross-check of the LR statistic at a specific point.
n, x, p = 100, 30, 0.20
pi = x / n
lr_manual = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)
                  - (n - x) * np.log(1 - pi) - x * np.log(pi))
e_fix = np.zeros(n, dtype=int); e_fix[:x] = 1
# tolerance is 5e-4 because the reported LR is rounded to 3 dp for display;
# the p-value is computed from the unrounded statistic before rounding
check("Kupiec LR matches the closed form",
      np.isclose(kupiec_pof(e_fix, p)["lr"], lr_manual, atol=5e-4),
      f"{kupiec_pof(e_fix, p)['lr']:.6f} vs {lr_manual:.6f}")

# Size of the test: on truly nominal data it should reject ~5% of the time.
rej = sum(1 for _ in range(400)
          if (kupiec_pof(rng.random(500) < 0.20, 0.20)["p_value"] or 1) < 0.05)
check("Kupiec size ~5% under H0", 0.02 <= rej / 400 <= 0.10,
      f"rejected {rej/400:.1%} of 400 nominal samples")

# ----------------------------------------------------- Christoffersen ind --
# i.i.d. exceedances: independence must NOT be rejected.
iid = (rng.random(4000) < 0.2).astype(int)
c_iid = christoffersen(iid, 0.2)
check("Christoffersen: independence holds on i.i.d. exceedances",
      c_iid["p_value_ind"] is not None and c_iid["p_value_ind"] > 0.05,
      f"p_ind={c_iid['p_value_ind']}")

# Clustered exceedances at the CORRECT overall rate. This is the entire reason
# Christoffersen is here: Kupiec sees a perfectly calibrated 20% and passes it,
# while the failures actually arrive in bursts of 20 — which operationally is a
# drawdown, not scattered noise. Burst probability is solved so the long-run
# rate is exactly 20%:  L*p / (1 - p + L*p) = 0.2  =>  p = 0.2 / (L - 19*0.2).
#
# Built so the rate is EXACTLY 20% by construction rather than in expectation:
# a random burst process drifts (it came out at 15.3%), and then Kupiec rejects
# the rate, which destroys the very thing being demonstrated.
N, L = 6000, 20
seg = N // (N // 5 // L)              # one burst per segment, 20 ones per 100
clustered = np.zeros(N, dtype=int)
for s in range(0, N, seg):
    off = int(rng.integers(0, seg - L + 1))
    clustered[s + off:s + off + L] = 1
c_cl = christoffersen(clustered, 0.2)
k_cl = kupiec_pof(clustered, 0.2)
check("Christoffersen rejects CLUSTERING that Kupiec passes",
      c_cl["p_value_ind"] is not None and c_cl["p_value_ind"] < 0.01
      and k_cl["p_value"] > 0.05,
      f"rate {k_cl['rate_pct']}% (Kupiec p={k_cl['p_value']}, PASSES) "
      f"but p_ind={c_cl['p_value_ind']} (REJECTS)")

# ------------------------------------------------------- Diebold-Mariano --
check("DM: identical loss series is indistinguishable",
      diebold_mariano(np.arange(100.0), np.arange(100.0))["better"]
      == "indistinguishable")

a = rng.normal(1.0, 1.0, 500)          # A has clearly lower loss
b = rng.normal(3.0, 1.0, 500)
dm = diebold_mariano(a, b)
check("DM: detects a genuinely better forecast (A)",
      dm["better"] == "A" and dm["p_value"] < 1e-6,
      f"stat={dm['stat']}, p={dm['p_value']}")

# Size under H0: two draws from the same distribution should reject ~5%.
rej = sum(1 for _ in range(400)
          if (diebold_mariano(rng.normal(1, 1, 250),
                              rng.normal(1, 1, 250))["p_value"] or 1) < 0.05)
check("DM size ~5% under H0", 0.02 <= rej / 400 <= 0.10,
      f"rejected {rej/400:.1%} of 400 equal-loss pairs")

# The h parameter must actually account for serial correlation. h-step forecast
# errors are MA(h-1) with positive autocorrelation, and treating them as
# independent understates the variance and so OVERSTATES significance. Build a
# loss difference with strong positive autocorrelation and confirm that telling
# the test about it shrinks the statistic. (On i.i.d. data this comparison is
# not meaningful: the sample autocovariances are noise and can move the HAC
# variance either way, which is what an earlier version of this check got
# wrong.)
w = rng.standard_normal(600)
d_ac = np.array([0.25 * w[t:t + 5].sum() for t in range(len(w) - 5)]) + 0.15
zeros = np.zeros_like(d_ac)
d1 = diebold_mariano(d_ac, zeros, h=1)
d5 = diebold_mariano(d_ac, zeros, h=5)
rho1 = float(np.corrcoef(d_ac[:-1], d_ac[1:])[0, 1])
check("DM: accounting for serial correlation shrinks |stat|",
      abs(d5["stat"]) < abs(d1["stat"]),
      f"lag-1 autocorr {rho1:+.2f}   h=1 {d1['stat']}, h=5 {d5['stat']}")

# And the HLN small-sample factor itself, checked against its closed form.
def hln(n, h):
    return np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
check("HLN factor matches its closed form and is < 1",
      np.isclose(hln(100, 5), np.sqrt(91.2 / 100)) and hln(100, 5) < hln(100, 1) < 1,
      f"n=100: h=1 {hln(100,1):.5f}, h=5 {hln(100,5):.5f}")

# ------------------------------------------------------------------ verdict --
bad = [n for n, ok in results if not ok]
print(f"\n{len(results) - len(bad)}/{len(results)} checks passed")
if bad:
    print("FAILED: " + "; ".join(bad))
raise SystemExit(1 if bad else 0)
