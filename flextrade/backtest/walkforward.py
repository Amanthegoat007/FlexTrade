"""Rolling-origin evaluation — the harness every published number goes through.

WHY THIS EXISTS
---------------
Every accuracy figure on this site used to be measured the same way: one
chronological split, fit on the front, score on the back. That is the standard
machine-learning protocol and it is the wrong one for a forecasting system,
because it reports the performance of ONE window and says nothing about the
variance across windows.

It cost us. The price band was published at 94.4% coverage against an 80%
target. Re-measured under a rolling origin — refitting at each origin so the
scoring window is never inside the training data — the same construction
delivered 74.7% mean, 51.4% in its worst month. A 20-point error in a headline
metric, in the dangerous direction, invisible to the protocol that produced it.

Rolling-origin evaluation (Tashman 2000) is the standard answer and has been for
decades. The point is not a better average; it is that you get a DISTRIBUTION of
performance, so "worst window" becomes reportable instead of unknown.

WHAT IT MEASURES, AND WHY THESE METRICS
---------------------------------------
Coverage and width must never be reported alone as a pair, because they trade
off and either can be gamed by moving the other. The interval score
(Gneiting & Raftery 2007, eq. 43) is the PROPER scoring rule that combines them:

    S(l,u;y) = (u - l) + (2/a)(l - y)1{y<l} + (2/a)(y - u)1{y>u}

Proper means it is minimised in expectation by reporting the true interval, so
it cannot be gamed at all. It is the headline for any band here.

Coverage still gets tested, but as a hypothesis rather than a number:
  - Kupiec (1995) POF: is the exceedance RATE consistent with nominal?
  - Christoffersen (1998): are exceedances INDEPENDENT, or do they cluster?

The second is the one operators care about and almost nobody reports. A band can
have textbook 80% unconditional coverage and still fail six days in a row during
a heatwave — same rate, catastrophically different consequence. Clustered
failure is a different risk from scattered failure and the tests separate them.

Model comparison uses Diebold-Mariano (1995) with the Harvey-Leybourne-Newbold
(1997) small-sample correction, so "model A beats model B" is a test result and
not an eyeballed difference of two means.

CAVEAT, STATED UP FRONT
-----------------------
Rolling origins overlap in their training data, so per-origin results are not
independent draws. Aggregates here are descriptive; the significance tests are
applied WITHIN an origin (across its blocks), where the sample is a genuine time
series, not ACROSS origins. Nothing in this file claims a p-value on the
origin-level mean.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
OUT = HERE.parent / "output"
OUT.mkdir(exist_ok=True)


# ----------------------------------------------------------------- metrics --

def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    """Pinball (quantile) loss — the proper scoring rule for a single quantile."""
    e = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                   alpha: float) -> float:
    """Winkler / interval score for a central (1-alpha) interval. Lower better.

    Gneiting & Raftery (2007), eq. 43. Width plus a penalty of 2/alpha per unit
    of miss on whichever side was missed. Proper, so a forecaster cannot improve
    it by shading the band either wider or narrower than its honest value —
    which is exactly the failure mode that "coverage 94%" concealed.
    """
    y, lo, hi = np.asarray(y), np.asarray(lo), np.asarray(hi)
    return float(np.mean((hi - lo)
                         + (2 / alpha) * (lo - y) * (y < lo)
                         + (2 / alpha) * (y - hi) * (y > hi)))


def _safe_ll(k: int, n: int, p: float) -> float:
    """k successes out of n under rate p, in logs, with the degenerate cases
    (p at 0 or 1) returning 0 for the terms that vanish rather than -inf."""
    out = 0.0
    if k:
        out += k * np.log(p) if p > 0 else -np.inf
    if n - k:
        out += (n - k) * np.log(1 - p) if p < 1 else -np.inf
    return out


def kupiec_pof(exceed: np.ndarray, p: float) -> dict:
    """Unconditional coverage test (Kupiec 1995, proportion-of-failures).

    H0: the exceedance rate equals its nominal level p. Rejecting means the band
    is the wrong SIZE. LR ~ chi2(1).
    """
    e = np.asarray(exceed).astype(int)
    n, x = len(e), int(e.sum())
    if n == 0:
        return {"n": 0, "exceedances": 0, "rate_pct": None, "lr": None, "p_value": None}
    pi = x / n
    lr = -2 * (_safe_ll(x, n, p) - _safe_ll(x, n, pi))
    lr = float(max(lr, 0.0)) if np.isfinite(lr) else None
    return {"n": n, "exceedances": x, "rate_pct": round(pi * 100, 2),
            "nominal_pct": round(p * 100, 2), "lr": None if lr is None else round(lr, 3),
            "p_value": None if lr is None else round(float(stats.chi2.sf(lr, 1)), 4)}


def christoffersen(exceed: np.ndarray, p: float) -> dict:
    """Independence and conditional-coverage tests (Christoffersen 1998).

    LR_ind asks whether an exceedance today predicts one tomorrow. Clustered
    failures are a different — and worse — risk than scattered ones at the same
    rate, because they are what a drawdown is made of. LR_cc = LR_uc + LR_ind
    tests both jointly, chi2(2).
    """
    e = np.asarray(exceed).astype(int)
    if len(e) < 2:
        return {"lr_ind": None, "p_value_ind": None, "lr_cc": None, "p_value_cc": None}
    prev, cur = e[:-1], e[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    tot = n00 + n01 + n10 + n11
    if tot == 0 or (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        # no transitions to learn from — independence is untestable, not passed
        uc = kupiec_pof(e, p)
        return {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
                "lr_ind": None, "p_value_ind": None,
                "lr_cc": None, "p_value_cc": None,
                "note": "too few transitions to test independence"}
    pi = (n01 + n11) / tot
    pi0, pi1 = n01 / (n00 + n01), n11 / (n10 + n11)
    ll_uncond = _safe_ll(n01 + n11, tot, pi)
    ll_markov = _safe_ll(n01, n00 + n01, pi0) + _safe_ll(n11, n10 + n11, pi1)
    lr_ind = float(max(-2 * (ll_uncond - ll_markov), 0.0))
    uc = kupiec_pof(e, p)
    lr_cc = None if uc["lr"] is None else float(uc["lr"] + lr_ind)
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "lr_ind": round(lr_ind, 3),
            "p_value_ind": round(float(stats.chi2.sf(lr_ind, 1)), 4),
            "lr_cc": None if lr_cc is None else round(lr_cc, 3),
            "p_value_cc": None if lr_cc is None else round(float(stats.chi2.sf(lr_cc, 2)), 4)}


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> dict:
    """Is forecast A significantly better than B? (Diebold & Mariano 1995.)

    Operates on LOSS SERIES, not errors, so it works for any loss — pinball,
    interval score, squared error. Variance is HAC-corrected to lag h-1 because
    h-step forecast errors are MA(h-1) by construction, and the small-sample
    correction of Harvey, Leybourne & Newbold (1997) is applied because the
    uncorrected statistic over-rejects badly at these sample sizes.

    Negative statistic => A has lower loss => A is better.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 8 or np.allclose(d, 0):
        return {"n": n, "stat": None, "p_value": None, "better": "indistinguishable"}
    dbar = float(d.mean())
    dc = d - dbar
    gamma0 = float(np.mean(dc * dc))
    var = gamma0 + 2.0 * sum(float(np.mean(dc[k:] * dc[:-k])) for k in range(1, h))
    if var <= 0:
        return {"n": n, "stat": None, "p_value": None, "better": "indistinguishable"}
    dm = dbar / np.sqrt(var / n)
    corr = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    dm_star = float(dm * corr)
    pval = float(2 * stats.t.sf(abs(dm_star), df=n - 1))
    better = "A" if dm_star < 0 else "B"
    return {"n": n, "stat": round(dm_star, 3), "p_value": round(pval, 4),
            "better": better if pval < 0.05 else "indistinguishable"}


# ------------------------------------------------------------- the harness --

@dataclass
class Origin:
    train_end: pd.Timestamp     # model may use data strictly before this
    test_start: pd.Timestamp
    test_end: pd.Timestamp      # exclusive


def origins(index: pd.DatetimeIndex, n: int = 6, test_days: int = 30,
            min_train_days: int = 180) -> list[Origin]:
    """Consecutive non-overlapping test windows, latest last.

    Non-overlapping TEST windows matter: overlapping ones would count the same
    day several times and make the worst-window statistic optimistic. Training
    windows do overlap — that is inherent to expanding-window evaluation and is
    why per-origin results are not independent draws (see module docstring).
    """
    end = index.max().normalize() + pd.Timedelta(days=1)
    start = index.min().normalize()
    out = []
    for k in range(n):
        te = end - pd.Timedelta(days=test_days * k)
        ts = te - pd.Timedelta(days=test_days)
        if (ts - start).days < min_train_days:
            break
        out.append(Origin(train_end=ts, test_start=ts, test_end=te))
    return list(reversed(out))


# A task fits on data strictly before origin.train_end and predicts the window.
# It returns a frame indexed by timestamp with an "actual" column plus either
# "point", or "lo"/"hi" (and optionally "mid") for an interval forecast.
Task = Callable[[Origin], pd.DataFrame]


@dataclass
class Spec:
    name: str
    task: Task
    alpha: float | None = None        # set for interval forecasts (0.2 => 80%)
    unit: str = ""
    benchmark: Task | None = None     # optional, for the Diebold-Mariano test
    quantiles: tuple = field(default_factory=tuple)
    # How much history this model needs before its first origin, and how many
    # origins are worth running. Per-model because the series differ by years:
    # the load panel has 5 years, RTM has 367 days. A single global floor of 730
    # days silently produced ZERO origins for RTM — the harness refused to audit
    # it at all, which is the one outcome worse than auditing it badly.
    min_train_days: int | None = None
    n_origins: int | None = None


def _point_metrics(y, p, unit) -> dict:
    e = y - p
    denom = np.maximum(np.abs(y), 1e-9)
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        # WAPE, not MAPE: one division at the end. MAPE explodes when the target
        # approaches zero, which RTM prices genuinely do.
        "wape_pct": float(np.sum(np.abs(e)) / np.sum(denom) * 100),
        "bias": float(np.mean(p - y)),
        "unit": unit,
    }


def _interval_metrics(y, lo, hi, alpha) -> dict:
    below, above = y < lo, y > hi
    exceed = below | above
    return {
        "interval_score": interval_score(y, lo, hi, alpha),
        "coverage_pct": float(np.mean(~exceed) * 100),
        "nominal_pct": float((1 - alpha) * 100),
        "width": float(np.mean(hi - lo)),
        "below_pct": float(np.mean(below) * 100),
        "above_pct": float(np.mean(above) * 100),
        "kupiec": kupiec_pof(exceed, alpha),
        "christoffersen": christoffersen(exceed, alpha),
    }


def run(spec: Spec, n_origins: int = 6, test_days: int = 30,
        min_train_days: int = 180, index: pd.DatetimeIndex | None = None,
        verbose: bool = True) -> dict:
    """Walk `spec` across rolling origins and summarise honestly."""
    if index is None:
        raise ValueError("run() needs the model's own data index to place origins")
    n_origins = spec.n_origins or n_origins
    min_train_days = spec.min_train_days or min_train_days
    orgs = origins(index, n=n_origins, test_days=test_days,
                   min_train_days=min_train_days)
    if not orgs:
        span = (index.max() - index.min()).days
        raise RuntimeError(
            f"{spec.name}: {span} days of history cannot support even one "
            f"{test_days}d origin behind a {min_train_days}d training minimum. "
            f"Lower Spec.min_train_days or wait for the series to grow.")

    per, losses_a, losses_b = [], [], []
    for o in orgs:
        try:
            df = spec.task(o)
        except Exception as e:
            per.append({"test_start": str(o.test_start.date()),
                        "error": f"{type(e).__name__}: {str(e)[:160]}"})
            if verbose:
                print(f"  {o.test_start:%d %b}  FAILED {type(e).__name__}: {str(e)[:90]}")
            continue
        df = df.dropna(subset=["actual"])
        if not len(df):
            continue
        y = df["actual"].values
        rec = {"test_start": str(o.test_start.date()),
               "test_end": str((o.test_end - pd.Timedelta(days=1)).date()),
               "n": int(len(df))}

        if spec.alpha is not None and {"lo", "hi"} <= set(df.columns):
            rec.update(_interval_metrics(y, df["lo"].values, df["hi"].values,
                                         spec.alpha))
            losses_a.append((df["hi"].values - df["lo"].values)
                            + (2 / spec.alpha) * (df["lo"].values - y) * (y < df["lo"].values)
                            + (2 / spec.alpha) * (y - df["hi"].values) * (y > df["hi"].values))
        pcol = "point" if "point" in df.columns else (
            "mid" if "mid" in df.columns else None)
        if pcol:
            rec.update(_point_metrics(y, df[pcol].values, spec.unit))
            if spec.alpha is None:
                # a point spec's loss series for Diebold-Mariano. Without this
                # the DM test silently never ran for point models even when a
                # benchmark was wired — losses_a stayed empty and the length
                # check below quietly skipped the comparison.
                losses_a.append(np.abs(y - df[pcol].values))
        for q in spec.quantiles:
            col = f"q{q * 100:02.0f}"
            if col in df.columns:
                rec[f"pinball_{col}"] = pinball(y, df[col].values, q)

        if spec.benchmark is not None:
            try:
                bdf = spec.benchmark(o).reindex(df.index)
                bcol = "point" if "point" in bdf.columns else "mid"
                rec["benchmark_wape_pct"] = float(
                    np.sum(np.abs(y - bdf[bcol].values))
                    / np.sum(np.maximum(np.abs(y), 1e-9)) * 100)
                losses_b.append(np.abs(y - bdf[bcol].values))
            except Exception as e:
                rec["benchmark_error"] = f"{type(e).__name__}: {str(e)[:100]}"

        per.append(rec)
        if verbose:
            head = (f"IS {rec['interval_score']:8.1f}  cov {rec['coverage_pct']:5.1f}%"
                    f"  w {rec['width']:8.1f}" if "interval_score" in rec
                    else f"WAPE {rec.get('wape_pct', float('nan')):6.2f}%"
                         f"  MAE {rec.get('mae', float('nan')):8.1f}")
            print(f"  {o.test_start:%d %b %Y}  n{rec['n']:6,}  {head}")

    ok = [r for r in per if "error" not in r and r.get("n")]
    if not ok:
        return {"model": spec.name, "origins": per,
                "error": "every origin failed — see per-origin errors"}

    def agg(key, signed=False):
        vals = [r[key] for r in ok if r.get(key) is not None]
        if not vals:
            return None
        # "worst" means largest MAGNITUDE for a signed quantity like bias —
        # taking np.max of a signed series reported a +201 MW window as the
        # worst case for a model whose mean bias was -34 MW, which reads as the
        # opposite of the truth.
        worst = float(max(vals, key=abs)) if signed else float(np.max(vals))
        return {"mean": float(np.mean(vals)), "worst": worst,
                "best": float(np.min(np.abs(vals) if signed else vals)),
                "std": float(np.std(vals))}

    summary = {"model": spec.name, "unit": spec.unit,
               "origins_run": len(ok), "test_days": test_days,
               "window": f"{ok[0]['test_start']} -> {ok[-1]['test_end']}",
               "blocks": int(sum(r["n"] for r in ok))}

    if spec.alpha is not None:
        covs = [r["coverage_pct"] for r in ok if "coverage_pct" in r]
        if covs:
            # pooled tests over all origins: the honest headline, since a band is
            # a claim about the long run and not about one lucky month
            summary["coverage_pct_mean"] = round(float(np.mean(covs)), 1)
            summary["coverage_pct_worst"] = round(float(np.min(covs)), 1)
            summary["nominal_pct"] = ok[0]["nominal_pct"]
            summary["origins_below_nominal"] = int(sum(
                c < ok[0]["nominal_pct"] for c in covs))
            summary["interval_score_mean"] = round(
                float(np.mean([r["interval_score"] for r in ok])), 1)
            summary["width_mean"] = round(float(np.mean([r["width"] for r in ok])), 1)
            bad_uc = [r["test_start"] for r in ok
                      if (r["kupiec"] or {}).get("p_value") is not None
                      and r["kupiec"]["p_value"] < 0.05]
            bad_ind = [r["test_start"] for r in ok
                       if (r["christoffersen"] or {}).get("p_value_ind") is not None
                       and r["christoffersen"]["p_value_ind"] < 0.05]
            summary["kupiec_rejected_origins"] = bad_uc
            summary["independence_rejected_origins"] = bad_ind
    for k in ("wape_pct", "mae", "rmse", "bias"):
        a = agg(k, signed=(k == "bias"))
        if a:
            summary[k] = {kk: round(vv, 3) for kk, vv in a.items()}

    if losses_a and losses_b and len(losses_a) == len(losses_b):
        summary["vs_benchmark"] = diebold_mariano(np.concatenate(losses_a),
                                                  np.concatenate(losses_b))
    return {**summary, "origins": per}


def report(res: dict) -> str:
    """One human-readable block per model, for the metrics file and the console."""
    if res.get("error"):
        return f"{res['model']}: {res['error']}"
    L = [f"{res['model']}  —  {res['origins_run']} rolling origins "
         f"({res['test_days']}d each), {res['blocks']:,} blocks, {res['window']}"]
    if "coverage_pct_mean" in res:
        L.append(f"  interval score {res['interval_score_mean']:,.1f} "
                 f"(proper; width + miss penalty, lower is better)")
        L.append(f"  coverage {res['coverage_pct_mean']}% mean / "
                 f"{res['coverage_pct_worst']}% worst  vs {res['nominal_pct']}% nominal"
                 f"   width {res['width_mean']:,.1f}")
        L.append(f"  origins below nominal: {res['origins_below_nominal']}"
                 f"/{res['origins_run']}")
        if res.get("kupiec_rejected_origins"):
            L.append(f"  Kupiec REJECTS correct rate at: "
                     f"{', '.join(res['kupiec_rejected_origins'])}")
        if res.get("independence_rejected_origins"):
            L.append(f"  Christoffersen REJECTS independence (failures CLUSTER) at: "
                     f"{', '.join(res['independence_rejected_origins'])}")
    for k, lab in (("wape_pct", "WAPE %"), ("mae", "MAE"), ("rmse", "RMSE"),
                   ("bias", "bias")):
        if k in res:
            a = res[k]
            L.append(f"  {lab:8s} mean {a['mean']:>10,.3f}   worst {a['worst']:>10,.3f}"
                     f"   std {a['std']:>9,.3f}")
    if res.get("vs_benchmark"):
        b = res["vs_benchmark"]
        L.append(f"  Diebold-Mariano vs benchmark: stat {b.get('stat')} "
                 f"p={b.get('p_value')} -> {b.get('better')}")
    return "\n".join(L)


def save(results: list[dict], path: Path | None = None) -> Path:
    path = path or (OUT / "walkforward.json")
    path.write_text(json.dumps(
        {"generated_at": pd.Timestamp.now().isoformat(sep=" ", timespec="seconds"),
         "method": ("rolling origin, non-overlapping test windows, model refitted "
                    "at every origin; interval score is Gneiting & Raftery (2007), "
                    "coverage tests are Kupiec (1995) and Christoffersen (1998)"),
         "models": results}, indent=1, default=float), encoding="utf-8")
    return path
