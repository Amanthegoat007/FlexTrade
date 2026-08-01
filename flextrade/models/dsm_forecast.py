"""Day-ahead DSM exposure forecast -- what deviation will cost tomorrow.

models/dsm.py settles a deviation that has already happened. This module
answers the question a DISCOM scheduler actually has at 12:00 on D-1:

    if I submit THIS schedule, what will DSM cost me tomorrow, and what
    schedule costs me least?

Nothing here needs a new data feed. It composes what is already built:

  the load model's        183 days of REALISED forecast error on a held-out
  held-out errors         window, kept as whole-day curves -- this is the
                          only measured error distribution we own, and it is
                          what makes the deviation a real random variable
  models/price_model      tomorrow's DAM curve
  models/rtm_model        tomorrow's RTM curve (day-ahead horizon)
                          -> together these give Reg 14's Normal Rate, which
                             is what CERC 2024 prices deviation at
  models/dsm              the settlement rule itself, versioned by profile
                          and effective date

Two things this module deliberately does NOT do, both recorded here because
the reasons are the interesting part:

  * it does not price an RE developer's plant, which is the case with the
    strongest commercial pull. We hold zero rows of realised RE generation
    (re_weather is forecast-only), so a plant's error distribution would
    have to be invented, and an invented distribution priced in rupees is
    worse than no feature.
  * it does not recommend a schedule bias -- see `bias_sensitivity`.

A note on frequency, because it changes what this module needed to be. The
obvious design is "forecast grid frequency, then price deviation off it" --
that is how DSM worked for years and it is what we assumed we would have to
backfill NRLDC history for. It is no longer how the regulation works: CERC
DE-LINKED deviation charges from frequency in the 2022 regulations, and
nothing in the notified 2024 text reinstates it (see the research note at
the top of models/dsm.py). So no frequency history is required, and the
7 days we hold is not a blocker for this feature at all. The frequency-linked
variant stays available behind `freq_linked=True` for scenario work only.

The output is a distribution, not a number: expected charge, P90 bad day,
and the probability of ending up outside the tolerance band at all.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from models import dsm  # noqa: E402

OUT = HERE.parent / "output"
OUT.mkdir(exist_ok=True)

DEFAULT_PROFILE = "CERC_2024"
N_SCENARIOS = 60          # historical error days sampled per run
_ERROR_CACHE: pd.DataFrame | None = None


def error_scenarios(n: int = N_SCENARIOS) -> pd.DataFrame:
    """Empirical day-shaped forecast errors, one row per historical day.

    This is the part that has to be real, and it is the part that is easy to
    get wrong. The first version of this module used the load model's OWN
    quantiles as the "actual" scenarios -- which makes the P50 case deviate
    from a P50 schedule by exactly zero and turns the whole calculation into
    a tautology with a runaway optimum.

    What actually prices DSM is how far realised drawal lands from the
    forecast, so the scenarios are the model's realised errors on its
    held-out test window: 183 days it never trained on, each kept as a whole
    96-block CURVE rather than a pool of independent blocks. That matters --
    forecast errors are strongly correlated within a day (a hot afternoon is
    under-forecast across every evening block at once), and sampling blocks
    independently would cancel that out and understate the exposure badly.

    Returns a frame of relative errors (actual/forecast - 1), indexed by
    historical day, with 96 columns.
    """
    global _ERROR_CACHE
    if _ERROR_CACHE is not None:
        return _ERROR_CACHE.sample(min(n, len(_ERROR_CACHE)), random_state=42)

    import importlib.util
    import numpy as _np
    from models import load_model

    lf_dir = HERE.parent.parent / "load_forecast"
    spec = importlib.util.spec_from_file_location(
        "lf_train", lf_dir / "02_train_model.py")
    lf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lf)

    df = pd.read_parquet(lf_dir / "data" / "model_table.parquet")
    f = lf.build_features(df).dropna(subset=lf.FEATURES + ["load_mw"])
    test_start = f.index.max() - pd.DateOffset(months=6)
    te = f[f.index >= test_start]
    pred = _np.mean([b.predict(te[lf.FEATURES]) for b in load_model._boosters()],
                    axis=0)
    rel = pd.Series((te["load_mw"].values - pred) / _np.maximum(pred, 1.0),
                    index=te.index)
    block = rel.index.hour * 4 + rel.index.minute // 15
    wide = rel.groupby([rel.index.normalize(), block]).mean().unstack()
    wide = wide.reindex(columns=range(96)).dropna(thresh=90)
    wide = wide.interpolate(axis=1, limit_direction="both")
    _ERROR_CACHE = wide
    return wide.sample(min(n, len(wide)), random_state=42)


def _prices(target: date) -> tuple[pd.Series, pd.Series]:
    """Tomorrow's DAM and RTM curves -- the two inputs to the Normal Rate."""
    from models import price_model
    dam = price_model.forecast_day(target)["forecast_mcp"]
    try:
        from models import rtm_model
        rtm = rtm_model.forecast_day(target, horizon="dayahead")["forecast_rtm"]
        rtm = rtm.reindex(dam.index)
        if rtm.isna().all():
            raise ValueError("empty RTM forecast")
        rtm = rtm.fillna(dam)
        rtm_basis = "rtm_model day-ahead"
    except Exception:
        # Without an RTM view the Normal Rate degrades to the DAM curve.
        # Saying so matters: NR averages DAM, RTM and ancillary, so proxying
        # two of three legs with one price understates its dispersion.
        rtm = dam.copy()
        rtm_basis = "DAM proxy (RTM model unavailable)"
    dam.attrs["rtm_basis"] = rtm_basis
    return dam, rtm


def _load_band(target: date) -> pd.DataFrame:
    from models import load_quantile
    return load_quantile.forecast_day(target)


def exposure(target: date | None = None, schedule_mw: pd.Series | None = None,
             profile: str = DEFAULT_PROFILE, seller: str = "buyer",
             technology: str = "general",
             available_capacity_mw: float | None = None) -> dict:
    """DSM cost distribution for `target` under `schedule_mw`.

    `schedule_mw` defaults to the P50 load forecast -- i.e. "schedule what
    you expect", the behaviour we are trying to price.
    """
    target = target or (date.today() + timedelta(days=1))
    band = _load_band(target)
    dam, rtm = _prices(target)

    idx = band.index.intersection(dam.index)
    band, dam, rtm = band.loc[idx], dam.loc[idx], rtm.loc[idx]
    forecast = band["p50"]
    sched = (forecast if schedule_mw is None
             else pd.Series(schedule_mw).reindex(idx).ffill().bfill())

    # Scale the whole entity to `available_capacity_mw` if one is given, so
    # the same machinery prices a 20 MW battery and a 6 GW drawal book. The
    # deviation PERCENTAGE is what the regulation acts on, so scaling is
    # exact rather than an approximation.
    scale = 1.0
    if available_capacity_mw:
        peak = float(forecast.max())
        scale = available_capacity_mw / peak if peak > 0 else 1.0
    forecast, sched = forecast * scale, sched * scale
    cap = available_capacity_mw or float(forecast.max())

    errs = error_scenarios()
    block_of = np.asarray(idx.hour * 4 + idx.minute // 15)

    nets, payables, outside, per_scenario = [], [], [], []
    for day, row in errs.iterrows():
        actual = forecast * (1.0 + row.to_numpy()[block_of])
        settled = dsm.settle(
            profile, actual_mw=actual, scheduled_mw=sched,
            frequency_hz=50.0, dam_price=dam, rtm_price=rtm,
            available_capacity_mw=cap, seller=seller, technology=technology,
            settlement_date=target)
        s = dsm.summarize(settled)
        nets.append(s["net_dsm_rs"])
        payables.append(s["charge_payable_rs"])
        outside.append(s["blocks_outside_band"])
        per_scenario.append({"error_day": str(pd.Timestamp(day).date()),
                             "net_dsm_rs": s["net_dsm_rs"],
                             "payable_rs": s["charge_payable_rs"],
                             "blocks_outside_band": s["blocks_outside_band"],
                             "mae_mw": s["mae_mw"]})
    if not nets:
        raise RuntimeError("no scenarios produced a settlement")

    net = np.array(nets, dtype=float)
    pay = np.array(payables, dtype=float)
    out_arr = np.array(outside, dtype=float)
    nblocks = len(idx)
    # charge_rs is negative for a credit, so the BAD tail is the high end
    per_scenario.sort(key=lambda r: -r["payable_rs"])

    return {
        "day": str(target),
        "profile": profile,
        "seller": seller,
        "technology": technology,
        "blocks": int(nblocks),
        "n_scenarios": int(len(net)),
        "entity_peak_mw": round(float(forecast.max()), 1),
        "schedule_mwh": float(sched.sum() * dsm.BLOCK_H),
        "expected_dsm_rs": float(net.mean()),
        # The number a scheduler is actually exposed to. Credits earned by
        # deviating the profitable way are deliberately NOT netted off here:
        # our settlement engine implements the general-seller Normal Rate
        # without the over-injection caps the real regulation carries, so a
        # net-of-credits objective would happily recommend under-scheduling
        # forever. Payable charge is the honest, non-gameable objective.
        "expected_payable_rs": float(pay.mean()),
        "p90_payable_rs": float(np.quantile(pay, 0.90)),
        "worst_payable_rs": float(pay.max()),
        "p90_dsm_rs": float(np.quantile(net, 0.90)),
        "p10_dsm_rs": float(np.quantile(net, 0.10)),
        "worst_scenario_rs": float(net.max()),
        "best_scenario_rs": float(net.min()),
        "expected_blocks_outside_band": float(out_arr.mean()),
        "expected_pct_outside_band": float(out_arr.mean() / nblocks * 100),
        "normal_rate_mean_rs_mwh": float(((dam + rtm + rtm) / 3).mean()),
        "rtm_basis": dam.attrs.get("rtm_basis", "unknown"),
        "scenario_basis": (
            f"{len(net)} whole-day forecast-error curves sampled from the load "
            f"model's 6-month held-out window (day shape preserved)"),
        "worst_scenarios": per_scenario[:5],
    }


def bias_sensitivity(target: date | None = None, profile: str = DEFAULT_PROFILE,
                     seller: str = "buyer", technology: str = "general",
                     available_capacity_mw: float | None = None,
                     biases_pct=(-4, -3, -2, -1, 0, 1, 2, 3, 4)) -> dict:
    """How exposure varies if the schedule is biased off the forecast.

    Reported as a SENSITIVITY CURVE, and deliberately NOT as a
    recommendation. The intended product was "here is the schedule bias
    that minimises your DSM cost", and it is not shipped, for a reason
    worth stating plainly:

    the optimum sits at whichever end of the sweep we stop at. Under the
    general-seller Normal Rate as implemented in models/dsm.py, deviating
    in the profitable direction earns credit at the full Normal Rate with
    no cap, so scheduling lower always looks better -- on net AND on
    payable charge, since payable only accrues on the other side. A real
    settlement is two-sided: CERC constrains over-injection precisely so
    this is not free money. Our engine does not implement those caps, so an
    optimum derived from it would be an artifact of a missing rule, and
    following it would be advising a customer to game a settlement on the
    strength of an incomplete model.

    So the curve is published as information -- it shows the genuine
    trade-off between expected charge and how often you leave the band --
    and the recommendation is withheld until the over-injection limits are
    sourced and implemented. What IS sound here is `exposure()`: the
    distribution of what a given schedule costs, driven by 183 days of real
    measured forecast error.
    """
    target = target or (date.today() + timedelta(days=1))
    band = _load_band(target)
    p50 = band["p50"]

    rows = []
    for b in biases_pct:
        r = exposure(target, schedule_mw=p50 * (1 + b / 100.0), profile=profile,
                     seller=seller, technology=technology,
                     available_capacity_mw=available_capacity_mw)
        rows.append({"bias_pct": b,
                     "expected_payable_rs": r["expected_payable_rs"],
                     "p90_payable_rs": r["p90_payable_rs"],
                     "expected_dsm_rs": r["expected_dsm_rs"],
                     "pct_outside_band": r["expected_pct_outside_band"]})
    flat = next(r for r in rows if r["bias_pct"] == 0)
    argmin = min(rows, key=lambda r: r["expected_payable_rs"])["bias_pct"]
    boundary = argmin in (min(biases_pct), max(biases_pct))
    return {
        "day": str(target),
        "profile": profile,
        "curve": rows,
        "unconstrained_argmin_pct": argmin,
        "argmin_at_boundary": bool(boundary),
        "objective": "expected payable DSM charge (credits not netted)",
        "p50_schedule_payable_rs": flat["expected_payable_rs"],
        "p50_pct_outside_band": flat["pct_outside_band"],
        "recommendation": None,
        "recommendation_withheld_because": (
            "the minimum sits at the edge of the sweep: the implemented "
            "general-seller Normal Rate credits favourable deviation without "
            "the caps the real regulation applies, so any optimum would be an "
            "artifact of a missing rule rather than a real saving"
            if boundary else
            "interior optimum found, but schedule-bias advice is withheld "
            "until over-injection limits are sourced and implemented"),
    }


if __name__ == "__main__":
    import json
    tgt = date.today() + timedelta(days=1)
    ENTITY_MW = 200.0     # a mid-size scheduled generator / trading book
    e = exposure(tgt, available_capacity_mw=ENTITY_MW)
    print(f"DSM exposure for {e['day']}  ({e['profile']}, {e['seller']})")
    print(f"  schedule {e['schedule_mwh']:,.0f} MWh | normal rate "
          f"Rs {e['normal_rate_mean_rs_mwh']:,.0f}/MWh | RTM: {e['rtm_basis']}")
    print(f"  expected PAYABLE charge  Rs {e['expected_payable_rs']:>10,.0f}")
    print(f"  P90 bad day              Rs {e['p90_payable_rs']:>10,.0f}")
    print(f"  worst day in sample      Rs {e['worst_payable_rs']:>10,.0f}")
    print(f"  net incl. credits        Rs {e['expected_dsm_rs']:>10,.0f}"
          "   (not the objective -- see module docstring)")
    print(f"  blocks outside band: {e['expected_pct_outside_band']:.1f}% expected")
    print()
    print(f"  basis: {e['scenario_basis']}")
    print("  worst error days in the sample:")
    for s in e["worst_scenarios"]:
        print(f"    {s['error_day']}  payable Rs {s['payable_rs']:>10,.0f}   "
              f"{s['blocks_outside_band']:>3} blocks outside band   "
              f"MAE {s['mae_mw']:.1f} MW")

    print()
    o = bias_sensitivity(tgt, available_capacity_mw=ENTITY_MW)
    print("schedule-bias SENSITIVITY (information, not a recommendation):")
    for r in o["curve"]:
        mark = "  <- flat P50 schedule" if r["bias_pct"] == 0 else ""
        print(f"  {r['bias_pct']:+3d}%  payable Rs {r['expected_payable_rs']:>10,.0f}"
              f"   outside band {r['pct_outside_band']:5.1f}%{mark}")
    print(f"\n  no schedule-bias recommended. "
          f"{o['recommendation_withheld_because']}")

    (OUT / "dsm_forecast.json").write_text(json.dumps(
        {"exposure": e, "bias_sensitivity": o}, indent=2, default=float))
