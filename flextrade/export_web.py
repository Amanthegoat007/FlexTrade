"""Export layer for the FlexTrade web app (MERN stack in ../flextrade-web).

Dumps everything the Node/React app needs as JSON under output/web/:

    meta.json       generated_at, per-table data freshness, dataset stats,
                    parsed model metrics — the source of every date shown
                    in the UI (nothing is hardcoded there)
    live.json       realtime Delhi snapshot + Northern Region states +
                    BRPL BESS latest (written by --live, called by the
                    Node server's /api/refresh on a rate limit)
    plan.json       tomorrow's plan + bid sheet + price quantile band
    backtest.json   arbitrage + risk backtests, daily rows + summaries
    dsm.json        yesterday's DSM comparison under both CERC profiles
                    + current Alerts & Revision output
    states.json     the state adapter registry
    bess.json       BRPL telemetry history + validation results

Run modes:
    python export_web.py          # full export (models, backtests, dsm)
    python export_web.py --live   # fast path: live.json only (~3 s)

The daily pipeline calls the full export at the end of its run; the Node
server calls --live on demand. Single implementation of every fetch stays
in Python — Node only reads files and the SQLite DB, never re-implements
a scraper.
"""
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ingest import store  # noqa: E402

OUT = HERE / "output"
WEB = OUT / "web"
WEB.mkdir(parents=True, exist_ok=True)


def _sanitise(o):
    """Replace NaN/Infinity with null, recursively.

    Python's json.dumps happily emits bare `NaN`/`Infinity`, which are NOT
    valid JSON: the browser's JSON.parse throws and the whole endpoint dies
    in the client while looking fine on disk. One NaN age_hours took out
    /api/meta this way. Every artifact goes through here.
    """
    if isinstance(o, float):
        return None if (o != o or o in (float("inf"), float("-inf"))) else o
    if isinstance(o, dict):
        return {k: _sanitise(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitise(v) for v in o]
    return o


def _dump(name: str, obj) -> None:
    def default(o):
        if isinstance(o, (pd.Timestamp, datetime, date)):
            return o.isoformat(sep=" ") if not isinstance(o, date) or isinstance(o, datetime) else o.isoformat()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if np.isnan(v) or np.isinf(v) else v
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(f"{type(o)} not serializable")

    # allow_nan=False turns any remaining non-finite float into a loud
    # error here rather than silently invalid JSON in the browser
    text = json.dumps(_sanitise(obj), default=default, indent=1, allow_nan=False)
    (WEB / name).write_text(text, encoding="utf-8")
    print(f"  wrote {name}")


def _records(df: pd.DataFrame, round_to: int = 2) -> list[dict]:
    df = df.reset_index()
    if "ts" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "ts"})
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].astype(str)
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(round_to)
    return df.replace({np.nan: None}).to_dict("records")


# ---------------------------------------------------------------- meta ----

TABLE_NOTES = {
    "load_5min": ("Delhi SLDC 5-min load by DISCOM", "delhisldc.org day curves"),
    "dam_price": ("IEX Day-Ahead Market, 96 blocks/day", "iexindia.com scrape"),
    "rtm_price": ("IEX Real-Time Market", "iexindia.com scrape"),
    "gdam_price": ("IEX Green DAM incl. fuel split", "iexindia.com scrape"),
    "weather": ("Delhi hourly weather, actual + forecast", "Open-Meteo API"),
    "re_weather": ("Irradiance + hub-height wind for the RE twin", "Open-Meteo API"),
    "frequency": ("Grid frequency, 5-min (sampled daily, no history exists upstream)", "delhisldc.org image-map"),
    "bess_telemetry": ("BRPL Kilokari BESS: MW, kVAr, SoC (sampled every 5 min)", "delhisldc.org/bess.aspx"),
    "northern_region_snapshot": ("8 Northern Grid states: schedule/drawl/load", "delhisldc.org states table"),
    "state_live": ("23 states: demand met / own generation / import (MW)", "meritindia.in (Ministry of Power)"),
    "national_snapshot": ("All-India demand met + generation by fuel (MW)", "meritindia.in (Ministry of Power)"),
}


def export_meta():
    freshness, datasets = {}, []
    with store.connect() as con:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        for t in tables:
            try:
                n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
                ts_col = "ts" if "ts" in cols else ("fetched_at" if "fetched_at" in cols else None)
                lo = hi = None
                if ts_col:
                    lo, hi = con.execute(
                        f"SELECT MIN({ts_col}), MAX({ts_col}) FROM {t}").fetchone()
                freshness[t] = {"rows": n, "from": lo, "to": hi}
                desc, src = TABLE_NOTES.get(t, ("", ""))
                datasets.append({"table": t, "rows": n, "from": lo, "to": hi,
                                 "columns": cols, "description": desc, "source": src})
            except Exception:
                continue

    def read_txt(p: Path):
        return p.read_text().strip() if p.exists() else None

    metrics = {
        "load_model": read_txt(HERE.parent / "load_forecast" / "output" / "metrics.txt"),
        "price_model": read_txt(OUT / "metrics_price.txt"),
        "price_quantiles": read_txt(OUT / "metrics_price_quantile.txt"),
        "backtest_summary": read_txt(OUT / "backtest_summary.txt"),
        "risk_backtest_summary": read_txt(OUT / "risk_backtest_summary.txt"),
    }
    conformal = OUT / "price_conformal.json"
    if conformal.exists():
        metrics["conformal"] = json.loads(conformal.read_text())
    lab = OUT / "model_lab.json"
    if lab.exists():
        metrics["model_lab"] = json.loads(lab.read_text())
    try:
        from models import forecast_monitor
        ra = forecast_monitor.realized()
        metrics["realized_accuracy"] = ra.replace({np.nan: None}).to_dict("records") \
            if len(ra) else []
    except Exception as e:
        metrics["realized_accuracy"] = {"error": str(e)[:120]}

    # pipeline health — so a broken run shows up on the dashboard instead
    # of hiding in a log file (see the 27-28 Jul DNS outage)
    health = {}
    hf = OUT / "pipeline_health.json"
    if hf.exists():
        # utf-8-sig: tolerate a BOM. A BOM-prefixed file made json.loads
        # throw, the health block came back empty, and the banner stayed
        # hidden — i.e. the staleness alarm itself failed silently.
        try:
            health = json.loads(hf.read_text(encoding="utf-8-sig"))
        except Exception as e:
            health = {"ok": False, "stage": "health-file-unreadable",
                      "detail": f"{type(e).__name__}: {e}"}
        age = None
        try:
            last = pd.Timestamp(health.get("last_run_at"))
            if pd.notna(last):
                age = round((pd.Timestamp.now() - last).total_seconds() / 3600, 1)
        except Exception:
            age = None
        health["age_hours"] = age  # never NaN — see _dump's sanitiser
    # the plan is the pipeline's product: if it isn't for tomorrow, say so
    plan_f = OUT / "plan_latest.csv"
    if plan_f.exists():
        try:
            pl = pd.read_csv(plan_f, parse_dates=["ts"], nrows=1)
            pday = pl["ts"].iloc[0].date()
            health["plan_delivery_day"] = str(pday)
            health["plan_is_current"] = pday >= date.today() + timedelta(days=1)
            health["plan_days_stale"] = max(
                0, (date.today() + timedelta(days=1) - pday).days)
        except Exception:
            pass

    _dump("meta.json", {
        "generated_at": datetime.now(),
        "freshness": freshness,
        "datasets": datasets,
        "metrics": metrics,
        "health": health,
    })


# ---------------------------------------------------------------- live ----

def export_live():
    from ingest import bess, iex, sldc, states
    out = {"generated_at": datetime.now()}

    snap, meta = sldc.get_realtime()
    out["delhi"] = {**{k: v for k, v in snap.items() if k != "fetched_at"},
                    "live": meta["live"], "asof": str(meta["asof"])}

    nr, nmeta = states.get_northern_region_snapshot()
    out["northern_region"] = {"live": nmeta["live"], "asof": str(nmeta["asof"]),
                              "states": _records(nr.set_index("state")) if len(nr) else []}

    india, imeta = states.get_india_snapshot()
    out["india"] = {"live": imeta["live"], "asof": str(imeta["asof"]),
                    "national": india["national"],
                    "states": _records(india["states"].set_index("code"))
                    if len(india["states"]) else []}

    b, bmeta = bess.poll_once()
    out["bess"] = {**{k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in b.items()},
                   "live": bmeta["live"],
                   "rated": {"power_mw": bess.RATED_POWER_MW,
                             "energy_mwh": bess.RATED_ENERGY_MWH}}

    for name, fn in [("dam", iex.get_today), ("rtm", iex.get_rtm_today),
                     ("gdam", iex.get_gdam_today)]:
        df, m = fn()
        if len(df):
            out[name] = {"live": m["live"], "asof": str(m["asof"]),
                         "avg_mcp": round(float(df["mcp_rs_mwh"].mean()), 0),
                         "max_mcp": round(float(df["mcp_rs_mwh"].max()), 0),
                         "min_mcp": round(float(df["mcp_rs_mwh"].min()), 0),
                         "blocks": _records(df[["mcp_rs_mwh", "mcv_mw"]])}
    _dump("live.json", out)


# ---------------------------------------------------------------- plan ----

def export_plan():
    plan_f = OUT / "plan_latest.csv"
    if not plan_f.exists():
        return
    plan = pd.read_csv(plan_f, parse_dates=["ts"], index_col="ts")
    bids = pd.read_csv(OUT / "bid_sheet_latest.csv") \
        if (OUT / "bid_sheet_latest.csv").exists() else pd.DataFrame()

    quantiles = None
    try:
        from models import price_model
        qf = price_model.forecast_day_quantiles()
        quantiles = _records(qf)
    except Exception as e:
        print(f"  (quantiles skipped: {e})")

    _dump("plan.json", {
        "generated_at": datetime.now(),
        "delivery_day": str(plan.index[0].date()),
        "blocks": _records(plan),
        "bid_sheet": bids.replace({np.nan: None}).to_dict("records"),
        "price_quantiles": quantiles,
        "expected_pnl_rs": round(float(
            ((plan["discharge_mw"] - plan["charge_mw"]) * 0.25 * plan["forecast_mcp"]).sum()
        ), 0) if "discharge_mw" in plan else None,
        "peak_load_mw": round(float(plan["forecast_load_mw"].max()), 0)
        if "forecast_load_mw" in plan else None,
    })


# ------------------------------------------------------------ backtests ----

def export_backtests():
    out = {"generated_at": datetime.now()}
    f = OUT / "backtest_daily.csv"
    if f.exists():
        d = pd.read_csv(f, parse_dates=["date"])
        d["date"] = d["date"].astype(str)
        out["arbitrage"] = {
            "daily": d.round(0).replace({np.nan: None}).to_dict("records"),
            "totals": {c: round(float(d[c].sum()), 0)
                       for c in d.columns if c.startswith("pnl")},
        }
    f = OUT / "risk_backtest_daily.csv"
    if f.exists():
        d = pd.read_csv(f, parse_dates=["date"])
        d["date"] = d["date"].astype(str)
        out["risk"] = {
            "daily": d.round(0).replace({np.nan: None}).to_dict("records"),
            "totals": {c: round(float(d[c].sum()), 0)
                       for c in d.columns if c.startswith("pnl")},
        }
    _dump("backtest.json", out)


# ----------------------------------------------------------------- dsm ----

def export_dsm():
    from models import dsm_alerts, re_model
    out = {"generated_at": datetime.now(),
           "settlement_day": str(date.today() - timedelta(days=1))}
    for profile in ["CERC_2024", "CERC_2022"]:
        try:
            df, flex, naive = re_model.dsm_comparison_cerc(profile=profile)
            out[profile] = {
                "flextrade": flex, "naive": naive,
                "saved_rs": round(naive["net_dsm_rs"] - flex["net_dsm_rs"], 0)
                if flex else None,
                "blocks": _records(df[["schedule_mw", "actual_mw", "naive_mw",
                                       "solar_schedule_mw", "solar_actual_mw",
                                       "wind_schedule_mw", "wind_actual_mw"]])
                if len(df) else [],
            }
        except Exception as e:
            out[profile] = {"error": str(e)}
    try:
        al = dsm_alerts.next_gate_alerts()
        out["alerts"] = {
            "asof": al["asof"], "lead_minutes": al["lead_minutes"],
            "total_benefit_rs": round(al["total_benefit_rs"], 0),
            "schedule_basis": al["schedule_basis"],
            "items": [{
                "block": a.block_ts.strftime("%H:%M"), "action": a.action,
                "scheduled_mw": round(a.current_schedule_mw, 1),
                "forecast_mw": round(a.forecast_mw, 1),
                "benefit_rs": round(a.benefit_rs, 0), "reason": a.reason,
            } for a in al["alerts"]],
        }
    except Exception as e:
        out["alerts"] = {"error": str(e)}
    _dump("dsm.json", out)


# --------------------------------------------------------------- states ----

def export_states():
    from ingest import states
    out = {
        "generated_at": datetime.now(),
        "registry": [{
            "code": s.code, "name": s.name, "grid_region": s.grid_region,
            "status": s.status, "peak_load_gw": s.peak_load_gw, "notes": s.notes,
        } for s in states.list_states()],
    }
    snap, meta = states.get_india_snapshot()
    out["india"] = {
        "live": meta["live"], "asof": str(meta["asof"]),
        "national": snap["national"],
        "states": _records(snap["states"].set_index("code"))
        if len(snap["states"]) else [],
    }
    try:
        out["gujarat_direct"] = {**states.fetch_gujarat_realtime(),
                                 "source": "sldcguj.com homepage (no login)"}
    except Exception as e:
        out["gujarat_direct"] = {"error": str(e)[:120]}
    # Rajasthan deep-adapter health, reported honestly either way
    try:
        out["rajasthan_direct"] = {**states.fetch_rajasthan_overview(),
                                   "source": "sldc.rajasthan.gov.in read-sftp"}
    except Exception as e:
        out["rajasthan_direct"] = {
            "error": str(e)[:120],
            "note": ("endpoint mapped (freq/DSM rate/load/generation tags) but "
                     "currently 500s upstream — their own homepage widget is "
                     "equally broken; MERIT covers Rajasthan demand meanwhile")}

    # import-dependence analytics: which states most need price forecasts.
    # The ₹/day figure values current import MW at today's DAM average for
    # 24 h — an EXPOSURE PROXY (real supply is mostly long-term PPAs, with
    # exchange purchases at the margin), labelled as such in the UI.
    try:
        sdf = snap["states"]
        dam_avg = None
        dam = store.read("dam_price")
        if len(dam):
            today_dam = dam[dam.index.date == dam.index.date.max()]["mcp_rs_mwh"]
            dam_avg = float(today_dam.mean())
        rows = []
        for _, s in sdf.iterrows():
            if s["demand_mw"] and s["import_mw"] is not None:
                share = 100 * s["import_mw"] / s["demand_mw"]
                rows.append({
                    "code": s["code"], "name": s["name"],
                    "demand_mw": s["demand_mw"], "import_mw": s["import_mw"],
                    "import_share_pct": round(share, 1),
                    "exposure_rs_day": round(s["import_mw"] * 24 * dam_avg, 0)
                    if dam_avg and s["import_mw"] > 0 else None,
                })
        rows.sort(key=lambda r: r["import_share_pct"], reverse=True)
        out["import_dependence"] = {
            "dam_avg_rs_mwh": round(dam_avg, 0) if dam_avg else None,
            "note": ("exposure = import MW × 24 h × today's DAM average — a "
                     "volatility-exposure proxy, not a bill (most supply is "
                     "long-term PPA; exchange purchases sit at the margin)"),
            "rows": rows,
        }
    except Exception as e:
        out["import_dependence"] = {"error": str(e)[:120]}

    # per-state forecast readiness (the data-accrual story, told honestly)
    try:
        from models import state_model
        out["forecast_readiness"] = {
            "gate": f"{state_model.MIN_DAYS} days @ ≥{state_model.MIN_COVERAGE:.0%} coverage",
            "recipe_proof": "Delhi load model: 4.98% test MAPE on the same recipe",
            "rows": state_model.readiness().to_dict("records"),
        }
    except Exception as e:
        out["forecast_readiness"] = {"error": str(e)[:120]}

    _dump("states.json", out)


def export_state_forecast():
    """The pooled multi-state forecaster + per-state profiles (workspace page)."""
    from ingest import merit_history

    # the coverage tiers, stated plainly — what each state can actually support
    tiers = {
        "intraday_native": {
            "states": ["DL"],
            "resolution": "15-minute, day-ahead",
            "basis": "~5 years of Delhi SLDC 5-min load history",
            "accuracy": "4.33% test MAPE",
            "note": ("the only state that publishes intraday load history, so "
                     "the only one with a block-level model"),
        },
        "daily_pooled": {
            "states": merit_history.HISTORY_FULL,
            "resolution": "daily energy + exchange purchases, day-ahead",
            "basis": "MERIT daily history, pooled global LightGBM",
            "note": ("one model trained across these states at once — short "
                     "series borrow strength from each other (M4/M5 result)"),
        },
        "daily_pooled_partial": {
            "states": merit_history.HISTORY_PARTIAL,
            "resolution": "daily, gappy",
            "basis": "MERIT history exists but is intermittent",
            "note": "included in the pooled model with fewer rows; metrics shown per state",
        },
        "live_monitoring_only": {
            "states": merit_history.HISTORY_NONE,
            "resolution": "live demand / own generation / import",
            "basis": "MERIT live endpoint (verified), no historical series published",
            "note": ("MERIT returns well-formed responses with null energy values "
                     "for these states — we monitor them and say so rather than "
                     "inventing a forecast"),
        },
    }

    f = OUT / "state_forecast.json"
    if not f.exists():
        _dump("state_forecast.json", {
            "generated_at": datetime.now(), "tiers": tiers,
            "error": "not trained yet — run models/state_forecast.py"})
        return
    obj = json.loads(f.read_text(encoding="utf-8-sig"))
    obj["generated_at"] = datetime.now()
    obj["tiers"] = tiers
    try:
        obj["history_coverage"] = _records(merit_history.coverage())
    except Exception:
        pass
    _dump("state_forecast.json", obj)


# -------------------------------------------------------------- modules ----

def export_modules():
    """RTM re-optimization, physics degradation, C&I peak shaving."""
    from optimize import degradation, peak_shave, rtm_reopt
    from optimize.dispatch import Bess
    out = {"generated_at": datetime.now()}

    # intraday RTM re-optimization for the rest of today
    try:
        r = rtm_reopt.reoptimize()
        sched = r.pop("schedule", None)
        out["rtm"] = {**r, "schedule": _records(sched) if sched is not None else []}
    except Exception as e:
        out["rtm"] = {"error": str(e)}

    # physics-aware degradation on the latest full day of actual DAM prices
    try:
        dam = store.read("dam_price")
        day = dam.index.date.max()
        prices = dam[dam.index.date == day]["mcp_rs_mwh"]
        r = degradation.optimize_physical(prices)
        sched = r.pop("schedule")
        p = degradation.DegParams()
        out["degradation"] = {
            **r, "day": str(day),
            "schedule": _records(sched[["price", "soc_mwh", "bess_mw"]]),
            "marginal_curve": [{"dod_pct": round(d * 100), "rs_per_mwh": round(p.rs_per_mwh(d), 0)}
                               for d in np.arange(0.1, 1.01, 0.1)],
        }
    except Exception as e:
        out["degradation"] = {"error": str(e)}

    # three-way DAM+RTM+DSM co-optimization (compliant envelope)
    try:
        from optimize import threeway
        r = threeway.cooptimize()
        sched = r.pop("schedule", None)
        out["threeway"] = {**r, "schedule": _records(sched) if sched is not None else []}
    except Exception as e:
        out["threeway"] = {"error": str(e)[:150]}

    # warranty & availability guard (real telemetry + tomorrow's plan)
    try:
        from models import warranty
        aud = warranty.audit_telemetry()
        plan_f = OUT / "plan_latest.csv"
        plan_chk = None
        if plan_f.exists():
            plan = pd.read_csv(plan_f, parse_dates=["ts"], index_col="ts")
            if "soc_mwh" in plan:
                plan_chk = warranty.audit_schedule(plan["soc_mwh"], 40.0)
        out["warranty"] = {**aud, "plan_check": plan_chk}
    except Exception as e:
        out["warranty"] = {"error": str(e)[:150]}

    # thermal derating on the committed plan
    try:
        from optimize import thermal
        out["thermal"] = thermal.heat_cost()
    except Exception as e:
        out["thermal"] = {"error": str(e)[:150]}

    # frequency-response readiness on our sampled history
    try:
        from models import freq_response
        out["freq_response"] = freq_response.readiness()
    except Exception as e:
        out["freq_response"] = {"error": str(e)[:150]}

    # sizing & bankability curves (precomputed; refresh with models/sizing.py)
    try:
        from models import sizing as _sizing
        out["sizing"] = _sizing.export()
    except Exception as e:
        out["sizing"] = {"error": str(e)[:150]}

    # C&I peak shaving (illustrative profile, clearly labelled)
    try:
        day_ts = pd.Timestamp.today().normalize()
        load = peak_shave.factory_profile(day_ts, peak_mw=5.0)
        cni_bess = Bess(power_mw=2.0, energy_mwh=4.0, degradation_rs_mwh=843.0)
        r = peak_shave.optimize_peak_shave(load, cni_bess)
        sched = r.pop("schedule")
        out["cni"] = {
            **r,
            "profile_note": ("illustrative 5 MW two-shift factory with a 14:00-17:00 "
                             "process peak — replaceable by a pilot customer's meter data"),
            "bess": {"power_mw": cni_bess.power_mw, "energy_mwh": cni_bess.energy_mwh},
            "schedule": _records(sched),
        }
    except Exception as e:
        out["cni"] = {"error": str(e)}

    _dump("modules.json", out)


# ----------------------------------------------------------------- bess ----

def export_bess():
    from ingest import bess
    hist = bess.read_history()
    out = {"generated_at": datetime.now(),
           "rated": {"power_mw": bess.RATED_POWER_MW,
                     "energy_mwh": bess.RATED_ENERGY_MWH},
           "readings": len(hist)}
    if len(hist):
        try:
            recent = hist[hist.index >= hist.index.max() - pd.Timedelta("48h")]
        except Exception:
            recent = hist.tail(576)
        out["history"] = _records(recent[["discharge_mw", "soc_pct"]], 3)
        out["span"] = {"from": str(hist.index.min()), "to": str(hist.index.max())}
    vf = OUT / "bess_validation.csv"
    if vf.exists():
        v = pd.read_csv(vf)
        out["validation"] = v.replace({np.nan: None}).to_dict("records")
    _dump("bess.json", out)


def _metrics_text(name: str) -> str | None:
    f = OUT / name
    try:
        return f.read_text(encoding="utf-8-sig") if f.exists() else None
    except Exception:
        return None


def export_forecasts():
    """The forecast lab: RTM, probabilistic load, peak, DSM exposure.

    Every block here is allowed to be absent. A model that has not been
    trained yet must leave a null and an explanation rather than break the
    export, because a half-populated dashboard is recoverable and a failed
    pipeline stage at 11:00 is not.
    """
    out = {"generated_at": datetime.now()}

    # ---- RTM price + DAM->RTM spread ---------------------------------
    try:
        champs = json.loads((OUT / "rtm_champions.json").read_text(encoding="utf-8-sig"))
        out["rtm"] = {
            "horizons": {
                h: {"champion": v.get("champion"),
                    "incumbent": v.get("incumbent"),
                    "served_wape_pct": v.get("served", {}).get("wape"),
                    "served_direction_pct": v.get("served", {}).get("direction_pct"),
                    "model_wape_pct": v.get("model", {}).get("wape"),
                    "model_direction_pct": v.get("model", {}).get("direction_pct"),
                    "incumbent_wape_pct": v.get("incumbent_scores", {}).get("wape"),
                    "incumbent_direction_pct":
                        v.get("incumbent_scores", {}).get("direction_pct"),
                    "test_from": v.get("test_from"), "test_to": v.get("test_to"),
                    "n_train": v.get("n_train"), "n_test": v.get("n_test")}
                for h, v in champs.items()},
            "metric_note": (
                "WAPE (sum of absolute error / sum of actual), not MAPE: RTM "
                "clears at Rs 0 on real blocks and 3.2% of the test window is "
                "below Rs 100, so a per-block percentage divides by near-zero "
                "and reports nonsense."),
            "why_direction": (
                "Direction is the share of blocks where we get the SIGN of "
                "(RTM - DAM) right. It is the number that decides money: a "
                "level forecast that is close on average but wrong about the "
                "sign tells the optimizer to trade the wrong way."),
            "metrics_text": _metrics_text("metrics_rtm.txt"),
        }
    except Exception as e:
        out["rtm"] = {"error": f"not trained: {e}"}

    # ---- probabilistic load ------------------------------------------
    try:
        conf = json.loads((OUT / "load_conformal.json").read_text(encoding="utf-8-sig"))
        band = None
        try:
            from models import load_quantile
            band = _records(load_quantile.forecast_day())
        except Exception as e:
            print(f"  (load band skipped: {e})")
        out["load_quantiles"] = {
            "conformal": conf,
            "tomorrow": band,
            "metrics_text": _metrics_text("metrics_load_quantile.txt"),
        }
    except Exception as e:
        out["load_quantiles"] = {"error": f"not trained: {e}"}

    # ---- peak timing + magnitude --------------------------------------
    try:
        summary = json.loads((OUT / "peak_summary.json").read_text(encoding="utf-8-sig"))
        try:
            from models import peak_model
            summary["tomorrow"] = peak_model.forecast()
        except Exception as e:
            print(f"  (peak forecast skipped: {e})")
        summary["metrics_text"] = _metrics_text("metrics_peak.txt")
        out["peak"] = summary
    except Exception as e:
        out["peak"] = {"error": f"not trained: {e}"}

    # ---- DSM exposure --------------------------------------------------
    f = OUT / "dsm_forecast.json"
    if f.exists():
        try:
            out["dsm_exposure"] = json.loads(f.read_text(encoding="utf-8-sig"))
        except Exception as e:
            out["dsm_exposure"] = {"error": str(e)}
    else:
        out["dsm_exposure"] = {"error": "not computed yet"}

    _dump("forecasts.json", out)


def export_sqlite_series():
    """Pre-render the two endpoints the Node server answers from SQLite.

    A static build has no server and no 88 MB database, so without this the
    Overview load chart and the BESS telemetry chart would be the only empty
    panels on an otherwise complete page. Both queries are the same ones in
    flextrade-web/server/index.js, kept deliberately small (a few hundred
    rows each) so the whole static site stays around a megabyte.
    """
    with store.connect() as con:
        load = pd.read_sql("""
            SELECT strftime('%Y-%m-%d %H:', ts) ||
                   printf('%02d', (CAST(strftime('%M', ts) AS INT) / 15) * 15)
                   || ':00' AS block,
                   ROUND(AVG(delhi), 1) AS delhi_mw
            FROM load_5min
            WHERE ts >= datetime((SELECT MAX(ts) FROM load_5min), '-3 days')
            GROUP BY block ORDER BY block""", con)
        bess = pd.read_sql("""
            SELECT ts, discharge_mw, soc_pct FROM bess_telemetry
            WHERE ts >= datetime((SELECT MAX(ts) FROM bess_telemetry),
                                 '-48 hours')
            ORDER BY ts""", con)
    _dump("load_recent.json", {"days": 3,
                               "rows": load.replace({np.nan: None}).to_dict("records")})
    _dump("bess_history.json", {"hours": 48,
                                "rows": bess.replace({np.nan: None}).to_dict("records")})


if __name__ == "__main__":
    if "--live" in sys.argv:
        export_live()
    elif "--static" in sys.argv:
        # everything a server-less build needs, in one pass
        print("static export:")
        export_meta()
        export_live()
        export_plan()
        export_backtests()
        export_dsm()
        export_states()
        export_bess()
        export_modules()
        export_state_forecast()
        export_forecasts()
        export_sqlite_series()
        export_meta()
    else:
        print("full web export:")
        export_meta()
        export_live()
        export_plan()
        export_backtests()
        export_dsm()
        export_states()
        export_bess()
        export_modules()
        export_state_forecast()
        export_forecasts()
        export_meta()  # re-dump so freshness reflects the fetches above
