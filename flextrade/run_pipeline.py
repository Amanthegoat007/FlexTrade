"""FlexTrade daily cycle — what runs at ~11:00 IST before DAM gate closure.

  1. Refresh live data: weather forecast, today's IEX DAM, latest SLDC
     load curves, realtime snapshot.
  2. Forecast tomorrow's 96 blocks: Delhi load + DAM MCP.
  3. LP-optimize the BESS schedule on the price forecast.
  4. Emit the DAM bid sheet (output/bid_sheet_<date>.csv) and the full
     plan (output/plan_<date>.csv) that the dashboard displays.
"""
import json
import socket
import sys
import time
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

# Never let a console encoding kill a pipeline run. Windows defaults stdout to
# cp1252, which cannot encode the arrows and rupee signs these modules print;
# _run_pipeline.cmd sets PYTHONIOENCODING=utf-8 but anything else invoking this
# (a shell, a cron, a human) does not, and the run then dies at stage
# "refresh_live" on a PRINT STATEMENT with every fetch already succeeded.
# Observed exactly that: UnicodeEncodeError on '→'.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest import iex, sldc, states, store, weather
from models import load_model, price_model
from optimize.dispatch import Bess, bid_sheet, optimize_dispatch

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
HEALTH = OUT / "pipeline_health.json"


def wait_for_network(hosts=("api.open-meteo.com", "www.iexindia.com"),
                     max_wait_s: int = 300) -> bool:
    """Block until DNS resolves, or give up after max_wait_s.

    Why this exists: Windows fires the scheduled task the moment the
    machine is awake, which is often BEFORE Wi-Fi/DNS is ready. On
    27 and 28 Jul the 11:00 run died with `getaddrinfo failed` while a
    manual run 25 minutes later worked perfectly — four days of plans
    lost to a race with the network stack.
    """
    deadline = time.time() + max_wait_s
    delay = 5
    while time.time() < deadline:
        for h in hosts:
            try:
                socket.getaddrinfo(h, 443)
                return True
            except OSError:
                continue
        print(f"  network not ready — retrying in {delay}s")
        time.sleep(delay)
        delay = min(delay * 2, 60)
    return False


def write_health(stage: str, ok: bool, detail: str = "",
                 extra: dict | None = None) -> None:
    """Record the outcome of every run so a failure is VISIBLE.

    The 4-day outage was not caused by the DNS error alone — it was
    caused by that error being invisible. The dashboard reads this file
    and shows a banner when the last run failed or is stale.
    """
    rec = {"last_run_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
           "stage": stage, "ok": ok, "detail": detail[:400], **(extra or {})}
    try:
        HEALTH.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    except Exception:
        pass


def refresh_live() -> dict:
    status = {}
    _, status["weather"] = weather.get_forecast(days=2)
    _, status["re_wx"] = weather.get_re_forecast(days=2)
    _, status["iex_dam"] = iex.get_today()
    _, status["iex_rtm"] = iex.get_rtm_today()
    _, status["iex_gdam"] = iex.get_gdam_today()
    _, status["sldc"] = sldc.get_realtime()
    # frequency drives the DSM charge rate; SLDC serves no history, so
    # each run samples today's curve and the record builds up over time
    _, status["sldc_freq"] = sldc.get_frequency()
    # all-India + 23-state live position (MERIT, Ministry of Power)
    _, status["merit"] = states.get_india_snapshot()
    # UP's own SLDC — the only state besides Delhi with a first-party feed, and
    # the only one anywhere that publishes schedule/drawal/deviation. Non-fatal:
    # it enriches the panel, it is not on the path to a bid sheet.
    # First-party state feeds. Both are non-fatal: they enrich the panel and
    # are not on the path to a bid sheet. Both refuse to store an implausible
    # reading rather than poison the series with it.
    for name, mod in (("upsldc", "ingest.upsldc"), ("kptcl", "ingest.kptcl")):
        try:
            m = __import__(mod, fromlist=["poll"])
            snap = m.poll()
            status[name] = {"live": True,
                            "asof": snap.get("source_updated")
                            or datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        except Exception as e:
            print(f"  {name:8s} SKIPPED  {type(e).__name__}: {str(e)[:90]}")
    for k, v in status.items():
        print(f"  {k:8s} {'LIVE' if v['live'] else 'CACHED':6s} asof {v['asof']}")

    # keep the load history current — the model's lag features need D-2/D-3,
    # so a silent gap here quietly degrades every forecast downstream
    failed = sldc.ensure_load_current()
    if failed:
        print(f"  WARNING: {len(failed)} day(s) still missing: "
              f"{', '.join(str(d) for d in failed)}")
    _, status["iex_hist"] = iex.ensure_prices_current()
    return status


def plan_tomorrow(bess: Bess = Bess(), retries: int = 2) -> pd.DataFrame:
    target = date.today() + timedelta(days=1)
    print(f"planning delivery day {target}")

    # A missing target-day feature almost always means a transient fetch
    # failure upstream, so re-pull the inputs and try again rather than
    # abandoning the day (see the 27-28 Jul DNS outage).
    for attempt in range(retries + 1):
        try:
            return _plan(target, bess)
        except Exception as e:
            if attempt >= retries:
                raise
            print(f"  plan attempt {attempt + 1} failed ({str(e)[:120]}) — "
                  f"refreshing inputs and retrying")
            wait_for_network()
            try:
                weather.get_forecast(days=3)
                weather.get_re_forecast(days=3)
                sldc.ensure_load_current()
                iex.ensure_prices_current()
            except Exception as re_err:
                print(f"  refresh during retry failed: {str(re_err)[:120]}")
            time.sleep(5)


def _plan(target: date, bess: Bess) -> pd.DataFrame:
    load_fc = load_model.forecast_day(target)
    price_fc = price_model.forecast_day(target)
    plan = load_fc.join(price_fc)

    sched, exp_pnl = optimize_dispatch(plan["forecast_mcp"], bess)
    plan = plan.join(sched[["charge_mw", "discharge_mw", "soc_mwh", "bess_mw"]])
    bids = bid_sheet(sched, plan["forecast_mcp"])

    plan.to_csv(OUT / f"plan_{target}.csv", index_label="ts")
    plan.to_csv(OUT / "plan_latest.csv", index_label="ts")
    bids.to_csv(OUT / f"bid_sheet_{target}.csv", index=False)
    bids.to_csv(OUT / "bid_sheet_latest.csv", index=False)

    n_trades = (bids["side"] != "-").sum()
    print(f"  expected P&L Rs {exp_pnl:,.0f} | {n_trades} block bids "
          f"| peak load fc {plan['forecast_load_mw'].max():,.0f} MW")
    return plan


def refresh_forecast_lab(target: date | None = None) -> dict:
    """Recompute the day-ahead artifacts of the newer models.

    Deliberately non-fatal, one stage per model. These enrich the dashboard;
    none of them is on the critical path to a bid sheet, so a failure here
    must degrade the Forecast Lab page rather than take down the plan the
    trading desk actually needs at 12:00. Each outcome is recorded so a
    quietly-missing panel is still visible as a failure somewhere.
    """
    target = target or (date.today() + timedelta(days=1))
    results: dict[str, str] = {}

    steps = [
        ("peak", lambda: __import__(
            "models.peak_model", fromlist=["x"]).forecast(target)),
        ("dsm_exposure", _refresh_dsm_exposure),
        ("stress", _refresh_stress),
    ]
    for name, fn in steps:
        try:
            fn()
            results[name] = "ok"
        except Exception as e:
            results[name] = f"{type(e).__name__}: {str(e)[:120]}"
            print(f"  forecast lab / {name}: FAILED {results[name]}")
    ok = [k for k, v in results.items() if v == "ok"]
    print(f"  forecast lab: {len(ok)}/{len(results)} ok ({', '.join(ok) or 'none'})")
    return results


def _refresh_stress():
    """State Grid Stress reads live MERIT + the latest coal/outage days, so it
    has to be rebuilt each run or the dashboard shows yesterday's exposure."""
    import json as _json
    from models import stress
    (OUT / "stress.json").write_text(
        _json.dumps(stress.build(), indent=2, default=float))


def _refresh_dsm_exposure():
    import json as _json
    from models import dsm_forecast
    tgt = date.today() + timedelta(days=1)
    entity_mw = 200.0
    payload = {
        "exposure": dsm_forecast.exposure(tgt, available_capacity_mw=entity_mw),
        "bias_sensitivity": dsm_forecast.bias_sensitivity(
            tgt, available_capacity_mw=entity_mw),
    }
    (OUT / "dsm_forecast.json").write_text(
        _json.dumps(payload, indent=2, default=float))


def export_for_web():
    """Refresh every JSON artifact the MERN app serves (../flextrade-web)."""
    import subprocess
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "export_web.py")],
                       capture_output=True, text=True, timeout=600)
    print("  web export:", "ok" if r.returncode == 0 else f"FAILED\n{r.stderr[-400:]}")


if __name__ == "__main__":
    # every stage records its outcome; a silent failure is the one bug
    # this pipeline is not allowed to have again
    stage = "startup"
    try:
        stage = "network"
        if not wait_for_network():
            raise RuntimeError("no network after 5 minutes of waiting")

        stage = "refresh_live"
        refresh_live()

        stage = "plan_tomorrow"
        plan = plan_tomorrow()

        # write health BEFORE the export so the export captures it — the
        # dashboard reads health out of meta.json
        summary = {
            "delivery_day": str(date.today() + timedelta(days=1)),
            "peak_load_mw": round(float(plan["forecast_load_mw"].max()), 0)
            if plan is not None and "forecast_load_mw" in plan else None,
        }
        write_health("complete", True, extra=summary)

        stage = "forecast_lab"
        summary["forecast_lab"] = refresh_forecast_lab()

        stage = "export_for_web"
        export_for_web()
        write_health("complete", True, extra=summary)
        print("pipeline OK")
    except Exception as exc:
        write_health(stage, False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        print(f"pipeline FAILED at stage '{stage}' — recorded in "
              f"{HEALTH.name}; the dashboard will show a staleness banner")
        sys.exit(1)
