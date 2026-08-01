"""FlexTrade operator dashboard — multi-segment platform view.

    streamlit run app.py

One tab per customer segment from the business model (BESS operator,
RE developer, DISCOM & C&I, energy trader) plus a business tab that maps
the running system to the revenue streams (SaaS, Forecast-as-a-Service,
revenue share, transaction fees).
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ingest import bess as bess_live  # noqa: E402
from ingest import iex, sldc, store, weather  # noqa: E402
from models import re_model  # noqa: E402
from optimize.dispatch import Bess, bid_sheet, greedy_dispatch, optimize_dispatch, settle  # noqa: E402

OUT = HERE / "output"

# dataviz palette (light mode)
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"
RED, GRAY, GRID = "#e34948", "#898781", "#e1e0d9"

st.set_page_config(page_title="FlexTrade — Energy Trading Platform",
                   page_icon="⚡", layout="wide")


def badge(name: str, meta: dict) -> str:
    if meta.get("live"):
        return f":green-badge[⚡ {name}: LIVE]"
    asof = str(meta.get("asof") or "never")[:16]
    return f":orange-badge[● {name}: CACHED {asof}]"


def chart_layout(fig, height=380):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#fcfcfb", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, Segoe UI, sans-serif", color="#52514e"),
        legend=dict(orientation="h", y=1.08, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


@st.cache_data(ttl=300, show_spinner=False)
def live_data():
    wx, wx_meta = weather.get_forecast(days=2)
    dam, dam_meta = iex.get_today()
    rtm, rtm_meta = iex.get_rtm_today()
    snap, snap_meta = sldc.get_realtime()
    return dict(wx=wx, wx_meta=wx_meta, dam=dam, dam_meta=dam_meta,
                rtm=rtm, rtm_meta=rtm_meta, snap=snap, snap_meta=snap_meta)


@st.cache_data(ttl=600, show_spinner=False)
def re_data(profile: str = "CERC_2024"):
    fc = re_model.forecast_day()
    df, flex, naive = re_model.dsm_comparison_cerc(profile=profile)
    return fc, df, flex, naive


@st.cache_data(ttl=120, show_spinner=False)
def dsm_alerts_data():
    from models import dsm_alerts
    return dsm_alerts.next_gate_alerts()


@st.cache_data(ttl=180, show_spinner=False)
def northern_region_data():
    from ingest import states
    return states.get_northern_region_snapshot()


@st.cache_data(ttl=300, show_spinner=False)
def gdam_data():
    return iex.get_gdam_today()


@st.cache_data(ttl=600, show_spinner=False)
def price_bands():
    from models import price_model
    return price_model.forecast_day_quantiles()


@st.cache_data(ttl=120, show_spinner=False)
def brpl_bess():
    row, meta = bess_live.poll_once()
    return row, meta, bess_live.read_history()


@st.cache_data(ttl=300, show_spinner=False)
def latest_plan():
    f = OUT / "plan_latest.csv"
    if not f.exists():
        return None
    return pd.read_csv(f, parse_dates=["ts"], index_col="ts")


# ---------- header ----------
st.title("⚡ FlexTrade — AI Energy Trading & Optimization Platform")
st.caption("Delhi grid · IEX DAM + RTM · live data with cached fallback")

L = live_data()
st.markdown(" ".join([
    badge("Delhi SLDC", L["snap_meta"]), badge("IEX DAM", L["dam_meta"]),
    badge("IEX RTM", L["rtm_meta"]), badge("Open-Meteo", L["wx_meta"]),
]))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Delhi load now", f"{L['snap'].get('delhi_load', 0):,.0f} MW")
c2.metric("Grid frequency", f"{L['snap'].get('frequency', 0):.2f} Hz")
if len(L["dam"]):
    c3.metric("DAM avg MCP today", f"₹{L['dam']['mcp_rs_mwh'].mean():,.0f}/MWh")
if len(L["rtm"]):
    c4.metric("RTM avg MCP today", f"₹{L['rtm']['mcp_rs_mwh'].mean():,.0f}/MWh")
plan = latest_plan()
if plan is not None:
    c5.metric("Peak load forecast (D+1)", f"{plan['forecast_load_mw'].max():,.0f} MW")

# ---------- sidebar ----------
st.sidebar.header("BESS asset")
power = st.sidebar.slider("Power (MW)", 5.0, 100.0, 20.0, 5.0)
hours = st.sidebar.slider("Duration (h)", 1.0, 6.0, 2.0, 0.5)
eff = st.sidebar.slider("Round-trip efficiency", 0.80, 0.98, 0.90, 0.01)
deg = st.sidebar.slider("Degradation cost (₹/MWh)", 0, 1000, 200, 50)
share = st.sidebar.slider("FlexTrade revenue share %", 5, 40, 20, 5)
bess = Bess(power_mw=power, energy_mwh=power * hours,
            round_trip_eff=eff, degradation_rs_mwh=deg)

st.sidebar.header("Risk appetite")
lam = st.sidebar.slider(
    "λ — risk aversion", 0.0, 1.0, 0.5, 0.1,
    help="0 = maximize expected profit (risk-neutral). "
         "1 = maximize the worst 10% of outcomes (pure CVaR). "
         "In between trades a little mean profit for a better bad day.")

tab_bess, tab_re, tab_grid, tab_trader, tab_states, tab_biz = st.tabs([
    "🔋 BESS Operator", "☀️ RE Developer", "🏭 DISCOM & C&I",
    "📈 Energy Trader", "🗺️ Multi-State", "💼 Business Model"])

# ================= BESS OPERATOR =================
with tab_bess:
    # ---- live telemetry from a real operating battery ----
    st.subheader("Live asset — BRPL Kilokari BESS")
    st.caption("India's first utility-scale standalone BESS (20 MW / 40 MWh, "
               "COD Apr 2025, Kilokari 33/11 kV substation). Telemetry "
               "published by Delhi SLDC — this is a real battery, not a "
               "simulation.")
    try:
        row, bmeta, bhist = brpl_bess()
        state = ("⚡ Discharging" if row["discharge_mw"] > 0.05 else
                 "🔌 Charging" if row["discharge_mw"] < -0.05 else "⏸ Idle")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("State", state)
        r2.metric("Net power", f"{row['discharge_mw']:+.2f} MW",
                  help="Positive = exporting to grid")
        r3.metric("State of Charge", f"{row['soc_pct']:.0f}%",
                  help=f"≈ {row['soc_mwh']:.1f} MWh of 40 MWh")
        r4.metric("Telemetry readings", f"{len(bhist):,}",
                  help="Sampled by FlexTrade — SLDC publishes no history")
        st.progress(min(row["soc_pct"] / 100, 1.0),
                    text=f"SoC {row['soc_pct']:.0f}% of 40 MWh")

        if len(bhist) > 3:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_scatter(x=bhist.index, y=bhist["soc_pct"], mode="lines",
                            name="SoC (%)", line=dict(color=YELLOW, width=2))
            fig.add_scatter(x=bhist.index, y=bhist["discharge_mw"], mode="lines",
                            name="Net MW", line=dict(color=BLUE, width=1.5),
                            secondary_y=True)
            fig.update_layout(title="Observed BRPL BESS behaviour")
            st.plotly_chart(chart_layout(fig, 260), width="stretch")

        vfile = OUT / "bess_validation.csv"
        if vfile.exists():
            vdf = pd.read_csv(vfile, parse_dates=["day"], index_col="day")
            if len(vdf) and vdf["blocks_observed"].sum() >= 8:
                st.markdown("**Head-to-head: what the real battery earned vs "
                            "what FlexTrade's schedule would have earned**, "
                            "both valued at the same actual IEX prices.")
                v1, v2, v3 = st.columns(3)
                v1.metric("Real dispatch revenue",
                          f"₹{vdf['real_revenue_rs'].sum():,.0f}")
                v2.metric("FlexTrade schedule",
                          f"₹{vdf['flex_revenue_rs'].sum():,.0f}")
                v3.metric("Arbitrage value unmonetised",
                          f"₹{vdf['uplift_rs'].sum():,.0f}")
                st.caption("BRPL's battery is a regulated DISCOM asset "
                           "dispatched for grid support, not pure arbitrage — "
                           "the gap sizes the opportunity, it is not a "
                           "judgement of the operator.")
            else:
                st.info(f"Validation accumulating — {int(vdf['blocks_observed'].sum())} "
                        "market blocks observed so far. Run `python poll_bess.py` "
                        "to keep sampling.")
        else:
            st.info("Run `python poll_bess.py` to accumulate telemetry, then "
                    "`python validate/bess_validate.py` for the head-to-head.")
    except Exception as e:
        st.warning(f"BESS telemetry unavailable: {e}")

    st.divider()
    st.subheader("Tomorrow's optimized plan")
    if plan is None:
        st.info("No plan yet — run `python run_pipeline.py`.")
    else:
        sched, exp_pnl = optimize_dispatch(plan["forecast_mcp"], bess)
        target = plan.index[0].date()

        m1, m2, m3 = st.columns(3)
        m1.metric(f"Expected arbitrage P&L ({target})", f"₹{exp_pnl:,.0f}")
        m2.metric("Energy traded", f"{sched['discharge_mw'].sum() * 0.25:,.0f} MWh")
        m3.metric("Avg spread captured",
                  f"₹{(sched.loc[sched.discharge_mw > 0, 'price'].mean() - sched.loc[sched.charge_mw > 0, 'price'].mean()):,.0f}/MWh")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.4, 0.32, 0.28], vertical_spacing=0.06,
                            subplot_titles=("Delhi load forecast (MW)",
                                            "DAM price forecast (₹/MWh)",
                                            "BESS schedule (MW) & SoC (MWh)"))
        fig.add_scatter(x=plan.index, y=plan["forecast_load_mw"], mode="lines",
                        name="Load forecast", line=dict(color=BLUE, width=2),
                        row=1, col=1)
        fig.add_scatter(x=plan.index, y=plan["forecast_mcp"], mode="lines",
                        name="MCP forecast", line=dict(color=AQUA, width=2),
                        fill="tozeroy", fillcolor="rgba(27,175,122,0.08)",
                        row=2, col=1)
        fig.add_bar(x=sched.index, y=sched["discharge_mw"], name="Discharge (sell)",
                    marker_color=BLUE, row=3, col=1)
        fig.add_bar(x=sched.index, y=-sched["charge_mw"], name="Charge (buy)",
                    marker_color=RED, row=3, col=1)
        fig.add_scatter(x=sched.index, y=sched["soc_mwh"], mode="lines",
                        name="SoC (MWh)",
                        line=dict(color=YELLOW, width=2, dash="dot"), row=3, col=1)
        fig.update_layout(barmode="relative")
        st.plotly_chart(chart_layout(fig, 620), width="stretch")

        # ---- risk-aware dispatch on the forecast distribution ----
        st.divider()
        st.subheader("Risk-aware bidding")
        st.caption("A DAM bid is committed before the price is known, and our "
                   "price forecast carries real uncertainty. So instead of "
                   "optimizing against one guessed price curve, we forecast the "
                   "*distribution*, draw scenarios, and pick the schedule that "
                   "balances expected profit against the worst outcomes (CVaR).")
        try:
            from optimize import stochastic as sto
            qf = price_bands()
            scen = sto.make_scenarios(qf, n_scenarios=24)
            sched_cv, stats_cv = sto.optimize_cvar(qf, bess, lam=lam, scenarios=scen)
            pnl_pt = sto.evaluate(sched, scen, bess.degradation_rs_mwh)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Expected P&L (risk-aware)",
                      f"₹{stats_cv['expected_pnl']:,.0f}",
                      delta=f"{stats_cv['expected_pnl'] - pnl_pt.mean():+,.0f} vs point")
            cv_pt = pnl_pt[pnl_pt <= np.quantile(pnl_pt, 0.10)].mean()
            k2.metric("CVaR (worst 10%)", f"₹{stats_cv['cvar_pnl']:,.0f}",
                      delta=f"{stats_cv['cvar_pnl'] - cv_pt:+,.0f} vs point",
                      help="Mean P&L across the worst 10% of price scenarios")
            k3.metric("Worst scenario", f"₹{stats_cv['worst_pnl']:,.0f}",
                      delta=f"{stats_cv['worst_pnl'] - pnl_pt.min():+,.0f} vs point")
            k4.metric("P&L volatility", f"₹{stats_cv['std_pnl']:,.0f}",
                      delta=f"{stats_cv['std_pnl'] - pnl_pt.std():+,.0f} vs point",
                      delta_color="inverse")

            fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                                row_heights=[0.58, 0.42], vertical_spacing=0.14,
                                subplot_titles=(
                                    "Price forecast with conformal P10–P90 band",
                                    "P&L distribution across 24 price scenarios"))
            lo, mid, hi = qf.columns[0], qf.columns[1], qf.columns[-1]
            fig.add_scatter(x=qf.index, y=qf[hi], mode="lines", name="P90",
                            line=dict(width=0), showlegend=False, row=1, col=1)
            fig.add_scatter(x=qf.index, y=qf[lo], mode="lines", name="P10–P90",
                            line=dict(width=0), fill="tonexty",
                            fillcolor="rgba(42,120,214,0.18)", row=1, col=1)
            fig.add_scatter(x=qf.index, y=qf[mid], mode="lines", name="P50 (median)",
                            line=dict(color=BLUE, width=2), row=1, col=1)
            fig.add_histogram(x=pnl_pt, name="Point forecast", opacity=0.65,
                              marker_color=GRAY, nbinsx=14, row=2, col=1)
            fig.add_histogram(x=stats_cv["scenario_pnl"], name="Risk-aware",
                              opacity=0.65, marker_color=AQUA, nbinsx=14,
                              row=2, col=1)
            fig.update_layout(barmode="overlay")
            fig.update_xaxes(title="₹/MWh", row=1, col=1)
            fig.update_xaxes(title="day P&L (₹)", row=2, col=1)
            st.plotly_chart(chart_layout(fig, 560), width="stretch")

            rb = OUT / "risk_backtest_summary.txt"
            if rb.exists():
                with st.expander("Does it hold up? 60-day backtest"):
                    st.code(rb.read_text())
        except FileNotFoundError:
            st.info("Quantile models not trained yet — run "
                    "`python models/price_model.py quantiles`.")
        except Exception as e:
            st.warning(f"Risk-aware panel unavailable: {e}")

        st.divider()
        st.subheader("DAM bid sheet")
        bids = bid_sheet(sched, plan["forecast_mcp"])
        st.dataframe(bids[bids["side"] != "-"], width="stretch", height=240)
        st.download_button("Download 96-block bid sheet (CSV)",
                           bids.to_csv(index=False),
                           file_name=f"bid_sheet_{target}.csv")

        st.subheader("Backtest & revenue share")
        bt_file = OUT / "backtest_daily.csv"
        if bt_file.exists():
            daily = pd.read_csv(bt_file, parse_dates=["date"], index_col="date")
            uplift = daily["pnl_lp"].sum() - daily["pnl_greedy"].sum()
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Cumulative P&L (61d)", f"₹{daily['pnl_lp'].sum() / 1e5:,.1f} lakh")
            b2.metric("Uplift vs static EMS", f"₹{uplift / 1e5:,.1f} lakh")
            b3.metric("Capture vs perfect", f"{daily['pnl_lp'].sum() / daily['pnl_perfect'].sum():.0%}")
            b4.metric(f"FlexTrade fee @ {share}% of uplift",
                      f"₹{uplift * share / 100 / 1e5:,.1f} lakh",
                      help="Asset Optimization Revenue Share (business model §3.4): "
                           "FlexTrade bills a % of incremental profit vs the "
                           "customer's baseline strategy.")
            fig = go.Figure()
            fig.add_scatter(x=daily.index, y=daily["pnl_perfect"].cumsum() / 1e5,
                            name="Perfect foresight (bound)", mode="lines",
                            line=dict(color=GRAY, width=1.5, dash="dash"))
            fig.add_scatter(x=daily.index, y=daily["pnl_lp"].cumsum() / 1e5,
                            name="FlexTrade LP", mode="lines",
                            line=dict(color=BLUE, width=2.5))
            fig.add_scatter(x=daily.index, y=daily["pnl_greedy"].cumsum() / 1e5,
                            name="Static EMS baseline", mode="lines",
                            line=dict(color=AQUA, width=2))
            fig.update_yaxes(title="Cumulative P&L (₹ lakh)")
            st.plotly_chart(chart_layout(fig, 340), width="stretch")

# ================= RE DEVELOPER =================
with tab_re:
    st.caption("Renewable scheduling & deviation settlement — reference "
               "portfolio: 50 MW solar + 50 MW wind (Delhi NCR), digital twin "
               "on live NWP weather. Solar and wind are settled separately "
               "against their own DSM bands.")

    dsm_profile = st.radio(
        "DSM regulation profile", ["CERC_2024", "CERC_2022"], horizontal=True,
        help="CERC_2024: currently in force (Normal Rate mechanism, notified "
             "Aug 2024). CERC_2022: the mechanism it replaced (volume-band "
             "multipliers), kept for comparison and for backtesting periods "
             "before Sep 2024. See models/dsm.py's module docstring for "
             "exactly what is and isn't independently source-verified.")

    try:
        fc, dsm_df, flex, naive = re_data(profile=dsm_profile)
        saved = naive["net_dsm_rs"] - flex["net_dsm_rs"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RE energy forecast (D+1)",
                  f"{fc['re_total_mw'].sum() * 0.25:,.0f} MWh")
        m2.metric("DSM charge (FlexTrade fcst)", f"₹{flex['net_dsm_rs']:,.0f}",
                  help=f"{dsm_profile} mechanism, mean NR "
                       f"₹{flex['mean_normal_rate_rs_mwh']:,.0f}/MWh")
        m3.metric("DSM charge (naive persistence)", f"₹{naive['net_dsm_rs']:,.0f}")
        m4.metric("DSM saved / day", f"₹{saved:,.0f}",
                  delta=f"{saved:,.0f}", delta_color="normal")

        if dsm_profile == "CERC_2024":
            caption = (
                f"Settled with the **CERC 2024 Normal Rate mechanism** "
                f"(`models/dsm.py::settle_2024`): NR = ⅓ DAM + ⅓ RTM + ⅓ "
                f"ancillary (mean ₹{flex['mean_normal_rate_rs_mwh']:,.0f}/MWh); "
                f"solar tolerance ±5%, wind ±10% (tightened from ±10%/±15% on "
                f"2026-04-01); deviation measured against available capacity "
                f"(X-factor = 1.0 through FY27, per CERC's confirmed glide "
                f"path). Frequency-linkage is OFF by default — it appears in "
                f"a CERC consultation memo but is not independently confirmed "
                f"as gazetted; see the module docstring. "
            )
        else:
            caption = (
                f"Settled with the **CERC 2022 volume-band mechanism** "
                f"(`models/dsm.py::settle_2022`), superseded Aug 2024 — shown "
                f"for comparison / pre-2024 backtesting. NR = mean(DAM, RTM) "
                f"per block, no frequency linkage (2022 explicitly de-linked "
                f"it). Over-injection 0–5% at full NR, 5–10% at 90% NR, "
                f">10% uncompensated; under-injection free to 10%, then 10% "
                f"of NR. "
            )
        caption += ("System frequency observed from SLDC. "
                   if flex.get("frequency_observed") else
                   "⚠️ Frequency not sampled for that day. ")
        st.caption(caption)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.08,
                            subplot_titles=(f"Day-ahead RE generation forecast — {fc.index[0].date()}",
                                            "Yesterday: schedule vs actual (DSM view)"))
        fig.add_scatter(x=fc.index, y=fc["solar_mw"], mode="lines", name="Solar",
                        stackgroup="re", line=dict(color=YELLOW, width=1.5),
                        row=1, col=1)
        fig.add_scatter(x=fc.index, y=fc["wind_mw"], mode="lines", name="Wind",
                        stackgroup="re", line=dict(color=AQUA, width=1.5),
                        row=1, col=1)
        fig.add_scatter(x=dsm_df.index, y=dsm_df["schedule_mw"], mode="lines",
                        name="Scheduled (day-ahead)",
                        line=dict(color=BLUE, width=2), row=2, col=1)
        fig.add_scatter(x=dsm_df.index, y=dsm_df["actual_mw"], mode="lines",
                        name="Actual", line=dict(color=GRAY, width=1.5, dash="dot"),
                        row=2, col=1)
        st.plotly_chart(chart_layout(fig, 520), width="stretch")
        with st.expander("Regulatory provenance — what's confirmed vs. not"):
            st.markdown("""
This module is built as a **configurable rule engine**, not a fixed
calculator, because CERC's DSM framework keeps changing (2014 → 2015 →
2019 → 2022 → 2024 → Third Amendment draft, May 2026). Every rate, band
and effective date is a named constant in `models/dsm.py`, cross-checked
against multiple independent sources where possible:

- **Confirmed by 3+ independent sources**: 2022 volume bands (5%/10% WS
  over-injection, 10% under-injection tolerance, buyer 10–15%/>15% bands).
- **Confirmed by CERC's own regulatory-history summary**: 2022 de-linked
  charges from frequency; 2024 introduced the Normal Rate concept.
- **Confirmed, dated amendment**: solar/hybrid tolerance ±10%→±5%, wind
  ±15%→±10%, effective 2026-04-01; new WS projects treated as general
  sellers from the same date.
- **NOT independently confirmed**: the frequency-linked rate curve some
  earlier drafts of this module used — it traces to a CERC consultation
  *memo* (commentary proposing changes), not the gazetted text. Off by
  default here; toggle `freq_linked=True` only for exploration.
- **Draft, not confirmed effective**: the Third Amendment's proposed
  switch from block-wise to daily-average Normal Rate.

Before this prices a single rupee of real settlement, every number needs
checking against the CERC gazette notification directly by counsel —
which is also exactly what the DSM feature spec's own risk table (§6)
requires.
""")
    except Exception as e:
        st.warning(f"RE module unavailable: {e}")

    st.divider()
    st.subheader("Alerts & Revision Engine")
    st.caption("Forecast-driven gate-closure alerts: for each remaining block "
               "today, compares the DSM exposure of keeping the current "
               "schedule vs. revising it to match the latest forecast.")
    try:
        alerts = dsm_alerts_data()
        a1, a2, a3 = st.columns(3)
        a1.metric("Blocks to next gate", f"{alerts['lead_minutes']} min lead",
                  help="Target: ≥15 min lead time (DSM spec §5 success metric)")
        n_revise = sum(1 for a in alerts["alerts"] if a.action == "REVISE")
        a2.metric("Blocks flagged REVISE", n_revise)
        a3.metric("Estimated benefit if all revised",
                  f"₹{alerts['total_benefit_rs']:,.0f}")
        if alerts["alerts"]:
            adf = pd.DataFrame([{
                "block": a.block_ts.strftime("%H:%M"), "action": a.action,
                "scheduled_mw": round(a.current_schedule_mw, 1),
                "forecast_mw": round(a.forecast_mw, 1),
                "deviation_%cap": round(a.deviation_if_unrevised_pct, 1),
                "benefit_rs": round(a.benefit_rs, 0),
                "reason": a.reason,
            } for a in alerts["alerts"]])
            st.dataframe(adf, width="stretch", height=min(300, 40 + 35 * len(adf)))
        else:
            st.info("No material deviation flagged for the remaining blocks today.")
        st.caption(f"Schedule basis: {alerts['schedule_basis']}")
    except Exception as e:
        st.warning(f"Alerts engine unavailable: {e}")

# ================= DISCOM & C&I =================
with tab_grid:
    st.caption("Demand forecasting for procurement planning and predictive "
               "peak management.")
    if plan is not None:
        peak_thr = plan["forecast_load_mw"].quantile(0.95)
        peaks = plan[plan["forecast_load_mw"] >= peak_thr]
        m1, m2, m3 = st.columns(3)
        m1.metric("Forecast peak (D+1)", f"{plan['forecast_load_mw'].max():,.0f} MW")
        m2.metric("Peak window",
                  f"{peaks.index.min():%H:%M} – {peaks.index.max():%H:%M}")
        m3.metric("Load forecast accuracy", "4.98% MAPE",
                  help="6-month holdout, 15-min blocks, bid-time-valid features")

        fig = go.Figure()
        fig.add_scatter(x=plan.index, y=plan["forecast_load_mw"], mode="lines",
                        name="Load forecast", line=dict(color=BLUE, width=2))
        fig.add_hline(y=peak_thr, line=dict(color=RED, dash="dash", width=1.5),
                      annotation_text="peak alert threshold")
        for t in peaks.index:
            fig.add_vrect(x0=t, x1=t + pd.Timedelta(minutes=15),
                          fillcolor=RED, opacity=0.12, line_width=0)
        fig.update_yaxes(title="MW")
        fig.update_layout(title=f"Delhi day-ahead load forecast — {plan.index[0].date()} "
                                "(shaded: predictive peak-shaving windows)")
        st.plotly_chart(chart_layout(fig, 380), width="stretch")

    lf_metrics = HERE.parent / "load_forecast" / "output" / "metrics.txt"
    pm_metrics = OUT / "metrics_price.txt"
    q1, q2 = st.columns(2)
    if lf_metrics.exists():
        q1.subheader("Load model")
        q1.code(lf_metrics.read_text())
    if pm_metrics.exists():
        q2.subheader("Price model")
        q2.code(pm_metrics.read_text())

# ================= ENERGY TRADER =================
with tab_trader:
    st.caption("Market intelligence across IEX products — live.")
    if len(L["dam"]) and len(L["rtm"]):
        both = pd.DataFrame({"DAM": L["dam"]["mcp_rs_mwh"],
                             "RTM": L["rtm"]["mcp_rs_mwh"]}).dropna()
        spread = (both["RTM"] - both["DAM"])
        m1, m2, m3 = st.columns(3)
        m1.metric("DAM peak today", f"₹{L['dam']['mcp_rs_mwh'].max():,.0f}/MWh")
        m2.metric("RTM peak today", f"₹{L['rtm']['mcp_rs_mwh'].max():,.0f}/MWh")
        m3.metric("Mean RTM−DAM spread", f"₹{spread.mean():,.0f}/MWh",
                  help="Positive: RTM richer than DAM — sell RTM / buy DAM bias")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.08,
                            subplot_titles=("Today's clearing prices (₹/MWh)",
                                            "RTM − DAM spread (₹/MWh)"))
        fig.add_scatter(x=both.index, y=both["DAM"], mode="lines", name="DAM",
                        line=dict(color=BLUE, width=2, shape="hv"), row=1, col=1)
        fig.add_scatter(x=both.index, y=both["RTM"], mode="lines", name="RTM",
                        line=dict(color=AQUA, width=2, shape="hv"), row=1, col=1)
        fig.add_bar(x=spread.index, y=spread, name="Spread",
                    marker_color=[RED if v < 0 else BLUE for v in spread],
                    showlegend=False, row=2, col=1)
        st.plotly_chart(chart_layout(fig, 500), width="stretch")

    # ---- Green Day-Ahead Market ----
    st.divider()
    st.subheader("Green Day-Ahead Market (GDAM)")
    st.caption("Renewable energy clears in its own segment. The GDAM−DAM "
               "spread is the premium (or discount) green power carries — a "
               "direct signal for RE developers deciding where to sell.")
    try:
        gdam, gmeta = gdam_data()
        if len(gdam) and len(L["dam"]):
            gd = gdam["mcp_rs_mwh"].to_frame("GDAM").join(
                L["dam"]["mcp_rs_mwh"].rename("DAM")).dropna()
            gspread = gd["GDAM"] - gd["DAM"]
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("GDAM avg MCP", f"₹{gd['GDAM'].mean():,.0f}/MWh")
            g2.metric("Mean GDAM−DAM spread", f"₹{gspread.mean():,.0f}/MWh",
                      delta=f"{gspread.mean():+,.0f}",
                      help="Positive = green power at a premium")
            g3.metric("Green volume cleared",
                      f"{gdam['mcv_mw'].sum() * 0.25:,.0f} MWh")
            wind_share = (gdam["sell_wind_mw"].sum() /
                          max(gdam[["sell_wind_mw", "sell_other_re_mw",
                                    "sell_hydro_mw"]].sum().sum(), 1) * 100)
            g4.metric("Wind share of green sell bids", f"{wind_share:.0f}%")

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.08,
                                subplot_titles=("GDAM vs DAM clearing price",
                                                "GDAM − DAM spread (₹/MWh)"))
            fig.add_scatter(x=gd.index, y=gd["DAM"], mode="lines", name="DAM",
                            line=dict(color=BLUE, width=2, shape="hv"), row=1, col=1)
            fig.add_scatter(x=gd.index, y=gd["GDAM"], mode="lines", name="GDAM",
                            line=dict(color=YELLOW, width=2, shape="hv"), row=1, col=1)
            fig.add_bar(x=gspread.index, y=gspread, name="Spread",
                        marker_color=[RED if v < 0 else AQUA for v in gspread],
                        showlegend=False, row=2, col=1)
            st.plotly_chart(chart_layout(fig, 460), width="stretch")
        else:
            st.info("GDAM data unavailable right now.")
    except Exception as e:
        st.warning(f"GDAM panel unavailable: {e}")

    if plan is not None:
        fig = go.Figure()
        fig.add_scatter(x=plan.index, y=plan["forecast_mcp"], mode="lines",
                        name="DAM MCP forecast",
                        line=dict(color=AQUA, width=2))
        fig.update_layout(title=f"Day-ahead MCP forecast — {plan.index[0].date()}")
        fig.update_yaxes(title="₹/MWh")
        st.plotly_chart(chart_layout(fig, 300), width="stretch")

# ================= MULTI-STATE =================
with tab_states:
    from ingest import states as states_mod
    st.caption("Delhi is one instance of a general pattern, not a hardcoded "
               "destination. This tab shows what expanding beyond Delhi "
               "actually looks like today — real coverage, not a slide.")

    st.subheader("Northern Region — live snapshot")
    st.caption("Delhi SLDC's own real-time page already publishes load/"
               "schedule/drawl for 8 neighbouring states, at zero extra "
               "scraping cost. Same live-fetch-with-cache-fallback pattern "
               "as everything else on this platform.")
    try:
        nrdf, nmeta = northern_region_data()
        st.markdown(badge("Northern Region snapshot", nmeta))
        if len(nrdf):
            show = nrdf[["state", "schedule_mw", "drawl_mw", "od_ud_mw", "load_mw"]].copy()
            show.columns = ["State", "Schedule (MW)", "Drawl (MW)", "OD/UD (MW)", "Load (MW)"]
            st.dataframe(show.sort_values("Load (MW)", ascending=False),
                        width="stretch", height=320, hide_index=True)

            fig = go.Figure()
            sorted_df = nrdf.sort_values("load_mw", ascending=True)
            fig.add_bar(y=sorted_df["state"], x=sorted_df["load_mw"],
                       orientation="h", marker_color=BLUE, name="Load")
            fig.update_layout(title="Current load by state (MW)")
            st.plotly_chart(chart_layout(fig, 340), width="stretch")

            biggest = nrdf.loc[nrdf["load_mw"].idxmax()]
            st.caption(f"**{biggest['state'].title()}** currently carries "
                      f"{biggest['load_mw']:,.0f} MW — "
                      f"{biggest['load_mw'] / max(L['snap'].get('delhi_load', 1), 1):.1f}x "
                      f"Delhi's current load, from the same fetch.")
        else:
            st.info("Northern Region snapshot not available right now.")
    except Exception as e:
        st.warning(f"Northern Region panel unavailable: {e}")

    st.divider()
    st.subheader("State coverage registry")
    st.caption("`verified` = live-tested fetch, same discipline as Delhi's own "
               "adapter. `identified` = data source located, adapter not yet "
               "built — shown honestly rather than claimed.")
    reg_rows = []
    for s in states_mod.list_states():
        reg_rows.append({
            "Code": s.code, "State": s.name, "Grid region": s.grid_region,
            "Peak load": f"{s.peak_load_gw:.1f} GW" if s.peak_load_gw else "—",
            "Status": s.status, "Notes": s.notes,
        })
    reg_df = pd.DataFrame(reg_rows)
    st.dataframe(reg_df, width="stretch", height=380, hide_index=True,
                column_config={"Notes": st.column_config.TextColumn(width="large")})
    v = (reg_df["Status"] == "verified").sum()
    st.caption(f"**{v} of {len(reg_df)}** states have a live-tested data path "
              f"today; the rest are scoped, not claimed.")

# ================= BUSINESS MODEL =================
with tab_biz:
    st.subheader("How this running system maps to the FlexTrade business model")
    st.markdown("""
| Revenue stream (Business Model §3) | Live in this demo |
|---|---|
| **SaaS subscription** | This dashboard — per-seat platform access |
| **Forecast-as-a-Service** | REST API on `:8100` — tiered keys, metered usage (`/docs`) |
| **Asset optimization revenue share** | BESS tab: fee = % × uplift vs customer baseline (sidebar slider) |
| **Transaction fee / brokerage** | Bid sheets are the execution artifact — fee per MWh routed |
| **Data & market intelligence** | Trader tab: DAM/RTM spreads, price history (13 months scraped) |
| **Consulting & advisory** | Backtest engine produces the ROI case studies |
""")
    st.subheader("Forecast-as-a-Service — try it")
    st.code("""# Starter tier: market data only
curl -H "X-API-Key: demo-starter" http://localhost:8100/v1/prices/dam

# Professional tier: forecasts
curl -H "X-API-Key: demo-professional" http://localhost:8100/v1/forecast/load
curl -H "X-API-Key: demo-professional" http://localhost:8100/v1/forecast/renewables

# Enterprise tier: optimization + DSM
curl -X POST -H "X-API-Key: demo-enterprise" -H "Content-Type: application/json" \\
     -d '{"power_mw": 50, "energy_mwh": 100}' \\
     http://localhost:8100/v1/optimize/dispatch""", language="bash")

    with store.connect() as con:
        try:
            usage = pd.read_sql(
                "SELECT key, endpoint, COUNT(*) calls FROM api_usage "
                "GROUP BY key, endpoint ORDER BY calls DESC", con)
        except Exception:
            usage = pd.DataFrame()
    if len(usage):
        st.subheader("API usage metering (billing feed)")
        st.dataframe(usage, width="stretch", height=200)
