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

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
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
def re_data():
    fc = re_model.forecast_day()
    df, flex, naive = re_model.dsm_comparison()
    return fc, df, flex, naive


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

tab_bess, tab_re, tab_grid, tab_trader, tab_biz = st.tabs([
    "🔋 BESS Operator", "☀️ RE Developer", "🏭 DISCOM & C&I",
    "📈 Energy Trader", "💼 Business Model"])

# ================= BESS OPERATOR =================
with tab_bess:
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
               "on live NWP weather.")
    try:
        fc, dsm_df, flex, naive = re_data()
        saved = naive["penalty_rs"] - flex["penalty_rs"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RE energy forecast (D+1)",
                  f"{fc['re_total_mw'].sum() * 0.25:,.0f} MWh")
        m2.metric("DSM penalty (FlexTrade fcst)", f"₹{flex['penalty_rs']:,.0f}",
                  help="Yesterday, penalty on deviations beyond ±10% of schedule")
        m3.metric("DSM penalty (naive persistence)", f"₹{naive['penalty_rs']:,.0f}")
        m4.metric("Penalty saved / day", f"₹{saved:,.0f}",
                  delta=f"{saved:,.0f}", delta_color="normal")

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
        st.caption("DSM mechanics simplified to a single ±10%-of-schedule band "
                   "at ₹1,500/MWh — swap in the applicable CERC/SERC slabs for "
                   "production.")
    except Exception as e:
        st.warning(f"RE module unavailable: {e}")

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

    if plan is not None:
        fig = go.Figure()
        fig.add_scatter(x=plan.index, y=plan["forecast_mcp"], mode="lines",
                        name="DAM MCP forecast",
                        line=dict(color=AQUA, width=2))
        fig.update_layout(title=f"Day-ahead MCP forecast — {plan.index[0].date()}")
        fig.update_yaxes(title="₹/MWh")
        st.plotly_chart(chart_layout(fig, 300), width="stretch")

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
