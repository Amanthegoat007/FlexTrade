"""Project-finance model for a BESS — the model a lender actually reads.

models/sizing.py answers "how much revenue". That is not the question a bank
asks. A bank asks whether the project services its debt in the worst year it
can imagine, and it asks in DSCR, IRR and LCOS. Revenue with no O&M, no
capacity fade, no augmentation and no debt schedule is a revenue calculator
wearing the word "bankability".

What this adds, and why each one changes the answer:

  CAPACITY FADE, derived rather than assumed
      A battery does not earn year-15 revenue in year 15. Most models bolt on
      "2%/yr" from a slide. We take it from our own physics: the Wohler curve
      L(d) = L100 * d^-k gives cycles-to-80% at the depth the LP actually
      cycles at (0.434 measured -> ~15,000 cycles), and the LP's measured
      throughput gives cycles per year. Annual fade is then
      0.20 * cycles_per_year / L(d) -- consistent with the same curve that
      sets the Rs 800/MWh the optimizer is charged.

  O&M
      1.5% of capex a year, escalating. Absent from the previous model
      entirely, which alone overstated returns.

  AUGMENTATION
      Real projects re-cell when usable capacity falls below a floor, at a
      cost that declines with the cell-price learning curve. Ignoring it makes
      a 15-year model look like a 15-year annuity, which no lender believes.

  DEBT, and therefore DSCR
      Annuity debt service on a 70:30 structure. DSCR = cash available for
      debt service / debt service, reported per year with its minimum, because
      the minimum is the covenant. Below ~1.20x the project does not fund.

  TAX
      Indian new-regime corporate rate with written-down-value depreciation,
      so the tax shield is real rather than assumed away.

  LCOS
      Levelised cost of storage: PV of all lifetime costs / PV of energy
      discharged. The one number that compares a battery to any other
      flexibility option.

Every default is stated as a parameter, not buried, and every one is an
INPUT ASSUMPTION rather than a measurement -- flagged as such in the output.
The revenue side is the only part measured from our own operating history.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from optimize import degradation as dg  # noqa: E402

OUT = HERE.parent / "output"


# ----------------------------------------------------------------- helpers --

def irr(cashflows: list[float], lo: float = -0.95, hi: float = 3.0,
        tol: float = 1e-7) -> float | None:
    """Internal rate of return by bisection.

    Bisection rather than Newton because project cash-flow sign patterns
    (augmentation years go negative again) can give Newton a derivative of
    zero and send it somewhere meaningless. Bisection is slower and cannot.
    Returns None when no sign change exists — an honest "undefined" instead
    of a number that looks like an answer.
    """
    def npv(r):
        return sum(cf / (1 + r) ** t for t, cf in enumerate(cashflows))

    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return None
    for _ in range(300):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def npv(rate: float, cashflows: list[float]) -> float:
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))


def annuity_payment(principal: float, rate: float, years: int) -> float:
    if years <= 0:
        return 0.0
    if rate == 0:
        return principal / years
    return principal * rate / (1 - (1 + rate) ** -years)


# ------------------------------------------------------------------ inputs --

@dataclass
class Assumptions:
    # --- asset ---
    power_mw: float = 20.0
    duration_h: float = 2.0
    life_years: int = 15

    # --- capex, TWO-PART (Indian grid BESS, FY25-26 tender range 1.3-1.8 Cr/MWh)
    #
    # Split into a power-related and an energy-related component, because a
    # single Rs/MWh figure makes every duration comparison meaningless. Under a
    # flat Rs/MWh a 1h system costs a QUARTER of a 4h system while earning far
    # more than a quarter of the revenue, so short duration wins by construction
    # rather than on merit — an error a project-finance reviewer finds in one
    # pass, and one this model previously made.
    #
    #   power-related  PCS, transformer, switchgear, grid connection, EPC, land
    #                  -> scales with MW, indifferent to how many hours you store
    #   energy-related cells, modules, racks, containers, BMS
    #                  -> scales with MWh
    #
    # Defaults are CALIBRATED so a 2-hour system — the configuration Indian
    # tenders actually discover — reproduces Rs 1.5 Cr/MWh:
    #     (1.0e7 + 2 x 1.0e7) / 2 = 1.5e7  per MWh
    # The split itself is an assumption, not a measurement. duration_irr_sweep()
    # in sizing.py reports how much the answer moves with it, which is the
    # honest way to use a number we cannot source precisely.
    capex_power_rs_per_mw: float = 1.0e7
    capex_energy_rs_per_mwh: float = 1.0e7
    gst_pct: float = 18.0
    # share of capex that is the battery itself, i.e. what augmentation replaces
    cell_share_of_capex: float = 0.60
    cell_price_decline_pct_yr: float = 6.0

    # --- operating costs ---
    om_pct_of_capex: float = 1.5          # per year
    om_escalation_pct: float = 5.0
    insurance_pct_of_capex: float = 0.5

    # --- revenue ---
    revenue_escalation_pct: float = 3.0
    availability_pct: float = 97.0

    # --- degradation / augmentation ---
    cycles_per_year: float = 412.0        # measured from the dispatch LP
    mean_cycle_depth: float = 0.434       # measured, rainflow on the LP schedule
    augment_at_capacity_pct: float = 80.0

    # --- financing ---
    debt_share_pct: float = 70.0
    interest_pct: float = 9.5
    debt_tenor_years: int = 12
    cost_of_equity_pct: float = 14.0

    # --- tax (India, new regime) ---
    tax_rate_pct: float = 25.17
    depreciation_pct_wdv: float = 40.0    # written-down value, accelerated

    # --- discounting ---
    discount_rate_pct: float = 10.0

    @property
    def energy_mwh(self) -> float:
        return self.power_mw * self.duration_h

    @property
    def capex_ex_gst_rs(self) -> float:
        """Total capex before GST — power component plus energy component."""
        return (self.capex_power_rs_per_mw * self.power_mw
                + self.capex_energy_rs_per_mwh * self.energy_mwh)

    @property
    def capex_rs_per_mwh(self) -> float:
        """Blended Rs/MWh, for reporting and for comparison against tender
        discovery. DERIVED — it is an output of the two-part model, not an
        input, so it moves with duration exactly as a real quote does."""
        return self.capex_ex_gst_rs / max(self.energy_mwh, 1e-9)


MEASURED = {"cycles_per_year", "mean_cycle_depth"}   # from our own operations


# ------------------------------------------------------------------- model --

def annual_fade_pct(a: Assumptions, params: dg.DegParams = dg.DegParams()) -> float:
    """Capacity fade per year, derived from the SAME curve that prices cycling.

    L(d) = L100 * d^-k is cycles to 80% capacity at depth d. Doing N cycles a
    year at that depth consumes N/L(d) of life per year, and the full life is
    a 20-point capacity loss — so fade is 0.20 * N / L(d) per year.
    """
    life = params.cycle_life_100 * a.mean_cycle_depth ** -params.kp
    return 100.0 * 0.20 * a.cycles_per_year / life


def build(a: Assumptions, base_annual_revenue_rs: float,
          params: dg.DegParams = dg.DegParams()) -> dict:
    """Full lifetime cash-flow model. Returns per-year rows plus headline metrics."""
    capex_ex_gst = a.capex_ex_gst_rs        # two-part: power MW + energy MWh
    capex = capex_ex_gst * (1 + a.gst_pct / 100)
    debt = capex * a.debt_share_pct / 100
    equity = capex - debt
    r_debt = a.interest_pct / 100
    service = annuity_payment(debt, r_debt, a.debt_tenor_years)
    fade = annual_fade_pct(a, params) / 100

    rows: list[dict] = []
    balance = debt
    wdv = capex_ex_gst        # depreciation base excludes recoverable GST
    capacity = 1.0
    energy_pv = 0.0
    cost_pv = capex           # LCOS numerator starts at capex
    disc = a.discount_rate_pct / 100

    for yr in range(1, a.life_years + 1):
        # --- augmentation before the year runs, if capacity has fallen through
        augment = 0.0
        if capacity < a.augment_at_capacity_pct / 100:
            restore = 1.0 - capacity
            # Augmentation buys CELLS, and cells sit entirely inside the energy
            # component — you do not re-buy the PCS or the grid connection to
            # restore capacity. Pricing this off the blended Rs/MWh overstated
            # augmentation for short-duration systems (whose blended figure is
            # inflated by power-related cost) and understated it for long ones.
            unit = (a.capex_energy_rs_per_mwh * a.cell_share_of_capex
                    * (1 - a.cell_price_decline_pct_yr / 100) ** (yr - 1))
            augment = restore * a.energy_mwh * unit * (1 + a.gst_pct / 100)
            wdv += augment / (1 + a.gst_pct / 100)
            capacity = 1.0

        revenue = (base_annual_revenue_rs
                   * capacity
                   * (a.availability_pct / 100)
                   * (1 + a.revenue_escalation_pct / 100) ** (yr - 1))
        om = (capex_ex_gst * a.om_pct_of_capex / 100
              * (1 + a.om_escalation_pct / 100) ** (yr - 1))
        insurance = capex_ex_gst * a.insurance_pct_of_capex / 100
        ebitda = revenue - om - insurance

        interest = balance * r_debt
        principal = max(min(service - interest, balance), 0.0)
        debt_service = interest + principal

        depreciation = wdv * a.depreciation_pct_wdv / 100
        wdv -= depreciation

        taxable = ebitda - depreciation - interest
        tax = max(0.0, taxable) * a.tax_rate_pct / 100
        # unlevered tax ignores the interest shield (project view)
        tax_unlev = max(0.0, ebitda - depreciation) * a.tax_rate_pct / 100

        cfads = ebitda - tax
        dscr = cfads / debt_service if debt_service > 1e-6 else None
        equity_cf = cfads - debt_service - augment
        project_cf = ebitda - tax_unlev - augment

        energy_mwh_yr = (a.energy_mwh * a.cycles_per_year * capacity
                         * a.availability_pct / 100)
        energy_pv += energy_mwh_yr / (1 + disc) ** yr
        cost_pv += (om + insurance + augment) / (1 + disc) ** yr

        rows.append({
            "year": yr,
            "capacity_pct": round(capacity * 100, 1),
            "revenue_rs": round(revenue, 0),
            "om_rs": round(om + insurance, 0),
            "ebitda_rs": round(ebitda, 0),
            "interest_rs": round(interest, 0),
            "principal_rs": round(principal, 0),
            "debt_service_rs": round(debt_service, 0),
            "tax_rs": round(tax, 0),
            "cfads_rs": round(cfads, 0),
            "dscr": round(dscr, 3) if dscr else None,
            "augmentation_rs": round(augment, 0),
            "equity_cf_rs": round(equity_cf, 0),
            "project_cf_rs": round(project_cf, 0),
        })

        balance -= principal
        capacity *= (1 - fade)

    proj_flows = [-capex] + [r["project_cf_rs"] for r in rows]
    eq_flows = [-equity] + [r["equity_cf_rs"] for r in rows]
    dscrs = [r["dscr"] for r in rows if r["dscr"] is not None]

    # payback on undiscounted project cash flow
    cum, payback = -capex, None
    for r in rows:
        cum += r["project_cf_rs"]
        if cum >= 0 and payback is None:
            payback = r["year"]

    p_irr, e_irr = irr(proj_flows), irr(eq_flows)
    min_dscr = min(dscrs) if dscrs else None

    return {
        "assumptions": asdict(a),
        "measured_inputs": sorted(MEASURED),
        "capex_rs": round(capex, 0),
        "capex_ex_gst_rs": round(capex_ex_gst, 0),
        "debt_rs": round(debt, 0),
        "equity_rs": round(equity, 0),
        "annual_debt_service_rs": round(service, 0),
        "annual_fade_pct": round(annual_fade_pct(a, params), 2),
        "cycle_life_at_depth": round(
            params.cycle_life_100 * a.mean_cycle_depth ** -params.kp, 0),
        "base_annual_revenue_rs": round(base_annual_revenue_rs, 0),
        "project_irr_pct": round(p_irr * 100, 2) if p_irr is not None else None,
        "equity_irr_pct": round(e_irr * 100, 2) if e_irr is not None else None,
        "npv_at_discount_rs": round(npv(disc, proj_flows), 0),
        "min_dscr": round(min_dscr, 3) if min_dscr else None,
        "avg_dscr": round(float(np.mean(dscrs)), 3) if dscrs else None,
        "payback_years": payback,
        "lcos_rs_per_mwh": round(cost_pv / energy_pv, 0) if energy_pv else None,
        "bankable": bool(min_dscr and min_dscr >= 1.20),
        "years": rows,
    }


def sensitivity(a: Assumptions, base_revenue: float,
                revenue_range=(-30, -20, -10, 0, 10, 20)) -> list[dict]:
    """What a credit committee actually asks: how far can revenue fall?"""
    out = []
    for pct in revenue_range:
        m = build(a, base_revenue * (1 + pct / 100))
        out.append({
            "revenue_delta_pct": pct,
            "equity_irr_pct": m["equity_irr_pct"],
            "min_dscr": m["min_dscr"],
            "bankable": m["bankable"],
        })
    return out


def breakeven_revenue(a: Assumptions, base_revenue: float,
                      target_dscr: float = 1.20) -> float | None:
    """Lowest annual revenue that still clears the DSCR covenant."""
    lo, hi = base_revenue * 0.2, base_revenue * 2.0
    if (build(a, hi)["min_dscr"] or 0) < target_dscr:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if (build(a, mid)["min_dscr"] or 0) >= target_dscr:
            hi = mid
        else:
            lo = mid
    return round(hi, 0)


def _base_revenue_from_sizing(a: Assumptions) -> tuple[float, str]:
    """Annual DAM arbitrage revenue for this asset, from the sizing curves."""
    try:
        from models import sizing
        c = sizing.compute()
        key = min(c["curves"], key=lambda k: abs(float(k.rstrip("h")) - a.duration_h))
        daily = [d["pnl_rs"] for d in c["curves"][key]]
        # per-MW perfect-foresight daily P&L, scaled by the capture ratio we
        # actually achieve — the sizing curves are solved at 1 MW so this is
        # exactly linear in power at fixed duration
        per_mw_day = float(np.mean(daily)) * c["capture_ratio"]
        rev = per_mw_day * a.power_mw * 365
        return rev, (f"sizing curve {key} over {c['n_days']} days "
                     f"({c['first_day']}..{c['last_day']}), x {a.power_mw:.0f} MW "
                     f"x 365, x measured capture {c['capture_ratio']:.3f}")
    except Exception as e:
        # fall back to the backtest's own annualised figure, scaled by size
        rev = 8.23e7 * (a.power_mw / 20.0) * (a.duration_h / 2.0)
        return rev, f"backtest annualisation fallback ({type(e).__name__})"


# Revenue stacking. A merchant battery running DAM arbitrage alone is not a
# bankable asset in India at current capex, and the model should say so rather
# than be tuned until it agrees. These are the other legs an operating BESS
# actually earns, each expressed as a share of the DAM arbitrage base so the
# stack scales with the asset:
#
#   rtm      measured. The intraday re-optimizer's realised uplift on the
#            issued book, as a fraction of DAM revenue.
#   dsm      ASSUMPTION. Deviation savings for a co-located RE or DISCOM
#            client; we can price the exposure but have not operated it.
#   capacity ASSUMPTION. A capacity/ancillary contract, which is how nearly
#            every funded Indian BESS to date actually closes. Modelled at a
#            conservative Rs/MW/yr rather than as a multiple.
STACK_DEFAULTS = {
    "rtm_pct_of_dam": 12.0,        # measured-ish: see trade book uplift
    "dsm_pct_of_dam": 8.0,         # assumption
    "capacity_rs_per_mw_yr": 0.0,  # off by default; set to model a contract
}


def stacked_revenue(base_dam_rs: float, a: Assumptions,
                    stack: dict | None = None) -> tuple[float, dict]:
    s = {**STACK_DEFAULTS, **(stack or {})}
    rtm = base_dam_rs * s["rtm_pct_of_dam"] / 100
    dsm = base_dam_rs * s["dsm_pct_of_dam"] / 100
    cap = s["capacity_rs_per_mw_yr"] * a.power_mw
    total = base_dam_rs + rtm + dsm + cap
    return total, {"dam_rs": round(base_dam_rs, 0), "rtm_rs": round(rtm, 0),
                   "dsm_rs": round(dsm, 0), "capacity_rs": round(cap, 0),
                   "total_rs": round(total, 0), "shares": s}


def capacity_payment_for_bankability(a: Assumptions, base_dam_rs: float,
                                     target_dscr: float = 1.20) -> float | None:
    """The capacity payment (Rs/MW/yr) that would make this project fund.

    This is the number a developer takes into a tender, and it is the honest
    output when merchant revenue alone does not clear the covenant.
    """
    lo, hi = 0.0, 5.0e7
    stack = dict(STACK_DEFAULTS)
    stack["capacity_rs_per_mw_yr"] = hi
    rev, _ = stacked_revenue(base_dam_rs, a, stack)
    if (build(a, rev)["min_dscr"] or 0) < target_dscr:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        stack["capacity_rs_per_mw_yr"] = mid
        rev, _ = stacked_revenue(base_dam_rs, a, stack)
        if (build(a, rev)["min_dscr"] or 0) >= target_dscr:
            hi = mid
        else:
            lo = mid
    return round(hi, 0)


def run(a: Assumptions | None = None) -> dict:
    a = a or Assumptions()
    base, basis = _base_revenue_from_sizing(a)

    # headline case: DAM arbitrage only, which is what we can actually prove
    model = build(a, base)
    model["revenue_basis"] = basis

    # and the stacked case, clearly separated so the two are never conflated
    stacked, breakdown = stacked_revenue(base, a)
    st = build(a, stacked)
    model["stacked"] = {
        "revenue_breakdown": breakdown,
        "project_irr_pct": st["project_irr_pct"],
        "equity_irr_pct": st["equity_irr_pct"],
        "min_dscr": st["min_dscr"],
        "bankable": st["bankable"],
        "note": ("DAM arbitrage is the only leg we can prove from our own "
                 "operating record. RTM is measured on a short book; DSM and "
                 "capacity are assumptions. Shown separately for that reason."),
    }
    model["capacity_payment_for_bankability_rs_per_mw_yr"] =         capacity_payment_for_bankability(a, base)
    model["sensitivity"] = sensitivity(a, base)
    model["breakeven_revenue_rs"] = breakeven_revenue(a, base)
    if model["breakeven_revenue_rs"]:
        head = round((1 - model["breakeven_revenue_rs"] / base) * 100, 1)
        model["revenue_headroom_pct"] = head
        model["headroom_note"] = (
            f"revenue can fall {head:.1f}% before DSCR breaks 1.20x" if head > 0
            else f"revenue must RISE {abs(head):.1f}% to reach a 1.20x DSCR")
    (OUT / "bankability.json").write_text(json.dumps(model, indent=2, default=float))
    return model


if __name__ == "__main__":
    m = run()
    cr = lambda v: f"Rs {v/1e7:,.2f} Cr"          # noqa: E731
    a = m["assumptions"]
    print(f"BESS project finance — {a['power_mw']:.0f} MW / "
          f"{a['power_mw']*a['duration_h']:.0f} MWh, {a['life_years']}-year life")
    print(f"  revenue basis: {m['revenue_basis']}")
    print(f"  capex {cr(m['capex_rs'])} (incl {a['gst_pct']:.0f}% GST)"
          f"  debt {cr(m['debt_rs'])}  equity {cr(m['equity_rs'])}")
    print(f"  capacity fade {m['annual_fade_pct']}%/yr, derived from "
          f"{m['cycle_life_at_depth']:,.0f} cycles to 80% at depth "
          f"{a['mean_cycle_depth']}")
    print()
    print(f"  project IRR   {m['project_irr_pct']}%")
    print(f"  equity IRR    {m['equity_irr_pct']}%")
    print(f"  min DSCR      {m['min_dscr']}   (avg {m['avg_dscr']})"
          f"   -> {'BANKABLE' if m['bankable'] else 'NOT BANKABLE at 1.20x'}")
    print(f"  NPV @ {a['discount_rate_pct']:.0f}%   {cr(m['npv_at_discount_rs'])}")
    print(f"  payback       {m['payback_years']} years")
    print(f"  LCOS          Rs {m['lcos_rs_per_mwh']:,.0f}/MWh")
    if m.get("breakeven_revenue_rs"):
        print(f"  {m.get('headroom_note','')}")
    st = m["stacked"]; bd = st["revenue_breakdown"]
    print()
    print("  revenue stacking (DAM proven; RTM short-book; DSM/capacity assumed):")
    print(f"    DAM {cr(bd['dam_rs'])}  + RTM {cr(bd['rtm_rs'])}  "
          f"+ DSM {cr(bd['dsm_rs'])}  = {cr(bd['total_rs'])}")
    print(f"    stacked -> equity IRR {st['equity_irr_pct']}%, "
          f"min DSCR {st['min_dscr']} -> "
          f"{'BANKABLE' if st['bankable'] else 'still not bankable'}")
    cp = m.get("capacity_payment_for_bankability_rs_per_mw_yr")
    if cp:
        print(f"    capacity payment needed for 1.20x DSCR: "
              f"Rs {cp/1e5:,.1f} lakh/MW/yr")
    print()
    print("  sensitivity (what a credit committee asks):")
    print(f"    {'rev delta':>10} {'equity IRR':>12} {'min DSCR':>10}  verdict")
    for s in m["sensitivity"]:
        print(f"    {s['revenue_delta_pct']:+9d}% {str(s['equity_irr_pct'])+'%':>12} "
              f"{s['min_dscr']:>10}  {'ok' if s['bankable'] else 'fails covenant'}")
    print()
    print(f"  first 6 years:")
    print(f"    {'yr':>3} {'cap%':>6} {'revenue':>12} {'EBITDA':>12} "
          f"{'debt svc':>12} {'DSCR':>7} {'augment':>12}")
    for r in m["years"][:6]:
        print(f"    {r['year']:3d} {r['capacity_pct']:6.1f} "
              f"{r['revenue_rs']/1e5:11,.1f}L {r['ebitda_rs']/1e5:11,.1f}L "
              f"{r['debt_service_rs']/1e5:11,.1f}L {r['dscr']:7.2f} "
              f"{r['augmentation_rs']/1e5:11,.1f}L")
