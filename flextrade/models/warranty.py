"""Warranty & Availability Guard — protect the asset while monetizing it.

Battery warranties come with fine print the trading optimizer can quietly
violate: a maximum cycle count per day, an approved SoC operating window,
throughput caps. Contract-side, capacity agreements (SECI-style tolling)
carry availability guarantees with penalties. Operators today discover
violations at claim time — the worst possible moment.

This module audits BOTH our own schedules and the real BRPL Kilokari
telemetry against a configurable warranty profile:

  - cycles/day (rainflow FCE) vs the warranty limit
  - observed SoC envelope vs the approved window
  - telemetry availability (honest label: OUR sampling coverage — a gap
    proves nothing about the battery; a reading outside limits does)

The same terms can be handed to the dispatch LP as constraints (cycle
cap -> throughput cap; SoC window -> tighter SoC bounds), so "warranty-
safe dispatch" is a solver setting, not a hope.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ingest import bess  # noqa: E402
from optimize.degradation import rainflow  # noqa: E402


@dataclass
class WarrantyTerms:
    """Typical LFP grid-storage warranty envelope — configurable per
    asset; defaults are indicative of common OEM terms, labelled so."""
    max_cycles_per_day: float = 1.5     # FCE/day averaged over the period
    soc_min_pct: float = 5.0
    soc_max_pct: float = 95.0
    max_throughput_mwh_year_per_mwh: float = 1100.0  # ~1.5 FCE/day x 2E
    note: str = ("indicative OEM terms — replace with the asset's actual "
                 "warranty schedule for a pilot")


def audit_telemetry(terms: WarrantyTerms = WarrantyTerms(),
                    energy_mwh: float = bess.RATED_ENERGY_MWH) -> dict:
    """Audit the sampled BRPL telemetry, day by day."""
    hist = bess.read_history()
    if not len(hist):
        return {"days": [], "note": "no telemetry yet"}

    days = []
    for d, g in hist.groupby(hist.index.date):
        soc = g["soc_pct"].dropna()
        if len(soc) < 3:
            continue
        fce = sum(depth * cnt for depth, cnt in rainflow(soc.values / 100.0))
        # sampling coverage: share of the day's 5-min slots we captured
        coverage = min(len(g) / 288.0, 1.0)
        viol = []
        if soc.min() < terms.soc_min_pct:
            viol.append(f"SoC {soc.min():.0f}% < {terms.soc_min_pct:.0f}% floor")
        if soc.max() > terms.soc_max_pct:
            viol.append(f"SoC {soc.max():.0f}% > {terms.soc_max_pct:.0f}% ceiling")
        # cycle check only meaningful with decent coverage — partial
        # sampling UNDERcounts cycles, so flag, don't clear
        fce_flag = (fce > terms.max_cycles_per_day) if coverage >= 0.6 else None
        if fce_flag:
            viol.append(f"{fce:.2f} FCE > {terms.max_cycles_per_day} limit")
        days.append({
            "day": str(d), "samples": len(g),
            "coverage_pct": round(coverage * 100, 1),
            "fce": round(fce, 2),
            "fce_reliable": coverage >= 0.6,
            "soc_min_pct": round(float(soc.min()), 1),
            "soc_max_pct": round(float(soc.max()), 1),
            "violations": viol,
        })

    full = [d for d in days if d["fce_reliable"]]
    return {
        "terms": {**terms.__dict__},
        "days": days,
        "summary": {
            "days_observed": len(days),
            "days_with_reliable_cycle_count": len(full),
            "mean_fce_reliable_days": round(float(np.mean([d["fce"] for d in full])), 2)
            if full else None,
            "worst_soc_min_pct": min((d["soc_min_pct"] for d in days), default=None),
            "worst_soc_max_pct": max((d["soc_max_pct"] for d in days), default=None),
            "total_violations": sum(len(d["violations"]) for d in days),
            "telemetry_note": ("coverage % is OUR sampling coverage of the "
                               "day — partial days undercount cycles and are "
                               "excluded from the cycle average"),
        },
    }


def audit_schedule(soc_mwh: pd.Series, energy_mwh: float,
                   terms: WarrantyTerms = WarrantyTerms()) -> dict:
    """Audit one planned schedule (e.g. tomorrow's LP output) BEFORE it
    runs — the pre-trade compliance check."""
    soc_pct = 100.0 * np.asarray(soc_mwh, dtype=float) / energy_mwh
    fce = sum(d * c for d, c in rainflow(soc_pct / 100.0))
    viol = []
    if soc_pct.min() < terms.soc_min_pct - 1e-6:
        viol.append(f"planned SoC {soc_pct.min():.0f}% below warranty floor")
    if soc_pct.max() > terms.soc_max_pct + 1e-6:
        viol.append(f"planned SoC {soc_pct.max():.0f}% above warranty ceiling")
    if fce > terms.max_cycles_per_day:
        viol.append(f"planned {fce:.2f} FCE exceeds {terms.max_cycles_per_day}/day")
    return {"fce": round(fce, 2), "soc_min_pct": round(float(soc_pct.min()), 1),
            "soc_max_pct": round(float(soc_pct.max()), 1),
            "violations": viol, "compliant": not viol}


if __name__ == "__main__":
    r = audit_telemetry()
    s = r["summary"]
    print("Warranty & Availability audit — BRPL Kilokari (real telemetry)")
    print(f"  terms: <= {r['terms']['max_cycles_per_day']} FCE/day, SoC "
          f"{r['terms']['soc_min_pct']:.0f}-{r['terms']['soc_max_pct']:.0f}% "
          f"({r['terms']['note']})")
    for d in r["days"]:
        flag = " | ".join(d["violations"]) if d["violations"] else "ok"
        rel = "" if d["fce_reliable"] else " (partial sampling — undercount)"
        print(f"  {d['day']}  cov {d['coverage_pct']:5.1f}%  FCE {d['fce']:.2f}{rel}"
              f"  SoC {d['soc_min_pct']:.0f}-{d['soc_max_pct']:.0f}%  -> {flag}")
    print(f"  summary: {s['total_violations']} violation(s) across "
          f"{s['days_observed']} observed days")

    # pre-trade check on tomorrow's plan, if one exists
    plan_f = HERE.parent / "output" / "plan_latest.csv"
    if plan_f.exists():
        plan = pd.read_csv(plan_f, parse_dates=["ts"], index_col="ts")
        if "soc_mwh" in plan:
            chk = audit_schedule(plan["soc_mwh"], 40.0)
            print(f"\n  tomorrow's plan: {chk['fce']} FCE, SoC "
                  f"{chk['soc_min_pct']}-{chk['soc_max_pct']}% -> "
                  f"{'COMPLIANT' if chk['compliant'] else chk['violations']}")
