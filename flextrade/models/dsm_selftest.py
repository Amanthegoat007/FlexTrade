"""Correctness checks for models/dsm.py, run before this engine is trusted
with real settlement numbers. Not exhaustive regulatory validation (see
the module docstring's disclaimer) -- this checks internal consistency:
band arithmetic, sign conventions, and the profile-isolation bug fixed
during development (the 2026-04-01 tightening was briefly, incorrectly,
leaking into the frozen 2022 profile).

    python models/dsm_selftest.py
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import dsm  # noqa: E402

FAILS = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        FAILS.append(name)


def _series(vals, start="2026-07-21", freq="15min"):
    idx = pd.date_range(start, periods=len(vals), freq=freq)
    return pd.Series(vals, idx, dtype=float)


def main():
    print("1. Band arithmetic: partial-span blending")
    # 8% over-injection should blend 5%@100% + 3%@90%, not jump to a flat rate
    r = dsm._apply_bands(8.0, dsm.WS_OVER_2022)
    expected = (5 * 1.00 + 3 * 0.90) / 8
    check(f"8% WS over-injection blended rate ({r:.4f} == {expected:.4f})",
          abs(r - expected) < 1e-9)

    print("\n2. Boundary inclusivity")
    check("exactly 5% over-injection is fully in the 100% band",
          abs(dsm._apply_bands(5.0, dsm.WS_OVER_2022) - 1.00) < 1e-9)
    check("exactly 10% under-injection still free (inclusive upper bound)",
          abs(dsm._apply_bands(10.0, dsm.WS_UNDER_2022)) < 1e-9)

    print("\n3. 2022 profile is frozen -- unaffected by the 2026-04-01 amendment")
    act = _series([45.0])          # -5 MW vs 50 MW schedule = -10% deviation
    sch = _series([50.0])
    dam = _series([5000.0]); rtm = _series([5000.0])
    pre = dsm.settle_2022(act, sch, dam, rtm, technology="solar", seller="ws",
                          settlement_date=date(2025, 1, 1))
    post = dsm.settle_2022(act, sch, dam, rtm, technology="solar", seller="ws",
                           settlement_date=date(2026, 7, 21))
    check("2022 settlement identical before/after the 2024-lineage cutover date",
          abs(pre["charge_rs"].iloc[0] - post["charge_rs"].iloc[0]) < 1e-9)
    check("-10% under-injection is exactly at the free boundary -> zero charge",
          abs(post["charge_rs"].iloc[0]) < 1e-9)

    print("\n4. 2024 profile DOES respond to the 2026-04-01 tightening")
    # 7 MW deviation on 100 MW available capacity = 7% -- deliberately
    # between the pre-cutover (10%) and post-cutover (5%) tolerance, so
    # the tightening flips it from within-band to chargeable.
    act7 = _series([43.0])  # 50 - 7
    freq = 50.0
    before = dsm.settle_2024(act7, sch, freq, dam, rtm, available_capacity_mw=100,
                             seller="ws", technology="solar",
                             settlement_date=date(2026, 3, 31))
    after = dsm.settle_2024(act7, sch, freq, dam, rtm, available_capacity_mw=100,
                            seller="ws", technology="solar",
                            settlement_date=date(2026, 4, 1))
    check(f"solar tolerance narrows 10%->5% at cutover "
          f"(before tol={before['tolerance_mw'].iloc[0]:.1f} MW, "
          f"after tol={after['tolerance_mw'].iloc[0]:.1f} MW)",
          before["tolerance_mw"].iloc[0] == 10.0 and after["tolerance_mw"].iloc[0] == 5.0)
    check("a 7% deviation is within the old 10% band but outside the new 5% band",
          before["outside_band"].iloc[0] == False and after["outside_band"].iloc[0] == True)

    print("\n5. Sign conventions (seller pays vs receives)")
    over = dsm.settle_2022(_series([55.0]), sch, dam, rtm, seller="ws",
                           settlement_date=date(2026, 7, 21))
    under = dsm.settle_2022(_series([45.0]), sch, dam, rtm, seller="ws",
                            settlement_date=date(2026, 7, 21))
    check("WS over-injection within band is a net credit to the seller (charge_rs <= 0)",
          over["charge_rs"].iloc[0] <= 0)
    buyer_over = dsm.settle_2022(_series([65.0]), sch, dam, rtm, seller="buyer",
                                 settlement_date=date(2026, 7, 21))
    check("buyer over-drawal is a net debit (charge_rs >= 0)",
          buyer_over["charge_rs"].iloc[0] >= 0)

    print("\n6. X-factor glide path")
    check("X=1.0 (pure available capacity) for FY26-27, unchanged as confirmed by CERC",
          dsm.x_factor("solar", date(2026, 7, 21)) == 1.00)
    check("X starts declining from FY27-28 (1 Apr 2027) for solar",
          dsm.x_factor("solar", date(2027, 4, 1)) == 0.90)
    check("wind glide path is slower than solar/hybrid at the same date",
          dsm.x_factor("wind", date(2027, 4, 1)) > dsm.x_factor("solar", date(2027, 4, 1)))

    print("\n7. Unified dispatch matches direct calls")
    a = dsm.settle("CERC_2022", actual_mw=act, scheduled_mw=sch, dam_price=dam,
                   rtm_price=rtm, seller="ws", settlement_date=date(2026, 7, 21))
    b = dsm.settle_2022(act, sch, dam, rtm, seller="ws",
                        settlement_date=date(2026, 7, 21))
    check("settle('CERC_2022', ...) matches settle_2022(...)",
          abs(a["charge_rs"].iloc[0] - b["charge_rs"].iloc[0]) < 1e-9)

    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nAll checks passed.")
    if FAILS:
        sys.exit(1)


if __name__ == "__main__":
    main()
