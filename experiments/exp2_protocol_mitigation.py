"""Experiment 2: Prototype verification of Proposition 1 (O(1/E^2) break-even hit rate),
the grid-flooding vulnerability, and our proposed Trustworthy Emission-Normalized Protocol (TENP),
plus Dice>=0.4 vs Point-Adjusted (PA) F1 comparison.
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from official_eval.evaluator import (
    FaultInterval,
    evaluate_intervals,
    evaluate_point_adjusted,
)


def generate_synthetic_truths(n_true: int = 292, n_minutes: int = 20160, seed: int = 7) -> list[FaultInterval]:
    rng = np.random.default_rng(seed)
    # D1-like duration distribution: min~2, p25~8, p50~13.6, p75~21, max~29, gap >= 20 min
    elements = [f"r{i}-e{j}" for i in range(8) for j in range(9)]
    majors = ["firewall", "routing", "link", "resource", "service"]
    major_probs = [0.25, 0.22, 0.18, 0.20, 0.15]
    # Place non-overlapping intervals with >= 20 min spacing
    slots = np.linspace(30, n_minutes - 60, n_true)
    jitter = rng.uniform(-12.0, 12.0, size=n_true)
    starts = np.clip(slots + jitter, 5.0, n_minutes - 35.0)
    starts.sort()
    truths = []
    for s in starts:
        dur = float(np.clip(rng.gamma(shape=3.2, scale=4.5), 2.0, 29.0))
        maj = str(rng.choice(majors, p=major_probs))
        sub = f"{maj}_sub{int(rng.integers(0, 4))}"
        elem = str(rng.choice(elements[:15] if rng.random() < 0.6 else elements))
        truths.append(FaultInterval(float(s), float(s + dur), elem, maj, sub))
    return truths


def build_base_predictions(
    truths: list[FaultInterval],
    recall: float = 0.583,
    n_fp: int = 256,
    n_minutes: int = 20160,
    seed: int = 11,
) -> list[FaultInterval]:
    rng = np.random.default_rng(seed)
    n_tp = int(round(len(truths) * recall))
    chosen = rng.choice(len(truths), size=n_tp, replace=False)
    preds = []
    for idx in sorted(chosen):
        tr = truths[idx]
        ds = float(rng.normal(0.0, 1.4))
        de = float(rng.normal(0.0, 1.4))
        p0 = max(0.0, tr.start_min + ds)
        p1 = min(float(n_minutes), max(p0 + 2.0, tr.end_min + de))
        # High-quality RCA/classification on detected base events
        if rng.random() < 0.88:
            top5 = (tr.element, "r0-e0", "r1-e0", "r2-e0", "r3-e0")
        elif rng.random() < 0.70:
            top5 = ("r0-e0", tr.element, "r1-e0", "r2-e0", "r3-e0")
        else:
            top5 = ("r0-e0", "r1-e0", "r2-e0", "r3-e0", "r4-e0")
        maj = tr.major if rng.random() < 0.90 else "firewall"
        sub = tr.sub if (maj == tr.major and rng.random() < 0.82) else f"{maj}_sub0"
        preds.append(FaultInterval(p0, p1, top5[0], maj, sub, top5))

    # Add false positives away from truth intervals
    for _ in range(n_fp):
        s = float(rng.uniform(0, n_minutes - 15))
        d = float(rng.uniform(4.0, 18.0))
        preds.append(FaultInterval(s, s + d, "r0-e0", "firewall", "firewall_sub0",
                                   ("r0-e0", "r1-e0", "r2-e0", "r3-e0", "r4-e0")))
    return preds


def build_grid_predictions(n_minutes: int = 20160, scales=((30, 15), (10, 5))) -> list[FaultInterval]:
    # Content prior: modal top5 + modal (firewall, firewall_sub0)
    top5 = ("r0-e0", "r0-e1", "r0-e2", "r0-e3", "r0-e4")
    grid = []
    for L, S in scales:
        t = 0.0
        while t < n_minutes:
            e = min(t + L, float(n_minutes))
            if e - t >= 1.0:
                grid.append(FaultInterval(t, e, top5[0], "firewall", "firewall_sub0", top5))
            t += S
    return grid


def run_exp2() -> dict:
    truths = generate_synthetic_truths(n_true=292, n_minutes=20160, seed=7)

    # 2A: Verify Proposition 1 O(1/E^2) break-even marginal hit rate curve
    # Using exact D1-fitted parameters (N=292, a=1.0, B=54.203, T=170) and empirical finite-diff
    N = 292.0
    a_rec = 1.0
    B_rec = 54.203
    T_op = 170.0
    E_grid = [250, 426, 853, 1673, 3000, 5102, 8000, 10000]
    breakeven_curve = []
    for E in E_grid:
        # Exact analytical derivatives of S(T, E) = (T/N)*(40*a*(0.7 + 0.3*T/E) + B)
        dS_dE_orig = -(T_op / N) * 40.0 * a_rec * 0.3 * T_op / (E ** 2)
        dS_dT_orig = (40.0 * a_rec * 0.7 + B_rec) / N + (2.0 * T_op / N) * 40.0 * a_rec * 0.3 / E
        p_be_orig = (-dS_dE_orig) / dS_dT_orig

        # Under TENP (gamma=1.0): S_tenp(T, E) = (T/N)*(T/E)*(40*a + B) = (T^2 / (N*E))*(40*a + B)
        # dS/dE = -(T^2/(N*E^2))*(40*a+B), dS/dT = (2*T/(N*E))*(40*a+B) => p_be = T / (2*E) = precision/2!
        p_be_tenp = T_op / (2.0 * E)

        breakeven_curve.append({
            "E": E,
            "orig_dS_dE": round(dS_dE_orig, 7),
            "orig_dS_dT": round(dS_dT_orig, 4),
            "orig_breakeven_pct": round(p_be_orig * 100.0, 4),
            "tenp_breakeven_pct": round(p_be_tenp * 100.0, 4),
        })

    # 2B: Strategy comparison across Original vs No-Floor vs TENP(gamma=0.5) vs TENP(gamma=1.0)
    base = build_base_predictions(truths, recall=0.583, n_fp=256, seed=11)
    # Second mechanism-distinct detector (complementary recall, moderate FP)
    det2 = build_base_predictions(truths, recall=0.35, n_fp=400, seed=29)
    union_preds = base + det2
    grid = build_grid_predictions(20160, scales=((30, 15), (10, 5)))
    flooded = base + grid
    # High-precision PROVE-like S0 detector
    s0_clean = build_base_predictions(truths, recall=0.974, n_fp=3, seed=41)

    strategies = {
        "Clean_HighPrec (S0-like)": s0_clean,
        "Base_Degraded (S2-like)": base,
        "Union_2Mech (Base+Det2)": union_preds,
        "Grid_Only (30/15+10/5)": grid,
        "Base_Plus_GridFlood": flooded,
    }

    protocol_results = {}
    for sname, preds in strategies.items():
        r_orig = evaluate_intervals(truths, preds, protocol="original", alpha_floor=0.7)
        r_nofl = evaluate_intervals(truths, preds, protocol="no_floor")
        r_tenp05 = evaluate_intervals(truths, preds, protocol="normalized", norm_gamma=0.5)
        r_tenp10 = evaluate_intervals(truths, preds, protocol="normalized", norm_gamma=1.0)
        r_pa = evaluate_point_adjusted(truths, preds, n_minutes=20160)
        protocol_results[sname] = {
            "E": len(preds),
            "TP": r_orig["TP"],
            "precision": round(r_orig["precision"], 4),
            "recall": round(r_orig["recall"], 4),
            "F1": round(r_orig["f1"], 4),
            "PA_F1": round(r_pa["PA_F1"], 4),
            "Score_Original": round(r_orig["Total"], 2),
            "Score_NoFloor_AD": round(r_nofl["Total"], 2),
            "Score_TENP_g05": round(r_tenp05["Total"], 2),
            "Score_TENP_g10": round(r_tenp10["Total"], 2),
        }

    # Flooding delta summary
    flood_deltas = {
        "Original_alpha07": round(
            protocol_results["Base_Plus_GridFlood"]["Score_Original"]
            - protocol_results["Base_Degraded (S2-like)"]["Score_Original"], 2
        ),
        "NoFloor_AD_only": round(
            protocol_results["Base_Plus_GridFlood"]["Score_NoFloor_AD"]
            - protocol_results["Base_Degraded (S2-like)"]["Score_NoFloor_AD"], 2
        ),
        "TENP_gamma05": round(
            protocol_results["Base_Plus_GridFlood"]["Score_TENP_g05"]
            - protocol_results["Base_Degraded (S2-like)"]["Score_TENP_g05"], 2
        ),
        "TENP_gamma10": round(
            protocol_results["Base_Plus_GridFlood"]["Score_TENP_g10"]
            - protocol_results["Base_Degraded (S2-like)"]["Score_TENP_g10"], 2
        ),
    }

    return {
        "exp2a_breakeven_curve": breakeven_curve,
        "exp2b_protocol_comparison": protocol_results,
        "exp2c_flood_exploit_delta": flood_deltas,
    }


if __name__ == "__main__":
    res = run_exp2()
    print(json.dumps(res, indent=2))
