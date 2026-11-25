"""Clean-room implementation of the Dice + Hungarian full-stack fault diagnosis evaluator,
together with trustworthy protocol mitigation variants evaluated in the PROVE BlockSys'2027 paper.

Protocol variants:
  - "original"   : Official protocol with precision gate floor alpha_0 = 0.7 applied ONLY to AD.
  - "no_floor"   : Floorless precision gate (alpha_0 = 0.0) applied to AD only.
  - "normalized" : Proposed Trustworthy Emission-Normalized Protocol (TENP):
                   floorless precision gate alpha_norm = precision^gamma applied across ALL four
                   terms (AD, RCA, Major, Minor), eliminating both grid-flooding and unpenalized
                   over-emission while preserving true-positive discrimination.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class FaultInterval:
    start_min: float
    end_min: float
    element: str
    major: str
    sub: str
    top5: tuple[str, ...] = ()


def dice_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    denom = max(a1 - a0, 0.0) + max(b1 - b0, 0.0)
    return (2.0 * inter / denom) if denom > 0.0 else 0.0


def evaluate_intervals(
    truths: list[FaultInterval],
    preds: list[FaultInterval],
    *,
    dice_gate: float = 0.4,
    t_max_s: float = 180.0,
    protocol: str = "original",
    alpha_floor: float = 0.7,
    norm_gamma: float = 1.0,
) -> dict[str, float]:
    """One-to-one maximum-weight Dice matching (Hungarian) and full-stack scoring."""
    n_true = len(truths)
    n_pred = len(preds)
    if n_true == 0:
        return {
            "Total": 0.0, "AD": 0.0, "RCA": 0.0, "Major": 0.0, "Minor": 0.0,
            "TP": 0, "FP": n_pred, "FN": 0, "precision": 0.0, "recall": 0.0,
            "f1": 0.0, "alpha_fp": 0.0, "mean_a": 0.0, "mean_rca": 0.0,
        }
    if n_pred == 0:
        return {
            "Total": 0.0, "AD": 0.0, "RCA": 0.0, "Major": 0.0, "Minor": 0.0,
            "TP": 0, "FP": 0, "FN": n_true, "precision": 0.0, "recall": 0.0,
            "f1": 0.0, "alpha_fp": alpha_floor if protocol == "original" else 0.0,
            "mean_a": 0.0, "mean_rca": 0.0,
        }

    # Vectorised Dice matrix (n_true x n_pred)
    t0 = np.array([t.start_min for t in truths], dtype=np.float64)[:, None]
    t1 = np.array([t.end_min for t in truths], dtype=np.float64)[:, None]
    p0 = np.array([p.start_min for p in preds], dtype=np.float64)[None, :]
    p1 = np.array([p.end_min for p in preds], dtype=np.float64)[None, :]

    inter = np.maximum(0.0, np.minimum(t1, p1) - np.maximum(t0, p0))
    denom = np.maximum(t1 - t0, 0.0) + np.maximum(p1 - p0, 0.0)
    dice = np.where(denom > 0.0, 2.0 * inter / np.maximum(denom, 1e-12), 0.0)
    valid = dice >= dice_gate

    # Maximize sum of Dice over valid edges via linear_sum_assignment on negated weights
    cost = np.where(valid, -dice, 0.0)
    row_ind, col_ind = linear_sum_assignment(cost)

    tp = 0
    sum_a = 0.0
    sum_rca = 0.0
    sum_maj = 0.0
    sum_min = 0.0
    for r, c in zip(row_ind, col_ind):
        if not valid[r, c]:
            continue
        tp += 1
        tr = truths[r]
        pr = preds[c]
        dt_s = (abs(pr.start_min - tr.start_min) + abs(pr.end_min - tr.end_min)) * 60.0
        s_time = max(0.0, 1.0 - dt_s / (2.0 * t_max_s))
        sum_a += 0.7 + 0.3 * s_time

        top5 = pr.top5 if pr.top5 else (pr.element,)
        if tr.element in top5:
            rank = top5.index(tr.element) + 1
            # Standard discounted Top-5 credit (1.0, 0.8, 0.6, 0.4, 0.2)
            sum_rca += max(0.0, 1.0 - 0.2 * (rank - 1))
        if pr.major == tr.major:
            sum_maj += 1.0
            if pr.sub == tr.sub:
                sum_min += 1.0

    fp = n_pred - tp
    fn = n_true - tp
    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_true
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    if protocol == "original":
        alpha_fp = alpha_floor + (1.0 - alpha_floor) * precision
        ad = 40.0 * (sum_a / n_true) * alpha_fp
        rca = 40.0 * (sum_rca / n_true)
        maj = 10.0 * (sum_maj / n_true)
        sub = 10.0 * (sum_min / n_true)
    elif protocol == "no_floor":
        alpha_fp = precision
        ad = 40.0 * (sum_a / n_true) * alpha_fp
        rca = 40.0 * (sum_rca / n_true)
        maj = 10.0 * (sum_maj / n_true)
        sub = 10.0 * (sum_min / n_true)
    elif protocol == "normalized":
        # Trustworthy Emission-Normalized Protocol (TENP): precision^gamma gates all 4 terms
        alpha_fp = precision ** norm_gamma
        ad = 40.0 * (sum_a / n_true) * alpha_fp
        rca = 40.0 * (sum_rca / n_true) * alpha_fp
        maj = 10.0 * (sum_maj / n_true) * alpha_fp
        sub = 10.0 * (sum_min / n_true) * alpha_fp
    else:
        raise ValueError(f"Unknown protocol: {protocol!r}")

    return {
        "Total": float(ad + rca + maj + sub),
        "AD": float(ad),
        "RCA": float(rca),
        "Major": float(maj),
        "Minor": float(sub),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "alpha_fp": float(alpha_fp),
        "mean_a": float(sum_a / tp) if tp else 0.0,
        "mean_rca": float(sum_rca / tp) if tp else 0.0,
    }


def evaluate_point_adjusted(
    truths: list[FaultInterval],
    preds: list[FaultInterval],
    n_minutes: int = 20160,
) -> dict[str, float]:
    """Point-adjusted (PA) F1 scoring (Kim et al., AAAI 2022) for comparison against Dice>=0.4."""
    y_true = np.zeros(n_minutes, dtype=bool)
    y_pred = np.zeros(n_minutes, dtype=bool)
    for p in preds:
        a = max(0, int(np.floor(p.start_min)))
        b = min(n_minutes, int(np.ceil(p.end_min)))
        y_pred[a:b] = True
    y_pa = y_pred.copy()
    for t in truths:
        a = max(0, int(np.floor(t.start_min)))
        b = min(n_minutes, int(np.ceil(t.end_min)))
        y_true[a:b] = True
        if y_pred[a:b].any():
            y_pa[a:b] = True
    tp = int((y_true & y_pa).sum())
    fp = int((~y_true & y_pa).sum())
    fn = int((y_true & ~y_pa).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"PA_precision": prec, "PA_recall": rec, "PA_F1": f1}
