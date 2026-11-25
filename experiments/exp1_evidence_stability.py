"""Experiment 1: Prototype verification of PROVE's Evidence Plane stability and
offline excursion-contraction transfer failure reproduction.

Uses the actual `src.evidence` and `src.detect` modules on a controlled 8-region x 10-role
multi-source telemetry stream (no private competition data required).
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config as C
from src import evidence as E
from src import detect as D


def run_exp1(seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)

    # 1A: Four-way scale floor vs unfloored MAD robust z-score on quasi-constant & zero-inflated channels
    n_min = 20160
    n_ch = 40
    # 10 quasi-constant channels (MAD=0), 15 zero-inflated iface channels (95% zeros), 15 Gaussian channels
    X = np.zeros((n_min, n_ch), dtype=np.float64)
    # Quasi-constant: baseline 100.0 with tiny rare step +0.05 on 0.2% of minutes
    for j in range(10):
        X[:, j] = 100.0
        spike_idx = rng.choice(n_min, size=40, replace=False)
        X[spike_idx, j] += 0.05
    # Zero-inflated interface error/drop counters: 95% exact zero, 5% small noise, plus 20 real fault bursts
    fault_starts = np.linspace(500, n_min - 500, 20, dtype=int)
    for j in range(10, 25):
        nz = rng.random(n_min) < 0.05
        X[nz, j] = rng.exponential(0.5, size=nz.sum())
        for fs in fault_starts:
            X[fs:fs + 14, j] += rng.uniform(8.0, 15.0)
    # Continuous Gaussian metrics
    for j in range(25, 40):
        X[:, j] = rng.normal(50.0, 2.0, size=n_min)
        for fs in fault_starts:
            X[fs:fs + 14, j] += rng.uniform(10.0, 16.0)

    # Unfloored MAD z-score (clipped at 1e6 for numerical inspection)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    z_unfloored = np.abs(X - med) / np.maximum(1.4826 * mad, 1e-9)
    spurious_unfloored_max = float(z_unfloored[:, :10].max())
    spurious_unfloored_gt40 = float((z_unfloored[:, :25] > 40.0).mean())

    # PROVE four-way scale floor (Eq. 3 in paper)
    q95 = np.quantile(np.abs(X - med), 0.95, axis=0)
    q99 = np.quantile(np.abs(X - med), 0.99, axis=0)
    denom_4way = np.maximum.reduce([
        1.4826 * mad,
        q95 / 1.96,
        0.01 * np.abs(med),
        0.02 * q99,
        np.full(n_ch, 1e-6),
    ])
    z_prove = np.clip(np.abs(X - med) / denom_4way, 0.0, 40.0)
    spurious_prove_max_qconst = float(z_prove[:, :10].max())
    # Check signal-to-background ratio on fault windows vs normal minutes for zero-inflated channels
    fault_mask = np.zeros(n_min, dtype=bool)
    for fs in fault_starts:
        fault_mask[fs:fs + 14] = True
    sbr_unfloored = float(np.median(z_unfloored[fault_mask, 10:25]) / max(np.quantile(z_unfloored[~fault_mask, 10:25], 0.99), 1e-9))
    sbr_prove = float(np.median(z_prove[fault_mask, 10:25]) / max(np.quantile(z_prove[~fault_mask, 10:25], 0.99), 1e-9))

    # 1B: Excursion-contraction transfer failure reproduction (Seg0 anchor vs Seg2 degraded)
    # Simulate 2 segments of 20,160 min each on 4 surviving sources (node, iface, route, scrape)
    # with D1-measured excursion contraction factors: iface=0.18, scrape=0.59, node=0.82, route=0.87
    contraction = {"iface": 0.18, "scrape": 0.59, "node": 0.82, "route": 0.87}
    n_faults = 120
    # Generate ground-truth intervals (duration p50 ~ 14 min, gap >= 60 min)
    starts_s0 = np.sort(rng.choice(np.arange(100, n_min - 100, 120), size=n_faults, replace=False))
    durs_s0 = rng.integers(6, 24, size=n_faults)
    starts_s2 = np.sort(rng.choice(np.arange(100, n_min - 100, 120), size=n_faults, replace=False))
    durs_s2 = rng.integers(6, 24, size=n_faults)

    def build_seg_envelope(starts, durs, scale_map, noise_boost=1.0):
        # 4 source channels per segment
        raw = np.zeros((n_min, 4), dtype=np.float64)
        kinds = ["node", "iface", "route", "scrape"]
        for k_i, kname in enumerate(kinds):
            # Heavy-tailed background noise
            raw[:, k_i] = np.abs(rng.standard_t(df=4, size=n_min)) * 0.6 * noise_boost
            sc = scale_map[kname]
            for s, d in zip(starts, durs):
                # Each fault activates a subset of sources with a flat-top + tapered edges
                if rng.random() < 0.45:
                    taper = np.hanning(d + 2)[1:-1]
                    amp = rng.uniform(5.0, 11.0) * sc
                    raw[s:s + d, k_i] += amp * (0.4 + 0.6 * taper)
        # Apply PROVE envelope_excess on single-segment array
        med_ = np.median(raw, axis=0)
        mad_ = np.median(np.abs(raw - med_), axis=0)
        Z = np.abs(raw - med_) / np.maximum(1.4826 * mad_, 1e-6)
        env = np.quantile(Z, 0.99, axis=0)
        ex = np.maximum(Z / np.maximum(env, 1e-6) - 1.0, 0.0)
        # 9-minute moving average track (key smooth track in PROVE)
        kernel = np.ones(9) / 9.0
        ex_w9 = np.column_stack([np.convolve(ex[:, j], kernel, mode="same") for j in range(4)])
        return ex, ex_w9

    ex0_w0, ex0_w9 = build_seg_envelope(starts_s0, durs_s0, {k: 1.0 for k in contraction})
    ex2_w0, ex2_w9 = build_seg_envelope(starts_s2, durs_s2, contraction, noise_boost=1.15)

    # Measure excursion ratios (q999 / median_nz) between S2 and S0
    excursion_measured = {}
    for k_i, kname in enumerate(["node", "iface", "route", "scrape"]):
        v0 = ex0_w9[:, k_i]
        v2 = ex2_w9[:, k_i]
        e0 = np.quantile(v0, 0.999) / max(np.median(v0[v0 > 0]), 1e-6)
        e2 = np.quantile(v2, 0.999) / max(np.median(v2[v2 > 0]), 1e-6)
        excursion_measured[kname] = round(float(e2 / e0), 3)

    # Monotone quantile alignment of S2 onto S0 (mirrors src.evidence.align_segments_to_anchor)
    ex2_aligned = np.zeros_like(ex2_w9)
    qgrid = E.CDF_ALIGN_Q
    for j in range(4):
        qr = np.maximum.accumulate(np.quantile(ex0_w9[:, j], qgrid))
        qs = np.maximum.accumulate(np.quantile(ex2_w9[:, j], qgrid))
        keep = np.concatenate([[True], np.diff(qs) > 1e-12])
        xs, ys = qs[keep], qr[keep]
        new = np.interp(ex2_w9[:, j], xs, ys)
        ex2_aligned[:, j] = new

    # Check p999 alignment accuracy
    p999_s0 = float(np.quantile(ex0_w9.max(axis=1), 0.999))
    p999_s2_raw = float(np.quantile(ex2_w9.max(axis=1), 0.999))
    p999_s2_aligned = float(np.quantile(ex2_aligned.max(axis=1), 0.999))

    # Evaluate detection recall (Dice >= 0.4) at fixed top-K=160 peaks on S0, S2_raw, S2_aligned
    def eval_env_recall(env_mat, starts, durs, top_k=160):
        score = env_mat.max(axis=1)
        pks = D.local_maxima_nms(score, radius=3, thr=0.1)
        if len(pks) > top_k:
            pks = pks[np.argsort(-score[pks])[:top_k]]
        # Flat-top crossing windows
        wins = []
        for t in pks:
            lvl = 0.5 * score[t]
            a = int(t)
            while a > 0 and score[a - 1] >= lvl and (t - a) < 15:
                a -= 1
            b = int(t) + 1
            while b < n_min and score[b] >= lvl and (b - t) < 15:
                b += 1
            wins.append((a, max(b, a + 2)))
        # Count truths matched at Dice >= 0.4
        hits = 0
        for s, d in zip(starts, durs):
            best = 0.0
            for a, b in wins:
                inter = max(0, min(s + d, b) - max(s, a))
                dc = 2.0 * inter / (d + (b - a))
                if dc > best:
                    best = dc
            if best >= 0.4:
                hits += 1
        med_len = float(np.median([b - a for a, b in wins])) if wins else 0.0
        return round(hits / len(starts), 4), round(med_len, 2)

    rec_s0, len_s0 = eval_env_recall(ex0_w9, starts_s0, durs_s0)
    rec_s2_raw, len_s2_raw = eval_env_recall(ex2_w9, starts_s2, durs_s2)
    rec_s2_aligned, len_s2_aligned = eval_env_recall(ex2_aligned, starts_s2, durs_s2)

    return {
        "exp1a_scale_floor": {
            "unfloored_qconst_max_z": round(spurious_unfloored_max, 1),
            "unfloored_spurious_frac_gt40": round(spurious_unfloored_gt40, 4),
            "prove_4way_qconst_max_z": round(spurious_prove_max_qconst, 4),
            "zero_inflated_sbr_unfloored": round(sbr_unfloored, 3),
            "zero_inflated_sbr_prove": round(sbr_prove, 3),
        },
        "exp1b_excursion_and_alignment": {
            "measured_excursion_ratios": excursion_measured,
            "p999_s0": round(p999_s0, 3),
            "p999_s2_raw": round(p999_s2_raw, 3),
            "p999_s2_aligned": round(p999_s2_aligned, 3),
            "p999_rel_gap_after_align_pct": round(abs(p999_s2_aligned - p999_s0) / max(p999_s0, 1e-6) * 100.0, 2),
            "recall_s0": rec_s0,
            "recall_s2_raw": rec_s2_raw,
            "recall_s2_aligned": rec_s2_aligned,
            "median_win_s0": len_s0,
            "median_win_s2_raw": len_s2_raw,
            "median_win_s2_aligned": len_s2_aligned,
        },
    }


if __name__ == "__main__":
    res = run_exp1()
    print(json.dumps(res, indent=2))
