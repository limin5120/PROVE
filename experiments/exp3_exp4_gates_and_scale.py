"""Experiment 3: Prototype verification of Sentinel-Subject Isolation (Anti-Vacuity Gate)
in `src.verify` and Experiment 4: Scalability & Online Latency Benchmark across 80..800 elements.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import tracemalloc
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config as C
from src import evidence as E
from src import detect as D
from src import verify as V


def run_exp3_anti_vacuity() -> dict:
    """Verify that `src.verify.e1b_is_vacuous` and `src.verify.gates` reject vacuous frozen shards."""
    with tempfile.TemporaryDirectory(prefix="prove_gate_test_") as tmp:
        frozen_dir = os.path.join(tmp, "frozen")
        os.makedirs(frozen_dir, exist_ok=True)

        # Create a mock frozen shard for Segment 0 (2026-08-19 04:00 .. 2026-09-02 04:00)
        rec = {
            "prediction_id": "p-0001",
            "start_time": "2026-08-20T10:00:00.000Z",
            "end_time": "2026-08-20T10:14:00.000Z",
            "root_cause_top5": [{"rank": 1, "network_element_id": "beida-fw"}],
            "fault_category": {"major_category": "firewall", "sub_category": "port_block"},
        }
        frozen_path = os.path.join(frozen_dir, "submission_stage1.jsonl")
        shipped_path = os.path.join(tmp, "shipped_mixed.jsonl")
        with open(frozen_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        with open(shipped_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        # Case 1: Shipped file uses frozen shard, no --e1b-file supplied -> must detect vacuity & FAIL
        vac_msg = V.e1b_is_vacuous(shipped_path, frozen_dir=frozen_dir)
        shape_ok = {"outside_any_segment": 0}
        schema_ok = {"records": 1, "invalid": {}, "duplicate_prediction_id": 0}
        fake_frozen_score = {"Total": 92.3469, "AD": 35.38, "RCA": 38.50, "Major": 9.25,
                             "Minor": 9.21, "TP": 297, "FP": 3, "FN": 8}
        g_vac = V.gates(shape_ok, fake_frozen_score, schema_ok, e1b=None, vacuous=vac_msg)
        e1b_vac_pass = [ok for nm, ok, _ in g_vac if "E1b" in nm][0]

        # Case 2: Shipped file uses frozen shard + pure unified run scoring 70.56 -> must FAIL
        e1b_low = {"Total": 70.5554, "AD": 24.1, "RCA": 31.2, "Major": 8.0, "Minor": 7.25,
                   "TP": 269, "FP": 131, "FN": 36}
        g_low = V.gates(shape_ok, fake_frozen_score, schema_ok, e1b=e1b_low, vacuous=vac_msg)
        e1b_low_pass = [ok for nm, ok, _ in g_low if "E1b" in nm][0]

        # Case 3: Pure unified run scoring 89.14 (>= 88.0) -> must PASS
        e1b_good = {"Total": 89.14, "AD": 33.8, "RCA": 37.5, "Major": 9.0, "Minor": 8.84,
                    "TP": 296, "FP": 15, "FN": 9}
        g_good = V.gates(shape_ok, fake_frozen_score, schema_ok, e1b=e1b_good, vacuous=vac_msg)
        e1b_good_pass = [ok for nm, ok, _ in g_good if "E1b" in nm][0]

        return {
            "case1_vacuous_detected": vac_msg is not None,
            "case1_gate_passed": bool(e1b_vac_pass),
            "case2_low_unified_passed": bool(e1b_low_pass),
            "case3_valid_unified_passed": bool(e1b_good_pass),
            "all_assertions_ok": (vac_msg is not None) and (not e1b_vac_pass) and (not e1b_low_pass) and e1b_good_pass,
        }


def run_exp4_scalability(seed: int = 123) -> list[dict]:
    """Benchmark PROVE L1 evidence plane + smoothing + NMS + windowing from 80 to 800 elements."""
    rng = np.random.default_rng(seed)
    n_min = 20160  # 14 days at 1-min resolution
    results = []
    for n_elem in (80, 160, 320, 640, 800):
        # 4 surviving source channels per element (node, iface, route, scrape)
        n_ch = n_elem * 4
        X = rng.standard_normal((n_min, n_ch)).astype(np.float32)
        # Inject realistic fault bursts (~20 per region) so NMS exercises peak sorting & vetoing
        for r_i in range(n_elem // 10):
            for s_i in range(400, n_min - 400, 950):
                X[s_i:s_i + 12, r_i * 40 + (s_i % 40)] += 7.5

        tracemalloc.start()
        t0 = time.perf_counter()

        # L1a: Vectorised robust z-score + 99th-percentile envelope excess (14-day segment)
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0)
        scale = np.where(mad > 1e-12, 1.4826 * mad, 1.0)
        Z = np.abs(X - med) / scale
        env = np.quantile(Z, 0.99, axis=0)
        ex = np.maximum(Z / np.maximum(env, 1e-6) - 1.0, 0.0).astype(np.float32)

        # L1b: 9-min centred moving average via cumsum (identical to src.evidence.movmean_segmented)
        cv = np.concatenate([np.zeros((1, n_ch), dtype=np.float32), np.cumsum(ex, axis=0)])
        idx = np.arange(n_min)
        a = np.maximum(idx - 4, 0)
        b = np.minimum(idx + 5, n_min)
        ex_w9 = (cv[b] - cv[a]) / (b - a)[:, None]

        # L1c: Per-region max envelope + NMS peak detection across n_elem/10 regions
        n_reg = n_elem // 10
        total_peaks = 0
        for r in range(n_reg):
            reg_env = ex_w9[:, r * 40:(r + 1) * 40].max(axis=1)
            pks = D.local_maxima_nms(reg_env, radius=3, thr=0.25)
            total_peaks += len(pks)

        elapsed_s = time.perf_counter() - t0
        _, peak_mem_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        ms_per_min = (elapsed_s / n_min) * 1000.0
        results.append({
            "n_elements": n_elem,
            "n_channels": n_ch,
            "batch_14d_s": round(elapsed_s, 3),
            "ms_per_minute_tick": round(ms_per_min, 4),
            "peak_mem_mb": round(peak_mem_bytes / (1024 * 1024), 1),
            "peaks_detected": int(total_peaks),
        })
    return results


if __name__ == "__main__":
    out = {
        "exp3_anti_vacuity": run_exp3_anti_vacuity(),
        "exp4_scalability": run_exp4_scalability(),
    }
    print(json.dumps(out, indent=2))
