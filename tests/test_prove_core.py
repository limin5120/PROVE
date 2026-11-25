"""Automated unit and integration tests for the PROVE open-source artifact."""
from __future__ import annotations

import os
import sys
import unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config as C
from src import evidence as E
from src import detect as D
from official_eval.evaluator import FaultInterval, dice_overlap, evaluate_intervals
from experiments.exp3_exp4_gates_and_scale import run_exp3_anti_vacuity


class TestProveCore(unittest.TestCase):
    def test_canonical_axis_and_segments(self):
        self.assertEqual(C.N_MINUTES, 50400)
        self.assertEqual(len(C.REGIONS), 8)
        self.assertEqual(len(C.ROLES), 10)
        self.assertEqual(len(C.ELEMENT_IDS), 80)
        self.assertEqual(len(C.CATEGORY_PAIRS), 28)
        # Verify clamping never straddles segments
        t0, t1 = C.clamp_to_segment(20150, 20175, ref_m=20155)
        self.assertLessEqual(t1, 20160)
        # Verify exclusive end timestamp stays inside seg0 at boundary 20160
        end_str = C.fmt_end_utc(20160)
        self.assertEqual(end_str, "2026-09-02T04:00:00.000Z")

    def test_envelope_excess_and_movmean_segmented(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0.0, 1.0, size=(C.N_MINUTES, 3)).astype(np.float64)
        # Inject strong fault in seg0
        X[1000:1015, 0] += 12.0
        # Make column 2 perfectly constant (dead channel)
        X[:, 2] = 42.0
        ex = E.envelope_excess(X)
        self.assertEqual(ex.shape, (C.N_MINUTES, 3))
        self.assertGreater(ex[1005, 0], 1.0)
        self.assertEqual(float(ex[:, 2].max()), 0.0)
        sm = E.movmean_segmented(ex, 9)
        self.assertGreater(sm[1007, 0], 0.5)

    def test_nms_and_flat_top_width(self):
        v = np.zeros(C.N_MINUTES, dtype=np.float32)
        v[500:512] = 5.0
        v[505] = 8.0
        pks = D.local_maxima_nms(v, radius=3, thr=0.25)
        self.assertIn(505, pks)
        a, b = D.flat_top_width(v, 505, frac=0.5, maxlen=30)
        self.assertEqual((a, b), (500, 512))
        hyps = D.hypotheses(505, v, anchor=505)
        self.assertGreaterEqual(len(hyps), 2)

    def test_evaluator_and_tenp_mitigation(self):
        self.assertAlmostEqual(dice_overlap(0, 10, 5, 15), 0.5, places=6)
        truths = [FaultInterval(100, 114, "beida-fw", "firewall", "port_block")]
        good = [FaultInterval(100, 114, "beida-fw", "firewall", "port_block", ("beida-fw",))]
        r_orig = evaluate_intervals(truths, good, protocol="original")
        r_tenp = evaluate_intervals(truths, good, protocol="normalized", norm_gamma=1.0)
        self.assertAlmostEqual(r_orig["Total"], 100.0, places=4)
        self.assertAlmostEqual(r_tenp["Total"], 100.0, places=4)

        # Flooding with 100 FPs should be sharply penalized by TENP
        flood = good + [FaultInterval(500 + 20 * i, 510 + 20 * i, "beida-fw", "firewall", "port_block")
                        for i in range(100)]
        f_orig = evaluate_intervals(truths, flood, protocol="original")
        f_tenp = evaluate_intervals(truths, flood, protocol="normalized", norm_gamma=1.0)
        self.assertGreater(f_orig["Total"], 80.0)   # Original protocol barely penalizes!
        self.assertLess(f_tenp["Total"], 2.0)       # TENP crushes the flooding exploit!

    def test_anti_vacuity_gate(self):
        res = run_exp3_anti_vacuity()
        self.assertTrue(res["all_assertions_ok"])


if __name__ == "__main__":
    unittest.main()
