"""One-command runner for all PROVE BlockSys'2027 prototype verification experiments."""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.exp1_evidence_stability import run_exp1
from experiments.exp2_protocol_mitigation import run_exp2
from experiments.exp3_exp4_gates_and_scale import run_exp3_anti_vacuity, run_exp4_scalability
from experiments.exp5_baseline_benchmark import run_exp5


def main() -> int:
    print("[PROVE] Running Experiment 1: Evidence plane stability & excursion contraction...")
    r1 = run_exp1()
    print("[PROVE] Running Experiment 2: Protocol incentive & TENP mitigation verification...")
    r2 = run_exp2()
    print("[PROVE] Running Experiment 3: Anti-vacuity sentinel-subject isolation gate...")
    r3 = run_exp3_anti_vacuity()
    print("[PROVE] Running Experiment 4: Scalability & online latency benchmark (80..800 elements)...")
    r4 = run_exp4_scalability()
    print("[PROVE] Running Experiment 5: Controlled multi-source baseline comparison (5 seeds)...")
    r5 = run_exp5()

    summary = {
        "exp1": r1,
        "exp2": r2,
        "exp3": r3,
        "exp4": r4,
        "exp5": r5,
    }
    out_path = os.path.join(ROOT, "results", "prototype_verification_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[PROVE] All experiments completed. Saved -> {out_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
