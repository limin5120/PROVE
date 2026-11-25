# PROVE: Trustworthy Full-Stack Fault Diagnosis under Degraded Observability

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Conference](https://img.shields.io/badge/BlockSys-2027-green.svg)](https://blocksys.info/2027/)

Official open-source replication package and prototype verification artifact for the BlockSys'2027 paper:

> **PROVE: Trustworthy Full-Stack Fault Diagnosis and Verification under Degraded Observability**  
> Chen Wang and Min Li, Institute of Computer Application, China Academy of Engineering Physics, Mianyang, China.

## Overview

Modern backbone networks rely on multi-source telemetry (element probes, device metrics, logs, and NetFlow) for full-stack fault diagnosis. In operational deployments, however, the observation plane degrades persistently as probes are retired, log collection is suspended for compliance, and instrumentation evolves. **PROVE** provides:

1. **Source-Homogeneous Evidence Plane (`src/evidence.py`, `src/detect.py`)**: Converts heterogeneous multi-source telemetry into scale-bounded channels using a four-way robust z-score floor and per-segment envelope excess, eliminating spurious excursions on quasi-constant and zero-inflated streams.
2. **Layered Full-Stack Decoding (`src/features.py`, `src/models.py`, `src/train.py`, `src/infer.py`)**: Decouples event detection (`event` head), root-cause attribution (`root` head + NetFlow remote attribution), legal two-level classification (`cat` head with role-to-major constraints), and boundary duration debiasing (`dur` head).
3. **Observability-Ablation Surrogate & Anti-Vacuity Verification (`src/verify.py`, `src/diag.py`)**: Enforces **Sentinel-Subject Isolation** (`e1b_is_vacuous`) so frozen shards cannot vacuously pass quality gates, and quantifies $\Delta = \Delta_{\text{source}} + \Delta_{\text{mismatch}} + \Delta_{\text{composition}}$.
4. **Protocol Incentive Audit & Trustworthy Mitigation (`official_eval/evaluator.py`, `src/fuse_all.py`, `src/flood.py`)**: Implements both the original Dice $\ge 0.4$ Hungarian evaluator and our proposed **Trustworthy Emission-Normalized Protocol (TENP)** that neutralizes grid-flooding exploits.

## Repository Structure

```text
PROVE-BlockSys2027-Artifact/
├── src/                        # Core L0-L4 PROVE pipeline (16 modules)
│   ├── config.py               # Canonical 50,400-min multi-segment axis & official enums
│   ├── rawio.py                # Encoding-safe CSV ingestion (UTF-8/GBK boundary-safe)
│   ├── build_cache.py          # L0 per-region telemetry & NetFlow cache builder
│   ├── evidence.py             # L1 source-homogeneous evidence plane, calibration & alignment
│   ├── build_evidence.py       # L1 driver across all regions and segments
│   ├── detect.py               # Segment-balanced NMS peak detection & flat-top windowing
│   ├── features.py             # 70-dim element fingerprint & 142-dim peak fingerprint
│   ├── models.py               # 5-fold LightGBM head wrappers (root, event, cat, dur)
│   ├── train.py                # Segment-0 anchor training with event-grouped CV
│   ├── infer.py                # Unified multi-segment inference & legal hierarchy decoding
│   ├── verify.py               # Hard quality gates, anti-vacuity sentinel check & curve tools
│   ├── diag.py                 # Offline detection ceiling & feature-shift diagnostics
│   ├── fuse.py / fuse_all.py   # Multi-mechanism detector union & break-even pricing
│   ├── flood.py                # Protocol-audit grid flooding probe (diagnostic only)
│   └── assemble.py             # Segment-safe answer assembly & schema validation
├── official_eval/              # Clean-room Dice+Hungarian evaluator, TENP & schemas
├── models_unified/             # Trained 5-fold LightGBM model artifacts & training metadata
├── experiments/                # Self-contained prototype verification suite (no private data needed)
│   ├── run_all_verifications.py
│   ├── exp1_evidence_stability.py
│   ├── exp2_protocol_mitigation.py
│   └── exp3_exp4_gates_and_scale.py
├── results/                    # Verified JSON outputs from prototype verification runs
├── tests/                      # Automated unit & integration tests
└── docs/
    ├── NEGATIVE_RESULTS.md     # Structured ledger of falsified hypotheses & negative results
    ├── DATA_AVAILABILITY.md    # Formal Data Availability & Artifact Evaluation statement
    └── archive_zh/             # Original experimental logs preserved for provenance
```

## Quickstart (Out-of-the-Box Prototype Verification)

No restricted competition data is required to reproduce our prototype verification experiments and unit tests.

```bash
# 1. Install core dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Run the automated unit & anti-vacuity gate test suite
python3 -m unittest discover -s tests -v

# 3. Reproduce all four prototype verification experiments
python3 experiments/run_all_verifications.py
```

The runner executes:
- **Exp 1 (`exp1_evidence_stability.py`)**: Verifies the four-way scale floor on quasi-constant/zero-inflated channels and reproduces excursion-contraction transfer failure under monotone quantile alignment.
- **Exp 2 (`exp2_protocol_mitigation.py`)**: Verifies Proposition 1's $O(1/E^2)$ break-even hit rate, reproduces the grid-flooding exploit under $\alpha_0=0.7$, validates our proposed **TENP** mitigation, and compares Dice $\ge 0.4$ against Point-Adjusted F1.
- **Exp 3 (`exp3_exp4_gates_and_scale.py`)**: Verifies **Sentinel-Subject Isolation** (`src.verify.e1b_is_vacuous`) against vacuous frozen-shard passes.
- **Exp 4 (`exp3_exp4_gates_and_scale.py`)**: Benchmarks L1 evidence construction + NMS scaling from **80 to 800 network elements** (14-day streams), confirming sub-millisecond per-minute-tick latency.

## Full Pipeline Reproduction (With D1 Backbone Dataset)

When the CCF International AIOps Challenge 2026 backbone dataset (D1) is placed under `data/stage{1,2,3}/` and `gt/96GT.jsonl`, the full pipeline runs via:

```bash
python -m src.build_cache
python -m src.build_evidence
python -m src.train --out models_unified
python -m src.infer --models models_unified --out output/result_unified.jsonl
python -m src.verify output/result_unified.jsonl --e1b-file output/result_unified.jsonl
```

## License

Released under the [MIT License](LICENSE).
