# Data Availability and Artifact Evaluation Statement

## 1. Source Code and Pre-Trained Model Artifacts
All source code for the PROVE pipeline (`src/`), the clean-room Dice+Hungarian evaluator and proposed Trustworthy Emission-Normalized Protocol (`official_eval/evaluator.py`), the trained 5-fold LightGBM model files and training metadata (`models_unified/`), the negative-result ledger (`docs/NEGATIVE_RESULTS.md`), and the self-contained prototype verification suite (`experiments/`) are released openly under the MIT License in this repository.

## 2. Dataset Availability
- **D1 (Carrier Backbone Fault-Injection Dataset)**: Released by the CCF International AIOps Challenge 2026 Organizing Committee (`https://competition.aiops.cn/`). Due to dataset size (~90 GB, $2.8\times 10^8$ NetFlow rows) and organizer redistribution terms, raw telemetry CSVs are not redistributed inside this repository; researchers can obtain D1 directly from the official challenge portal and place the archives under `data/stage{1,2,3}/` to execute the full L0–L4 pipeline.
- **D2 (RCAEval RE2 Microservice Benchmark)**: Publicly available from Pham et al. (2024), *RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data* (`https://github.com/phamquiluan/RCAEval`).
- **Prototype Verification & Synthetic Replay Suite (`experiments/`)**: To guarantee full, zero-barrier reproducibility without access to the 90 GB D1 archive, `experiments/run_all_verifications.py` generates controlled multi-source synthetic telemetry and fault-interval streams matching D1's statistical structure, verifying: (i) four-way scale floor stability, (ii) excursion-contraction transfer failure, (iii) Proposition 1's $O(1/E^2)$ scaling and TENP protocol mitigation, (iv) anti-vacuity sentinel-subject isolation, and (v) linear scalability from 80 to 800 network elements.

## 3. Reproducibility Verification
All verification results reported in the paper's prototype-validation tables and figures are stored verbatim in `results/prototype_verification_results.json` and can be regenerated deterministically on any standard CPU machine in under 30 seconds via:

```bash
python3 -m unittest discover -s tests -v
python3 experiments/run_all_verifications.py
```
