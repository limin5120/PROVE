"""L1 driver: build the aligned evidence layer for the whole 50,400-minute axis.

Order of operations is the science, not just plumbing:

1. ``build_all`` -- per region, reduce each source to one channel per node, then form the four
   window tracks (0/1/3/9 minutes).  Statistics are computed **inside each segment** so no
   sliding window or robust scale ever straddles a batch boundary.
2. ``calibrate`` -- nominal alarm levels pooled over **Segment 0 only**.  Segment 0 is the batch
   with a public answer sheet, so its levels are the reference the detector can be validated
   against.  ``_level`` is deliberately unmodified: capping its zero-inflation floor would crush
   the anchor's own legitimate ``0|route`` denominator (floor/q ~ 180 there).
3. ``align_segments_to_anchor`` -- monotonically map segments 1/2 onto Segment 0 per
   (window, kind).  Segment 0 is identity by construction, which is what keeps step 4's
   regression comparable to the anchor's own history.
4. Print the E1a gate readings: per-segment ``frac(score>=1)`` and score quantiles.  Before
   alignment batch 3 read frac 0.1451 / p999 16.3 against batch 1's 0.0904 / 187.9 -- a flat,
   over-ignited envelope.  Alignment should pull every segment into one comparable range.

Writes ``cache/all/<region>/evidence.npz`` and ``cache/all/calibration.json``.  Rerunning
overwrites both; nothing accumulates.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C, evidence as E  # noqa: E402

META_PATH = os.path.join(C.cache_dir(), "calibration.json")


def save_evidence(evs: dict[str, E.RegionEvidence]) -> None:
    for region, ev in evs.items():
        d = C.region_cache_dir(region)
        os.makedirs(d, exist_ok=True)
        payload = {f"A_{w}": ev.A[w] for w in C.WINDOWS}
        payload.update({"channels": np.array(ev.channels), "kind_idx": ev.kind_idx,
                        "node_of": np.array(ev.node_of)})
        np.savez_compressed(os.path.join(d, "evidence.npz"), **payload)


def load_evidence(regions=None) -> dict[str, E.RegionEvidence]:
    out = {}
    for r in regions or C.REGIONS:
        p = os.path.join(C.region_cache_dir(r), "evidence.npz")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}; run build_evidence first")
        with np.load(p, allow_pickle=False) as z:
            A = {w: z[f"A_{w}"] for w in C.WINDOWS}
            out[r] = E.RegionEvidence(r, [str(c) for c in z["channels"]],
                                      z["kind_idx"], [str(c) for c in z["node_of"]], A)
    return out


def envelope_report(S: np.ndarray, label: str) -> dict:
    """E1a readings: how much of each segment sits above the alarm level, and how loud it gets."""
    out = {}
    print(f"  [{label}] element-score envelope by segment")
    for lo, hi, name in C.SEGMENT_BOUNDS:
        B = S[lo:hi]
        q = np.quantile(B, (0.99, 0.999))
        d = {"frac_ge_1": round(float((B >= 1.0).mean()), 5),
             "p99": round(float(q[0]), 4), "p999": round(float(q[1]), 4),
             "max": round(float(B.max()), 2)}
        out[name] = d
        print(f"      {name}: frac>=1={d['frac_ge_1']:.5f} p99={d['p99']:.3f} "
              f"p999={d['p999']:.3f} max={d['max']:.1f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="*", default=None, choices=C.REGIONS)
    ap.add_argument("--no-align", action="store_true",
                    help="calibrate on Segment 0 but skip the cross-segment quantile mapping "
                         "(diagnostic: shows what alignment actually changes)")
    args = ap.parse_args()
    os.makedirs(C.LOG_ROOT, exist_ok=True)
    log_path = os.path.join(C.LOG_ROOT, "build_evidence.log")
    t0 = time.time()

    with open(log_path, "w", encoding="utf-8", buffering=1) as lf:
        def emit(s: str = "") -> None:
            print(s, flush=True)
            lf.write(s + "\n")

        emit(f"[build_evidence] {time.strftime('%Y-%m-%d %H:%M:%S')} axis={C.N_MINUTES} "
             f"segments={[(n, hi - lo) for lo, hi, n in C.SEGMENT_BOUNDS]} align={not args.no_align}")
        evs = E.build_all(args.regions, verbose=False)
        for r, ev in evs.items():
            kinds = {E.KINDS[k]: int((ev.kind_idx == k).sum()) for k in np.unique(ev.kind_idx)}
            emit(f"  [evidence] {r:10s} channels={ev.n_channels} {kinds}")

        thr = E.calibrate(evs)
        emit(f"\n[calibrate] Segment 0 anchor levels ({len(thr)} (window, kind) pairs):")
        for (w, k), v in sorted(thr.items()):
            emit(f"    {w}|{E.KINDS[k]:8s} = {v:.6g}")

        S_pre, elems = E.element_scores(evs, thr)
        emit("\n[E1a] BEFORE cross-segment alignment:")
        pre = envelope_report(S_pre, "pre")
        del S_pre

        report = {} if args.no_align else E.align_segments_to_anchor(evs, verbose=False)
        if report:
            emit(f"\n[align] {len(report)} (window, kind, segment) mappings applied")
            for key in sorted(report):
                if key.startswith(("0|", "9|")):
                    d = report[key]
                    emit(f"    {key:22s} q99 {d['q99_before']:.4g}->{d['q99_after']:.4g}  "
                         f"q999 {d['q999_before']:.4g}->{d['q999_after']:.4g}  "
                         f"tail={d['tail_slope']:.2f}")

        S, _ = E.element_scores(evs, thr)
        emit("\n[E1a] AFTER cross-segment alignment:")
        post = envelope_report(S, "post")
        del S

        save_evidence(evs)
        os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump({"thr": {f"{w}|{k}": v for (w, k), v in thr.items()},
                       "align": report, "envelope_pre": pre, "envelope_post": post,
                       "windows": list(C.WINDOWS), "kinds": list(E.KINDS),
                       "n_minutes": C.N_MINUTES,
                       "segments": {n: [lo, hi] for lo, hi, n in C.SEGMENT_BOUNDS}},
                      f, indent=1)
        emit(f"\n[build_evidence] wrote {META_PATH} and per-region evidence.npz "
             f"({time.time() - t0:.1f}s)")

        # E1a verdict, stated as numbers rather than an impression.
        ref = post.get("seg0", {})
        bad = []
        for name, d in post.items():
            if not (0.075 <= d["frac_ge_1"] <= 0.105):
                bad.append(f"{name} frac>=1={d['frac_ge_1']}")
            if ref.get("p999") and abs(d["p999"] / ref["p999"] - 1) > 0.15:
                bad.append(f"{name} p999={d['p999']} vs seg0 {ref['p999']}")
        emit("[E1a gate] " + ("PASS: all segments inside frac>=1 [0.075,0.105] and p999 "
                              "within +-15% of the anchor" if not bad else
                              "CHECK: " + "; ".join(bad)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
