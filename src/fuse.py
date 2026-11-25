"""Fuse the two independent batch-3 detectors into one answer file.

Why fusion is the only lever left
---------------------------------
Batch 3 has no labels, so every property of a candidate answer can only be priced with the
scoring equation.  With ``S = (T/N)(40*a*alpha_fp + B)`` and ``alpha_fp = 0.7 + 0.3*T/E``:

    dS3/dT = +0.2916        dS3/dE = -0.003876        break-even marginal hit rate = 1.33%

(``a = 0.8864``, ``B = 56.87`` measured on the labelled stand-in; ``N3 = 292``, validated by
predicting the known server point ``(T=139, E=426) -> 40.54`` against the measured 40.9948.)

The two pipelines now on disk are architecturally unrelated -- the old per-batch system
(``models_selrawo``, 426 records, T=139, precision 0.326) and this rebuild (unified axis,
quantile alignment, 1247 records, T=84, precision 0.067).  The rebuild's precision is *five
times* the break-even rate, so its records are worth adding even though the rebuild on its own
scores far worse (S3 22.63 vs 40.99).  The only thing that can make fusion lose is overlap:
under one-to-one Hungarian matching a truth already claimed by the old system cannot be claimed
again, so a redundant record adds E without adding T.

Worst case (rebuild's 84 hits all already in the old 139) costs 0.47 total; break-even needs
just 12 novel hits out of 84 (14%).  The leaderboard is max-semantics, so the downside is
bounded by "does not beat the banked best" and the upside is several points.

Variants
--------
``full``    every rebuild seg2 record, appended after the old answer.
``dice``    drop rebuild records with Dice >= 0.4 against any old record -- the official
            matching threshold, so a dropped record could only have competed for a truth the
            old answer already contests.
``gap20``   drop rebuild records starting within 20 min of any old record, using the FAQ prior
            that distinct faults are usually >= 20 min apart.  Strictest, best worst case.

    python -m src.fuse --old <path> --rebuild output/result_curve_dur.jsonl --variant dice ^
        --out output/fused_batch3.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C  # noqa: E402

N3 = 292
A_REC = 0.8864          # mean per-record AD score, measured on the labelled stand-in
B_REC = 56.8716         # mean per-record RCA+Major+Minor, ditto
T_OLD = 139             # reverse-solved from the server's S3=40.9948 at E=426
T_REBUILD = 84          # reverse-solved from the server's S3=22.63 at E=1247
FROZEN = 44.20942       # 0.2*S1 + 0.4*S2 with S1=90.0911, S2=65.478
BEST_TOTAL = 61.3052


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def ts(rec: dict, key: str) -> pd.Timestamp:
    t = pd.Timestamp(rec[key])
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def seg2_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    spec = C.BATCHES["seg2"]
    tol = pd.Timedelta(minutes=5.0)
    return (pd.Timestamp(spec["grid_start"], tz="UTC") - tol,
            pd.Timestamp(spec["grid_end"], tz="UTC") + tol)


def dice(a0, a1, b0, b1) -> float:
    ov = (min(a1, b1) - max(a0, b0)).total_seconds()
    if ov <= 0:
        return 0.0
    la = (a1 - a0).total_seconds()
    lb = (b1 - b0).total_seconds()
    return 2.0 * ov / (la + lb) if (la + lb) > 0 else 0.0


def predict(T: float, E: int) -> tuple[float, float]:
    """(S3, total) for a batch-3 answer emitting E records with T true positives."""
    if T <= 0 or E <= 0:
        return 0.0, FROZEN
    s3 = (T / N3) * (40.0 * A_REC * (0.7 + 0.3 * T / E) + B_REC)
    return s3, FROZEN + 0.4 * s3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True, help="old per-batch system's batch-3 answer")
    ap.add_argument("--rebuild", required=True,
                    help="unified-axis answer file (all segments; seg2 is filtered out here)")
    ap.add_argument("--variant", default="dice", choices=("full", "dice", "gap20"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--survey", action="store_true",
                    help="also list the seg2 record count of every output/result_*.jsonl")
    args = ap.parse_args()

    lo, hi = seg2_window()
    print(f"seg2 window (UTC, +/-5 min): {lo} .. {hi}")

    if args.survey:
        print("\n[survey] seg2 records in each rebuild candidate:")
        d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        for fn in sorted(os.listdir(d)):
            if not (fn.startswith("result") and fn.endswith(".jsonl")):
                continue
            p = os.path.join(d, fn)
            try:
                recs = read_jsonl(p)
            except Exception as e:                      # a truncated/partial run is not fatal
                print(f"  {fn:34s} unreadable: {e}")
                continue
            n2 = sum(1 for r in recs if lo <= ts(r, "start_time") < hi)
            note = "  <== matches the measured E=1247 (S3=22.63)" if n2 == 1247 else ""
            print(f"  {fn:34s} total {len(recs):5d}   seg2 {n2:5d}{note}")
        print()

    old = [r for r in read_jsonl(args.old) if lo <= ts(r, "start_time") < hi]
    rb_all = read_jsonl(args.rebuild)
    rb = [r for r in rb_all if lo <= ts(r, "start_time") < hi]
    print(f"old answer      : {len(old)} records in seg2 (file has {len(read_jsonl(args.old))})")
    print(f"rebuild answer  : {len(rb)} records in seg2 (file has {len(rb_all)})")
    if not old or not rb:
        raise SystemExit("one side contributed no seg2 records -- check the paths")

    o0 = [ts(r, "start_time") for r in old]
    o1 = [ts(r, "end_time") for r in old]

    kept, drop_dice, drop_gap = [], 0, 0
    for r in rb:
        s, e = ts(r, "start_time"), ts(r, "end_time")
        dmax = max((dice(s, e, a, b) for a, b in zip(o0, o1)), default=0.0)
        gmin = min((abs((s - a).total_seconds()) / 60.0 for a in o0), default=1e9)
        if dmax >= 0.4:
            drop_dice += 1
        if gmin < 20.0:
            drop_gap += 1
        redundant = {"full": False,
                     "dice": dmax >= 0.4,
                     "gap20": gmin < 20.0}[args.variant]
        if not redundant:
            kept.append(r)

    print(f"\nredundant with the old answer: Dice>=0.4 -> {drop_dice} "
          f"({drop_dice / len(rb):.1%}),  start gap <20min -> {drop_gap} ({drop_gap / len(rb):.1%})")
    print(f"variant {args.variant!r}: keeping {len(kept)} of {len(rb)} rebuild records")

    E = len(old) + len(kept)
    print(f"\nfused emission E = {len(old)} + {len(kept)} = {E}")
    print(f"{'novel hits':>11s} {'T_union':>8s} {'alpha_fp':>9s} {'S3':>8s} {'total':>9s} "
          f"{'vs 61.3052':>11s}")
    print("-" * 62)
    for frac in (0.0, 0.14, 0.25, 0.50, 0.75, 1.0):
        T = T_OLD + frac * T_REBUILD
        s3, tot = predict(T, E)
        afp = 0.7 + 0.3 * T / E
        flag = "  <== beats the banked best" if tot > BEST_TOTAL else ""
        print(f"{frac * T_REBUILD:11.1f} {T:8.1f} {afp:9.4f} {s3:8.3f} {tot:9.3f} "
              f"{tot - BEST_TOTAL:+11.3f}{flag}")
    print("-" * 62)
    s3_0, tot_0 = predict(T_OLD, 426)
    print(f"reference (old answer alone, E=426, T=139): S3={s3_0:.3f} total={tot_0:.3f} "
          f"(server measured 40.9948 / 60.6073; model bias {40.9948 - s3_0:+.3f})")
    need = None
    for t in range(T_OLD, T_OLD + T_REBUILD + 1):
        if predict(t, E)[1] > BEST_TOTAL:
            need = t
            break
    print(f"break-even: T_union >= {need} "
          f"(i.e. >= {need - T_OLD} novel hits out of the rebuild's {T_REBUILD}, "
          f"{(need - T_OLD) / T_REBUILD:.0%})" if need else
          f"break-even: unreachable at this E even with all {T_REBUILD} hits novel")

    with open(args.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(old + kept, 1):
            rec = {k: r[k] for k in ("prediction_id", "start_time", "end_time",
                                     "root_cause_top5", "fault_category") if k in r}
            rec["prediction_id"] = f"fuse-{i:05d}"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n[fuse] wrote {len(old) + len(kept)} records -> {args.out}")
    print("       (old records first, then the kept rebuild records; prediction_id renumbered)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
