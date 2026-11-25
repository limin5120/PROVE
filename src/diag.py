"""Measure the detection ceiling before tuning anything downstream.

The score is proportional to rho = TP / N_true, and rho is bounded above by a purely
pre-model quantity: the fraction of true fault intervals that contain at least one candidate
peak at all.  No amount of ranking quality, window placement or category accuracy can recover
a fault that never entered the candidate pool.

So this probe answers, on the only labelled segment, for a grid of evidence thresholds:

* **coverage** -- fraction of the 96GT intervals with >=1 peak inside them (the rho ceiling);
* **pool** -- how many peaks that threshold admits, i.e. how much the ranker has to sift;
* **pool precision** -- fraction of pooled peaks that land on a true interval, which bounds how
  good alpha_fp can possibly get at a given emission count;
* **tail census** -- how many cells sit above large multiples of the level, to check whether a
  handful of pathological channels could monopolise the top of the ranking.

Reading it: pick the lowest threshold whose coverage has saturated, not the one with the
prettiest envelope.  A threshold that is too high caps rho silently -- the local score looks
fine on the labelled batch only because the batch was used to choose it, and the unlabelled
batches inherit the cap.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C, detect, evidence as E  # noqa: E402

THRS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
PAD = 2


def feature_shift(model_dir: str, score_thr: float, top: int = 12) -> None:
    """Per-segment distribution of the selection head's inputs, and of its output.

    This is the transfer diagnostic.  The anchor is the only labelled segment, so the models are
    fitted there; if a feature that the selection head leans on is distributed differently on the
    unlabelled segments, the head is scoring them out of distribution and its ranking -- hence
    rho, hence the whole score -- degrades with no local symptom at all.  That is what happened
    on the first submission: anchor 63.43, batch 3 solved back to 21.04, a transfer ratio of
    0.33, while every envelope statistic the alignment targeted looked fine.

    Prints, per segment: the head's output quantiles, and for each of its most important inputs
    the p50/p90 ratio against the anchor.  A ratio far from 1 on a high-gain feature is the
    mechanism; a ratio near 1 on all of them means the collapse is somewhere else.
    """
    from src import features as F, models as M

    evs = E.build_all(verbose=False)
    thr = E.calibrate(evs)
    E.align_segments_to_anchor(evs, verbose=False)
    S, _ = E.element_scores(evs, thr)
    rint = E.region_intensity(S)
    peaks = detect.find_peaks(S, rint, score_thr=score_thr, verbose=False)
    del S
    fb = F.FeatureBuilder(evs, thr, verbose=False)
    feat = fb.build(peaks)
    with open(os.path.join(model_dir, "train_meta.json"), encoding="utf-8") as f:
        cols = json.load(f)["cols"]
    pk = F.peak_frame(feat, cols)
    ev = M.Head.load(model_dir, "event")
    p = ev.predict(pk)
    p = p[:, 0] if p.ndim > 1 else p
    pk = pk.assign(p_event=p)
    imp = [c for c, _ in ev.importance(top)]

    print(f"\n[feature-shift] {len(pk)} peaks, selection head '{ev.name}', "
          f"top {len(imp)} features by gain")
    print(f"  {'segment':8s} {'peaks':>7s} {'p_event p50':>12s} {'p90':>8s} "
          f"{'>=0.5':>7s} " + " ".join(f"{c.replace('__max', '/X').replace('__mean', '/M'):>15s}"
                                       for c in imp[:6]))
    base = {}
    for si, (lo, hi, name) in enumerate(C.SEGMENT_BOUNDS):
        s = pk[pk["seg"] == si]
        if not len(s):
            continue
        cells = []
        for c in imp[:6]:
            v = s[c].to_numpy(dtype=float) if c in s else np.zeros(len(s))
            p50, p90 = float(np.median(v)), float(np.quantile(v, 0.90))
            if si == 0:
                base[c] = (p50, p90)
                cells.append(f"{p50:7.3g}/{p90:<7.3g}")
            else:
                b = base.get(c, (0.0, 0.0))
                r50 = p50 / b[0] if b[0] else float("nan")
                r90 = p90 / b[1] if b[1] else float("nan")
                cells.append(f"x{r50:.2f}/x{r90:.2f}   ")
        days = (hi - lo) / 1440.0
        print(f"  {name:8s} {len(s):7d} {float(np.median(s['p_event'])):12.4f} "
              f"{float(np.quantile(s['p_event'], 0.90)):8.4f} "
              f"{int((s['p_event'] >= 0.5).sum()) / days:6.1f}/d " + " ".join(f"{c:>15s}"
                                                                               for c in cells))
    print("  (seg0 shows absolute p50/p90; other segments show the ratio to seg0 -- "
          "x1.00 means in distribution)")
    print(f"  features shown: {imp[:6]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nms", type=int, default=C.NMS_RADIUS)
    ap.add_argument("--no-align", action="store_true",
                    help="skip cross-segment alignment (shows what alignment is worth)")
    ap.add_argument("--features", action="store_true",
                    help="run the cross-segment feature-shift diagnostic instead of the "
                         "detection-ceiling sweep")
    ap.add_argument("--model-dir", default=C.MODEL_ROOT)
    ap.add_argument("--score-thr", type=float, default=C.SCORE_THR)
    args = ap.parse_args()

    if args.features:
        feature_shift(args.model_dir, args.score_thr)
        return 0

    evs = E.build_all(verbose=False)
    thr = E.calibrate(evs)
    if not args.no_align:
        E.align_segments_to_anchor(evs, verbose=False)
    S, elems = E.element_scores(evs, thr)
    rint = E.region_intensity(S)
    gt = C.load_gt()
    seg0_hi = C.SEGMENT_BOUNDS[0][1]
    gt0 = gt[gt["m0"] < seg0_hi]
    print(f"[ceiling] {len(gt)} labelled intervals, {len(gt0)} inside Segment 0; "
          f"align={not args.no_align}")

    # tail census: is the dynamic range carried by many cells or a few pathological ones?
    print("\n[tail census] element-score cells above multiples of the level, per segment")
    for mult in (1, 10, 100, 1000):
        counts = [int((S[lo:hi] >= mult).sum()) for lo, hi, _ in C.SEGMENT_BOUNDS]
        print(f"    >= {mult:5d}: " + "  ".join(f"{n}={c:,}" for (lo, hi, n), c
                                               in zip(C.SEGMENT_BOUNDS, counts)))

    print(f"\n[ceiling] Segment 0 coverage vs threshold (nms={args.nms}, pad={PAD} min)")
    print(f"    {'thr':>6s} {'pool':>7s} {'coverage':>9s} {'pool_prec':>10s} "
          f"{'peaks/region':>13s} {'missed':>7s}")
    best = None
    for t in THRS:
        pk = detect.find_peaks(S, rint, score_thr=t, nms_radius=args.nms, verbose=False)
        p0 = pk[pk["midx"] < seg0_hi]
        if not len(p0):
            print(f"    {t:6.2f} {0:7d} {'-':>9s}")
            continue
        hit_iv, hit_pk = set(), 0
        for region, g in gt0.groupby("region"):
            pr = p0[p0["region"] == region]
            if not len(pr):
                continue
            tm = pr["midx"].to_numpy()
            for i, (a, b) in enumerate(zip(g["m0"], g["m1"])):
                if ((tm >= a - PAD) & (tm <= b + PAD)).any():
                    hit_iv.add((region, i))
            lo = g["m0"].to_numpy() - PAD
            hi = g["m1"].to_numpy() + PAD
            on = np.zeros(len(tm), dtype=bool)
            for a, b in zip(lo, hi):
                on |= (tm >= a) & (tm <= b)
            hit_pk += int(on.sum())
        cov = len(hit_iv) / max(len(gt0), 1)
        prec = hit_pk / len(p0)
        print(f"    {t:6.2f} {len(p0):7d} {cov:9.4f} {prec:10.4f} "
              f"{len(p0) / len(C.REGIONS):13.1f} {len(gt0) - len(hit_iv):7d}")
        if best is None or cov > best[1] + 1e-6:
            best = (t, cov)
    if best:
        print(f"\n[ceiling] highest coverage {best[1]:.4f} at thr={best[0]}; "
              f"rho cannot exceed this whatever the models do")

    # Does the implicated element (argmax over roles) already point at the true root cause?
    # This bounds the RCA term independently of the learned head.
    pk = detect.find_peaks(S, rint, score_thr=1.0, nms_radius=args.nms, verbose=False)
    p0 = pk[pk["midx"] < seg0_hi]
    if len(p0):
        n = ok = ok5 = 0
        n_role = len(C.ROLES)
        for region, g in gt0.groupby("region"):
            ri = C.REGIONS.index(region)
            pr = p0[p0["region"] == region]
            for _, gt_row in g.iterrows():
                sel = pr[(pr["midx"] >= gt_row["m0"] - PAD) & (pr["midx"] <= gt_row["m1"] + PAD)]
                if not len(sel):
                    continue
                r = sel.iloc[0]
                n += 1
                block = S[r["midx"], ri * n_role:(ri + 1) * n_role]
                order = np.argsort(-block)
                ok += int(C.ROLES[order[0]] == gt_row["role"])
                ok5 += int(gt_row["role"] in [C.ROLES[i] for i in order[:5]])
        print(f"\n[evidence-only RCA] on {n} labelled intervals with a nearby peak: "
              f"loudest-element is the true root cause {ok / max(n, 1):.4f}, "
              f"inside top-5 {ok5 / max(n, 1):.4f}")
        print("    (this is the RCA ceiling before any learned re-ranking; the official term "
              "pays 1.0/0.8/0.6/0.4/0.2 by rank)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
