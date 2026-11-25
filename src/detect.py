"""L1 detection: segment-balanced peak finding on the aligned evidence score.

Two rules here are load-bearing, both learned from measured failures:

* **Budget per (segment, region), not globally.** A single global top-K over 50,400 minutes
  lets whichever batch happens to have the loudest evidence monopolise the candidate pool.
  Batch 2 ended up emitting 39.9 records/day against a physical density of ~21/day while
  batch 3 under-emitted, purely because their score scales differed.  Scaling the budget by
  segment length keeps candidate density equal across batches.
* **NMS across elements within a region.** The organiser FAQ states one fault at a time with
  injection gaps of 20 minutes or more, so two elements peaking in the same minute are one
  event, not two.  Collapsing them here (keeping the louder element as the implicated one)
  is what makes the emitted count interpretable as a fault count.

The implicated element is only a *starting point*: ``features.py`` scores all ten elements of
the region for every peak, and the element head produces the final Top-5.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C  # noqa: E402


def local_maxima_nms(v: np.ndarray, radius: int, thr: float) -> np.ndarray:
    """Greedy NMS: take the loudest maximum, veto its +-radius neighbourhood, repeat."""
    idx = np.flatnonzero(v >= thr)
    if not len(idx):
        return idx
    order = idx[np.argsort(-v[idx], kind="stable")]
    taken = np.zeros(len(v), dtype=bool)
    keep = []
    for i in order:
        a = i - radius if i - radius > 0 else 0
        b = i + radius + 1 if i + radius + 1 < len(v) else len(v)
        if taken[a:b].any():
            continue
        taken[a:b] = True
        keep.append(int(i))
    keep.sort()
    return np.asarray(keep, dtype=np.int64)


def find_peaks(S: np.ndarray, rint: dict[str, np.ndarray], score_thr: float = C.SCORE_THR,
               nms_radius: int = C.NMS_RADIUS, verbose: bool = True) -> pd.DataFrame:
    """Candidate peaks, one row per (segment, region, minute).

    Columns: ``midx, seg, region, rint, role, elem, score`` where ``role``/``elem`` are the
    loudest element at that minute (the implicated element) and ``score`` its evidence score.
    """
    n_role = len(C.ROLES)
    rows = []
    for si, (lo, hi, sname) in enumerate(C.SEGMENT_BOUNDS):
        budget = max(200, round(C.MAX_PEAKS_PER_SEG_REGION * (hi - lo) / 20160))
        n_seg_tot = 0
        for ri, region in enumerate(C.REGIONS):
            r = rint[region][lo:hi]
            pk = local_maxima_nms(r, nms_radius, score_thr)
            if len(pk) > budget:
                pk = pk[np.argsort(-r[pk], kind="stable")[:budget]]
                pk.sort()
            if not len(pk):
                continue
            t = pk + lo
            sl = slice(ri * n_role, (ri + 1) * n_role)
            block = S[np.ix_(t, np.arange(sl.start, sl.stop))]
            j = block.argmax(axis=1)
            rows.append(pd.DataFrame({
                "midx": t, "seg": si, "region": region,
                "rint": r[pk].astype(np.float32),
                "role": np.asarray(C.ROLES)[j],
                "elem": np.asarray(C.ELEMENT_IDS)[ri * n_role + j],
                "score": block.max(axis=1).astype(np.float32),
            }))
            n_seg_tot += len(pk)
        if verbose:
            print(f"  [peaks] {sname}: {n_seg_tot} candidates over {len(C.REGIONS)} regions "
                  f"(budget {budget}/region, thr={score_thr}, nms={nms_radius})", flush=True)
    if not rows:
        return pd.DataFrame(columns=["midx", "seg", "region", "rint", "role", "elem", "score"])
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["seg", "region", "midx"]).reset_index(drop=True)


def flat_top_width(v: np.ndarray, t: int, frac: float = C.CROSS_LEVEL,
                   maxlen: int = C.MAX_LEN) -> tuple[int, int]:
    """Widest window around ``t`` over which the envelope stays above ``frac`` of its peak.

    This is the physical window estimate: a real fault holds a flat top for its whole duration,
    whereas a spike falls off within a minute or two.  Losing this track (batch 3 under the old
    per-batch calibration, where the 9-minute denominator was inflated 12.86x) is what collapsed
    the median window to 7.8 min.

    ``maxlen`` bounds the *total* width, not the walk in each direction -- bounding each side
    separately admits windows up to 2*maxlen (measured: 69, 51, 48 minutes against a FAQ maximum
    of 30), and any window longer than the true fault both loses Dice and is an easy false
    positive.  The longer side is trimmed first so the peak stays inside the window.
    """
    seg_lo, seg_hi, _ = C.SEGMENT_BOUNDS[int(C.seg_of(t))]
    peak = float(v[t])
    if not np.isfinite(peak) or peak <= 0:
        return int(t), int(t) + C.MIN_LEN
    lvl = frac * peak
    a = int(t)
    while a - 1 >= max(seg_lo, t - maxlen) and v[a - 1] >= lvl:
        a -= 1
    b = int(t) + 1
    while b < min(seg_hi, t + maxlen + 1) and v[b] >= lvl:
        b += 1
    while b - a > maxlen:
        if (t - a) >= (b - t):
            a += 1
        else:
            b -= 1
    return a, max(b, a + C.MIN_LEN)


def hypotheses(t: int, v: np.ndarray, anchor: int) -> list[tuple[int, int, str]]:
    """Candidate windows for one peak: the crossing window plus a fixed-length ladder.

    ``anchor`` is the minute the window is centred on when a fixed length is used.  Every
    hypothesis is confined to the peak's own segment.
    """
    seg_lo, seg_hi, _ = C.SEGMENT_BOUNDS[int(C.seg_of(t))]
    out = []
    a, b = flat_top_width(v, t)
    out.append((a, b, "cross"))
    for L in C.FIXED_LENGTHS:
        t0 = max(seg_lo, min(anchor - L // 2, seg_hi - L))
        out.append((t0, min(t0 + L, seg_hi), f"fixed{L}"))
    seen, uniq = set(), []
    for t0, t1, nm in out:
        if t1 - t0 < C.MIN_LEN or (t0, t1) in seen:
            continue
        seen.add((t0, t1))
        uniq.append((t0, t1, nm))
    return uniq
