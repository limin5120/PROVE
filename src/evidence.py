"""L1 evidence: per-segment normalisation, golden-anchor calibration, cross-segment alignment.

The failure this layer exists to prevent
---------------------------------------
Calibrating each batch on its own statistics makes the batches mutually incomparable, and the
damage is not a uniform rescale -- it is *per window*.  `_level` returns
``max(quantile(v, q), quantile(v[v>0], 0.90))``; the second term guards zero-inflated channels
but is extremely sensitive to where the 90th percentile of the non-zero mass happens to fall.
On batch 3 the routing channels' non-zero fraction moved from 0.0672 to 0.0580, which pushed
that conditional quantile out of the dust and into the fault tail, inflating the `9|route`
denominator 12.86x while `0|route` simultaneously fell to 0.42x.

Because the score is ``max_w A_w / thr_w``, the consequence is a handover between tracks: the
smooth 9-minute track (whose 12-15 minute flat tops are what make a fault window measurable)
got suppressed 12.86x and the 1-minute spike track got amplified 2.37x.  Batch 3 therefore
ignited everywhere (frac>=1 rose to 0.145, 26k peaks), its windows collapsed to a 7.8 minute
median, and roughly half the true faults fell below the Dice 0.4 gate.

The fix, per the design document:

1. ``calibrate`` pools **Segment 0 only**.  Segment 0 is the batch with a public answer sheet,
   so its levels are the reference the detector was validated against.  `_level` itself is
   deliberately left untouched -- capping the floor would crush batch 1's own legitimate
   `0|route`/`1|route` denominators (floor/q ~ 180 there) and destroy the anchor.
2. ``align_segments_to_anchor`` monotonically maps each other segment's channel statistics onto
   Segment 0's distribution over a fixed quantile grid, per (window, kind).  Segment 0 is
   identity by construction.  This is an unsupervised operation: it uses feature distributions
   only, never labels, so it cannot leak the answer sheet into batches 2/3.

Everything here is segment-aware: sliding windows never straddle a batch boundary.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C  # noqa: E402

KINDS = C.CHANNEL_KINDS                       # ("node", "iface", "route", "scrape", "netflow")
KIND_IX = {k: i for i, k in enumerate(KINDS)}
WINDOWS = C.WINDOWS                           # (0, 1, 3, 9)

# Quantile grid for the cross-segment monotone map.  It has to span the whole envelope: the
# bulk sets the background level, the extreme tail sets how loud a fault can look.  Getting the
# tail wrong is exactly what produced the 11.5x p999 collapse.
CDF_ALIGN_Q = np.array([0.0, 0.80, 0.85, 0.90, 0.93, 0.95, 0.965, 0.975,
                        0.985, 0.990, 0.993, 0.995, 0.997, 0.9985, 0.9992, 0.9997],
                       dtype=np.float64)


class RegionEvidence:
    """Per-region channel statistics at every window, plus the channel->element mapping."""

    __slots__ = ("region", "channels", "kind_idx", "node_of", "A")

    def __init__(self, region: str, channels: list[str], kind_idx: np.ndarray,
                 node_of: list[str], A: dict[int, np.ndarray]):
        self.region = region
        self.channels = channels          # "<kind>|<node>"
        self.kind_idx = kind_idx          # per channel, index into KINDS
        self.node_of = node_of            # per channel, the raw node label
        self.A = A                        # window -> (N_MINUTES, n_ch) float32

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def cols_of_kind(self, k: int) -> np.ndarray:
        return np.flatnonzero(self.kind_idx == k)


# ---------------------------------------------------------------- per-segment statistics
def robust_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median / MAD-scale / alive mask, ignoring NaN."""
    with np.errstate(invalid="ignore", divide="ignore"):
        center = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - center), axis=0)
    scale = 1.4826 * mad
    alive = np.isfinite(center) & (scale > 1e-12)
    # A channel that is finite but perfectly constant carries no information; treat it as dead
    # so it cannot dominate a max-over-metrics statistic with a huge Z from a 1e-9 scale.
    scale = np.where(alive, scale, 1.0)
    center = np.where(np.isfinite(center), center, 0.0)
    return center, scale, alive


def envelope_excess(X: np.ndarray) -> np.ndarray:
    """How far each column sits above its own within-segment background envelope.

    Zero-inflated by construction: a minute inside the column's 99th-percent envelope reads
    exactly 0, so roughly 99% of the matrix is 0 and only genuine departures are positive.

    That property is not cosmetic -- it is what makes ``_level`` mean what it is supposed to
    mean.  ``_level`` returns ``max(quantile(v, q), quantile(v[v>0], 0.90))``; the second term
    exists to give a zero-inflated channel a sane denominator.  Feed it a *dense* statistic
    instead (e.g. ``max |robust Z|``, which is essentially never exactly 0) and the floor never
    fires while ``quantile(v, 0.99)`` is set by the faults themselves.  The level then describes
    the loudest 1% rather than the background, a fault normalises to ~1.0 instead of standing
    out, and the detector loses its dynamic range.  Measured on batch 1 with the dense
    statistic: ``frac(score>=1) = 0.0082`` and ``max = 5.7``, against 0.0904 and 1231 for a
    properly zero-inflated one.

    The envelope is per column and per segment, so it also absorbs the zero-inflation of the
    interface metrics (many are exactly 0 for long stretches, which made a MAD-based Z blow up
    and pinned the whole ``iface`` kind against the clip at 40, silently killing the ``link``
    fault category whose primary evidence is interface counters).
    """
    out = np.zeros(X.shape, dtype=np.float32)
    for lo, hi, _ in C.SEGMENT_BOUNDS:
        Xs = X[lo:hi]
        if not Xs.shape[0]:
            continue
        center, scale, alive = robust_stats(Xs)
        with np.errstate(invalid="ignore", divide="ignore"):
            Z = np.abs(Xs - center) / scale
        Z[~np.isfinite(Z)] = 0.0
        Z[:, ~alive] = 0.0
        env = np.quantile(Z, 0.99, axis=0)
        env = np.where(np.isfinite(env) & (env > 1e-9), env, 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ex = Z / env - 1.0
        ex[~np.isfinite(ex)] = 0.0
        out[lo:hi] = np.maximum(ex, 0.0).astype(np.float32)
    return out


def movmean_segmented(X: np.ndarray, w: int) -> np.ndarray:
    """Centred moving average confined to each segment (NaN-aware, edges shrink to the segment)."""
    if w <= 1:
        return X.astype(np.float32, copy=True)
    out = np.zeros(X.shape, dtype=np.float32)
    for lo, hi, _ in C.SEGMENT_BOUNDS:
        Xs = X[lo:hi].astype(np.float64)
        n = Xs.shape[0]
        V = np.where(np.isfinite(Xs), Xs, 0.0)
        M = np.isfinite(Xs).astype(np.float64)
        cv = np.concatenate([np.zeros((1, Xs.shape[1])), np.cumsum(V, axis=0)])
        cm = np.concatenate([np.zeros((1, Xs.shape[1])), np.cumsum(M, axis=0)])
        half = w // 2
        i0 = np.arange(n)
        a = np.maximum(i0 - half, 0)
        b = np.minimum(i0 + half + 1, n)
        s = cv[b] - cv[a]
        c = cm[b] - cm[a]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[lo:hi] = np.where(c > 0, s / np.maximum(c, 1), 0.0).astype(np.float32)
    return out


# ---------------------------------------------------------------- raw -> channel statistics
def _kind_matrix(region: str, kind: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Reduce one source's wide cache to a (N_MINUTES x n_node) statistic matrix.

    The reduction is ``max over that node's metric columns of |robust Z|``: a fault shows up in
    at least one metric of the affected node, and taking the max keeps the channel count at one
    per node instead of one per metric (which would make the evidence layer's dimensionality
    depend on how many interfaces a batch happens to expose).
    """
    if kind == "netflow":
        return _netflow_matrix(region)
    p = os.path.join(C.region_cache_dir(region), f"{kind}.parquet")
    if not os.path.exists(p):
        return np.zeros((C.N_MINUTES, 0), np.float32), [], []
    df = pd.read_parquet(p)
    cols = list(df.columns)
    X = df.to_numpy(dtype=np.float32)
    del df
    Ex = envelope_excess(X.astype(np.float64))
    del X
    # Column key is "<node>" or "<node>.<iface>"; group by the node part.
    nodes, node_of = [], {}
    for j, c in enumerate(cols):
        nd = c.split("|")[0].split(".")[0]
        node_of.setdefault(nd, []).append(j)
    order = sorted(node_of)
    out = np.zeros((C.N_MINUTES, len(order)), dtype=np.float32)
    for i, nd in enumerate(order):
        js = node_of[nd]
        out[:, i] = Ex[:, js].max(axis=1) if len(js) > 1 else Ex[:, js[0]]
    del Ex
    return out, [f"{kind}|{n}" for n in order], order


def _netflow_matrix(region: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Per-node traffic anomaly: |robust Z| of log-scale bytes, aggregated over peers/ports."""
    p = os.path.join(C.region_cache_dir(region), "netflow.parquet")
    if not os.path.exists(p):
        return np.zeros((C.N_MINUTES, 0), np.float32), [], []
    d = pd.read_parquet(p)
    if not len(d):
        return np.zeros((C.N_MINUTES, 0), np.float32), [], []
    g = d.groupby(["midx", "node"], sort=False)[["bytes", "records"]].sum()
    del d
    nodes = sorted(g.index.get_level_values(1).unique())
    out = np.zeros((C.N_MINUTES, len(nodes)), dtype=np.float32)
    for i, nd in enumerate(nodes):
        s = g.xs(nd, level=1).reindex(range(C.N_MINUTES))
        v = np.log1p(s["bytes"].fillna(0.0).to_numpy(dtype=np.float64))
        out[:, i] = envelope_excess(v[:, None])[:, 0]
    del g
    return out, [f"netflow|{n}" for n in nodes], nodes


# ---------------------------------------------------------------- build / calibrate / align
def build_region(region: str) -> RegionEvidence:
    mats, chans, kinds, nodes = [], [], [], []
    for kind in KINDS:
        M, names, nlist = _kind_matrix(region, kind)
        if M.shape[1] == 0:
            continue
        mats.append(M)
        chans.extend(names)
        nodes.extend(nlist)
        kinds.extend([KIND_IX[kind]] * M.shape[1])
    A0 = np.concatenate(mats, axis=1) if mats else np.zeros((C.N_MINUTES, 0), np.float32)
    del mats
    A = {0: A0}
    for w in WINDOWS[1:]:
        A[w] = movmean_segmented(A0, w)
    return RegionEvidence(region, chans, np.array(kinds, dtype=np.int16), nodes, A)


def build_all(regions=None, verbose: bool = True) -> dict[str, RegionEvidence]:
    out = {}
    for r in regions or C.REGIONS:
        ev = build_region(r)
        out[r] = ev
        if verbose:
            print(f"  [evidence] {r:10s} channels={ev.n_channels} "
                  f"kinds={ {KINDS[k]: int((ev.kind_idx == k).sum()) for k in np.unique(ev.kind_idx)} }",
                  flush=True)
    return out


def _level(v: np.ndarray, q: float = 0.99) -> float:
    """Nominal alarm level.  Unchanged from the historical definition on purpose: the floor
    term is what keeps a zero-inflated channel from scoring infinitely loud, and capping it
    would crush the anchor batch's own legitimate `0|route` denominator (floor/q ~ 180)."""
    v = v[np.isfinite(v)]
    if not len(v):
        return 1.0
    hi = float(np.quantile(v, q))
    nz = v[v > 0]
    floor = float(np.quantile(nz, 0.90)) if len(nz) else 1.0
    return max(hi, floor, 1e-6)


def calibrate(evs: dict[str, RegionEvidence], q: float = 0.99) -> dict:
    """Levels pooled over **Segment 0 only** -> identical to calibrating the anchor batch alone."""
    lo, hi, _ = C.SEGMENT_BOUNDS[0]
    buckets: dict[tuple[int, int], list[np.ndarray]] = {}
    for ev in evs.values():
        for w in WINDOWS:
            Aw = ev.A[w][lo:hi]
            for k in range(len(KINDS)):
                cols = ev.cols_of_kind(k)
                if len(cols):
                    buckets.setdefault((w, k), []).append(Aw[:, cols].ravel())
    thr = {}
    for (w, k), arrs in sorted(buckets.items()):
        thr[(w, k)] = _level(np.concatenate(arrs), q)
    return thr


def align_segments_to_anchor(evs: dict[str, RegionEvidence], windows=WINDOWS,
                             verbose: bool = True) -> dict:
    """Monotonically map segments 1..n channel statistics onto Segment 0's distribution.

    Applied per (window, kind) as a multiplicative correction clipped to [0.1, 20] so a single
    pathological quantile cannot blow a channel up.  Segment 0 is never touched, which is what
    makes the anchor's calibration and the labelled-batch regression comparable.
    """
    report = {}
    if len(C.SEGMENT_BOUNDS) <= 1:
        return report
    ref_lo, ref_hi, ref_name = C.SEGMENT_BOUNDS[0]
    for w in sorted({0, *windows}):
        for k, kind in enumerate(KINDS):
            ref_chunks = []
            for ev in evs.values():
                cols = ev.cols_of_kind(k)
                if len(cols):
                    s = ev.A[w][ref_lo:ref_hi][:, cols].ravel()
                    ref_chunks.append(s[np.isfinite(s)])
            if not ref_chunks:
                continue
            ref_all = np.concatenate(ref_chunks)
            del ref_chunks
            if float((ref_all > 0).mean()) < 1e-4:
                continue
            q_ref = np.maximum.accumulate(np.quantile(ref_all, CDF_ALIGN_Q))
            del ref_all
            for si, (lo, hi, name) in enumerate(C.SEGMENT_BOUNDS[1:], start=1):
                chunks = []
                for ev in evs.values():
                    cols = ev.cols_of_kind(k)
                    if len(cols):
                        s = ev.A[w][lo:hi][:, cols].ravel()
                        chunks.append(s[np.isfinite(s)])
                if not chunks:
                    continue
                seg_all = np.concatenate(chunks)
                del chunks
                if float((seg_all > 0).mean()) < 1e-4:
                    del seg_all
                    continue
                q_seg = np.maximum.accumulate(np.quantile(seg_all, CDF_ALIGN_Q))
                del seg_all
                if q_seg[-1] <= 1e-7 or q_ref[-1] <= 1e-7:
                    continue
                # Beyond the last grid point the map is linear, with a slope blended from the
                # two loudest grid ratios so an extreme outlier cannot set the tail gradient.
                tail = float(np.clip(0.6 * (q_ref[-2] / max(q_seg[-2], 1e-9))
                                     + 0.4 * (q_ref[-4] / max(q_seg[-4], 1e-9)), 0.25, 15.0))
                keep = np.concatenate([[True], np.diff(q_seg) > 1e-12])
                xs, ys = q_seg[keep], q_ref[keep]
                for ev in evs.values():
                    cols = ev.cols_of_kind(k)
                    if not len(cols):
                        continue
                    old = ev.A[w][lo:hi][:, cols].astype(np.float64)
                    new = np.interp(old, xs, ys)
                    above = old > xs[-1]
                    if above.any():
                        new[above] = ys[-1] + (old[above] - xs[-1]) * tail
                    with np.errstate(divide="ignore", invalid="ignore"):
                        mult = np.where(old > 1e-9, new / np.maximum(old, 1e-12), 1.0)
                    ev.A[w][lo:hi][:, cols] *= np.clip(mult, 0.1, 20.0).astype(np.float32)
                key = f"{w}|{kind}|{name}"
                report[key] = {"q99_before": float(q_seg[9]), "q99_after": float(q_ref[9]),
                               "q999_before": float(q_seg[-2]), "q999_after": float(q_ref[-2]),
                               "tail_slope": tail}
                if verbose and w in (0, 9):
                    print(f"  [align] {name}->{ref_name} {w}|{kind}: "
                          f"q99 {q_seg[9]:.4g}->{q_ref[9]:.4g} "
                          f"q999 {q_seg[-2]:.4g}->{q_ref[-2]:.4g} tail={tail:.2f}", flush=True)
    return report


def align_by_level(evs: dict[str, RegionEvidence], thr: dict, windows=WINDOWS,
                   verbose: bool = True) -> dict:
    """Per-segment calibration, expressed on the anchor's level scale.

    Multiplies each segment's channel statistics by ``level_anchor / level_own`` per
    (window, kind).  Dividing by ``thr`` afterwards then yields ``A / level_own`` -- which is
    exactly what calibrating each batch on its own statistics does, i.e. the pre-rebuild
    behaviour, reproduced on the unified axis so the two can be compared with everything else
    held constant.

    This is the control for :func:`align_segments_to_anchor`.  The difference between them is
    how much of each batch's own character is preserved:

    * this one corrects a single scale per (window, kind), so a batch whose faults are weak
      relative to its own noise stays weak after normalisation -- the ranking *within* the batch
      is untouched;
    * the quantile map reshapes the whole distribution onto the anchor's, which also rescales
      that batch's noise up to the anchor's loudness.  If the batch's signal-to-noise is worse
      than the anchor's, fault and noise become equally loud and the ranking degrades.

    Measured: with the quantile map the anchor rose 63.43 -> 73.19 across two rounds of model
    work while batch 3 fell 42.74 (per-batch predecessor) -> 21.04 -> 18.05.  That divergence is
    the reason this control exists.
    """
    report = {}
    if len(C.SEGMENT_BOUNDS) <= 1:
        return report
    for si, (lo, hi, name) in enumerate(C.SEGMENT_BOUNDS):
        if si == 0:
            continue
        for w in sorted({0, *windows}):
            for k, kind in enumerate(KINDS):
                chunks = []
                for ev in evs.values():
                    cols = ev.cols_of_kind(k)
                    if len(cols):
                        s = ev.A[w][lo:hi][:, cols].ravel()
                        chunks.append(s[np.isfinite(s)])
                if not chunks:
                    continue
                v = np.concatenate(chunks)
                del chunks
                if float((v > 0).mean()) < 1e-4:
                    del v
                    continue
                own = _level(v)
                del v
                ref = float(thr.get((w, k), 1.0))
                mult = float(np.clip(ref / max(own, 1e-9), 0.1, 20.0))
                for ev in evs.values():
                    cols = ev.cols_of_kind(k)
                    if len(cols):
                        ev.A[w][lo:hi][:, cols] *= np.float32(mult)
                report[f"{w}|{kind}|{name}"] = {"own_level": own, "anchor_level": ref,
                                               "mult": mult}
                if verbose and w in (0, 9) and kind in ("node", "iface", "route"):
                    print(f"  [level-align] {name} {w}|{kind}: own={own:.5g} "
                          f"anchor={ref:.5g} mult={mult:.4f}", flush=True)
    return report


def normalized(ev: RegionEvidence, thr: dict, w: int) -> np.ndarray:
    """A_w / thr[(w, kind)], broadcast per channel kind."""
    out = np.zeros_like(ev.A[w], dtype=np.float32)
    for k in range(len(KINDS)):
        cols = ev.cols_of_kind(k)
        if len(cols):
            out[:, cols] = ev.A[w][:, cols] / thr.get((w, k), 1.0)
    return out


def element_scores(evs: dict[str, RegionEvidence], thr: dict) -> tuple[np.ndarray, list[str]]:
    """(N_MINUTES x n_elements) evidence score, ``max`` over window and over observing kind.

    Max rather than sum: a sum rewards nodes that happen to be observed by many sources
    (br/cr appear in node+iface+route+scrape+netflow, service-vm in only three), which biases
    attribution toward the routers for reasons that have nothing to do with the fault.
    """
    elems = C.ELEMENT_IDS
    eix = {e: i for i, e in enumerate(elems)}
    S = np.zeros((C.N_MINUTES, len(elems)), dtype=np.float32)
    for region, ev in evs.items():
        best = np.zeros((C.N_MINUTES, ev.n_channels), dtype=np.float32)
        for w in WINDOWS:
            np.maximum(best, normalized(ev, thr, w), out=best)
        for j, nd in enumerate(ev.node_of):
            eid = f"{region}-{nd}"
            i = eix.get(eid)
            if i is None:
                continue
            np.maximum(S[:, i], best[:, j], out=S[:, i])
    return S, elems


def region_intensity(S: np.ndarray) -> dict[str, np.ndarray]:
    """Per-region loudest-element envelope -- what the boundary refiner walks along."""
    out = {}
    n = len(C.ROLES)
    for i, r in enumerate(C.REGIONS):
        out[r] = S[:, i * n:(i + 1) * n].max(axis=1).astype(np.float32)
    return out


def save(evs: dict[str, RegionEvidence], thr: dict, report: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = {"channels": {r: ev.channels for r, ev in evs.items()},
            "kind_idx": {r: ev.kind_idx.tolist() for r, ev in evs.items()},
            "node_of": {r: ev.node_of for r, ev in evs.items()},
            "thr": {f"{w}|{k}": v for (w, k), v in thr.items()},
            "align": report}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f)


def load_meta(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    m["thr"] = {tuple(int(x) for x in k.split("|")): v for k, v in m["thr"].items()}
    return m
