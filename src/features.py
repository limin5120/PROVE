"""L3 fingerprint: one feature row per (peak, candidate element).

Three batch-invariance fixes from the design document live here, all of them responses to
measured batch-3 failures rather than speculation:

1. **Absolute port slots, never ranked ports.** Batch 1's busy ports are 1053/80/0/8080/9343/
   9100/179; batch 3's are 80/1053/8080/179/5201 -- 9100 and 9343 vanish and ephemeral ports
   get promoted.  Any feature keyed on "the k-th busiest port" therefore silently changes
   meaning between batches (this is what made 157-164 of the old model's port-named columns go
   dead on batch 3 and dropped its weakest head's coverage to 96.7%).  Slots here are keyed on
   fixed port numbers, so a slot means the same service in every batch.
2. **Flanking-baseline detrending.** Absolute levels drift between batches (slow-moving
   counters, different load).  Every level feature is paired with its own ±30 minute flanking
   baseline, and the *difference* is what gets fed to the models.  The flank is clamped to the
   peak's own segment so a batch boundary can never leak into the baseline.
3. **Segment-adaptive far-end floor.** The absolute ``min_base=20`` consensus floor killed
   batch 3's remote-victim observations outright (beida/service-vm-1 went from 144 consensus
   minutes to 21) because batch 3 simply carries less traffic (0.32-0.55x the row count).  The
   floor here scales with the segment's own median volume, bounded to [0.30, 1.0] of the
   anchor's, so a quieter batch is not punished for being quiet.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C, detect, evidence as E  # noqa: E402

# Semantic port slots, keyed on absolute port numbers (see module docstring).
PORT_SLOTS: dict[str, tuple[int, ...]] = {
    "dns": (53,), "dns_alt": (1053,), "web": (80, 8080), "ntp": (123,),
    "bgp": (179,), "iperf": (5201,), "svc": C.SVC_PRIMARY_PORTS, "zero": (0,),
}
FAR_MIN_BASE = 20.0
FAR_FLOOR_MIN = 6.0
FLANK = 30                 # minutes of baseline on each side
FLANK_GAP = 8              # minutes around the peak excluded from the baseline


class NetflowCube:
    """Dense per-region netflow tensors plus the fixed semantic port slots."""

    def __init__(self, region: str):
        self.region = region
        self.own = C.REGIONS.index(region)
        p = os.path.join(C.region_cache_dir(region), "netflow.parquet")
        self.nodes = list(detect_nf_nodes())
        self.n_node = len(self.nodes)
        T = C.N_MINUTES
        self.tot = np.zeros((T, self.n_node), np.float32)
        self.far = np.zeros((T, self.n_node), np.float32)
        self.slot = {k: np.zeros((T, self.n_node), np.float32) for k in PORT_SLOTS}
        self.other = np.zeros((T, self.n_node), np.float32)
        if not os.path.exists(p):
            return
        d = pd.read_parquet(p)
        if not len(d):
            return
        nix = {n: i for i, n in enumerate(self.nodes)}
        ni = d["node"].map(nix).to_numpy()
        m = d["midx"].to_numpy(dtype=np.int64)
        ok = ni.notna().to_numpy() if hasattr(ni, "notna") else ~pd.isna(ni)
        ni = np.nan_to_num(pd.to_numeric(pd.Series(ni), errors="coerce")).astype(np.int64)
        ok = ok & (m >= 0) & (m < T) & (ni >= 0) & (ni < self.n_node)
        d, m, ni = d[ok], m[ok], ni[ok]
        byt = d["bytes"].to_numpy(np.float64)
        rec = d["records"].to_numpy(np.float64)
        dr = d["dst_region"].to_numpy(np.int64)
        pt = d["port"].to_numpy(np.int64)
        np.add.at(self.tot, (m, ni), byt)
        far_mask = (dr != self.own) & (dr < len(C.REGIONS))
        np.add.at(self.far, (m[far_mask], ni[far_mask]), rec[far_mask])
        covered = np.zeros(len(pt), dtype=bool)
        for name, ports in PORT_SLOTS.items():
            sel = np.isin(pt, ports)
            covered |= sel
            np.add.at(self.slot[name], (m[sel], ni[sel]), rec[sel])
        np.add.at(self.other, (m[~covered], ni[~covered]), rec[~covered])
        self.tot = np.log1p(self.tot)
        # Per-segment median volume, used to scale the far-end consensus floor.
        self.seg_med = np.array([np.median(self.tot[lo:hi]) for lo, hi, _ in C.SEGMENT_BOUNDS])

    def far_floor(self, seg: int, ref_med: float) -> float:
        """Segment-adaptive consensus floor (fix 3 in the module docstring)."""
        ratio = float(np.clip(self.seg_med[seg] / max(ref_med, 1e-9), 0.30, 1.0))
        return max(FAR_FLOOR_MIN, FAR_MIN_BASE * ratio)


def detect_nf_nodes() -> tuple:
    return ("br-1", "br-2", "cr-1", "cr-2")


def _flank_baseline(v: np.ndarray, t: int, seg: int) -> float:
    """Median of ``v`` over the +-FLANK minute flanks, excluding the peak, inside one segment."""
    lo, hi, _ = C.SEGMENT_BOUNDS[seg]
    a0, a1 = max(lo, t - FLANK), max(lo, t - FLANK_GAP)
    b0, b1 = min(hi, t + FLANK_GAP), min(hi, t + FLANK)
    parts = []
    if a1 > a0:
        parts.append(v[a0:a1])
    if b1 > b0:
        parts.append(v[b0:b1])
    if not parts:
        return 0.0
    x = np.concatenate(parts)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else 0.0


class FeatureBuilder:
    """Builds the (peak x element) feature frame shared by every model head."""

    def __init__(self, evs: dict[str, E.RegionEvidence], thr: dict, verbose: bool = True):
        self.evs = evs
        self.thr = thr
        self.nf: dict[str, NetflowCube] = {}
        for r in C.REGIONS:
            self.nf[r] = NetflowCube(r)
        self.ref_far_med = float(np.median([c.seg_med[0] for c in self.nf.values()]))
        # channel lookup: (region, kind, node) -> column index
        self.ch: dict[tuple[str, int, str], int] = {}
        for r, ev in evs.items():
            for j, (k, nd) in enumerate(zip(ev.kind_idx, ev.node_of)):
                self.ch[(r, int(k), nd)] = j
        if verbose:
            print(f"  [features] {len(evs)} regions, {sum(e.n_channels for e in evs.values())} "
                  f"channels, far-end ref median={self.ref_far_med:.3f}", flush=True)

    def _series(self, region: str, kind: int, node: str, w: int) -> np.ndarray:
        j = self.ch.get((region, kind, node))
        if j is None:
            return None
        return self.evs[region].A[w][:, j]

    def _norm(self, region: str, kind: int, node: str, w: int, t: int) -> float:
        s = self._series(region, kind, node, w)
        if s is None:
            return 0.0
        return float(s[t]) / self.thr.get((w, kind), 1.0)

    def _norm_series(self, region: str, role: str) -> tuple[dict, dict]:
        """Normalised channel series for one (region, role), computed once per role.

        Returns ``(per_kind, env)`` where ``per_kind[(k, w)]`` is ``A_w[:, ch] / thr[(w, k)]``
        (absent when that role is not observed by kind ``k``) and ``env[w]`` is the max over the
        kinds that do observe it.  Dividing each kind by its *own* level before combining is what
        keeps the combination meaningful -- the levels differ by orders of magnitude between
        kinds, so a raw max would always be won by whichever kind has the largest unit.
        """
        per_kind: dict = {}
        env: dict = {w: None for w in C.WINDOWS}
        for k in range(len(E.KINDS)):
            j = self.ch.get((region, k, role))
            if j is None:
                continue
            ev = self.evs[region]
            for w in C.WINDOWS:
                s = ev.A[w][:, j].astype(np.float64) / self.thr.get((w, k), 1.0)
                per_kind[(k, w)] = s
                env[w] = s.copy() if env[w] is None else np.maximum(env[w], s)
        for w in C.WINDOWS:
            if env[w] is None:
                env[w] = np.zeros(C.N_MINUTES, dtype=np.float64)
        return per_kind, env

    def build(self, peaks: pd.DataFrame) -> pd.DataFrame:
        """One row per (peak, role) for the ten official roles of the peak's region.

        The candidate role comes from the loop, NOT from ``pk.role``.  ``pk.role`` is the
        loudest element at that minute -- only one of the ten -- and reading the row's identity
        from it makes all ten rows of a peak claim the same element while carrying ten different
        feature vectors.  That silently destroys the element head: every row of a peak is then
        labelled root or not-root together (measured: is_root per peak was 10 for 284 peaks and
        0 for the rest, instead of exactly 1), and Top-1 equals Top-5 by construction.
        """
        rows = []
        for region, grp in peaks.groupby("region", sort=False):
            nf = self.nf[region]
            nf_idx = {n: i for i, n in enumerate(nf.nodes)}
            for role in C.ROLES:
                per_kind, env = self._norm_series(region, role)
                ni = nf_idx.get(role, -1)
                for pk in grp.itertuples(index=False):
                    rows.append(self._row(pk, role, per_kind, env, nf, ni))
        df = pd.DataFrame(rows)
        # peak_id must follow the row order actually produced (grouped by region, then role),
        # so recompute it from the frame rather than assuming the input order.
        key = pd.MultiIndex.from_arrays([df["midx"], df["region"]])
        df["peak_id"] = pd.factorize(key)[0].astype(np.int64)
        df["rank_in_region"] = (df.groupby("peak_id")["s_w0"]
                                .rank(ascending=False, method="first").astype(np.int16) - 1)
        df["is_loudest"] = (df["rank_in_region"] == 0).astype(np.int8)
        return df

    def _row(self, pk, role: str, per_kind: dict, env: dict,
             nf: "NetflowCube", ni: int) -> dict:
        t, seg, region = int(pk.midx), int(pk.seg), pk.region
        f: dict = {"midx": t, "seg": seg, "region": region, "role": role,
                   "elem": f"{region}-{role}"}
        # Whether this candidate is the element the detector found loudest is real information
        # (it is right ~45% of the time on the labelled segment), so keep it as a feature now
        # that `role` no longer aliases it.
        f["is_peak_role"] = 1.0 if role == pk.role else 0.0
        for k in range(len(E.KINDS)):
            for w in C.WINDOWS:
                s = per_kind.get((k, w))
                f[f"k{k}_w{w}"] = float(s[t]) if s is not None else 0.0
        for w in C.WINDOWS:
            f[f"s_w{w}"] = float(env[w][t])
        # flanking-baseline detrending (fix 2): the absolute level is not comparable across
        # batches, the departure from its own local baseline is.
        f["b30"] = _flank_baseline(env[0], t, seg)
        f["d_w0"] = f["s_w0"] - f["b30"]
        f["d_w9"] = f["s_w9"] - f["b30"]
        f["ratio30"] = f["s_w0"] / (f["b30"] + 0.1)
        f["rint"] = float(pk.rint)
        f["frac_region"] = f["s_w0"] / (f["rint"] + 1e-6)
        # flat_top_width only compares against a fraction of the peak, so float64 is fine;
        # converting the whole 50,400-minute envelope per row would allocate gigabytes.
        a, b = detect.flat_top_width(env[0], t)
        f["ftw"] = float(b - a)
        pk_lvl = max(float(env[0][t]), 1e-9)
        f["persist"] = float((env[0][max(0, t - 10):t + 11] >= 0.5 * pk_lvl).sum())
        f["nf_far_floor"] = nf.far_floor(seg, self.ref_far_med)
        if ni >= 0:
            f["nf_tot"] = float(nf.tot[t, ni])
            f["nf_tot_b30"] = _flank_baseline(nf.tot[:, ni], t, seg)
            f["nf_tot_d"] = f["nf_tot"] - f["nf_tot_b30"]
            f["nf_far"] = float(nf.far[t, ni])
            f["nf_far_b30"] = _flank_baseline(nf.far[:, ni], t, seg)
            f["nf_far_d"] = f["nf_far"] - f["nf_far_b30"]
            f["nf_far_cons"] = 1.0 if f["nf_far"] >= f["nf_far_floor"] else 0.0
            # Express the far-end volume *relative to* its segment floor rather than exposing the
            # floor itself: the floor is a per-segment constant, so as a raw feature it is a batch
            # identifier, and a model that splits on it learns batch-1-specific thresholds that
            # cannot transfer to the two unlabelled segments.
            f["nf_far_rel"] = f["nf_far"] / max(f["nf_far_floor"], 1e-9)
            tot = f["nf_tot"] + 1e-6
            for name in PORT_SLOTS:
                col = nf.slot[name][:, ni]
                v = float(col[t])
                f[f"nfslot_{name}"] = v
                f[f"nfslot_{name}_r"] = v / tot
                f[f"nfslot_{name}_d"] = v - _flank_baseline(col, t, seg)
            f["nfslot_other_r"] = float(nf.other[t, ni]) / tot
        else:
            for nm in ("nf_tot", "nf_tot_b30", "nf_tot_d", "nf_far", "nf_far_b30",
                       "nf_far_d", "nf_far_cons", "nfslot_other_r"):
                f[nm] = 0.0
            for name in PORT_SLOTS:
                f[f"nfslot_{name}"] = f[f"nfslot_{name}_r"] = f[f"nfslot_{name}_d"] = 0.0
        # No time-of-day features.  Faults are injected uniformly over the record, so minute of
        # day carries no real signal -- but batch 1's 305 labelled faults do fall at particular
        # clock times, and with tod_sin/tod_cos exposed the model spent 35% of its total gain on
        # them, i.e. it memorised *when* batch 1 broke.  That cannot transfer to the two
        # unlabelled segments, so the features are removed rather than regularised.
        f["score"] = float(pk.score)
        return f


FEATURE_COLS: list[str] | None = None


def peak_frame(feat: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Collapse the (peak x role) frame to one row per peak, for the selection head.

    Selection needs "is there a real fault here", which is a property of the *peak*, while the
    element head answers "which of the ten elements caused it".  Deriving the former as
    ``max over roles`` of the latter conflates the two: a peak that is a genuine fault gets one
    high role score and nine low ones, so the max is diluted by the element problem, and a peak
    that is not a fault can still produce a spuriously high single-role score.  Measured cost of
    the conflation: at E=1200 the anchor kept precision 0.217 (260 TP of 1200) where the
    previous architecture reached 0.912 (291 TP of 319).

    Aggregation is max and mean over the ten roles for every feature.  Max captures "some element
    here is loud", mean captures "the whole node group moved together" -- a real fault usually
    does both, a single spiky channel does only the first.  Both are permutation-invariant over
    roles, so the frame is identical at train and inference time.
    """
    num = [c for c in cols if c in feat.columns]
    g = feat.groupby("peak_id", sort=True)
    agg = g[num].max()
    agg.columns = [f"{c}__max" for c in agg.columns]
    mean = g[num].mean()
    mean.columns = [f"{c}__mean" for c in mean.columns]
    out = agg.join(mean)
    # identity/context columns that must survive the collapse unchanged
    for c in ("midx", "seg", "region"):
        out[c] = g[c].first()
    # how many of the ten roles show real evidence: a fault implicates a coherent group,
    # noise implicates one channel
    if "s_w0" in feat.columns:
        pid = feat["peak_id"]
        out["n_loud"] = (feat["s_w0"] > 1.0).astype(np.int16).groupby(pid).sum()
        out["s_w0_spread"] = feat.groupby(pid)["s_w0"].std().fillna(0.0)
    else:
        out["n_loud"] = 0
        out["s_w0_spread"] = 0.0
    return out.reset_index()


def peak_cols(feat: pd.DataFrame, cols: list[str]) -> list[str]:
    """Feature names produced by :func:`peak_frame`, in a deterministic order."""
    num = [c for c in cols if c in feat.columns]
    return ([f"{c}__max" for c in num] + [f"{c}__mean" for c in num]
            + ["n_loud", "s_w0_spread"])


def feature_cols(df: pd.DataFrame) -> list[str]:
    """Model inputs, excluding every identifier and every label column.

    Call this BEFORE attaching labels.  The exclusion list is defence in depth, not the primary
    guard: an earlier version computed the list after labelling and omitted ``cat_idx``, so the
    ground-truth category became the top feature of both heads (81% and 61% of total gain) and
    the reported ``cat_top1`` of 0.96 was pure leakage.  ``cat_idx >= 0`` is exactly
    ``is_root == 1``, i.e. a perfect copy of the target.
    """
    drop = {"midx", "seg", "region", "role", "elem", "peak_id", "p_root",
            "is_event", "is_root", "major", "sub", "cat_idx", "hyp", "gt_id", "gid",
            "gt_dur", "gt_key",
            # per-segment constant: as a feature it is a batch identifier (see _row)
            "nf_far_floor"}
    return [c for c in df.columns if c not in drop and df[c].dtype.kind in "fiub"]
