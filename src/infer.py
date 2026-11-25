"""L4 unified inference: one pass over 50,400 minutes, one answer file.

Segment discipline, which is where a naive compact-axis implementation goes wrong:

* The three batches are adjacent on the canonical axis (Segment 0 ends at midx 20159, Segment 1
  starts at 20160) but a month apart in wall-clock time.  Overlap suppression therefore runs
  **per segment** -- suppressing across the boundary would treat two independent faults as
  duplicates and delete one.
* Every window is clamped to the segment of its own peak, so no predicted interval spans a batch
  boundary and renders as a nonsense month-long fault.

Emission is **target-driven, not candidate-driven**.  An earlier version took the top-K peaks
per segment (K = 1280 per 14 days) and relied on overlap suppression to thin them to roughly a
quarter, which is what the previous architecture did.  Here the windows are narrower and the
peaks are already NMS-separated, so suppression removed only 12% (3200 -> 2818) and the answer
file came out 3.5x over target -- enough to drag alpha_fp to its 0.7 floor on its own.  So the
selection now walks peaks in descending fault probability, skips any that overlap an accepted
one, and stops when the segment's emission target is met.  The target comes from the organiser
FAQ's physical bound (at most one fault per 60 minutes) times the measured hedge: on the labelled
anchor, emitting 21% too few cost 14.07 points while 18% too many cost 1.42, so the budget errs
high rather than aiming at the estimated fault count.

``--freeze-stage1`` replaces the Segment 0 part of the output with the frozen, already-scored
batch-1 answer file.  Batch 1 ships a probe table and FRR syslog that batches 2/3 lack, so its
historical submission (90.0911 on the server) beats anything this 5-source model produces for
that segment; the leaderboard takes the historical maximum, so protecting it costs nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C, detect, evidence as E, features as F, models  # noqa: E402

TOP_K = 5


def attach_windows(peaks: pd.DataFrame, evs, thr, verbose: bool = True,
                   cross_level: float = C.CROSS_LEVEL,
                   win_min: int = 0, win_max: int = 10 ** 6,
                   widths: np.ndarray | None = None) -> pd.DataFrame:
    """Give every peak a physical window, measured on its implicated element's own envelope.

    The envelope must be built from the *smoothed* tracks.  Two separate failures are avoided
    here, both measured:

    * Using window 0 alone degenerates completely.  Window 0 is the zero-inflated instantaneous
      excess, so it reads exactly 0 for most peaks (which exist because a *long* window crossed
      the threshold).  With a zero peak the walk cannot expand and every window collapses to
      ``MIN_LEN``=2 min -- measured p25/p50/p75 = 2/2/2 against a truth of 7.8/13.6/21.2, which
      puts Dice at 2*2/(2+14) = 0.25, below the official 0.4 gate, so those records cannot match
      at all.
    * Taking the max over *all* windows (including 0) is scale-correct but shape-wrong.  The
      cross-segment alignment matches each channel's marginal distribution, which fixes how
      *loud* a batch is, but a pointwise monotone map cannot restore temporal autocorrelation --
      how fast the envelope decays either side of a peak, which is exactly what the crossing walk
      measures.  Batch 3's evidence is genuinely spikier in time, and including the w=0 track
      lets that spike set the window: measured Segment-2 duration p50 = 3.0 min (4.0 even for the
      300 highest-confidence records) against Segment 0's 14.75.  Restricting the walk to the
      smoothed tracks (``win_min``) is what keeps the flat top a flat top.

    The envelope depends only on (region, role) -- 80 distinct pairs -- so it is cached rather
    than recomputed per peak.
    """
    ch_index = {}
    for region, ev in evs.items():
        for j, (k, nd) in enumerate(zip(ev.kind_idx, ev.node_of)):
            ch_index[(region, int(k), nd)] = j
    cache: dict[tuple[str, str], np.ndarray] = {}
    use_windows = tuple(w for w in C.WINDOWS if win_min <= w <= win_max) \
        or (max(C.WINDOWS),)

    def envelope(region: str, role: str) -> np.ndarray:
        key = (region, role)
        if key in cache:
            return cache[key]
        env = None
        for w in use_windows:
            for k in range(len(E.KINDS)):
                j = ch_index.get((region, k, role))
                if j is None:
                    continue
                s = evs[region].A[w][:, j].astype(np.float64) / thr.get((w, k), 1.0)
                env = s if env is None else np.maximum(env, s)
        if env is None:
            env = np.zeros(C.N_MINUTES, dtype=np.float64)
        cache[key] = env
        return env

    t0s = np.empty(len(peaks), dtype=np.int64)
    t1s = np.empty(len(peaks), dtype=np.int64)
    degenerate = 0
    for i, pk in enumerate(peaks.itertuples(index=False)):
        env = envelope(pk.region, pk.role)
        if not (env[int(pk.midx)] > 0):
            degenerate += 1
        a, b = detect.flat_top_width(env, int(pk.midx), frac=cross_level)
        if widths is not None:
            # Keep the envelope-derived CENTRE, replace the WIDTH.  The crossing walk locates
            # the fault well but its width is set by the smoothing support (9 min for the w=9
            # track) rather than by the fault, and the official time score only forgives ~3 min
            # per edge, so a systematic width error is expensive.
            w = int(round(float(np.clip(widths[i], C.MIN_LEN, C.MAX_LEN))))
            c = (a + b) / 2.0
            a = int(round(c - w / 2.0))
            b = a + w
        a, b = C.clamp_to_segment(a, b, int(pk.midx))
        t0s[i], t1s[i] = a, max(b, a + C.MIN_LEN)
    if verbose:
        d = t1s - t0s
        print(f"  [windows] {len(peaks)} peaks @cross_level={cross_level} "
              f"tracks={use_windows}, length p25/p50/p75/mean = "
              f"{np.percentile(d, 25):.0f}/{np.percentile(d, 50):.0f}/"
              f"{np.percentile(d, 75):.0f}/{d.mean():.1f} min "
              f"(reference truth 7.8/13.6/21.2, mean 14.7); "
              f"degenerate(peak<=0)={degenerate}", flush=True)
        pk_seg = peaks["seg"].to_numpy()
        for si, (_, _, nm) in enumerate(C.SEGMENT_BOUNDS):
            m = pk_seg == si
            if m.any():
                print(f"      {nm}: p25/p50/p75 = {np.percentile(d[m], 25):.0f}/"
                      f"{np.percentile(d[m], 50):.0f}/{np.percentile(d[m], 75):.0f} min",
                      flush=True)
    return peaks.assign(t0=t0s, t1=t1s)


def emission_targets(hedge: float) -> dict[str, int]:
    """Per-segment emission target from the FAQ physical bound (<= 1 fault per 60 min)."""
    return {name: int(np.ceil((hi - lo) / 60.0 * hedge))
            for lo, hi, name in C.SEGMENT_BOUNDS}


def select_and_suppress(peaks: pd.DataFrame, targets: dict[str, int],
                        verbose: bool = True) -> pd.DataFrame:
    """Per segment: walk peaks by descending fault probability, skip overlaps, stop at target."""
    kept = []
    for si, (lo, hi, name) in enumerate(C.SEGMENT_BOUNDS):
        seg = peaks[(peaks["midx"] >= lo) & (peaks["midx"] < hi)]
        target = targets[name]
        if not len(seg):
            continue
        seg = seg.sort_values("p_event", ascending=False)
        taken: list[tuple[int, int]] = []
        dropped = 0
        for r in seg.itertuples(index=False):
            if len(taken) >= target:
                break
            if any(r.t0 < b and a < r.t1 for a, b in taken):
                dropped += 1
                continue
            taken.append((r.t0, r.t1))
            kept.append(r._asdict())
        pmin = min((k["p_event"] for k in kept), default=float("nan"))
        if verbose:
            print(f"  [select] {name} ({(hi - lo) // 1440}d): pool={len(seg)} "
                  f"target={target} -> emitted={len(taken)} (overlap-skipped {dropped}), "
                  f"p_event min={pmin:.4f}", flush=True)
    out = pd.DataFrame(kept)
    return out.sort_values("p_event", ascending=False).reset_index(drop=True) if len(out) else out


def build_records(feat: pd.DataFrame, sel: pd.DataFrame, cat_head) -> list[dict]:
    """Selected peaks -> official answer records (Top-5 elements + category from rank 1).

    Two constraints, both derived from the labelled anchor rather than tuned on a score:

    * Roles absent from ``role_major_prior()`` are dropped from the Top-5.  ``monitor-vm`` is the
      only such role: it is never a root cause in the 305-interval anchor and it is the one role
      that reports no metrics at all, so its feature row is identically zero and its rank is
      noise.  It was nonetheless rank 1 on 163/1200 Segment-0 records -- each a guaranteed RCA
      miss, since the official term pays by the rank of the *true* element.
    * The major category follows the rank-1 role, and the sub-category is the head's best choice
      *within* that major.  The head alone collapsed to 84% firewall while only 24% of rank-1
      roles were ``fw``: it is trained on true-root-cause rows only but applied to the rank-1 row
      of every emitted peak, most of which are not faults.  Role determines the major with purity
      0.97-1.00 for every role except service-vm-1 (0.63), so constraining is close to free and
      removes an out-of-distribution failure mode.
    """
    prior = C.role_major_prior()
    allowed = set(prior)
    feats_by_peak = {pid: g for pid, g in feat.groupby("peak_id", sort=False)}
    rows = []
    for r in sel.itertuples(index=False):
        g = feats_by_peak.get(r.peak_id)
        if g is None or not len(g):
            continue
        g = g[g["role"].isin(allowed)]
        if not len(g):
            continue
        g = g.sort_values("p_root", ascending=False).head(TOP_K)
        first = g.iloc[0]
        major = prior[str(first["role"])]
        sub = C.subs_of(major)[0]
        if cat_head is not None and cat_head.models:
            pr = cat_head.predict(first.to_frame().T)[0]
            ci = int(np.argmax(pr))
            cand_major, cand_sub = C.CATEGORY_PAIRS[ci]
            if cand_major == major:
                sub = cand_sub
            else:
                # keep the head's ranking but restrict it to sub-categories of the role's major
                idx = [i for i, (m, _) in enumerate(C.CATEGORY_PAIRS) if m == major]
                if idx:
                    sub = C.CATEGORY_PAIRS[max(idx, key=lambda i: pr[i])][1]
        rows.append({"t0": int(r.t0), "t1": int(r.t1), "seg": int(r.seg),
                     "p_event": float(r.p_event),
                     "top5": [C.element_id(r.region, x) for x in g["role"].tolist()],
                     "major": major, "sub": sub})
    return rows


def freeze_stage1(records: list[dict], frozen_path: str, verbose: bool = True) -> list[dict]:
    """Swap the unified Segment 0 records for the frozen, already-scored batch-1 answer file.

    Comparison is on parsed timestamps, not strings: the frozen shard and this writer can render
    the UTC offset differently, and a string compare mis-sorts records near the boundary.
    """
    if not os.path.exists(frozen_path):
        print(f"  [freeze] {frozen_path} not found; keeping the unified Segment 0 output",
              flush=True)
        return records
    s1_end = pd.Timestamp(C.BATCHES["seg0"]["grid_end"], tz="UTC")
    with open(frozen_path, encoding="utf-8") as f:
        frozen = [json.loads(x) for x in f if x.strip()]
    later = [r for r in records if C.to_utc([r["t0"]])[0] >= s1_end]
    if verbose:
        print(f"  [freeze] {len(frozen)} frozen batch-1 records + {len(later)} unified "
              f"(dropped {len(records) - len(later)} unified batch-1 records)", flush=True)
    return [{"_raw": r} for r in frozen] + later


def write_jsonl(records: list[dict], path: str) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = ("prediction_id", "start_time", "end_time", "root_cause_top5", "fault_category")
    with open(path, "w", encoding="utf-8") as f:
        for i, r in enumerate(records, 1):
            if "_raw" in r:
                rec = dict(r["_raw"])
            else:
                rec = {"start_time": C.fmt_utc(r["t0"]), "end_time": C.fmt_end_utc(r["t1"]),
                       "root_cause_top5": [{"rank": k + 1, "network_element_id": e}
                                           for k, e in enumerate(r["top5"][:TOP_K])],
                       "fault_category": {"major_category": r["major"],
                                          "sub_category": r["sub"]}}
            rec["prediction_id"] = f"pred-{i:05d}"
            f.write(json.dumps({k: rec[k] for k in keys}, ensure_ascii=False) + "\n")
    return len(records)


def run(model_dir: str, out_path: str, score_thr: float, freeze: str | None,
        hedge: float, verbose: bool = True,
        cross_level: float = C.CROSS_LEVEL, win_min: int = 0,
        win_max: int = 10 ** 6, align: str = "quantile") -> list[dict]:
    t0 = time.time()
    evs = E.build_all(verbose=False)
    thr = E.calibrate(evs)
    if align == "quantile":
        E.align_segments_to_anchor(evs, verbose=False)
    elif align == "level":
        # Per-segment calibration on the unified axis: each batch divided by its OWN level, which
        # is what the pre-rebuild pipeline did.  This is the control for the quantile map -- see
        # evidence.align_by_level for why the two are not interchangeable.
        rep = E.align_by_level(evs, thr, verbose=verbose)
        if verbose:
            ms = [d["mult"] for d in rep.values()]
            print(f"  [infer] per-segment calibration: {len(rep)} (window, kind, segment) "
                  f"rescales, multiplier min/median/max = {min(ms):.3f}/"
                  f"{float(np.median(ms)):.3f}/{max(ms):.3f}", flush=True)
    elif verbose:
        print("  [infer] NO cross-segment normalisation: segments 1/2 are divided by the "
              "anchor's levels while carrying their own scale", flush=True)
    S, _ = E.element_scores(evs, thr)
    rint = E.region_intensity(S)
    if verbose:
        print(f"  [infer] evidence+alignment {time.time() - t0:.0f}s", flush=True)

    peaks = detect.find_peaks(S, rint, score_thr=score_thr, verbose=verbose)
    del S

    fb = F.FeatureBuilder(evs, thr, verbose=verbose)
    feat = fb.build(peaks)                      # does not read t0/t1, so it can run first
    root = models.Head.load(model_dir, "root")
    pr = root.predict(feat)
    feat["p_root"] = pr[:, 0] if pr.ndim > 1 else pr
    del pr
    # Link peaks to their ten feature rows.  `peak_id` only exists on the feature frame, so join
    # on the (midx, region) identity rather than assuming column passthrough.
    key = feat[["midx", "region", "peak_id"]].drop_duplicates()
    peaks = peaks.merge(key, on=["midx", "region"], how="left")

    # The peak frame must be built from exactly the feature list used at training, otherwise the
    # aggregated column names drift and Head._X silently zero-fills the mismatch.
    with open(os.path.join(model_dir, "train_meta.json"), encoding="utf-8") as f:
        train_cols = json.load(f)["cols"]
    pk = F.peak_frame(feat, train_cols)
    pk_index = pk["peak_id"].to_numpy()

    # Selection ranks PEAKS, so use the dedicated peak-level head.
    p_event = feat.groupby("peak_id")["p_root"].max()      # fallback / comparison baseline
    try:
        event = models.Head.load(model_dir, "event")
        pe = event.predict(pk)
        p_event = pd.Series(pe[:, 0] if pe.ndim > 1 else pe, index=pk_index)
        if verbose:
            print(f"  [infer] selection score from the peak-level event head "
                  f"({len(pk)} peaks)", flush=True)
    except (FileNotFoundError, KeyError):
        if verbose:
            print("  [infer] no event head found; falling back to max-over-roles of the root "
                  "head (weaker: conflates 'is a fault' with 'is this element')", flush=True)
    peaks["p_event"] = np.nan_to_num(
        p_event.reindex(peaks["peak_id"].to_numpy()).to_numpy(dtype=np.float64), nan=0.0)

    # Window width, if a duration head was trained.  Aligned back to `peaks` row order by
    # peak_id: the peak frame is sorted by peak_id, `peaks` is not.
    widths = None
    try:
        dur = models.Head.load(model_dir, "dur")
        wd = dur.predict(pk)
        wd = wd[:, 0] if wd.ndim > 1 else wd
        widths = (pd.Series(wd, index=pk_index)
                  .reindex(peaks["peak_id"].to_numpy()).to_numpy(dtype=np.float64))
        widths = np.where(np.isfinite(widths), widths, C.MIN_LEN)
        if verbose:
            print(f"  [infer] window widths from the duration head: "
                  f"p25/p50/p75 = {np.percentile(widths, 25):.1f}/"
                  f"{np.percentile(widths, 50):.1f}/{np.percentile(widths, 75):.1f} min",
                  flush=True)
    except (FileNotFoundError, KeyError):
        if verbose:
            print("  [infer] no duration head; window width comes from the crossing walk only",
                  flush=True)

    peaks = attach_windows(peaks, evs, thr, verbose, cross_level=cross_level,
                           win_min=win_min, win_max=win_max, widths=widths)
    if verbose:
        print(f"  [infer] features {feat.shape}, scores in {time.time() - t0:.0f}s; "
              f"p_event>=0.5: {int((peaks['p_event'] >= 0.5).sum())} peaks", flush=True)

    targets = emission_targets(hedge)
    if verbose:
        print(f"  [infer] emission targets (hedge {hedge}): {targets}", flush=True)
    sel = select_and_suppress(peaks, targets, verbose)
    try:
        cat = models.Head.load(model_dir, "cat")
    except (FileNotFoundError, KeyError):
        cat = None
    records = build_records(feat, sel, cat)
    if freeze:
        records = freeze_stage1(records, freeze, verbose)
    n = write_jsonl(records, out_path)
    if verbose:
        print(f"  [infer] wrote {n} records -> {out_path} ({time.time() - t0:.1f}s)", flush=True)
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=C.MODEL_ROOT)
    ap.add_argument("--out", default=os.path.join(C.OUTPUT_ROOT, "result_unified.jsonl"))
    ap.add_argument("--score-thr", type=float, default=C.SCORE_THR)
    ap.add_argument("--hedge", type=float, default=C.EMIT_HEDGE,
                    help="emission target = ceil(segment_minutes/60 * hedge)")
    ap.add_argument("--cross-level", type=float, default=C.CROSS_LEVEL,
                    help="a window expands while the envelope stays above this fraction of its "
                         "peak; higher narrows windows (measured: 0.5 gives predicted length "
                         "p50 18.0 min against a 13.5 min reference)")
    ap.add_argument("--win-min", type=int, default=0, metavar="MIN",
                    help="place windows on evidence tracks smoothed over at least MIN minutes "
                         "(9 or 21 excludes the spiky instantaneous track; see attach_windows)")
    ap.add_argument("--win-max", type=int, default=10 ** 6, metavar="MAX",
                    help="upper bound on the smoothing track used for window placement; set "
                         "MIN=MAX=9 to use the 9-minute track alone, whose half-max width "
                         "tracks the true fault width instead of the smoothing width")
    ap.add_argument("--align", choices=["quantile", "level", "none"], default="quantile",
                    help="cross-segment normalisation: 'quantile' maps each segment's channel "
                         "distribution onto the anchor's; 'level' rescales by one factor per "
                         "(window, kind) so each batch is divided by its OWN level (the "
                         "pre-rebuild behaviour, and the control for 'quantile'); 'none' divides "
                         "every segment by the anchor's levels unchanged")
    ap.add_argument("--freeze-stage1", default=None,
                    help="path to the frozen batch-1 answer file; its records replace the "
                         "unified Segment 0 output")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    os.makedirs(C.LOG_ROOT, exist_ok=True)
    log_path = os.path.join(C.LOG_ROOT, "infer.log")
    with open(log_path, "w", encoding="utf-8", buffering=1) as lf:
        class Tee:
            def write(self, s):
                sys.__stdout__.write(s)
                lf.write(s)
                lf.flush()

            def flush(self):
                sys.__stdout__.flush()
                lf.flush()

        old = sys.stdout
        sys.stdout = Tee()
        try:
            print(f"[infer] {time.strftime('%Y-%m-%d %H:%M:%S')} models={args.model_dir} "
                  f"score_thr={args.score_thr} hedge={args.hedge} freeze={args.freeze_stage1}")
            run(args.model_dir, args.out, args.score_thr, args.freeze_stage1, args.hedge,
                verbose=not args.quiet, cross_level=args.cross_level, win_min=args.win_min,
                win_max=args.win_max, align=args.align)
        finally:
            sys.stdout = old
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
