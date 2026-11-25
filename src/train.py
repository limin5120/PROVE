"""L3 training: fit the root-cause and category heads on Segment 0 only.

Segment 0 is the only part of the unified axis with a public answer sheet (``gt/96GT.jsonl``),
so every label comes from there.  Peaks from segments 1/2 are excluded outright -- including
them would label real but unannotated faults as negatives, which is the single easiest way to
train a detector to ignore exactly the batch it will be judged on.

Grouped K-fold by ``peak_id`` matters: each peak contributes ten rows (one per candidate role)
and exactly one of them is the root cause.  If those ten rows could land in different folds the
model would see the same minute in both train and validation, and the OOF numbers would be
fiction rather than an estimate.

Reported metrics are chosen to be the ones the score actually depends on:
* ``peak_auc`` -- can the head separate real faults from noise peaks?  This drives the emission
  ranking, hence rho and alpha_fp, which together scale all four score components.
* ``role_top1`` / ``role_top5`` -- is the true root cause ranked first / inside the top five?
  These map directly onto the RCA term (1.0 / 0.8 / 0.6 / 0.4 / 0.2 / 0 by rank).
* ``cat_top1`` / ``major_top1`` -- the Minor and Major terms.
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

LABEL_PAD = 2          # a peak this many minutes outside a true interval still counts as on it


def load_labels() -> pd.DataFrame:
    """The batch-1 reference sheet plus its category index into ``config.CATEGORY_PAIRS``."""
    df = C.load_gt()
    pair_ix = {p: i for i, p in enumerate(C.CATEGORY_PAIRS)}
    df["cat_idx"] = [pair_ix.get((m, s), -1) for m, s in zip(df["major"], df["sub"])]
    unknown = int((df["cat_idx"] < 0).sum()) if len(df) else 0
    print(f"  [labels] {len(df)} intervals on the canonical axis "
          f"(midx {df['m0'].min()}..{df['m1'].max()}), "
          f"{unknown} with a category outside the official taxonomy", flush=True)
    return df


def attach_labels(feat: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Add ``is_event`` / ``is_root`` / ``cat_idx`` to a (peak x role) frame."""
    feat = feat.copy()
    feat["is_event"] = np.int8(0)
    feat["is_root"] = np.int8(0)
    feat["cat_idx"] = np.int16(-1)
    feat["gt_dur"] = np.float32(-1.0)      # matched interval length in minutes; -1 = unmatched
    feat["gt_key"] = ""                    # which labelled interval this peak matched
    for region, g in gt.groupby("region", sort=False):
        sel = feat["region"] == region
        if not sel.any():
            continue
        # searchsorted requires sorted interval starts; the reference sheet happens to be in
        # time order but nothing enforces it, and an unsorted `lo` silently mis-assigns peaks
        # to the wrong interval (which inflates the root-row count rather than raising).
        g = g.sort_values("m0", kind="stable")
        t = feat.loc[sel, "midx"].to_numpy()
        lo = (g["m0"] - LABEL_PAD).to_numpy()
        hi = (g["m1"] + LABEL_PAD).to_numpy()
        # nearest enclosing interval per peak (intervals do not overlap within a region: the
        # organiser FAQ states one fault at a time)
        j = np.searchsorted(lo, t, side="right") - 1
        ok = (j >= 0) & (t <= hi[np.clip(j, 0, len(hi) - 1)])
        j = np.where(ok, j, -1)
        sub = feat.loc[sel]
        ev = pd.Series(j >= 0, index=sub.index)
        feat.loc[sub.index, "is_event"] = ev.astype(np.int8)
        good = sub.index[j >= 0]
        if len(good):
            jj = j[j >= 0]
            want_role = g["role"].to_numpy()[jj]
            want_cat = g["cat_idx"].to_numpy()[jj]
            have_role = feat.loc[good, "role"].to_numpy()
            is_root = np.asarray(have_role == want_role).ravel()
            if _DIAG["n"] < 3:
                _DIAG["n"] += 1
                print(f"    [label-diag] {region}: len(good)={len(good)} "
                      f"len(have_role)={len(have_role)} len(want_role)={len(want_role)} "
                      f"is_root.shape={np.shape(is_root)} is_root.sum={int(is_root.sum())} "
                      f"have_dtype={have_role.dtype} want_dtype={want_role.dtype} "
                      f"have[:3]={list(have_role[:3])} want[:3]={list(want_role[:3])}")
            feat.loc[good, "is_root"] = is_root.astype(np.int8)
            sel_root = np.asarray(is_root, dtype=bool)
            feat.loc[good[sel_root], "cat_idx"] = np.asarray(want_cat)[sel_root].astype(np.int16)
            # truth duration for the interval this peak matched; the width target for the
            # duration head.  Set on all ten role rows so the peak-level collapse can take it.
            feat.loc[good, "gt_dur"] = (g["m1"].to_numpy()[jj]
                                        - g["m0"].to_numpy()[jj]).astype(np.float32)
            # Which labelled interval this peak matched.  Needed to de-bias the duration head:
            # a long interval attracts several peaks, so one row per matched peak silently
            # reweights the duration distribution toward long faults.  Measured: the matched-peak
            # duration p50 came out 19.0 min where the answer sheet's own p50 is 13.6.
            feat.loc[good, "gt_key"] = [f"{region}|{i}" for i in jj]
    return feat


_DIAG = {"n": 0}


def report_root(feat: pd.DataFrame, oof: np.ndarray) -> dict:
    """Score the head the way inference actually uses it.

    Two different rankings are being measured, and they must not be conflated:

    * ``peak_auc`` is **per peak**, on ``p_event = max over the ten roles``.  Inference selects
      the top-K peaks by exactly this quantity, so it is what governs rho and alpha_fp.  Scoring
      AUC per *row* instead is meaningless here: nine of the ten rows of a matched peak are
      non-root by construction and correctly get a low score, which drags a row-level AUC to
      ~0.54 no matter how good the head is.
    * ``role_top1`` / ``role_top5`` are **within a peak**, across its ten candidate roles.  These
      map onto the RCA term, which pays 1.0 / 0.8 / 0.6 / 0.4 / 0.2 by rank.
    """
    from sklearn.metrics import roc_auc_score

    p = oof[:, 0] if oof.ndim > 1 else oof
    d = feat.assign(p=p)
    per_peak = d.groupby("peak_id").agg(p_event=("p", "max"),
                                        is_event=("is_event", "max"))
    auc = (float(roc_auc_score(per_peak["is_event"], per_peak["p_event"]))
           if per_peak["is_event"].nunique() > 1 else float("nan"))
    ev = d[d["is_event"] == 1]
    top1 = top5 = n = 0
    for _, g in ev.groupby("peak_id"):
        truth = g.loc[g["is_root"] == 1, "role"]
        if not len(truth):
            continue
        n += 1
        order = g.sort_values("p", ascending=False)["role"].tolist()
        t = truth.iloc[0]
        top1 += int(order[0] == t)
        top5 += int(t in order[:5])
    return {"peak_auc": round(auc, 4), "matched_peaks": n,
            "role_top1": round(top1 / max(n, 1), 4), "role_top5": round(top5 / max(n, 1), 4),
            "n_peaks": int(len(per_peak)),
            "n_event_peaks": int(per_peak["is_event"].sum())}


def report_cat(feat: pd.DataFrame, oof: np.ndarray) -> dict:
    m = feat["cat_idx"].to_numpy() >= 0
    if not m.any() or oof.shape[0] != int(m.sum()):
        return {}
    pred = oof.argmax(axis=1)
    truth = feat.loc[m, "cat_idx"].to_numpy()
    pm, ps = zip(*[C.CATEGORY_PAIRS[i] for i in pred])
    tm, ts = zip(*[C.CATEGORY_PAIRS[i] for i in truth])
    return {"cat_top1": round(float((np.asarray(pred) == truth).mean()), 4),
            "major_top1": round(float((np.asarray(pm) == np.asarray(tm)).mean()), 4),
            "n_cat_rows": int(m.sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=C.MODEL_ROOT)
    ap.add_argument("--folds", type=int, default=models.N_FOLDS)
    ap.add_argument("--rounds", type=int, default=models.N_ROUNDS)
    ap.add_argument("--score-thr", type=float, default=C.SCORE_THR)
    args = ap.parse_args()
    os.makedirs(C.LOG_ROOT, exist_ok=True)
    t_start = time.time()
    log_path = os.path.join(C.LOG_ROOT, "train.log")

    with open(log_path, "w", encoding="utf-8", buffering=1) as lf:
        def em(s="") -> None:
            print(s, flush=True)
            lf.write(str(s) + "\n")
            lf.flush()

        em(f"[train] {time.strftime('%Y-%m-%d %H:%M:%S')} model_dir={args.model_dir} "
           f"folds={args.folds} rounds={args.rounds} score_thr={args.score_thr}")
        evs = E.build_all(verbose=False)
        thr = E.calibrate(evs)
        rep = E.align_segments_to_anchor(evs, verbose=False)
        em(f"[train] evidence built for {len(evs)} regions; {len(rep)} alignment mappings "
           f"applied ({time.time() - t_start:.0f}s)")

        S, elems = E.element_scores(evs, thr)
        rint = E.region_intensity(S)
        peaks = detect.find_peaks(S, rint, score_thr=args.score_thr, verbose=False)
        seg0_hi = C.SEGMENT_BOUNDS[0][1]
        tr_peaks = peaks[peaks["midx"] < seg0_hi].reset_index(drop=True)
        per_seg = ", ".join(
            "{}={}".format(n, int((peaks["seg"] == i).sum()))
            for i, (_, _, n) in enumerate(C.SEGMENT_BOUNDS))
        em(f"[train] peaks: all segments={len(peaks)} ({per_seg}); "
           f"Segment 0 used for training={len(tr_peaks)}")
        del S

        gt = load_labels()
        fb = F.FeatureBuilder(evs, thr, verbose=False)
        em(f"[train] building feature frame ({time.time() - t_start:.0f}s)")
        feat = fb.build(tr_peaks)
        # Lock the feature list BEFORE labelling.  Computing it afterwards relies on remembering
        # every label column name in an exclusion list, and one omission (cat_idx) silently made
        # the ground truth the strongest feature of both heads.
        cols = F.feature_cols(feat)
        feat["role_code"] = feat["role"].map({r: i for i, r in enumerate(C.ROLES)}).astype(np.int16)
        cols = cols + ["role_code"]
        feat = attach_labels(feat, gt)
        em(f"[train] frame={feat.shape} features={len(cols)} "
           f"events={int(feat.groupby('peak_id')['is_event'].max().sum())} "
           f"root_rows={int((feat['is_root'] == 1).sum())}")
        # One root cause per matched peak is an invariant of the labelling, not a hope: the
        # reference sheet gives exactly one root_cause per interval.  If this trips, the RCA
        # head is being trained on contradictory labels.
        rpg = feat.groupby("peak_id")["is_root"].sum()
        em(f"[train] is_root per peak: max={int(rpg.max())} "
           f"dist={rpg.value_counts().sort_index().to_dict()}")
        assert "cat_idx" not in cols and "is_root" not in cols, "label leaked into features"
        del evs, fb, rint

        # role is categorical information the tree should split on directly (the fault major
        # category is close to a deterministic function of the root element's role), so encode
        # it as a code column rather than dropping it.
        y_root = feat["is_root"].to_numpy(dtype=np.int8)
        grp = feat["peak_id"].to_numpy()
        root = models.Head("root", models.PARAMS_ROOT, cols)
        em(f"[train] fitting root head ({time.time() - t_start:.0f}s)")
        oof = root.fit_oof(feat, y_root, grp, args.folds, args.rounds)
        m_root = report_root(feat, oof)
        em(f"[root] {m_root}")
        em(f"[root] top features {root.importance(12)}")

        # ---- event head: peak-level "is there a real fault here" ------------------------
        # Selection ranks peaks, not (peak, role) pairs.  Using max-over-roles of the root head
        # as the ranking signal conflates the two problems and measured precision 0.217 at
        # E=1200 on the anchor.  This head is trained on the collapsed peak frame instead, and
        # the two AUCs are reported side by side so the comparison is explicit.
        pk = F.peak_frame(feat, cols)
        pcols = F.peak_cols(feat, cols)
        y_event = (feat.groupby("peak_id", sort=True)["is_event"].max()
                   .reindex(pk["peak_id"]).fillna(0).to_numpy(dtype=np.int8))
        em(f"[train] peak frame={pk.shape} features={len(pcols)} "
           f"positives={int(y_event.sum())} ({y_event.mean():.4f})")
        event = models.Head("event", models.PARAMS_EVENT, pcols)
        oof_ev = event.fit_oof(pk, y_event, pk["peak_id"].to_numpy(), args.folds, args.rounds)
        p_ev = oof_ev[:, 0] if oof_ev.ndim > 1 else oof_ev
        from sklearn.metrics import roc_auc_score
        auc_ev = float(roc_auc_score(y_event, p_ev)) if len(np.unique(y_event)) > 1 else float("nan")
        m_event = {"peak_auc": round(auc_ev, 4),
                   "peak_auc_max_of_root": m_root["peak_auc"],
                   "n_peaks": int(len(pk)), "n_event_peaks": int(y_event.sum())}
        em(f"[event] {m_event}")
        em(f"[event] top features {event.importance(12)}")

        # ---- duration head: how wide should this peak's window be -----------------------
        # Window width is the binding constraint on the AD term.  The official time score is
        # S_time = max(0, 1 - (|dStart|+|dEnd|)/360s), i.e. ~3 min of tolerance per edge, while
        # the w=9 smoothing track makes every window exactly 9 min regardless of the fault:
        # measured predicted p50 9.0 against a truth p50 14.5, giving |dStart| 212.7 s and
        # |dEnd| 242.9 s (455 s total, over the 360 s budget) and pinning `a` at 0.7675.
        # Trained on matched peaks only -- an unmatched peak has no duration to learn.
        pk_dur = (feat.groupby("peak_id", sort=True)["gt_dur"].max()
                  .reindex(pk["peak_id"]).fillna(-1.0).to_numpy(dtype=np.float64))
        pk_key = (feat.groupby("peak_id", sort=True)["gt_key"].first()
                  .reindex(pk["peak_id"]).fillna("").to_numpy())
        dur = models.Head("dur", models.PARAMS_DUR, pcols)
        # One row per labelled interval, not per matched peak: a long interval attracts several
        # peaks, and training on all of them reweights the target toward long faults (measured
        # p50 19.0 min against the answer sheet's own 13.6).  Keep the loudest peak of each
        # interval, since that is the one selection will actually pick at inference time.
        dd = pk.assign(_dur=pk_dur, _key=pk_key)
        dd = dd[dd["_dur"] > 0]
        sort_col = next((c for c in ("rint__max", "s_w9__max") if c in dd.columns), None)
        if sort_col:
            dd = dd.sort_values(sort_col, ascending=False)
        n_before = len(dd)
        dd = dd.drop_duplicates("_key")
        dm_rows = len(dd)
        em(f"[train] duration rows: {n_before} matched peaks -> {dm_rows} distinct labelled "
           f"intervals (de-biased); truth p25/p50/p75 = "
           f"{np.percentile(dd['_dur'], 25):.1f}/{np.percentile(dd['_dur'], 50):.1f}/"
           f"{np.percentile(dd['_dur'], 75):.1f} min")
        if dm_rows >= 40:
            y_dur = dd["_dur"].to_numpy(dtype=np.float64)
            oof_d = dur.fit_oof(dd, y_dur, dd["peak_id"].to_numpy(), args.folds, args.rounds)
            pred_d = np.clip(oof_d[:, 0] if oof_d.ndim > 1 else oof_d,
                             C.MIN_LEN, C.MAX_LEN)
            m_dur = {"n": dm_rows,
                     "mae_min": round(float(np.abs(pred_d - y_dur).mean()), 3),
                     "baseline_mae_min": round(float(np.abs(y_dur - y_dur.mean()).mean()), 3),
                     "pred_p25/p50/p75": [round(float(np.percentile(pred_d, q)), 1)
                                          for q in (25, 50, 75)],
                     "truth_p25/p50/p75": [round(float(np.percentile(y_dur, q)), 1)
                                           for q in (25, 50, 75)]}
            em(f"[dur] {m_dur}")
            if m_dur["mae_min"] >= m_dur["baseline_mae_min"]:
                em("[dur] WARNING: does not beat predicting the mean -- window width should "
                   "stay fixed rather than use this head")
            em(f"[dur] top features {dur.importance(10)}")
        else:
            m_dur = {"skipped": f"only {dm_rows} distinct labelled intervals"}
            em(f"[dur] {m_dur}")

        m_cat_mask = feat["cat_idx"].to_numpy() >= 0
        cat_rows = feat[m_cat_mask].reset_index(drop=True)
        y_cat = cat_rows["cat_idx"].to_numpy(dtype=np.int32)
        cat = models.Head("cat", models.PARAMS_CAT, cols)
        if len(np.unique(y_cat)) >= 2:
            em(f"[train] fitting cat head on {len(cat_rows)} rows / "
               f"{len(np.unique(y_cat))} classes ({time.time() - t_start:.0f}s)")
            oof_cat = cat.fit_oof(cat_rows, y_cat, cat_rows["peak_id"].to_numpy(),
                                  args.folds, args.rounds)
            m_cat = report_cat(cat_rows, oof_cat)
        else:
            m_cat = {"skipped": "fewer than 2 distinct categories"}
        em(f"[cat] {m_cat}")
        if hasattr(cat, "models") and cat.models:
            em(f"[cat] top features {cat.importance(12)}")

        os.makedirs(args.model_dir, exist_ok=True)
        root.save(args.model_dir)
        event.save(args.model_dir)
        cat.save(args.model_dir)
        if dur.models:
            dur.save(args.model_dir)
        with open(os.path.join(args.model_dir, "train_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"cols": cols, "peak_cols": pcols,
                       "metrics": {"root": m_root, "event": m_event, "cat": m_cat,
                                   "dur": m_dur},
                       "dur_head": bool(dur.models),
                       "n_peaks": int(len(tr_peaks)), "n_rows": int(len(feat)),
                       "score_thr": args.score_thr, "folds": args.folds,
                       "rounds": args.rounds,
                       "elapsed_s": round(time.time() - t_start, 1),
                       "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)
        em(f"[train] saved {args.model_dir} in {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
