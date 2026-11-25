"""Verification: official scoring of Segment 0, strict schema validation, and the shape gates.

Three separate jobs, deliberately kept in one place because they answer one question -- "may
this file be submitted?" -- and a partial answer is how bad files get shipped:

1. **Official scoring of Segment 0.** The unified axis has labels only for batch 1, so the
   records falling inside batch 1's window are scored with the organiser's own evaluator
   against ``gt/96GT.jsonl``.  This is the E1b gate.  It also back-solves the per-match quality
   terms (a, r, mj, sb) and rho, which is what makes one submission worth four measurements.
2. **Strict schema validation.** Every record through ``validate_prediction`` plus
   ``ensure_unique``.  A file that scores well locally but trips the schema scores zero.
3. **Shape gates.** Duration and gap distributions against the organiser FAQ's physical priors
   (faults last 1..30 min, injections are >=20 min apart), per-segment emission counts, and the
   major-category mix.  These are the symptoms that identified the batch-3 collapse before any
   score was known: dur_p50 7.8 min against a 13.6 min reference, 129 sub-20-minute gaps against
   5, and a 59% firewall share against 24%.

The official package is stored flattened (``evaluator__matcher.py`` etc.) with its layout
documented in ``official_eval/PROVENANCE.md``.  ``ensure_package`` rebuilds the real tree once,
idempotently, so nothing accumulates per run.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C  # noqa: E402

PKG = "aiops_challenge_2026"
E1B_MIN = 88.0        # labelled-anchor gate; the only hard quality gate in the system
SEG_TOL_MIN = 5.0     # segment-membership tolerance, must match assemble.seg_bounds_ts
LAYOUT = {
    "schema.py": "schema.py",
    "config/__init__.py": "config____init__.py",
    "config/fault_taxonomy.json": "fault_taxonomy.json",
    "config/network_elements.json": "network_elements.json",
    "evaluator/__init__.py": "evaluator____init__.py",
    "evaluator/__main__.py": "evaluator____main__.py",
    "evaluator/evaluator.py": "evaluator__evaluator.py",
    "evaluator/matcher.py": "evaluator__matcher.py",
    "evaluator/metrics.py": "evaluator__metrics.py",
}


def ensure_package() -> str:
    """Materialise the official package tree; returns the directory to put on sys.path."""
    root = os.path.join(C.OFFICIAL_DIR, "pkg")
    dest = os.path.join(root, PKG)
    need = not os.path.isdir(dest)
    if not need:
        for rel in LAYOUT:
            if not os.path.exists(os.path.join(dest, rel.replace("/", os.sep))):
                need = True
                break
    if need:
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        for rel, flat in LAYOUT.items():
            src = os.path.join(C.OFFICIAL_DIR, flat)
            if not os.path.exists(src):
                raise SystemExit(f"official_eval/{flat} missing; cannot rebuild the package")
            tgt = os.path.join(dest, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(tgt), exist_ok=True)
            shutil.copyfile(src, tgt)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def official_evaluate(pred_path: str, truth: list[dict] | None = None,
                      only_segment0: bool = True) -> dict:
    ensure_package()
    ev = importlib.import_module(f"{PKG}.evaluator.evaluator")
    # The reference sheet ships in prediction format; the evaluator validates ground-truth
    # schema, so it has to be converted (config.gt_as_official) or this raises instead of scoring.
    truth = truth if truth is not None else C.gt_as_official()
    with open(pred_path, encoding="utf-8") as f:
        preds = [json.loads(x) for x in f if x.strip()]
    if only_segment0:
        lo = pd.Timestamp(C.BATCHES["seg0"]["grid_start"], tz="UTC")
        hi = pd.Timestamp(C.BATCHES["seg0"]["grid_end"], tz="UTC")
        keep = []
        for r in preds:
            t = pd.Timestamp(r["start_time"])
            t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
            if lo <= t < hi:
                keep.append(r)
        preds = keep
    return ev.evaluate(truth, preds)


def validate_schema(pred_path: str) -> dict:
    ensure_package()
    schema = importlib.import_module(f"{PKG}.schema")
    bad: dict[str, int] = {}
    ids = []
    n = 0
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            r = json.loads(line)
            ids.append(r.get("prediction_id"))
            res = schema.validate_prediction(r)
            for k, v in res.items():
                if k.startswith("_invalid") and v:
                    bad[k] = bad.get(k, 0) + 1
    dup = n - len(set(ids))
    return {"records": n, "invalid": bad, "duplicate_prediction_id": dup}


def load_records(pred_path: str) -> pd.DataFrame:
    rows = []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            g = json.loads(line)
            st = pd.Timestamp(g["start_time"])
            en = pd.Timestamp(g["end_time"])
            st = st.tz_localize("UTC") if st.tzinfo is None else st.tz_convert("UTC")
            en = en.tz_localize("UTC") if en.tzinfo is None else en.tz_convert("UTC")
            top5 = [e["network_element_id"] for e in g["root_cause_top5"]]
            rows.append({"start": st, "end": en, "top1": top5[0] if top5 else None,
                         "major": g["fault_category"]["major_category"],
                         "sub": g["fault_category"]["sub_category"],
                         "pid": g["prediction_id"]})
    return pd.DataFrame(rows)


def segment_of(ts: pd.Timestamp) -> str:
    tol = pd.Timedelta(minutes=SEG_TOL_MIN)
    for name in C.SEGMENT_ORDER:
        lo = pd.Timestamp(C.BATCHES[name]["grid_start"], tz="UTC") - tol
        hi = pd.Timestamp(C.BATCHES[name]["grid_end"], tz="UTC") + tol
        if lo <= ts < hi:
            return name
    return "outside"


def shape_report(df: pd.DataFrame) -> dict:
    out = {}
    df = df.assign(seg=df["start"].map(segment_of))
    out["by_segment"] = {k: int(v) for k, v in df["seg"].value_counts().items()}
    out["outside_any_segment"] = int((df["seg"] == "outside").sum())
    dur = (df["end"] - df["start"]).dt.total_seconds() / 60.0
    out["dur_all"] = {"p25": float(dur.quantile(.25)), "p50": float(dur.quantile(.5)),
                      "p75": float(dur.quantile(.75)), "mean": float(dur.mean()),
                      "min": float(dur.min()), "max": float(dur.max()),
                      "outside_1_30min": int(((dur < 1) | (dur > 30)).sum())}
    for name in C.SEGMENT_ORDER:
        s = df[df["seg"] == name]
        if not len(s):
            continue
        d = (s["end"] - s["start"]).dt.total_seconds() / 60.0
        gaps = s.sort_values("start")["start"].diff().dt.total_seconds().dropna() / 60.0
        mix = s["major"].value_counts().to_dict()
        out[name] = {"n": int(len(s)), "dur_p50": round(float(d.quantile(.5)), 2),
                     "dur_mean": round(float(d.mean()), 2),
                     "outside_1_30": int(((d < 1) | (d > 30)).sum()),
                     "gap_min": round(float(gaps.min()), 1) if len(gaps) else None,
                     "gap_p50": round(float(gaps.median()), 1) if len(gaps) else None,
                     "gap_lt20": int((gaps < 20).sum()) if len(gaps) else 0,
                     "fw_share": round(float(mix.get("firewall", 0)) / len(s), 3),
                     "majors": mix,
                     "elems": int(s["top1"].nunique())}
    return out


def read_jsonl(path: str) -> list[dict]:
    """Raw records as dicts (``load_records`` below parses them into a frame instead)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def frozen_signature(path: str) -> set | None:
    """Content signature of a frozen shard, or None if it does not exist."""
    if not path or not os.path.exists(path):
        return None
    sig = set()
    for r in read_jsonl(path):
        top = r.get("root_cause_top5") or [{}]
        fc = r.get("fault_category") or {}
        sig.add((r.get("start_time"), r.get("end_time"), top[0].get("network_element_id"),
                 fc.get("major_category"), fc.get("sub_category")))
    return sig


def segment0_records(pred_path: str) -> list[dict]:
    lo, hi = (pd.Timestamp(C.BATCHES["seg0"]["grid_start"], tz="UTC"),
              pd.Timestamp(C.BATCHES["seg0"]["grid_end"], tz="UTC"))
    out = []
    for r in read_jsonl(pred_path):
        t = record_start(r)
        if lo <= t < hi:
            out.append(r)
    return out


def record_start(rec: dict) -> pd.Timestamp:
    t = pd.Timestamp(rec["start_time"])
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def e1b_is_vacuous(pred_path: str, frozen_dir: str = C.FROZEN_DIR) -> str | None:
    """Detect an E1b gate that would score the frozen shard instead of the model.

    This exists because the gate was silently vacuous once already.  E1b asks "does the unified
    model reproduce the labelled anchor?", so it must be scored on the model's *own* Segment-0
    output.  But the shipped file deliberately replaces Segment 0 with the frozen batch-1 shard
    (which scores 92.3469 by construction), and a gate that simply scores "whatever sits in
    batch 1's time window" then reports PASS while measuring nothing.  The gate and its subject
    must not be the same object.
    """
    seg0 = segment0_records(pred_path)
    if not seg0:
        return "no Segment-0 records in the file"
    for name in sorted(os.listdir(frozen_dir)) if os.path.isdir(frozen_dir) else []:
        if not name.endswith(".jsonl"):
            continue
        sig = frozen_signature(os.path.join(frozen_dir, name))
        if not sig:
            continue
        hit = 0
        for r in seg0:
            top = r.get("root_cause_top5") or [{}]
            fc = r.get("fault_category") or {}
            if (r.get("start_time"), r.get("end_time"), top[0].get("network_element_id"),
                    fc.get("major_category"), fc.get("sub_category")) in sig:
                hit += 1
        if hit == len(seg0):
            return (f"all {len(seg0)} Segment-0 records are byte-identical to frozen/{name}, "
                    f"which scores 92.3469 by construction")
    return None


def gates(shape: dict, score: dict | None, schema: dict,
          e1b: dict | None = None, vacuous: str | None = None) -> list[tuple[str, bool, str]]:
    """HARD gates.  Failing any of these means the file must not be submitted.

    Only three things are hard, because only three are either mechanically fatal or directly
    measured against labels:

    * schema clean -- an invalid record scores zero, whatever the model did;
    * no record outside a segment -- a stray timestamp cannot match anything;
    * E1b >= 88.0 on the model's OWN Segment-0 output -- the only labelled evidence available.

    The duration / gap / category-mix thresholds that used to live here were demoted to
    ``advisories`` because measurement showed they do not predict score.  Three concrete
    counter-examples: 9-minute windows scored 63.43 while 21-minute windows -- closer to the
    reference truth's 13.6 median -- scored 59.45; enforcing the FAQ's >=20 min injection gap
    cost 3.36 points on the anchor; and the frozen batch-2 shard has firewall share 0.734
    against a threshold of 0.34 yet scored 65.478 on the server.  Keeping them as hard gates
    would have blocked better-scoring files, and relaxing their numbers until they passed would
    just be moving the goalposts.
    """
    g = []
    g.append(("schema clean", not schema["invalid"] and schema["duplicate_prediction_id"] == 0,
              f"records={schema['records']} invalid={schema['invalid']} "
              f"dup_ids={schema['duplicate_prediction_id']}"))
    g.append(("no record outside a segment", shape["outside_any_segment"] == 0,
              f"outside={shape['outside_any_segment']} (segment membership uses the same "
              f"{SEG_TOL_MIN} min tolerance as assemble.py; a frozen shard renders window edges "
              f"with a -45 s offset)"))
    # E1b is scored on the model's own Segment-0 output, never on the shipped file when that
    # file's Segment 0 is a frozen shard.
    src = e1b if e1b is not None else score
    if vacuous and e1b is None:
        g.append(("E1b: Segment 0 total >= 88.0", False,
                  f"GATE VACUOUS -- {vacuous}; pass --e1b-file with the pure unified run"))
    elif src:
        g.append(("E1b: Segment 0 total >= 88.0", src["Total"] >= E1B_MIN,
                  f"Total={src['Total']:.4f} AD={src['AD']:.2f} RCA={src['RCA']:.2f} "
                  f"Major={src['Major']:.2f} Minor={src['Minor']:.2f} "
                  f"TP={src['TP']} FP={src['FP']} FN={src['FN']}"
                  + (f" [from --e1b-file; shipped Segment 0 is frozen: {vacuous}]"
                     if vacuous else "")))
    return g


def advisories(shape: dict) -> list[str]:
    """Shape statistics worth reading, but not submission blockers (see ``gates``)."""
    out = []
    for name in C.SEGMENT_ORDER:
        s = shape.get(name)
        if not s:
            out.append(f"{name}: NO RECORDS")
            continue
        out.append(f"{name}: n={s['n']} dur_p50={s['dur_p50']} mean={s['dur_mean']} "
                   f"outside_1_30={s['outside_1_30']} gap_min={s['gap_min']} "
                   f"gap_p50={s['gap_p50']} gap<20={s['gap_lt20']} "
                   f"fw_share={s['fw_share']} elems={s['elems']} majors={s['majors']}")
    return out


def decompose(score: dict, n_true: int) -> str:
    """Back out rho and the per-match quality terms -- the diagnosis, not just the total."""
    tp = score["TP"]
    e = tp + score["FP"]
    rho = tp / max(n_true, 1)
    a = score["AD"] / (40 * rho * score["alpha_fp"]) if rho and score["alpha_fp"] else 0.0
    r = score["RCA"] / (40 * rho) if rho else 0.0
    mj = score["Major"] / (10 * rho) if rho else 0.0
    sb = score["Minor"] / (10 * rho) if rho else 0.0
    return (f"rho={rho:.4f} (TP {tp}/{n_true})  E={e}  precision={score['precision']:.4f} "
            f"alpha_fp={score['alpha_fp']:.4f}\n"
            f"           per-match quality: a={a:.4f} r={r:.4f} mj={mj:.4f} sb={sb:.4f}")


def seg0_curve(pred_path: str, counts: list[int]) -> None:
    """Score Segment-0 prefixes of one answer file to locate the emission optimum.

    Selection walks peaks in descending fault probability and accepts or skips each one
    independently of the target, so the first K accepted Segment-0 records of a large-target run
    are *exactly* what a run targeted at K would have produced.  One inference therefore yields
    the whole emission curve, instead of one run per candidate count.

    This is the single most important curve in the system: rho multiplies all four score
    components while alpha_fp only scales AD and is floored at 0.7, so under-emitting is far
    more expensive than over-emitting, and the optimum has to be read off rather than assumed.
    """
    truth = C.gt_as_official()
    n_true = len(truth)
    lo = pd.Timestamp(C.BATCHES["seg0"]["grid_start"], tz="UTC")
    hi = pd.Timestamp(C.BATCHES["seg0"]["grid_end"], tz="UTC")
    with open(pred_path, encoding="utf-8") as f:
        recs = [json.loads(x) for x in f if x.strip()]
    seg0 = []
    for r in recs:                       # file order is descending p_event; filtering keeps it
        t = pd.Timestamp(r["start_time"])
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        if lo <= t < hi:
            seg0.append(r)
    print(f"[curve] {len(seg0)} Segment-0 records available (of {len(recs)} total), "
          f"{n_true} labelled truths")
    print(f"    {'E':>6s} {'Total':>8s} {'AD':>7s} {'RCA':>7s} {'Maj':>6s} {'Min':>6s} "
          f"{'TP':>5s} {'FP':>5s} {'rho':>6s} {'prec':>6s} {'a':>6s} {'r':>6s}")
    best = None
    for k in counts:
        if k > len(seg0):
            continue
        s = _evaluate_records(seg0[:k], truth)
        tp, e = s["TP"], k
        rho = tp / max(n_true, 1)
        a = s["AD"] / (40 * rho * s["alpha_fp"]) if rho and s["alpha_fp"] else 0.0
        r = s["RCA"] / (40 * rho) if rho else 0.0
        print(f"    {e:6d} {s['Total']:8.4f} {s['AD']:7.3f} {s['RCA']:7.3f} {s['Major']:6.3f} "
              f"{s['Minor']:6.3f} {tp:5d} {s['FP']:5d} {rho:6.4f} {s['precision']:6.4f} "
              f"{a:6.4f} {r:6.4f}")
        if best is None or s["Total"] > best[1]:
            best = (k, s["Total"])
    if best:
        print(f"[curve] best Segment-0 total {best[1]:.4f} at E={best[0]} "
              f"-> hedge = {best[0] / (C.SEG_MINUTES['seg0'] / 60.0):.3f}")


def _evaluate_records(records: list[dict], truth: list[dict]) -> dict:
    ensure_package()
    ev = importlib.import_module(f"{PKG}.evaluator.evaluator")
    return ev.evaluate(truth, records)


def edge_bias(pred_path: str) -> None:
    """Signed start/end error of matched records, in seconds.

    ``S_time = max(0, 1 - (|dStart| + |dEnd|) / 360s)`` forgives only ~3 minutes per edge, so a
    *systematic* offset is expensive in a way that random jitter is not: a constant +90 s on both
    edges costs 0.25 of S_time on every matched record, i.e. ~7.5 AD points at rho=0.7, while
    the same 90 s of zero-mean noise mostly lands inside the tolerance.  The sign therefore
    matters and is cheap to fix at render time, which is why this reports mean and median
    separately rather than just the absolute spread.

    Windows are built on a 1-minute grid and the reference sheet carries sub-minute timestamps,
    so some bias is expected by construction; this measures how much and in which direction.
    """
    truth = C.gt_as_official()
    res = official_evaluate(pred_path, truth, only_segment0=True)
    preds = {}
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                preds[r["prediction_id"]] = r
    by_id = {t["ground_truth_id"]: t for t in truth}
    ds, de, dl, tl = [], [], [], []
    for d in res["per_ground_truth"]:
        if not d.get("matched") or d["prediction_id"] not in preds:
            continue
        t = by_id.get(d["ground_truth_id"])
        p = preds[d["prediction_id"]]
        if t is None:
            continue
        f_ = lambda r, k: pd.Timestamp(r[k]).tz_convert("UTC") if pd.Timestamp(r[k]).tzinfo \
            else pd.Timestamp(r[k]).tz_localize("UTC")
        ds.append((f_(p, "start_time") - f_(t, "start_time")).total_seconds())
        de.append((f_(p, "end_time") - f_(t, "end_time")).total_seconds())
        dl.append((f_(p, "end_time") - f_(p, "start_time")).total_seconds() / 60.0)
        tl.append((f_(t, "end_time") - f_(t, "start_time")).total_seconds() / 60.0)
    if not ds:
        print("[edge-bias] no matched records")
        return
    a = lambda v: np.asarray(v, dtype=float)
    print(f"[edge-bias] {len(ds)} matched records (Segment 0)")
    for nm, v in (("dStart(s)", ds), ("dEnd(s)", de)):
        x = a(v)
        print(f"    {nm:10s} mean={x.mean():+8.1f} median={np.median(x):+8.1f} "
              f"p10={np.percentile(x, 10):+8.1f} p90={np.percentile(x, 90):+8.1f} "
              f"|mean|={np.abs(x).mean():7.1f}")
    print(f"    predicted len p50={np.median(a(dl)):.1f} min, truth len p50={np.median(a(tl)):.1f} min, "
          f"mean ratio={a(dl).mean() / max(a(tl).mean(), 1e-9):.3f}")
    print(f"    current S_time mean={res['AD'] / (40.0 * (res['TP'] / 305.0) * res['alpha_fp']):.4f}"
          f" -> a; implied mean |dStart|+|dEnd| = "
          f"{360.0 * (1.0 - ((res['AD'] / (40.0 * (res['TP'] / 305.0) * res['alpha_fp']) - 0.7) / 0.3)):.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--json", default=None, help="also write the report here (overwritten)")
    ap.add_argument("--seg0-curve", default=None, metavar="K1,K2,...",
                    help="score Segment-0 prefixes of this file at the given emission counts "
                         "and report the optimum; skips the normal report")
    ap.add_argument("--edge-bias", action="store_true",
                    help="report the signed start/end error of matched Segment-0 records")
    ap.add_argument("--e1b-file", default=None,
                    help="pure unified run (no frozen shard) to score the E1b gate on. "
                         "Required when the submitted file's Segment 0 is a frozen shard, "
                         "otherwise the gate would score that shard and pass vacuously.")
    args = ap.parse_args()

    if args.edge_bias:
        edge_bias(args.path)
        return 0
    if args.seg0_curve:
        seg0_curve(args.path, [int(x) for x in args.seg0_curve.split(",") if x.strip()])
        return 0

    print(f"[verify] {args.path}")
    sch = validate_schema(args.path)
    print(f"  schema: {sch}")
    df = load_records(args.path)
    shape = shape_report(df)
    print(f"  by segment: {shape['by_segment']}  outside={shape['outside_any_segment']}")
    print(f"  duration all: {shape['dur_all']}")
    for name in C.SEGMENT_ORDER:
        if name in shape:
            s = shape[name]
            print(f"  {name}: n={s['n']} dur_p50={s['dur_p50']} mean={s['dur_mean']} "
                  f"outside_1_30={s['outside_1_30']} gap_min={s['gap_min']} "
                  f"gap_p50={s['gap_p50']} gap<20={s['gap_lt20']} fw_share={s['fw_share']} "
                  f"elems={s['elems']}")
            print(f"        majors={s['majors']}")

    score = None
    if not args.no_score:
        n_true = sum(1 for l in open(C.GT_PATH, encoding="utf-8") if l.strip())
        score = official_evaluate(args.path)
        print(f"  [official] Segment 0 vs 96GT ({n_true} truths): "
              f"Total={score['Total']:.4f} AD={score['AD']:.4f} RCA={score['RCA']:.4f} "
              f"Major={score['Major']:.4f} Minor={score['Minor']:.4f}")
        print(f"             TP={score['TP']} FP={score['FP']} FN={score['FN']} "
              f"precision={score['precision']:.4f} alpha_fp={score['alpha_fp']:.4f}")
        print(f"  [decompose] {decompose(score, n_true)}")

    # E1b must be scored on the model's own Segment-0 output.  Detect the case where the shipped
    # file's Segment 0 is the frozen shard -- scoring that would pass the gate vacuously.
    vacuous = e1b_is_vacuous(args.path)
    e1b = None
    if vacuous:
        print(f"  [E1b] shipped Segment 0 is a frozen shard: {vacuous}")
        if args.e1b_file:
            e1b = official_evaluate(args.e1b_file)
            print(f"  [E1b] scored on the pure unified run {args.e1b_file}: "
                  f"Total={e1b['Total']:.4f} AD={e1b['AD']:.4f} RCA={e1b['RCA']:.4f} "
                  f"Major={e1b['Major']:.4f} Minor={e1b['Minor']:.4f} "
                  f"TP={e1b['TP']} FP={e1b['FP']} FN={e1b['FN']}")
            print(f"  [E1b decompose] {decompose(e1b, n_true)}")

    print("\n[gates]")
    fails = 0
    for name, ok, ev in gates(shape, score, sch, e1b=e1b, vacuous=vacuous):
        print(f"  {'PASS' if ok else 'FAIL'}  {name:38s} {ev}")
        fails += 0 if ok else 1
    print(f"\n[verify] {'PASSED' if fails == 0 else f'{fails} GATE(S) FAILED'}")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"path": args.path, "schema": sch, "shape": shape,
                       "score": {k: v for k, v in (score or {}).items()
                                 if k not in ("per_ground_truth", "unmatched_prediction_ids")},
                       "e1b": ({k: v for k, v in e1b.items()
                                if k not in ("per_ground_truth", "unmatched_prediction_ids")}
                               if e1b else None),
                       "e1b_vacuous": vacuous,
                       "gates": [{"name": n, "pass": bool(o), "evidence": e}
                                 for n, o, e in gates(shape, score, sch, e1b=e1b,
                                                      vacuous=vacuous)]}, f, indent=1)
        print(f"[verify] report -> {args.json}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
