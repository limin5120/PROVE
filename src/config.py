"""Unified three-batch configuration: compact 50,400-minute axis, segments, official enums.

The competition releases three batches over the same 8-region / 10-role topology but with
different source sets and different calendar windows.  Historically each batch was cached,
calibrated and inferred in isolation, which made the per-batch statistics incomparable: the
`_level` conditional-quantile floor for `9|route` came out 12.86x larger on batch 3 than on
batch 1 while `0|route` came out 0.42x, so batch 3's normalised evidence lost its smooth
9-minute track and degraded to a 1-minute spike track.  Everything downstream (window width,
Dice gate, recall) followed.

This module therefore defines ONE canonical axis for all three batches:

    Segment 0  batch 1  2026-08-19 04:00 .. 2026-09-02 04:00   midx      0 .. 20159
    Segment 1  batch 2  2026-09-17 04:00 .. 2026-09-24 04:00   midx  20160 .. 30239
    Segment 2  batch 3  2026-10-01 04:00 .. 2026-10-15 04:00   midx  30240 .. 50399

Segments are contiguous in `midx` but NOT in wall-clock time, so every differencing, sliding
window and candidate window must be confined to one segment (`seg_of`, `clamp_to_segment`).
Segment 0 carries the only public answer sheet (``gt/96GT.jsonl``) and is the golden anchor:
calibration is computed on it alone and the other segments are mapped onto it.

Batch 1 also ships a probe table and FRR syslog that batches 2/3 lack.  They are deliberately
NOT part of the unified source set -- a segment-0-only feature is pure covariate shift for the
other two -- and batch 1's submission is protected separately by ``--freeze-stage1``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
CACHE_ROOT = os.path.join(PROJECT_ROOT, "cache")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")
MODEL_ROOT = os.path.join(PROJECT_ROOT, "models_unified")
LOG_ROOT = os.path.join(PROJECT_ROOT, "logs")
GT_PATH = os.path.join(PROJECT_ROOT, "gt", "96GT.jsonl")
FROZEN_DIR = os.path.join(PROJECT_ROOT, "frozen")
OFFICIAL_DIR = os.path.join(PROJECT_ROOT, "official_eval")

GRID_FREQ_MIN = 1

# ---------------------------------------------------------------- batch windows
# `inner` is the directory between the bundle directory and the CSVs; batch 1 nests one level
# deeper and names it after the bundle, batches 2/3 name it after the region.
BATCHES = {
    "seg0": {
        "data_subdir": "stage1", "inner": "{bundle}_data",
        "grid_start": "2026-08-19 04:00:00", "grid_end": "2026-09-02 04:00:00",
        "gt": True,
    },
    "seg1": {
        "data_subdir": "stage2", "inner": "{region}",
        "grid_start": "2026-09-17 04:00:00", "grid_end": "2026-09-24 04:00:00",
        "gt": False,
    },
    "seg2": {
        "data_subdir": "stage3", "inner": "{region}",
        "grid_start": "2026-10-01 04:00:00", "grid_end": "2026-10-15 04:00:00",
        "gt": False,
    },
}
SEGMENT_ORDER = ("seg0", "seg1", "seg2")


def _dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _n_minutes(spec: dict) -> int:
    return int((_dt(spec["grid_end"]) - _dt(spec["grid_start"])).total_seconds() // 60)


SEG_MINUTES = {k: _n_minutes(v) for k, v in BATCHES.items()}

# [(lo, hi, name)] over the canonical axis; hi is exclusive.
SEGMENT_BOUNDS: list[tuple[int, int, str]] = []
_cur = 0
for _name in SEGMENT_ORDER:
    _n = SEG_MINUTES[_name]
    SEGMENT_BOUNDS.append((_cur, _cur + _n, _name))
    _cur += _n
N_MINUTES = _cur                                    # 50400
ANCHOR_LO, ANCHOR_HI, ANCHOR_NAME = SEGMENT_BOUNDS[0]

# Per-segment k budget for the initial candidate selection.  Derived from the labelled
# stand-in: k=1200 over 14 days resolved to ~320 emitted records, and emission count is the
# largest single lever on the score (under-emitting costs ~10x more per record than
# over-emitting), so the budget scales with segment length rather than being uniform.
SELECT_K_PER_14D = 1280


def seg_of(midx) -> np.ndarray:
    """Segment id per minute index (vectorised)."""
    m = np.asarray(midx)
    out = np.zeros(np.shape(m), dtype=np.int64)
    for i, (lo, hi, _) in enumerate(SEGMENT_BOUNDS):
        out = np.where((m >= lo) & (m < hi), i, out)
    return out


def seg_bounds(seg_id: int) -> tuple[int, int, str]:
    return SEGMENT_BOUNDS[seg_id]


def clamp_to_segment(t0, t1, ref_m, pad: int = 0) -> tuple[int, int]:
    """Clip a [t0, t1) window to the segment containing ``ref_m`` (never straddle a batch)."""
    lo, hi, _ = SEGMENT_BOUNDS[int(seg_of(int(ref_m)))]
    return max(int(t0), lo + pad), min(int(t1), hi - pad)


def canonical_index() -> pd.DatetimeIndex:
    """Length-N_MINUTES UTC index: each segment keeps its own calendar window."""
    out = None
    for lo, hi, name in SEGMENT_BOUNDS:
        part = pd.date_range(BATCHES[name]["grid_start"], periods=hi - lo, freq="1min", tz="UTC")
        out = part if out is None else out.append(part)
    return out


GRID = None            # lazily built DatetimeIndex (see grid())
_GRID_TS = None        # int64 nanoseconds, for fast lookup


def grid() -> pd.DatetimeIndex:
    global GRID, _GRID_TS
    if GRID is None:
        GRID = canonical_index()
        _GRID_TS = GRID.asi8
    return GRID


def to_midx(ts: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Minute index on the canonical axis; -1 when the timestamp is outside every segment.

    Input timestamps are floored to the minute first.  Canonical minute ``m`` denotes the
    interval ``[m, m+1)``, so an event at 04:48:11.306 belongs to minute 04:48 -- and the
    labelled reference sheet carries sub-minute precision throughout
    (``2026-08-19T04:48:11.306Z``), so matching on exact equality would map *every* label to
    -1 and silently discard the whole answer sheet.  Flooring is a no-op for the already
    minute-aligned ``minute_utc`` column in the netflow archives.
    """
    t = pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).floor("1min").asi8
    g = _grid_ts()
    pos = np.clip(np.searchsorted(g, t), 0, N_MINUTES - 1)
    ok = g[pos] == t
    return np.where(ok, pos, -1).astype(np.int64)


def _grid_ts() -> np.ndarray:
    grid()
    return _GRID_TS


def to_utc(midx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(grid().asi8[np.asarray(midx, dtype=np.int64)]).tz_localize("UTC")


def fmt_utc(midx) -> str:
    """Official answer-file timestamp format: ISO-8601 UTC with millisecond precision."""
    return to_utc([int(midx)])[0].strftime("%Y-%m-%dT%H:%M:%S.000Z")


def fmt_end_utc(midx_excl) -> str:
    """Render an *exclusive* end minute as a wall-clock timestamp.

    A window is [t0, t1) in canonical minutes, so its end is the minute after the last included
    one.  Rendering ``grid[t1]`` instead is wrong at a segment edge: ``t1`` may equal the
    segment's exclusive bound, whose canonical timestamp belongs to the *next* batch.  That
    produced a record reading 2026-09-24 03:57 -> 2026-10-01 04:00, i.e. a 10,083-minute fault
    spanning the gap between batches 2 and 3, which is an automatic false positive.  Going via
    ``grid[t1 - 1] + 1 min`` stays inside the segment for every legal ``t1 <= hi``.
    """
    t = int(midx_excl)
    if t <= 0:
        return fmt_utc(0)
    ts = to_utc([t - 1])[0] + pd.Timedelta(minutes=1)
    return ts.strftime("%Y-%m-%dT%H:%M:%S.000Z")


# ---------------------------------------------------------------- official enums
def _load_official(name: str) -> dict:
    with open(os.path.join(OFFICIAL_DIR, name), encoding="utf-8") as f:
        return json.load(f)


_ne = _load_official("network_elements.json")
REGIONS: list[str] = list(_ne["cities"])
ROLES: list[str] = list(_ne["device_roles"])
ELEMENT_IDS: list[str] = [f"{c}-{r}" for c in REGIONS for r in ROLES]
ELEMENT_SET = frozenset(ELEMENT_IDS)

_tax = _load_official("fault_taxonomy.json")
MAJORS: list[str] = list(_tax["major_categories"])
CATEGORY_PAIRS: list[tuple[str, str]] = [
    (c["major_category"], c["sub_category"]) for c in _tax["fault_categories"]
]
PAIR_SET = frozenset(CATEGORY_PAIRS)

# Roles present on every region's submitter list; the topology/attribution layer maps the raw
# `node` field (br-1, service-vm-2, ...) onto `<region>-<role>` public ids.
def element_id(region: str, role: str) -> str:
    return f"{region}-{role}"


def split_element_id(eid: str) -> tuple[str, str]:
    """`beida-service-vm-1` -> (`beida`, `service-vm-1`); region is the first token."""
    for r in REGIONS:
        if eid.startswith(r + "-"):
            return r, eid[len(r) + 1:]
    raise ValueError(f"unknown region in element id {eid!r}")


_ROLE_MAJOR: dict | None = None


def role_major_prior() -> dict[str, str]:
    """Most likely major category per device role, derived from the labelled anchor at runtime.

    LABEL-FITTED, and deliberately *not* hardcoded: it is recomputed from ``gt/96GT.jsonl`` so
    that swapping the reference sheet re-derives it instead of silently going stale.

    Measured on the 305-interval anchor (purity = share of the role's most common major):
    br-1/br-2/cr-1/cr-2 -> routing (1.000), service-vm-2/3 -> resource (1.000),
    fw -> firewall (0.987), traffic-vm -> link (0.972), service-vm-1 -> service (0.629,
    66 service vs 39 resource).  Overall 0.866 of intervals get the right major from the role
    alone, which is why the category is constrained by role rather than left to the head: the
    head is trained only on true-root-cause rows but applied at inference to the rank-1 row of
    every emitted peak, most of which are not faults, and out of distribution it collapsed to
    84% firewall while only 24% of rank-1 roles were actually ``fw``.

    ``monitor-vm`` is absent: it never appears as a root cause in the anchor, and it is also the
    one role with no metric channels at all (only 9 of the 10 roles report), so its feature row
    is identically zero.  Emitting it as rank 1 is a guaranteed RCA miss -- it was rank 1 for
    163/1200 Segment-0 and 224/1200 Segment-2 records before being excluded.
    """
    global _ROLE_MAJOR
    if _ROLE_MAJOR is None:
        gt = load_gt()
        # groupby("role") then a single-level idxmax: indexing the MultiIndex series by role
        # already drops that level, so `.idxmax()` returns the major *name*.  Taking `[1]` of it
        # (as an earlier version did) silently yields the name's second character, which is not
        # in the taxonomy and makes every downstream category lookup fail.
        _ROLE_MAJOR = {role: sub["major"].value_counts().idxmax()
                       for role, sub in gt.groupby("role")}
        bad = {r: m for r, m in _ROLE_MAJOR.items() if m not in MAJORS}
        if bad:
            raise SystemExit(f"role_major_prior produced non-taxonomy majors: {bad}")
    return _ROLE_MAJOR


def subs_of(major: str) -> list[str]:
    """Official sub-categories belonging to one major, in taxonomy order."""
    return [s for m, s in CATEGORY_PAIRS if m == major]


# ---------------------------------------------------------------- source schemas
# Columns are declared explicitly: the raw CSVs mix encodings (batch-1 `region` is GBK-encoded
# Chinese while `routing_metrics` is UTF-8), so identity must never be taken from `region`.
# Region comes from the directory name, element identity from `node` / `node_key`.
SOURCES = ("node", "iface", "route", "scrape", "netflow")

NODE_METRICS = ("cpu_usage", "load1", "load5", "memory_available_ratio", "swap_used_ratio",
                "disk_read_rate", "disk_write_rate", "disk_io_util", "filesystem_used_ratio",
                "inode_used_ratio", "open_fd_ratio", "process_count")
IFACE_METRICS = ("rx_bytes_rate", "tx_bytes_rate", "rx_packets_rate", "tx_packets_rate",
                 "rx_drop_rate", "tx_drop_rate", "rx_error_rate", "tx_error_rate",
                 "carrier_changes")
SCRAPE_METRICS = ("scrape_up", "scrape_duration_seconds", "scrape_samples", "scrape_error_flag")

# Cumulative counters: the level is uninformative, only the per-minute increment is.  Verified
# against batch-1/2/3 routing metrics; differencing is confined to a segment (see build_cache).
ROUTE_COUNTER_METRICS = frozenset({"bgp_peer_uptime_seconds", "ospf6_neighbor_uptime_seconds"})

# Netflow aggregation grain and the service ports that are stable across batches.  Batch 3
# dropped 9100/9343 from its busy-port table and promoted ephemeral ports, so any feature
# keyed on a *ranked* port silently changes meaning between batches; these are absolute.
SVC_PRIMARY_PORTS = (9100, 9343, 8081, 8443, 3306, 5432, 6379)
NF_TRACKED_PORTS = (53, 80, 123, 179, 1053, 8080, 5201) + SVC_PRIMARY_PORTS
NF_CHUNK_ROWS = 4_000_000

# Evidence windows in minutes: 0 = instantaneous spike track, then progressively smoothed
# tracks.  The long tracks are what carry the 12..15 minute flat-top envelope typical of a real
# fault (truth duration p50 13.6, p75 21.2), so the ladder reaches 21; losing the long track to
# a mis-scaled denominator is exactly the batch-3 failure mode this design exists to prevent.
# w=1 is deliberately absent: a 1-minute centred mean is the identity, so it would duplicate
# w=0 and add a redundant calibration entry that always equals it.
WINDOWS = (0, 3, 9, 21)

# Channel kinds: how a raw column family is grouped for calibration and attribution.
CHANNEL_KINDS = ("node", "iface", "route", "scrape", "netflow")

MISS_EXCESS = 3.0          # envelope excess charged to a dense channel that stops reporting
# Evidence score threshold for a minute to become a candidate peak.
#
# MEASURED, not assumed (src/diag.py, Segment 0, 305 labelled intervals).  Coverage -- the
# fraction of true intervals containing at least one candidate peak -- is a hard ceiling on rho
# that no amount of model quality can raise, and it falls off steeply with this threshold:
#   thr 0.25 -> coverage 0.9803, pool 16447    thr 1.00 -> coverage 0.8393, pool 8430
#   thr 0.50 -> coverage 0.9475, pool 13340    thr 2.00 -> coverage 0.5869, pool 4341
# Lowering 1.0 -> 0.25 grows the pool only 1.95x while lifting the ceiling 0.839 -> 0.980, so
# the cheap end of the curve is the right place to sit: ranking is the model's job, and a fault
# that never enters the pool cannot be recovered by any ranker.
SCORE_THR = 0.25
NMS_RADIUS = 3             # peaks closer than this (minutes) collapse into the louder one
MAX_PEAKS_PER_SEG_REGION = 4000
MIN_LEN, MAX_LEN = 2, 30   # candidate window length bounds, minutes.  MAX_LEN is the organiser
                           # FAQ's stated upper bound on fault duration (1..30 min) -- a task
                           # rule, not a fitted constant, so it does not get re-derived per batch.
FIXED_LENGTHS = (4, 8, 12, 17, 22, 28)
CROSS_LEVEL = 0.5          # crossing window expands while intensity > CROSS_LEVEL * peak
DICE_GATE = 0.4            # official matching threshold (task rule, not a tuned constant)
T_MAX_S = 180              # official S_time half-tolerance in seconds (task rule)

# Emission economics, measured on the labelled anchor: emitting 21% too few cost 14.07 points
# while 18% too many cost 1.42.  The target therefore aims slightly above the estimated fault
# count rather than at it.
EMIT_HEDGE = 1.09


def stage_env() -> str:
    return os.environ.get("AIOPS_STAGE", "all")


# ---------------------------------------------------------------- labelled reference sheet
def load_gt() -> "pd.DataFrame":
    """The batch-1 reference sheet, normalised onto the canonical axis.

    ``gt/96GT.jsonl`` is published in *prediction* format (``prediction_id`` +
    ``root_cause_top5``), not the official ground-truth format (``ground_truth_id`` +
    ``root_cause``), so this accepts either and takes rank 1 as the true root cause.

    It is a reference answer, not the organisers' truth: it holds 305 intervals where batch 1
    actually contains 292 faults, which is why scores computed against it run optimistic --
    the same file scored 92.3469 locally and 90.0911 on the server.  Treat every threshold
    quoted against it as ~2.3 points high.
    """
    rows = []
    dropped = 0
    with open(GT_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            g = json.loads(line)
            m = to_midx([g["start_time"], g["end_time"]])
            if m[0] < 0 or m[1] < 0:
                dropped += 1                   # outside every segment: unusable on this axis
                continue
            if "root_cause" in g:
                eid = g["root_cause"]["network_element_id"]
            else:
                top = sorted(g["root_cause_top5"], key=lambda e: e.get("rank", 99))
                eid = top[0]["network_element_id"]
            region, role = split_element_id(eid)
            fc = g["fault_category"]
            rows.append({"m0": int(m[0]), "m1": int(m[1]), "region": region, "role": role,
                         "elem": eid, "major": fc["major_category"],
                         "sub": fc["sub_category"],
                         "gid": g.get("ground_truth_id") or g.get("prediction_id")})
    if not rows:
        # Never return an empty sheet quietly: a timestamp-parsing regression shows up here as
        # "no labels" rather than as a plausible-looking model trained on nothing.
        raise SystemExit(f"load_gt: all {dropped} reference intervals fell outside the canonical "
                         f"axis (N_MINUTES={N_MINUTES}); check to_midx / segment windows")
    if dropped:
        print(f"[load_gt] WARNING {dropped} interval(s) outside the canonical axis were dropped")
    return pd.DataFrame(rows)


def gt_as_official() -> list[dict]:
    """The reference sheet converted to the official ground-truth schema.

    The organiser evaluator runs ``validate_ground_truth`` on these records and rejects the
    prediction-format keys, so the conversion is mandatory before local scoring -- feeding the
    raw file raises instead of scoring.
    """
    out = []
    with open(GT_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            g = json.loads(line)
            if "root_cause" in g:
                out.append(g)
                continue
            top = sorted(g["root_cause_top5"], key=lambda e: e.get("rank", 99))
            out.append({"ground_truth_id": g.get("prediction_id", f"gt-{len(out):05d}"),
                        "start_time": g["start_time"], "end_time": g["end_time"],
                        "root_cause": {"network_element_id": top[0]["network_element_id"]},
                        "fault_category": g["fault_category"]})
    return out


def cache_dir(seg: str | None = None) -> str:
    """Per-segment cache root (``cache/seg0`` ...) or the unified root (``cache/all``)."""
    return os.path.join(CACHE_ROOT, seg if seg else "all")


def region_cache_dir(region: str, seg: str | None = None) -> str:
    return os.path.join(cache_dir(seg), region)


def bundles(region: str, seg: str) -> list[str]:
    """Bundle directory names for one region, in chronological order (discovered, not hardcoded)."""
    spec = BATCHES[seg]
    root = os.path.join(DATA_ROOT, spec["data_subdir"], region)
    if not os.path.isdir(root):
        return []
    out = [d for d in os.listdir(root)
           if os.path.isdir(os.path.join(root, d)) and d.startswith(f"{region}_")]
    return sorted(out)


def bundle_dir(region: str, seg: str, bundle: str) -> str:
    spec = BATCHES[seg]
    inner = spec["inner"].format(region=region, bundle=bundle)
    return os.path.join(DATA_ROOT, spec["data_subdir"], region, bundle, inner)


def source_paths(region: str, seg: str, source: str) -> list[str]:
    """Absolute CSV paths for one (region, segment, source), chronologically ordered."""
    out = []
    for b in bundles(region, seg):
        d = bundle_dir(region, seg, b)
        if source == "netflow":
            p = os.path.join(d, "netflow_5tuple_minute_readable.csv")
        else:
            p = os.path.join(d, f"{_FILE_PREFIX[source]}_{b[len(region) + 1:]}.csv")
        if os.path.exists(p):
            out.append(p)
        elif source != "netflow":
            # batch 1 names traffic/syslog without a bundle suffix; tolerate a plain name too
            alt = os.path.join(d, f"{_FILE_PREFIX[source]}.csv")
            if os.path.exists(alt):
                out.append(alt)
    return out


_FILE_PREFIX = {"node": "node_metrics", "iface": "interface_metrics",
                "route": "routing_metrics", "scrape": "scrape_health",
                "netflow": "netflow_5tuple_minute_readable"}


def describe() -> str:
    lines = [f"canonical axis: {N_MINUTES} minutes over {len(SEGMENT_BOUNDS)} segments"]
    for lo, hi, name in SEGMENT_BOUNDS:
        spec = BATCHES[name]
        n_reg = len([r for r in REGIONS if bundles(r, name)])
        lines.append(f"  {name}: midx {lo}..{hi - 1} ({hi - lo} min = {(hi - lo) / 1440:.0f} d) "
                     f"{spec['grid_start']} .. {spec['grid_end']}  data/{spec['data_subdir']}  "
                     f"regions_with_data={n_reg}  gt={spec['gt']}")
    lines.append(f"  select_k per segment: "
                 + ", ".join(f"{name}={round(SELECT_K_PER_14D * (hi - lo) / 20160)}"
                             for lo, hi, name in SEGMENT_BOUNDS))
    lines.append(f"official enums: {len(REGIONS)} regions x {len(ROLES)} roles "
                 f"= {len(ELEMENT_IDS)} elements; {len(MAJORS)} majors, "
                 f"{len(CATEGORY_PAIRS)} (major, sub) pairs")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
