"""L0 cache builder: raw archives -> one unified per-region cache on the 50,400-minute axis.

Design notes that matter for correctness and cost:

* **Written straight onto the canonical axis.** Each row's timestamp is mapped to its segment
  and shifted by that segment's offset, so there is no per-batch cache to concatenate later and
  no intermediate files.  Segment membership is implicit in ``midx``; every downstream sliding
  window must still be confined to one segment (``config.clamp_to_segment``).
* **Netflow uses a fixed-size coded accumulator, not growing partial tables.** beida's batch-1
  archive alone is 21.5 GB (~100M+ rows).  Grain is
  ``(midx, node, dst_region, port_bucket)`` -> ``code``, accumulated with ``np.bincount`` into
  preallocated arrays, so peak memory is bounded (~0.7 GB) regardless of archive size.
* **Ports are an absolute, batch-invariant list.** Batch 1's busy ports are 1053/80/0/8080/9343/
  9100/179; batch 3's are 80/1053/8080/179/5201 with 9100/9343/0 gone and ephemeral ports
  promoted.  Anything keyed on a *ranked* port therefore silently changes meaning between
  batches, so the bucket list is fixed here and unknown ports fall into one `other` bucket.
* **Region identity never comes from the file.** The archives mix GBK and UTF-8 for the Chinese
  `region` value; region comes from the directory name and elements from `node` / `node_key`.
* **Cumulative counters are differenced inside their own segment only** (a fault reset is the
  signal; differencing across a batch boundary would be a huge spurious spike).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C, rawio as R  # noqa: E402

# netflow accumulation geometry (see module docstring)
NF_NODES = ("br-1", "br-2", "cr-1", "cr-2")       # only BR/CR roles export netflow
NF_NNODE = len(NF_NODES)
NF_NREG = len(C.REGIONS) + 1                      # 8 regions + one "external/unknown" bucket
NF_PORTS = tuple(sorted({0, *C.NF_TRACKED_PORTS}))
NF_NPORT = len(NF_PORTS) + 1                      # + one "other" bucket
_PORT_IDX = {p: i for i, p in enumerate(NF_PORTS)}
NF_SIZE = C.N_MINUTES * NF_NNODE * NF_NREG * NF_NPORT
REGION_CODE = {r: i for i, r in enumerate(C.REGIONS)}
EXT_CODE = len(C.REGIONS)


def _metric_frame(W: pd.DataFrame, names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Sort columns deterministically so the vocabulary is stable across regions and reruns."""
    names = sorted(names)
    return W.reindex(columns=names).astype(np.float32), names


def _difference_counters(W: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    """Replace genuine cumulative counters with their per-minute increment.

    ``W`` is segment-local (index 0..n_seg-1), so differencing never straddles a batch boundary.

    Which columns to difference is decided by Prometheus naming *and* empirical monotonicity,
    because both a name-only and a monotonicity-only rule are wrong here:

    * A name rule on ``_count`` would catch the gauges ``bgp_peer_count`` / ``ipv6_route_count``.
      A fault shows up there as a *drop*, and differencing plus clipping negatives would erase
      the strongest routing signal in the archive.
    * A pure monotonicity rule would catch ``bgp_peer_uptime_seconds``, which also must NOT be
      differenced: its information is the level falling back to ~0 when a session flaps.  The
      differenced series is a constant 60 with the reset clipped away -- pure loss.

    So only ``*_total`` event counters qualify (their per-minute increment *is* the churn
    signal), and each still has to verify as non-decreasing in this segment before it is
    differenced.
    """
    metric = pd.Index([c.rsplit("|", 1)[-1] for c in W.columns])
    cand = np.flatnonzero(np.asarray(metric.str.endswith("_total")))
    if not len(cand):
        return W
    sub = W.to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        d = np.diff(sub, axis=0)
        finite = np.isfinite(d)
        nonneg = (np.nan_to_num(d, nan=0.0) >= 0) & finite
        n_fin = finite.sum(axis=0)
        frac_nonneg = np.where(n_fin > 0, nonneg.sum(axis=0) / np.maximum(n_fin, 1), 0.0)
    idx = cand[frac_nonneg[cand] >= 0.99]
    if not len(idx):
        log.append(f"    {len(cand)} *_total columns failed the monotonicity check; left as levels")
        return W
    out = W.copy()
    dd = np.zeros_like(sub)
    dd[1:] = d
    dd[dd < 0] = 0.0                      # a counter reset is carried by the level channel
    out.iloc[:, idx] = np.nan_to_num(dd)[:, idx].astype(np.float32)
    log.append(f"    differenced {len(idx)} cumulative counters "
               f"({sorted(set(metric[idx]))})")
    return out


def build_metrics(region: str, seg: str, log: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for src, fn, metrics in (("node", R.read_node, C.NODE_METRICS),
                             ("iface", R.read_iface, C.IFACE_METRICS),
                             ("scrape", R.read_scrape, C.SCRAPE_METRICS),
                             ("route", R.read_route, None)):
        t0 = time.time()
        W, names = fn(region, seg)
        if not len(W) or not names:
            log.append(f"    [{src:6s}] NO DATA")
            continue
        W = W.reindex(columns=names).astype(np.float32)
        W, names = _metric_frame(W, names)
        if src == "route":
            W = _difference_counters(W, log)
        # canonical axis: place this segment's rows at their unified offset
        A = np.zeros((C.N_MINUTES, len(names)), dtype=np.float32)
        lo = C.SEGMENT_BOUNDS[C.SEGMENT_ORDER.index(seg)][0]
        n_seg = C.SEG_MINUTES[seg]
        A[lo:lo + n_seg] = W.to_numpy(dtype=np.float32)
        del W
        df = pd.DataFrame(A, index=pd.RangeIndex(C.N_MINUTES, name="midx"), columns=names)
        # NaN means "this segment does not carry this column"; 0 means "present and quiet".
        # Keep the distinction in a companion mask column count, but store finite values only.
        nan_frac = float(np.isnan(df.to_numpy()).mean())
        out[src] = df
        keys = sorted({c.split("|")[0] for c in names})
        mets = sorted({c.rsplit("|", 1)[-1] for c in names})
        log.append(f"    [{src:6s}] {df.shape} keys={len(keys)} metrics={len(mets)} "
                   f"nan_frac={nan_frac:.4f} {time.time() - t0:.1f}s")
        log.append(f"             keys={keys[:12]}")
        if src in ("route", "scrape"):
            log.append(f"             metrics={mets}")
        del A, df
        gc.collect()
    return out


_PORTS_ARR = np.array(NF_PORTS, dtype=np.int64)
_OTHER_PORT = len(NF_PORTS)
# `minute_utc` and `node_key` are very low cardinality (<=4320 minutes per bundle, 4 nodes), so
# factorising them and parsing only the uniques is an order of magnitude cheaper than parsing
# ~1.1e9 rows.  `dst_addr` is high cardinality, but its region hextet is not: slicing a fixed
# 4-char window and factorising that keeps the Python-level work to ~100 values per chunk.
_NF_USECOLS = ("minute_utc", "node_key", "dst_addr", "dst_port",
               "packets", "bytes", "flow_record_count")


def _region_code_of_prefix(pref: str) -> int:
    """`fd00:X:Y::Z` -> region code from a 4-char slice starting at X ('1:20', '100:', '7:61')."""
    x = pref.split(":", 1)[0]
    r = _region_of_code(x)
    return REGION_CODE[r] if r is not None else EXT_CODE


def build_netflow(region: str, log: list[str]) -> pd.DataFrame:
    """Stream every segment's netflow archive into one fixed-size coded accumulator.

    Fully vectorised per chunk: three ``np.bincount`` passes over an integer code, no growing
    intermediate tables and no per-row Python.  Peak memory is the accumulator itself.
    """
    rec = np.zeros(NF_SIZE, dtype=np.float64)
    byt = np.zeros(NF_SIZE, dtype=np.float64)
    pkt = np.zeros(NF_SIZE, dtype=np.float64)
    node_idx = {n: i for i, n in enumerate(NF_NODES)}
    minute_lut: dict = {}
    total_rows = kept_rows = 0
    t0 = time.time()
    # Test hook: process at most N chunks per segment so the coding geometry can be validated
    # in a couple of minutes instead of after a two-hour full run.
    max_chunks = int(os.environ.get("AIOPS_NF_MAX_CHUNKS", "0") or 0)
    for seg in C.SEGMENT_ORDER:
        seg_rows = 0
        n_chunk = 0
        for d in R.iter_netflow(region, seg, chunk=C.NF_CHUNK_ROWS, cols=list(_NF_USECOLS)):
            n_chunk += 1
            n = len(d)
            seg_rows += n
            total_rows += n

            mc, mu = pd.factorize(d["minute_utc"])
            # Parse only the minutes this chunk has not seen before.  Checking a single sentinel
            # here would silently drop every row whose minute was missing from the cache.
            missing = [u for u in mu if u not in minute_lut]
            if missing:
                vals = C.to_midx(pd.to_datetime(list(missing), utc=True))
                minute_lut.update(zip(missing, (int(v) for v in vals)))
            umidx = np.array([minute_lut.get(u, -1) for u in mu], dtype=np.int64)
            midx = umidx[mc]

            nc, nu = pd.factorize(d["node_key"])
            unode = np.array([node_idx.get(R.role_of(u) or "", -1) for u in nu], dtype=np.int64)
            role = unode[nc]

            pc, pu = pd.factorize(d["dst_addr"].str.slice(5, 9))
            ureg = np.array([_region_code_of_prefix(u if isinstance(u, str) else "")
                             for u in pu], dtype=np.int64)
            dcode = ureg[pc]

            pv = pd.to_numeric(d["dst_port"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
            si = np.searchsorted(_PORTS_ARR, pv)
            hit = (si < len(_PORTS_ARR)) & (_PORTS_ARR[np.clip(si, 0, len(_PORTS_ARR) - 1)] == pv)
            pcode = np.where(hit, si, _OTHER_PORT)

            ok = (midx >= 0) & (role >= 0)
            if ok.any():
                code = (((midx[ok] * NF_NNODE + role[ok]) * NF_NREG + dcode[ok]) * NF_NPORT
                        + pcode[ok])
                num = lambda col: pd.to_numeric(d[col], errors="coerce").fillna(0.0).to_numpy()[ok]  # noqa: E731
                rec += np.bincount(code, weights=num("flow_record_count"), minlength=NF_SIZE)
                byt += np.bincount(code, weights=num("bytes"), minlength=NF_SIZE)
                pkt += np.bincount(code, weights=num("packets"), minlength=NF_SIZE)
                kept_rows += int(ok.sum())
            del d, midx, role, dcode, pcode, pv, si, hit, ok
            if max_chunks and n_chunk >= max_chunks:
                log.append(f"    [netflow] {seg}: stopped after {n_chunk} chunk(s) "
                           f"(AIOPS_NF_MAX_CHUNKS test hook)")
                break
        log.append(f"    [netflow] {seg}: {seg_rows:,} rows, {time.time() - t0:.0f}s elapsed")
        gc.collect()

    nz = np.flatnonzero(rec > 0)
    log.append(f"    [netflow] {region}: {total_rows:,} rows -> {len(nz):,} cells "
               f"(in-grid {100.0 * kept_rows / max(total_rows, 1):.2f}%) {time.time() - t0:.1f}s")
    p = nz % NF_NPORT
    r = (nz // NF_NPORT) % NF_NREG
    nd = (nz // (NF_NPORT * NF_NREG)) % NF_NNODE
    m = nz // (NF_NPORT * NF_NREG * NF_NNODE)
    return pd.DataFrame({
        "midx": m.astype(np.int32),
        "node": np.array(NF_NODES)[nd],
        "dst_region": r.astype(np.int8),
        "port": np.array([*NF_PORTS, -1], dtype=np.int32)[p],
        "records": rec[nz].astype(np.float32),
        "bytes": byt[nz].astype(np.float32),
        "packets": pkt[nz].astype(np.float32),
    })


def _region_of_code(x) -> str | None:
    """fd00 first hextet -> region name.  Verified: X = cities.index(region) + 1, 100 = external."""
    try:
        i = int(x)
    except (TypeError, ValueError):
        return None
    if 1 <= i <= len(C.REGIONS):
        return C.REGIONS[i - 1]
    return None


def build_region(region: str, force: bool = False, log=None):
    """Build one region's unified cache.  ``log`` may be any object with ``.append`` --
    passing an ``Emitter`` streams progress live instead of buffering until the region ends."""
    if log is None:
        log = []
    out_dir = C.region_cache_dir(region)
    os.makedirs(out_dir, exist_ok=True)
    done = os.path.join(out_dir, "index.json")
    if os.path.exists(done) and not force:
        log.append(f"  [{region}] cache exists, skipping (use --force to rebuild)")
        return log
    t0 = time.time()
    log.append(f"  [{region}] building")
    mets = build_metrics(region, C.SEGMENT_ORDER[0], log)
    for seg in C.SEGMENT_ORDER[1:]:
        part = build_metrics(region, seg, log)
        for k, df in part.items():
            if k in mets:
                cols = sorted(set(mets[k].columns) | set(df.columns))
                a = mets[k].reindex(columns=cols).to_numpy(dtype=np.float32)
                b = df.reindex(columns=cols).to_numpy(dtype=np.float32)
                lo = C.SEGMENT_BOUNDS[C.SEGMENT_ORDER.index(seg)][0]
                n = C.SEG_MINUTES[seg]
                a[lo:lo + n] = b[lo:lo + n]
                mets[k] = pd.DataFrame(a, index=mets[k].index, columns=cols)
                del a, b
            else:
                mets[k] = df
        del part
        gc.collect()
    for src, df in mets.items():
        # Keep NaN.  It marks two distinct things that a 0-fill would erase: a column this
        # segment does not have at all (presence 0 across the segment), and a minute where a
        # normally-reporting channel went quiet.  The evidence layer turns both into excess 0
        # (i.e. "no anomaly"), so the value of preserving NaN is that a *later* change can
        # distinguish them -- exporter failure is already carried explicitly by the
        # `scrape_up` and `scrape_error_flag` columns rather than inferred from gaps.
        df = df.astype(np.float32)
        p = os.path.join(out_dir, f"{src}.parquet")
        df.to_parquet(p)
        log.append(f"    wrote {src}.parquet {df.shape} "
                   f"nan_frac={float(np.isnan(df.to_numpy()).mean()):.4f} "
                   f"{os.path.getsize(p) / 1e6:.1f} MB")
        del df
        gc.collect()
    nf = build_netflow(region, log)
    nf.to_parquet(os.path.join(out_dir, "netflow.parquet"), index=False)
    log.append(f"    wrote netflow.parquet {nf.shape} "
               f"{os.path.getsize(os.path.join(out_dir, 'netflow.parquet')) / 1e6:.1f} MB")
    del nf
    gc.collect()
    with open(done, "w", encoding="utf-8") as f:
        json.dump({"region": region, "n_minutes": C.N_MINUTES,
                   "segments": {s: C.SEG_MINUTES[s] for s in C.SEGMENT_ORDER},
                   "sources": sorted(mets) + ["netflow"],
                   "elapsed_s": round(time.time() - t0, 1)}, f, indent=1)
    log.append(f"  [{region}] done in {time.time() - t0:.1f}s")
    return log


class Emitter:
    """List-like sink that streams each line to stdout and the log file immediately.

    ``build_region`` accumulates log lines in a list and returns them, which means a multi-hour
    run shows nothing until a region completes -- and when a region dies part-way there is no
    record of how far it got.  Passing this instead keeps every existing ``log.append(...)``
    call site working while making progress (and the failure point) visible live.
    """

    def __init__(self, fh=None):
        self.fh = fh

    def append(self, s: str) -> None:
        print(s, flush=True)
        if self.fh is not None:
            self.fh.write(s + "\n")
            self.fh.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="*", default=None, choices=C.REGIONS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(C.LOG_ROOT, exist_ok=True)
    log_path = os.path.join(C.LOG_ROOT, "build_cache.log")
    regions = args.regions or C.REGIONS
    t_start = time.time()
    # Single overwritten log: one line per step, plus the discovered schema so a rerun can be
    # compared against history without keeping per-run files around.
    with open(log_path, "w", encoding="utf-8", buffering=1) as lf:
        em = Emitter(lf)
        em.append(f"[build_cache] {time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"regions={len(regions)} axis={C.N_MINUTES} min force={args.force}")
        em.append(f"[build_cache] netflow geometry: nodes={NF_NODES} dst_regions={NF_NREG} "
                  f"ports={NF_PORTS}+other size={NF_SIZE:,}")
        for i, region in enumerate(regions, 1):
            em.append(f"=== {i}/{len(regions)} {region} "
                      f"(elapsed {time.time() - t_start:.0f}s)")
            build_region(region, args.force, em)
        em.append(f"[build_cache] ALL DONE in {time.time() - t_start:.1f}s -> {C.cache_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
