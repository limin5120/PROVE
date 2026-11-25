"""Encoding-safe raw CSV access and schema discovery (L0).

Two facts about the released archives drive every decision here:

1. **Mixed encodings.** Batch 1's ``node_metrics`` / ``interface_metrics`` / ``netflow`` carry
   the Chinese ``region`` value in GBK bytes while ``routing_metrics`` / ``scrape_health`` are
   UTF-8.  Reading everything as UTF-8 therefore yields two different strings for the same
   region, and any join or groupby on that column silently splits one region in two.  So:
   encoding is detected per file, and **region identity is never taken from the file** -- it
   comes from the directory name, and element identity from ``node`` / ``node_key``.
2. **Element naming differs by source.** Metric files use ``br-1`` / ``service-vm-2`` while
   netflow uses ``node_key`` = ``br1`` (no hyphen) plus a composite ``node``.  Both are
   normalised to the official ``<city>-<role>`` public id here, once.

``python -m src.rawio --probe`` prints the discovered schema (distinct keys, label dimensions,
address space, all-zero columns) so a new batch can be characterised without reading code.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C  # noqa: E402

# Fields that are pure metadata (encoding-sensitive, non-numeric) and never used as features.
_DROP_TEXT = ("region", "node_type", "region_code", "if_role", "collector_port",
              "interface_id", "exporter_type", "target_id", "node", "node_key", "label",
              "metric_name", "src_addr", "dst_addr", "first_seen", "last_seen")


def detect_encoding(path: str, sample: int = 1 << 16) -> str:
    """UTF-8 (BOM-aware) when the head decodes, else GBK, else UTF-8 with replacement.

    Two traps here, both of which silently corrupt data rather than raising:

    * A UTF-8 BOM must be consumed, not decoded.  Read as plain ``utf-8`` the BOM becomes a
      zero-width prefix on the first column name (``\\ufefftimestamp``), so every later
      ``df["timestamp"]`` raises KeyError.
    * A fixed-size head sample can cut a multi-byte character in half.  The truncated tail then
      fails strict UTF-8 decoding and the file is misclassified as GBK -- and decoding a UTF-8
      archive as GBK turns the BOM plus the Chinese region value into mojibake.  Whether this
      fires depends on where byte 65536 happens to land, so it varies per file: batch 1 beida
      read fine while batch 1 shenyang did not.  An error whose start is within the last few
      bytes of the sample is therefore treated as truncation, not as evidence of GBK.
    """
    with open(path, "rb") as f:
        head = f.read(sample)
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    for enc in ("utf-8", "gbk"):
        try:
            head.decode(enc)
            return enc
        except UnicodeDecodeError as e:
            if e.start >= len(head) - 4:
                try:
                    head[:e.start].decode(enc)
                    return enc
                except UnicodeDecodeError:
                    pass
    return "utf-8"


def read_csv(path: str, usecols=None, **kw) -> pd.DataFrame:
    enc = detect_encoding(path)
    d = pd.read_csv(path, encoding=enc, encoding_errors="replace", usecols=usecols,
                    low_memory=False, **kw)
    # Safety net for any residual BOM/whitespace in the header, so key columns are addressable
    # by their documented names regardless of how the archive was written.
    d.columns = pd.Index([str(c).lstrip("\ufeff").strip() for c in d.columns])
    return d


def role_of(node: str) -> str | None:
    """Map a raw node label to an official device role (``br-1``, ``service-vm-2``, ...)."""
    if not isinstance(node, str):
        return None
    n = node.strip()
    if n in C.ROLES:
        return n
    low = n.lower()
    for r in C.ROLES:
        if low == r:
            return r
    # netflow `node_key` drops the hyphen: br1 -> br-1, servicevm2 -> service-vm-2
    compact = low.replace("-", "").replace("_", "")
    for r in C.ROLES:
        if compact == r.replace("-", ""):
            return r
    return None


def local_midx(ts, seg: str) -> np.ndarray:
    """Minute offset within one segment; -1 for timestamps outside that segment's window."""
    spec = C.BATCHES[seg]
    t = pd.to_datetime(ts, errors="coerce", utc=True)
    start = pd.Timestamp(spec["grid_start"], tz="UTC")
    n = C.SEG_MINUTES[seg]
    off = ((t - start).dt.total_seconds() // 60).fillna(-1).to_numpy(dtype=np.int64)
    off[t.isna().to_numpy() | (off < 0) | (off >= n)] = -1
    return off


def canonical_midx(ts, seg: str) -> np.ndarray:
    """``local_midx`` shifted onto the unified 50,400-minute axis."""
    lo = C.SEGMENT_BOUNDS[C.SEGMENT_ORDER.index(seg)][0]
    m = local_midx(ts, seg)
    return np.where(m < 0, -1, m + lo).astype(np.int64)


def to_num(s: pd.Series) -> np.ndarray:
    """Numeric coercion that treats the archives' sentinels (``NULL``, ``\\N``, ``NaN``) as missing."""
    if s.dtype.kind in "fiu":
        return s.to_numpy(dtype=np.float64)
    v = pd.to_numeric(s.astype("string").str.strip(), errors="coerce")
    return v.to_numpy(dtype=np.float64)


def _wide(df: pd.DataFrame, midx: np.ndarray, key_cols: list[str], metrics: list[str],
          n_min: int, agg: str = "last", key_fmt: str = "{k}|{m}",
          key_join: str = ".") -> tuple[pd.DataFrame, list[str]]:
    """Tidy frame -> (n_min x C) float32 wide table.

    One groupby for all metrics (a per-metric pivot is 15x slower on the routing archive), and
    the result is reindexed onto the *full* minute range -- a minute with no report is genuinely
    missing and must stay NaN, not shift every later row up by one.
    """
    d = df.assign(_m=midx)
    d = d[d["_m"] >= 0]
    mets = [m for m in metrics if m in d.columns]
    if not len(d) or not mets:
        return pd.DataFrame(), []
    keys = [d[c].astype("string").fillna("?") for c in key_cols]
    lab = keys[0] if len(keys) == 1 else keys[0] + key_join + keys[1].astype(str)
    num = pd.DataFrame({m: to_num(d[m]) for m in mets})
    num["_m"] = d["_m"].to_numpy()
    num["_k"] = lab.to_numpy()
    g = num.groupby(["_m", "_k"], sort=True)[mets].agg(agg)
    ks = g.index.get_level_values(1).unique()
    full = pd.RangeIndex(n_min)
    out, names = {}, []
    for m in mets:
        piv = g[m].unstack("_k").reindex(index=full, columns=ks)
        for k in ks:
            nm = key_fmt.format(k=k, m=m)
            out[nm] = piv[k].to_numpy(dtype=np.float32)
            names.append(nm)
    W = pd.DataFrame(out, index=pd.RangeIndex(n_min, name="midx"))
    return W, names


def read_node(region: str, seg: str) -> tuple[pd.DataFrame, list[str]]:
    n = C.SEG_MINUTES[seg]
    parts = [read_csv(p) for p in C.source_paths(region, seg, "node")]
    if not parts:
        return pd.DataFrame(), []
    df = pd.concat(parts, ignore_index=True)
    midx = local_midx(df["timestamp"], seg)
    return _wide(df, midx, ["node"], list(C.NODE_METRICS), n)


def read_iface(region: str, seg: str) -> tuple[pd.DataFrame, list[str]]:
    n = C.SEG_MINUTES[seg]
    parts = [read_csv(p) for p in C.source_paths(region, seg, "iface")]
    if not parts:
        return pd.DataFrame(), []
    df = pd.concat(parts, ignore_index=True)
    midx = local_midx(df["timestamp"], seg)
    return _wide(df, midx, ["node", "interface_id"], list(C.IFACE_METRICS), n)


def read_scrape(region: str, seg: str) -> tuple[pd.DataFrame, list[str]]:
    n = C.SEG_MINUTES[seg]
    parts = [read_csv(p) for p in C.source_paths(region, seg, "scrape")]
    if not parts:
        return pd.DataFrame(), []
    df = pd.concat(parts, ignore_index=True)
    midx = local_midx(df["timestamp"], seg)
    # `scrape_error` is free text (NULL when healthy); reduce it to a 0/1 flag so the wide
    # table stays numeric without inventing a vocabulary per batch.
    err = df["scrape_error"].astype("string").str.strip()
    df["scrape_error_flag"] = (~err.isin(["NULL", "", "nan", "None"]) & err.notna()).astype(float)
    return _wide(df, midx, ["node"], list(C.SCRAPE_METRICS), n, agg="max")


def read_route(region: str, seg: str) -> tuple[pd.DataFrame, list[str]]:
    n = C.SEG_MINUTES[seg]
    parts = [read_csv(p) for p in C.source_paths(region, seg, "route")]
    if not parts:
        return pd.DataFrame(), []
    df = pd.concat(parts, ignore_index=True)
    midx = local_midx(df["timestamp"], seg)
    # `label` carries extra dimensions (e.g. command="bgp_summary"); collapse onto the metric
    # with max, which is the right reducer for state/gauge semantics and keeps one column per
    # (node, metric) so the column vocabulary does not explode with label churn.
    df["_metric"] = df["metric_name"].astype("string").str.strip()
    # Name columns `<node>|<metric>` (not `<node>.<metric>|value`) so downstream counter
    # detection can read the metric name back off the column.
    return _wide(df, midx, ["node", "_metric"], ["value"], n, agg="max",
                 key_fmt="{k}", key_join="|")


# ---------------------------------------------------------------- netflow addressing
def parse_addr(a: str) -> tuple[str | None, str | None]:
    """``fd00:R:S::H`` -> (R, H).  R identifies the region block, H the host inside it."""
    if not isinstance(a, str) or not a.startswith("fd00:"):
        return None, None
    body = a[5:]
    left, _, right = body.partition("::")
    seg = left.split(":") if left else []
    host = right.split(":")[0] if right else None
    return (seg[0] if seg else None), host


def iter_netflow(region: str, seg: str, chunk: int = C.NF_CHUNK_ROWS, cols=None):
    """Stream the (very large) netflow archive in row chunks, yielding tidy numeric frames."""
    usecols = cols or ["minute_utc", "node_key", "node", "protocol", "src_addr", "src_port",
                       "dst_addr", "dst_port", "packets", "bytes", "flow_record_count",
                       "first_seen", "last_seen"]
    for p in C.source_paths(region, seg, "netflow"):
        for d in pd.read_csv(p, usecols=lambda c: c in usecols, chunksize=chunk,
                             encoding=detect_encoding(p), encoding_errors="replace",
                             low_memory=False):
            yield d


def probe(region: str, seg: str, only: str = "all") -> None:
    """Characterise one (region, segment): keys, label dimensions, address space, dead columns."""
    print(f"\n===== probe {region} / {seg} ({C.SEG_MINUTES[seg]} min) =====", flush=True)
    metric_srcs = () if only == "netflow" else (
        ("node", read_node), ("iface", read_iface),
        ("scrape", read_scrape), ("route", read_route))
    for src, fn in metric_srcs:
        W, names = fn(region, seg)
        if not len(W):
            print(f"  [{src:6s}] NO DATA")
            continue
        A = W.to_numpy(dtype=np.float64)
        nz = np.isfinite(A).mean(axis=0)
        alive = (np.nan_to_num(A) != 0).any(axis=0)
        dead = [n for n, a in zip(names, alive) if not a]
        nodes = sorted({n.split("|")[0].split(".")[0] for n in names})
        print(f"  [{src:6s}] shape={A.shape} cols={len(names)} nodes={len(nodes)} {nodes[:12]}")
        print(f"           finite_frac_min={nz.min():.4f} dead_cols={len(dead)} {dead[:8]}")
        if src == "route":
            mets = sorted({n.split("|")[-1] for n in names})
            print(f"           metrics({len(mets)})={mets}")
        if src == "iface":
            ifs = sorted({n.split("|")[0].split(".", 1)[1] for n in names if "." in n.split("|")[0]})
            print(f"           interfaces({len(ifs)})={ifs[:14]}")

    # netflow: sample-based characterisation (the archives are 20+ GB per region)
    keys, protos, sports, dports, rblocks, hosts, roles = (Counter() for _ in range(7))
    nrow = 0
    for d in iter_netflow(region, seg, chunk=400_000):
        nrow += len(d)
        keys.update(d["node_key"].astype(str))
        if "protocol" in d:
            protos.update(d["protocol"].astype(str))
        if "dst_port" in d:
            dports.update(d["dst_port"].astype(str).head(200_000))
        if "src_port" in d:
            sports.update(d["src_port"].astype(str).head(200_000))
        for a in d["dst_addr"].astype(str).head(200_000):
            r, h = parse_addr(a)
            if r:
                rblocks[r] += 1
                hosts[h] += 1
        if "node" in d:
            roles.update(str(x).split("-")[0].upper() for x in d["node"].astype(str))
        if nrow >= 1_200_000:
            break
    print(f"  [netflow] sampled_rows={nrow} node_key={dict(keys)}")
    print(f"            protocol={dict(protos.most_common(6))} node_role_prefix={dict(roles.most_common(8))}")
    print(f"            fd00_R_blocks={dict(rblocks.most_common(12))}")
    print(f"            hosts={dict(hosts.most_common(14))}")
    print(f"            top_dst_port={[p for p, _ in dports.most_common(18)]}")
    print(f"            top_src_port={[p for p, _ in sports.most_common(10)]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--only", default="all", choices=["all", "netflow"],
                    help="netflow: skip the metric sources (they are hundreds of MB each)")
    ap.add_argument("--region", default="beida", choices=C.REGIONS)
    ap.add_argument("--seg", default="seg2", choices=list(C.SEGMENT_ORDER))
    args = ap.parse_args()
    if args.probe:
        probe(args.region, args.seg, args.only)
        return 0
    print(C.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
