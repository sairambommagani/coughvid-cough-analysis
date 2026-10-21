"""Step 1: read the metadata, select recordings, decode every audio file once and cache its features.

  python -m cough.prepare --raw data\\raw

Output (data/cache/): meta.parquet (labels, split), eda.json, features.parquet (handcrafted features),
logmel.npy (N x 64 x 250, float16) and frames.npy. Work is saved in parts of 1,000 recordings, so an interrupted
run continues where it stopped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .audio import load_audio
from .data import eda, read_metadata, select
from .features import FRAMES, N_MELS, analyse

PART = 1000


def process(path: str) -> dict:
    try:
        x = load_audio(path)
        if len(x) < 1600:
            raise ValueError("shorter than 0.1 s")
        a = analyse(x)
        f = a["features"]
        f.update(region_start=a["region"][0] / 16000, region_end=a["region"][1] / 16000, ok=1.0)
        return {"f": f, "mel": a["logmel"], "n": a["frames"], "err": ""}
    except Exception as e:                                          # broken / empty file: keep a row, mark it
        return {"f": {"ok": 0.0}, "mel": np.zeros((N_MELS, FRAMES), np.float16), "n": 0, "err": str(e)[:200]}


def run(raw: Path, out: Path, workers: int, max_noncough: int, limit: int | None = None, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    full = read_metadata(raw)
    log(f"metadata: {len(full):,} recordings with audio ({time.time() - t0:.0f}s)")
    sel = select(full, max_noncough)
    if limit:
        sel = sel.groupby("split", group_keys=False).apply(lambda g: g.head(max(1, limit * len(g) // len(sel))))
        sel = sel.reset_index(drop=True)
    e = eda(full, sel)
    log(f"selected {len(sel):,} recordings (cough_detected >= 0.8, plus {int((sel['y_cough'] == 0).sum()):,} "
        f"non-coughs); split {e['selected']['by_split']}")
    keep = [c for c in sel.columns if not any(c.startswith(f"{f}_") for f in
            ["quality", "cough_type", "dyspnea", "wheezing", "stridor", "choking", "congestion", "nothing",
             "diagnosis", "severity"])]
    meta = sel[keep].copy()
    meta["datetime"] = meta["datetime"].astype(str)
    for c in meta.columns:
        if meta[c].dtype == object or str(meta[c].dtype) == "string":
            meta[c] = meta[c].astype("string")
    meta.to_parquet(out / "meta.parquet", index=False)

    parts = out / "parts"
    parts.mkdir(exist_ok=True)
    paths = sel["path"].tolist()
    stamp = parts / "selection.txt"                                   # parts belong to this exact selection
    sig = f"{len(paths)}:{hashlib.md5(','.join(sel['uuid']).encode()).hexdigest()}"
    if stamp.exists() and stamp.read_text() != sig:
        for old in parts.glob("part_*"):
            old.unlink()
    stamp.write_text(sig)
    n_parts = (len(paths) + PART - 1) // PART
    with Pool(workers) as pool:
        for k in range(n_parts):
            fp = parts / f"part_{k:03d}.npz"
            if fp.exists():
                continue
            t1 = time.time()
            res = pool.map(process, paths[k * PART:(k + 1) * PART], chunksize=8)
            feats = pd.DataFrame([r["f"] for r in res])
            np.savez(fp, mel=np.stack([r["mel"] for r in res]), n=np.array([r["n"] for r in res]),
                     err=np.array([r["err"] for r in res]))
            feats.to_parquet(parts / f"part_{k:03d}.parquet", index=False)
            done = min((k + 1) * PART, len(paths))
            rate = (time.time() - t1) / max(len(res), 1)
            log(f"  decoded {done:,}/{len(paths):,} ({rate * 1000:.0f} ms per file, "
                f"about {(len(paths) - done) * rate / 60:.0f} min left)")

    feats = pd.concat([pd.read_parquet(parts / f"part_{k:03d}.parquet") for k in range(n_parts)], ignore_index=True)
    feats.insert(0, "uuid", sel["uuid"].values)
    mel = np.lib.format.open_memmap(out / "logmel.npy", mode="w+", dtype=np.float16, shape=(len(sel), N_MELS, FRAMES))
    frames, errors = np.zeros(len(sel), np.int32), []
    for k in range(n_parts):
        z = np.load(parts / f"part_{k:03d}.npz")
        s = k * PART
        mel[s:s + len(z["n"])] = z["mel"]
        frames[s:s + len(z["n"])] = z["n"]
        errors += [x for x in z["err"].tolist() if x]
    mel.flush()
    del mel
    np.save(out / "frames.npy", frames)
    feats.to_parquet(out / "features.parquet", index=False)
    ok = feats["ok"] == 1
    m = sel[ok.values]
    f = feats[ok]
    e["audio"] = {
        "decoded": int(ok.sum()), "failed": int((~ok).sum()), "example_errors": errors[:3],
        "no_cough_segment_found": int((f["n_coughs"] == 0).sum()),
        "coughs_per_recording_median": float(f["n_coughs"].median()),
        "snr_db_median": round(float(f["snr_db"].median()), 1),
        "duration_s_median": round(float(f["duration"].median()), 1),
        "by_status": {str(s): {"n": int((m["status"] == s).sum()),
                               "coughs_median": float(f.loc[(m["status"] == s).values, "n_coughs"].median()),
                               "snr_db_median": round(float(f.loc[(m["status"] == s).values, "snr_db"].median()), 1)}
                      for s in m["status"].dropna().unique()},
    }
    (out / "eda.json").write_text(json.dumps(e, indent=2, default=str))
    log(f"done in {(time.time() - t0) / 60:.1f} min; {e['audio']['decoded']:,} decoded, {e['audio']['failed']} failed")
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/cache")
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--max-noncough", type=int, default=3000)
    ap.add_argument("--limit", type=int, default=None, help="only N recordings (quick test)")
    a = ap.parse_args()
    e = run(Path(a.raw), Path(a.out), a.workers, a.max_noncough, a.limit)
    print(json.dumps({k: e[k] for k in ["recordings", "cough_detected", "status_of_coughs", "task_sizes",
                                         "expert_agreement"]}, indent=2))


if __name__ == "__main__":
    main()
