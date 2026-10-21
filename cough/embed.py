"""Step 2: PANNs CNN14 embeddings of every cough region (run after cough.prepare).

  python -m cough.embed --checkpoint models\\Cnn14_mAP=0.431.pth

Output (data/cache/): panns_emb.npy (N x 2048, float16) and panns_tags.npy (N x 7: AudioSet probabilities of
speech, breathing, wheeze, cough, throat clearing, sneeze, silence). Saved in parts; an interrupted run continues.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .audio import load_audio
from .panns import AUDIOSET, SR, embed, load

PART, BATCH = 1000, 8


def _wave(args) -> np.ndarray:
    path, s, e = args
    try:
        x = load_audio(path, sr=SR)
        w = x[int(s * SR):int(e * SR)] if e > s else x[: 5 * SR]
        return w if len(w) >= SR // 4 else x[: 5 * SR]
    except Exception:
        return np.zeros(SR, np.float32)


def run(cache: Path, checkpoint: Path, threads: int, log=print):
    meta = pd.read_parquet(cache / "meta.parquet")
    feats = pd.read_parquet(cache / "features.parquet")
    model = load(checkpoint)
    torch.set_num_threads(threads)
    parts = cache / "panns_parts"
    parts.mkdir(exist_ok=True)
    jobs = list(zip(meta["path"], feats.get("region_start", pd.Series(0.0, index=feats.index)).fillna(0),
                    feats.get("region_end", pd.Series(0.0, index=feats.index)).fillna(0)))
    n_parts = (len(jobs) + PART - 1) // PART
    t0 = time.time()
    tag_idx = list(AUDIOSET.values())
    with ThreadPoolExecutor(4) as pool:
        for k in range(n_parts):
            fp = parts / f"part_{k:03d}.npz"
            if fp.exists():
                continue
            t1 = time.time()
            waves = list(pool.map(_wave, jobs[k * PART:(k + 1) * PART]))
            order = np.argsort([len(w) for w in waves])                   # similar lengths -> little padding
            E = np.zeros((len(waves), 2048), np.float32)
            T = np.zeros((len(waves), len(tag_idx)), np.float32)
            for b in range(0, len(order), BATCH):
                idx = order[b:b + BATCH]
                e, t = embed(model, [waves[i] for i in idx])
                E[idx], T[idx] = e, t[:, tag_idx]
            np.savez(fp, emb=E.astype(np.float16), tags=T)
            done = min((k + 1) * PART, len(jobs))
            rate = (time.time() - t1) / max(len(waves), 1)
            log(f"  embedded {done:,}/{len(jobs):,} ({rate * 1000:.0f} ms per file, "
                f"about {(len(jobs) - done) * rate / 60:.0f} min left)")
    E = np.concatenate([np.load(parts / f"part_{k:03d}.npz")["emb"] for k in range(n_parts)])
    T = np.concatenate([np.load(parts / f"part_{k:03d}.npz")["tags"] for k in range(n_parts)])
    np.save(cache / "panns_emb.npy", E)
    np.save(cache / "panns_tags.npy", T)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--checkpoint", default="models/Cnn14_mAP=0.431.pth")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    run(Path(a.cache), Path(a.checkpoint), a.threads)


if __name__ == "__main__":
    main()
