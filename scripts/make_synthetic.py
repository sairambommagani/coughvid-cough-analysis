"""Small synthetic COUGHVID-like dataset for tests: <uuid>.webm / .ogg (Opus, 48 kHz) + <uuid>.json +
metadata_compiled.csv with the real column names. Coughs are noise bursts with a sharp onset and a decaying tail;
"COVID-19" coughs are slightly more numerous and darker, "wet" coughs have a longer low-frequency tail, so the
pipeline has something to learn. Non-coughs are hum / speech-like tones with no bursts.

  python scripts/make_synthetic.py --out data/synthetic --n 300
"""
from __future__ import annotations

import argparse
import json
import subprocess
import uuid as uuidlib
from pathlib import Path

import numpy as np
import pandas as pd

SR = 48_000


def burst(rng, wet: bool, dark: bool) -> np.ndarray:
    dur = rng.uniform(0.25, 0.45) + (0.25 if wet else 0)
    n = int(dur * SR)
    t = np.arange(n) / SR
    env = np.minimum(t / 0.012, 1.0) * np.exp(-t / (0.12 if wet else 0.06))
    noise = rng.normal(size=n)
    k = 9 if dark else 3                                    # crude low-pass: darker sound
    noise = np.convolve(noise, np.ones(k) / k, mode="same")
    tone = np.sin(2 * np.pi * rng.uniform(150, 300) * t) * (0.6 if wet else 0.1)
    return (noise + tone) * env


def recording(rng, kind: str, wet: bool) -> np.ndarray:
    dur = rng.uniform(3, 9)
    x = rng.normal(scale=0.01, size=int(dur * SR))
    if kind == "noncough":
        t = np.arange(len(x)) / SR
        return x + 0.2 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    n = rng.integers(2, 5) + (1 if kind == "covid" else 0)
    pos = np.sort(rng.uniform(0.2, dur - 1.0, n))
    for p in pos:
        b = burst(rng, wet, dark=(kind == "covid"))
        s = int(p * SR)
        x[s:s + len(b)] += b[: len(x) - s] * rng.uniform(0.5, 1.0)
    return x


def write(out: Path, n: int = 300, seed: int = 0):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        u = str(uuidlib.UUID(int=int(rng.integers(0, 2 ** 62)) << 64 | i))
        r = rng.random()
        kind = "noncough" if r < 0.15 else ("covid" if r < 0.35 else ("symptomatic" if r < 0.5 else "healthy"))
        wet = bool(rng.random() < 0.4)
        x = recording(rng, kind, wet)
        x = (x / np.abs(x).max() * 0.8).astype(np.float32)
        ext = ".webm" if i % 2 == 0 else ".ogg"
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
               "-c:a", "libopus", "-b:a", "48k", str(out / f"{u}{ext}")]
        subprocess.run(cmd, input=x.tobytes(), check=True)
        status = {"covid": "COVID-19", "symptomatic": "symptomatic", "healthy": "healthy", "noncough": "healthy"}[kind]
        lat, lon = (round(float(rng.uniform(-40, 60)), 4), round(float(rng.uniform(-120, 120)), 4))
        row = {"uuid": u, "datetime": f"2020-{rng.integers(4, 13):02d}-{rng.integers(1, 28):02d}T10:00:00.000Z",
               "cough_detected": round(float(rng.uniform(0.85, 1.0) if kind != "noncough" else rng.uniform(0, 0.15)), 4),
               "SNR": round(float(rng.uniform(5, 30)), 2), "latitude": lat, "longitude": lon,
               "age": float(rng.integers(18, 80)), "gender": str(rng.choice(["male", "female"])),
               "respiratory_condition": bool(rng.random() < 0.2),
               "fever_muscle_pain": bool(rng.random() < (0.5 if kind == "covid" else 0.1)),
               "status": status if rng.random() > 0.1 else np.nan,
               "status_SSL": status if rng.random() > 0.2 else "symptomatic"}
        if kind != "noncough" and rng.random() < 0.4:            # expert-labelled subset, 2 experts
            for e in (1, 2):
                agree = rng.random() < 0.8
                row.update({f"quality_{e}": "good", f"cough_type_{e}": ("wet" if wet == agree else "dry"),
                            f"dyspnea_{e}": False, f"wheezing_{e}": bool(rng.random() < 0.1), f"stridor_{e}": False,
                            f"choking_{e}": False, f"congestion_{e}": wet, f"nothing_{e}": not wet,
                            f"diagnosis_{e}": str(rng.choice(["healthy_cough", "upper_infection", "COVID-19"])),
                            f"severity_{e}": str(rng.choice(["mild", "severe"]))})
        rows.append(row)
        js = {k: v for k, v in row.items() if k != "uuid" and not k[-2:-1] == "_" and not pd.isna(v)}
        (out / f"{u}.json").write_text(json.dumps(js))
    df = pd.DataFrame(rows)
    for e in (3, 4):                                         # v3 has columns for 4 experts
        for f in ["quality", "cough_type", "dyspnea", "wheezing", "stridor", "choking", "congestion", "nothing",
                  "diagnosis", "severity"]:
            df[f"{f}_{e}"] = np.nan
    df.to_csv(out / "metadata_compiled.csv")                 # the real file has an unnamed index column
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    write(Path(a.out), a.n)
    print(f"wrote {a.n} recordings to {a.out}")
