"""Small synthetic COUGHVID-like dataset for tests: <uuid>.webm / .ogg (Opus, 48 kHz) + <uuid>.json +
metadata_compiled.csv with the real column names. With --formats the recordings are written in other formats too
(wav, mp3, m4a, mp4, flac, aac, ogg-vorbis...), round-robin, for the format matrix tests. Coughs are noise bursts with a sharp onset and a decaying tail;
"COVID-19" coughs are slightly more numerous and darker, "wet" coughs have a longer low-frequency tail, so the
pipeline has something to learn. Non-coughs are hum / speech-like tones with no bursts.

  python scripts/make_synthetic.py --out data/synthetic --n 300
  python scripts/make_synthetic.py --out data/synthetic_formats --n 60 --formats all

Fixes: uses the same ffmpeg as the pipeline (PATH, else the copy bundled with imageio-ffmpeg; it used to call
"ffmpeg" directly and failed on machines without it on the PATH), checks that every file it writes decodes again,
and can write single samples in any format (sample()).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import uuid as uuidlib
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cough.audio import ffmpeg_exe                                     # noqa: E402

SR = 48_000

# name -> (file extension, ffmpeg encoder arguments). The names are what real recorders and apps produce.
FORMATS = {
    "webm_opus": (".webm", ["-c:a", "libopus", "-b:a", "48k"]),           # Chrome / Firefox / Edge MediaRecorder
    "ogg_opus": (".ogg", ["-c:a", "libopus", "-b:a", "48k"]),             # Firefox, Android, voice notes
    "ogg_vorbis": (".ogg", ["-c:a", "libvorbis", "-q:a", "4"]),
    "mp4_aac": (".mp4", ["-c:a", "aac", "-b:a", "96k"]),                  # Safari / iPhone MediaRecorder
    "m4a_aac": (".m4a", ["-c:a", "aac", "-b:a", "96k"]),                  # iPhone voice memos
    "aac_adts": (".aac", ["-c:a", "aac", "-b:a", "96k", "-f", "adts"]),
    "mp3": (".mp3", ["-c:a", "libmp3lame", "-b:a", "96k"]),
    "wav_pcm16": (".wav", ["-c:a", "pcm_s16le"]),
    "wav_8k": (".wav", ["-c:a", "pcm_s16le", "-ar", "8000"]),             # telephone-quality audio
    "flac": (".flac", ["-c:a", "flac"]),
}
DEFAULT_FORMATS = ("webm_opus", "ogg_opus")                               # what COUGHVID itself contains


def encode(x: np.ndarray, path: Path, fmt: str, sr: int = SR) -> Path:
    """Write float audio `x` to `path` in format `fmt` (a FORMATS key) with ffmpeg."""
    exe = ffmpeg_exe()
    if exe is None:
        raise RuntimeError("ffmpeg not found: pip install imageio-ffmpeg")
    ext, args = FORMATS[fmt]
    path = Path(path).with_suffix(ext)
    cmd = [exe, "-nostdin", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-", *args, str(path)]
    r = subprocess.run(cmd, input=np.asarray(x, np.float32).tobytes(), capture_output=True)
    if r.returncode != 0 or not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg could not write {fmt}: {r.stderr.decode(errors='ignore')[:200]}")
    return path


def sample(path: Path, fmt: str = "webm_opus", kind: str = "healthy", seed: int = 0, wet: bool = False) -> Path:
    """One synthetic recording (cough, or kind="noncough") in any format, for tests and demos."""
    rng = np.random.default_rng(seed)
    x = recording(rng, kind, wet)
    return encode((x / np.abs(x).max() * 0.8).astype(np.float32), Path(path), fmt)


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


def write(out: Path, n: int = 300, seed: int = 0, formats=DEFAULT_FORMATS, verify: bool = False):
    """verify=True decodes every written file again with the pipeline's loader (slower, used by the tests)."""
    out.mkdir(parents=True, exist_ok=True)
    formats = list(FORMATS) if formats == "all" else list(formats)
    unknown = [f for f in formats if f not in FORMATS]
    if unknown:
        raise ValueError(f"unknown formats {unknown}; choose from {list(FORMATS)}")
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        u = str(uuidlib.UUID(int=int(rng.integers(0, 2 ** 62)) << 64 | i))
        r = rng.random()
        kind = "noncough" if r < 0.15 else ("covid" if r < 0.35 else ("symptomatic" if r < 0.5 else "healthy"))
        wet = bool(rng.random() < 0.4)
        x = recording(rng, kind, wet)
        x = (x / np.abs(x).max() * 0.8).astype(np.float32)
        f = encode(x, out / u, formats[i % len(formats)])
        if verify:
            from cough.audio import load_audio
            assert len(load_audio(f)) > 0, f"{f.name} does not decode"
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
    ap.add_argument("--formats", default=",".join(DEFAULT_FORMATS),
                    help=f"comma-separated list or 'all'; available: {', '.join(FORMATS)}")
    ap.add_argument("--verify", action="store_true", help="decode every written file again")
    a = ap.parse_args()
    write(Path(a.out), a.n, formats="all" if a.formats == "all" else a.formats.split(","), verify=a.verify)
    print(f"wrote {a.n} recordings to {a.out}")
