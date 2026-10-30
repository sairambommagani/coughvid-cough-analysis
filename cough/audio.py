"""Audio loading and cough segmentation.

COUGHVID recordings are .webm or .ogg files (Opus codec, 48 kHz, up to 10 s) recorded on phones and laptops.
They are decoded with ffmpeg to mono float32 at a fixed sample rate and peak-normalised (devices record at very
different volumes).

Cough segmentation follows the idea of the COUGHVID authors' preprocessing (hysteresis on the signal energy):
a cough starts when the short-time energy rises above a high threshold and ends when it falls below a low one.
Short gaps are merged, very short events dropped, and each cough is padded a little so its onset and tail are kept.
The SNR is the energy of the cough parts relative to the rest of the recording.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

SR = 16_000
FFMPEG_TIMEOUT_S = 30
AUDIO_EXT = (".webm", ".ogg", ".wav", ".mp3", ".m4a", ".mp4", ".aac", ".flac", ".3gp", ".amr", ".aiff", ".opus")


def ffmpeg_exe() -> str | None:
    """ffmpeg on the PATH, otherwise the copy bundled with the imageio-ffmpeg package (pip install imageio-ffmpeg)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def ffmpeg_available() -> bool:
    return ffmpeg_exe() is not None


def load_audio(path: str | Path, sr: int = SR, max_seconds: float = 12.0) -> np.ndarray:
    """Decode any audio file to mono float32 at `sr` Hz, peak-normalised to 1."""
    path = str(path)
    exe = ffmpeg_exe()
    if exe:
        cmd = [exe, "-nostdin", "-v", "error", "-i", path, "-t", str(max_seconds), "-ac", "1", "-ar", str(sr),
               "-f", "f32le", "-"]
        try:                                      # RC-09: a malformed file must not hang the service
            out = subprocess.run(cmd, capture_output=True, check=False, timeout=FFMPEG_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise ValueError(f"decoding {Path(path).name} took longer than {FFMPEG_TIMEOUT_S} s")
        if out.returncode != 0:
            raise ValueError(f"ffmpeg could not decode {Path(path).name}: {out.stderr.decode(errors='ignore')[:200]}")
        x = np.frombuffer(out.stdout, dtype=np.float32).copy()
    else:                                                     # wav / flac / ogg-vorbis only
        import soundfile as sf
        x, file_sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(1)
        if file_sr != sr:
            import librosa
            x = librosa.resample(x, orig_sr=file_sr, target_sr=sr)
        x = x[: int(max_seconds * sr)]
    return normalise(x)


def normalise(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float32))
    x = x - x.mean() if len(x) else x
    peak = np.abs(x).max() if len(x) else 0.0
    return x / peak if peak > 1e-6 else x


def frame_rms(x: np.ndarray, sr: int = SR, frame: float = 0.02, hop: float = 0.01) -> np.ndarray:
    n, h = int(frame * sr), int(hop * sr)
    if len(x) < n:
        return np.array([np.sqrt(np.mean(x ** 2))]) if len(x) else np.zeros(1)
    idx = np.arange(0, len(x) - n + 1, h)
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[idx]
    return np.sqrt((frames ** 2).mean(1) + 1e-12)


def segment_coughs(x: np.ndarray, sr: int = SR, hi_mult: float = 2.0, lo_mult: float = 0.25, pad: float = 0.15,
                   min_len: float = 0.15, merge_gap: float = 0.1, hop: float = 0.01) -> tuple[list[tuple[int, int]], float]:
    """Return cough segments as (start_sample, end_sample) and the SNR in dB.

    Thresholds are relative to the recording's own energy: the high threshold is hi_mult x the RMS of the whole
    recording (and at least 4x the background level), the low threshold is lo_mult x the high one (and at least
    2x the background level). The background level is the 20th percentile of the frame RMS."""
    if len(x) < int(0.1 * sr):
        return [], 0.0
    env = frame_rms(x, sr, hop=hop)
    background = np.percentile(env, 20)
    hi = max(hi_mult * np.sqrt(np.mean(x ** 2)), 4.0 * background)
    lo = max(lo_mult * hi, 2.0 * background)
    segs, inside, start = [], False, 0
    for i, e in enumerate(env):
        if not inside and e > hi:
            inside, start = True, i
            while start > 0 and env[start - 1] > lo:                 # walk back to the onset
                start -= 1
        elif inside and e < lo:
            inside = False
            segs.append([start, i])
    if inside:
        segs.append([start, len(env)])
    merged: list[list[int]] = []
    gap = int(merge_gap / hop)
    for s in segs:
        if merged and s[0] - merged[-1][1] <= gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    p, step, n = int(pad / hop), int(hop * sr), len(x)
    out = []
    for s, e in merged:
        if (e - s) * hop < min_len:
            continue
        out.append((max(0, (s - p) * step), min(n, (e + p) * step)))
    # padded segments may overlap again
    final: list[tuple[int, int]] = []
    for s, e in out:
        if final and s <= final[-1][1]:
            final[-1] = (final[-1][0], max(final[-1][1], e))
        else:
            final.append((s, e))
    mask = np.zeros(n, bool)
    for s, e in final:
        mask[s:e] = True
    if mask.any() and (~mask).any():
        snr = 10 * np.log10((np.mean(x[mask] ** 2) + 1e-12) / (np.mean(x[~mask] ** 2) + 1e-12))
    else:
        snr = 0.0
    return final, float(snr)


def cough_region(x: np.ndarray, segs: list[tuple[int, int]], sr: int = SR, max_seconds: float = 5.0) -> tuple[int, int]:
    """Region from the first cough to the last (at most max_seconds); the whole start of the file if no cough."""
    limit = int(max_seconds * sr)
    if not segs:
        return 0, min(len(x), limit)
    s = segs[0][0]
    e = segs[-1][1]
    return s, min(e, s + limit)


def cough_signal(x: np.ndarray, segs: list[tuple[int, int]]) -> np.ndarray:
    """Only the cough parts, concatenated (the whole signal if no cough was found)."""
    if not segs:
        return x
    return np.concatenate([x[s:e] for s, e in segs])
