"""Features of one recording.

Handcrafted features (computed on the cough parts only, ~130 numbers): cough count and timing, energy, spectral
shape, band energies and MFCCs. They describe what clinicians listen for: how many coughs, how long, how sharp
the onset, how "wet" (low-frequency energy, long tail) or "dry" (short, broadband) they sound.

Log-mel spectrogram (64 mel bands x 250 frames of 20 ms = 5 s) of the region from the first cough to the last,
input of the CNN.
"""
from __future__ import annotations

import numpy as np

from .audio import SR, cough_region, cough_signal, frame_rms, segment_coughs

N_MELS, HOP, N_FFT, WIN, MAX_SECONDS = 64, 320, 1024, 640, 5.0
FRAMES = int(MAX_SECONDS * SR / HOP)                    # 250
BANDS = [(50, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]

READABLE = {  # names used in the API explanations
    "n_coughs": "number of coughs", "cough_rate": "coughs per second", "cough_dur_mean": "average cough length",
    "cough_dur_std": "variation in cough length", "cough_total": "total cough time", "snr_db": "recording quality (SNR)",
    "duration": "recording length", "rms_mean": "loudness", "rms_std": "loudness variation", "crest": "peakiness",
    "attack_mean": "cough onset sharpness", "tail_mean": "cough tail length", "zcr_mean": "noisiness (zero crossings)",
    "centroid_mean": "spectral brightness", "centroid_std": "brightness variation", "bandwidth_mean": "spectral spread",
    "rolloff_mean": "high-frequency content", "flatness_mean": "noise-likeness (flatness)",
}


def _stats(prefix: str, m: np.ndarray, out: dict):
    out[f"{prefix}_mean"] = float(np.mean(m)) if m.size else 0.0
    out[f"{prefix}_std"] = float(np.std(m)) if m.size else 0.0


def handcrafted(x: np.ndarray, segs: list[tuple[int, int]], snr: float, sr: int = SR) -> dict:
    import librosa
    f: dict[str, float] = {}
    durs = np.array([(e - s) / sr for s, e in segs]) if segs else np.zeros(0)
    f["n_coughs"] = float(len(segs))
    f["duration"] = len(x) / sr
    f["cough_rate"] = len(segs) / max(f["duration"], 1e-3)
    f["cough_dur_mean"] = float(durs.mean()) if len(durs) else 0.0
    f["cough_dur_std"] = float(durs.std()) if len(durs) else 0.0
    f["cough_total"] = float(durs.sum())
    f["snr_db"] = snr
    c = cough_signal(x, segs)
    if len(c) < N_FFT:
        c = np.pad(c, (0, N_FFT - len(c)))
    env = frame_rms(c, sr)
    f["rms_mean"], f["rms_std"] = float(env.mean()), float(env.std())
    f["crest"] = float(np.abs(c).max() / (np.sqrt(np.mean(c ** 2)) + 1e-9))
    attack, tail = [], []                              # shape of each cough's energy envelope
    for s, e in segs:
        ev = frame_rms(x[s:e], sr)
        k = int(np.argmax(ev))
        attack.append(k / max(len(ev), 1))
        above = np.flatnonzero(ev[k:] > 0.3 * ev[k])
        tail.append((above[-1] if len(above) else 0) * 0.01)
    f["attack_mean"] = float(np.mean(attack)) if attack else 0.0
    f["tail_mean"] = float(np.mean(tail)) if tail else 0.0
    S = np.abs(librosa.stft(c, n_fft=N_FFT, hop_length=HOP, win_length=WIN)) ** 2
    _stats("zcr", librosa.feature.zero_crossing_rate(c, frame_length=WIN, hop_length=HOP)[0], f)
    _stats("centroid", librosa.feature.spectral_centroid(S=S, sr=sr)[0], f)
    _stats("bandwidth", librosa.feature.spectral_bandwidth(S=S, sr=sr)[0], f)
    _stats("rolloff", librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0], f)
    _stats("flatness", librosa.feature.spectral_flatness(S=S)[0], f)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr, fmin=100.0, n_bands=6)
    for i, row in enumerate(contrast):
        f[f"contrast{i}_mean"] = float(row.mean())
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    total = S.sum() + 1e-12
    for lo, hi in BANDS:
        f[f"band_{lo}_{hi}"] = float(S[(freqs >= lo) & (freqs < hi)].sum() / total)
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=N_MELS, fmin=50, fmax=sr // 2)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    d = librosa.feature.delta(mfcc, width=3) if mfcc.shape[1] >= 3 else np.zeros_like(mfcc)
    for i in range(20):
        f[f"mfcc{i}_mean"], f[f"mfcc{i}_std"] = float(mfcc[i].mean()), float(mfcc[i].std())
        f[f"dmfcc{i}_std"] = float(d[i].std())
    return f


def logmel(x: np.ndarray, region: tuple[int, int], sr: int = SR) -> tuple[np.ndarray, int]:
    """(64, 250) float16 log-mel of the cough region, padded with the minimum value; also the real length."""
    import librosa
    s, e = region
    y = x[s:e]
    if len(y) < WIN:
        y = np.pad(y, (0, WIN - len(y)))
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, win_length=WIN, n_mels=N_MELS,
                                       fmin=50, fmax=sr // 2)
    m = librosa.power_to_db(m, ref=1.0, amin=1e-10, top_db=80.0)[:, :FRAMES]
    n = m.shape[1]
    out = np.full((N_MELS, FRAMES), m.min(), dtype=np.float32)
    out[:, :n] = m
    return out.astype(np.float16), n


def analyse(x: np.ndarray, sr: int = SR) -> dict:
    """Everything the pipeline and the API need from one decoded recording."""
    segs, snr = segment_coughs(x, sr)
    region = cough_region(x, segs, sr, MAX_SECONDS)
    feats = handcrafted(x, segs, snr, sr)
    mel, n = logmel(x, region, sr)
    return {"segments": segs, "snr": snr, "region": region, "features": feats, "logmel": mel, "frames": n}
