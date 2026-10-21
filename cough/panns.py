"""PANNs CNN14 (Kong et al., 2020): a CNN pretrained on AudioSet (2M YouTube clips, 527 sound classes,
including "Cough", "Throat clearing", "Breathing", "Wheeze" and "Sneeze").

Used two ways:
  * its 2048-d embedding of a cough is a general "how does this sound" description for a classifier;
  * its AudioSet "Cough" probability is an independent check that a recording really contains a cough.

The network is defined here (same layer names as the official code, so the official checkpoint loads directly);
the log-mel front end uses torch.stft + librosa's mel filters instead of the torchlibrosa package, matching the
original settings (32 kHz, 1024-point FFT, hop 320, 64 mel bands, 50 Hz - 14 kHz).

Checkpoint (327 MB, CC BY 4.0): https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 32_000
CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
AUDIOSET = {"speech": 0, "breathing": 41, "wheeze": 42, "cough": 47, "throat_clearing": 48, "sneeze": 49,
            "silence": 500}


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn1, self.bn2 = nn.BatchNorm2d(cout), nn.BatchNorm2d(cout)

    def forward(self, x, pool=(2, 2)):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, pool)


class Cnn14(nn.Module):
    def __init__(self, n_fft: int = 1024, hop: int = 320, n_mels: int = 64, fmin: int = 50, fmax: int = 14_000):
        super().__init__()
        import librosa
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        mel = librosa.filters.mel(sr=SR, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax).T
        self.register_buffer("melW", torch.from_numpy(mel.astype(np.float32)), persistent=False)
        self.bn0 = nn.BatchNorm2d(64)
        chans = [1, 64, 128, 256, 512, 1024, 2048]
        for i in range(6):
            setattr(self, f"conv_block{i + 1}", ConvBlock(chans[i], chans[i + 1]))
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, 527)

    def logmel(self, wav: torch.Tensor) -> torch.Tensor:                 # (B, samples) -> (B, 1, T, 64)
        spec = torch.stft(wav, self.n_fft, self.hop, self.n_fft, self.window, center=True, pad_mode="reflect",
                          return_complex=True).abs() ** 2
        mel = spec.transpose(1, 2) @ self.melW
        return (10 * torch.log10(mel.clamp_min(1e-10)))[:, None]

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.logmel(wav)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for i in range(1, 7):
            x = getattr(self, f"conv_block{i}")(x, (2, 2) if i < 6 else (1, 1))
        x = x.mean(3)
        x = x.max(2)[0] + x.mean(2)
        emb = F.relu(self.fc1(x))
        return emb, torch.sigmoid(self.fc_audioset(emb))


def load(checkpoint: str | Path) -> Cnn14:
    m = Cnn14()
    try:
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception:                                     # older torch / pickled extras
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ck.get("model", ck)
    missing, _ = m.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith(("window", "melW"))]
    if missing:
        raise RuntimeError(f"checkpoint does not match CNN14, missing weights: {missing[:5]}")
    return m.eval()


@torch.no_grad()
def embed(model: Cnn14, waves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Batch of 32 kHz waveforms (any lengths, padded to the longest) -> (B, 2048) embeddings, (B, 527) tags."""
    n = max(max(len(w) for w in waves), SR // 2)
    x = np.zeros((len(waves), n), np.float32)
    for i, w in enumerate(waves):
        x[i, :len(w)] = w
    emb, tags = model(torch.from_numpy(x))
    return emb.numpy(), tags.numpy()
