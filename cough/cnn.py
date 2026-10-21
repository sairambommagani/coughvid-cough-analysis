"""A small CNN on log-mel spectrograms, trained from scratch for all tasks at once (multi-task).

Input: 64 mel bands x up to 250 frames (5 s) of the cough region. Four conv blocks (conv-BN-ReLU x2, 2x2 pooling)
with 16-32-64-128 channels, then mean over frequency and (max + mean) over time, so any length works. One output
per task; recordings without a label for a task simply do not count in that task's loss.

Augmentation (training only): random 4 s crop, random gain (+-6 dB), SpecAugment (2 frequency masks of up to
8 bands, 2 time masks of up to 20 frames).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                         nn.AvgPool2d(2))


class CoughCNN(nn.Module):
    def __init__(self, n_tasks: int, width: int = 16, dropout: float = 0.3):
        super().__init__()
        c = [1, width, 2 * width, 4 * width, 8 * width]
        self.features = nn.Sequential(*[block(c[i], c[i + 1]) for i in range(4)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(c[-1], n_tasks)

    def embed(self, x: torch.Tensor) -> torch.Tensor:              # x: (B, 64, T)
        h = self.features(x[:, None]).mean(2)                         # (B, C, T')
        return h.max(2)[0] + h.mean(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.embed(x)))


def _augment(x: np.ndarray, n: np.ndarray, rng, crop: int = 200) -> np.ndarray:
    B, M, T = x.shape
    out = np.empty((B, M, crop), np.float32)
    for i in range(B):
        L = max(int(n[i]), 1)
        s = rng.integers(0, max(L - crop, 0) + 1)
        seg = x[i, :, s:s + crop]
        if seg.shape[1] < crop:
            seg = np.pad(seg, ((0, 0), (0, crop - seg.shape[1])), constant_values=x[i].min())
        seg = seg + rng.uniform(-6, 6)
        for _ in range(2):
            f = rng.integers(0, 9)
            f0 = rng.integers(0, M - f + 1)
            seg[f0:f0 + f] = seg.mean()
            t = rng.integers(0, 21)
            t0 = rng.integers(0, crop - t + 1)
            seg[:, t0:t0 + t] = seg.mean()
        out[i] = seg
    return out


class Normaliser:
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean, self.std = mean, std

    def fit(self, mel, rows):
        sample = np.asarray(mel[rows[:2000]], np.float32)
        self.mean, self.std = float(sample.mean()), float(sample.std() + 1e-6)
        return self

    def __call__(self, x):
        return (np.asarray(x, np.float32) - self.mean) / self.std


@torch.no_grad()
def predict(model: CoughCNN, mel, rows: np.ndarray, norm: Normaliser, batch: int = 128) -> np.ndarray:
    """Logits (len(rows), n_tasks) on the full 5 s input."""
    model.eval()
    out = []
    for s in range(0, len(rows), batch):
        idx = rows[s:s + batch]
        out.append(model(torch.from_numpy(norm(mel[np.sort(idx)])))[np.argsort(np.argsort(idx))].numpy())
    return np.vstack(out) if out else np.zeros((0, model.head.out_features), np.float32)


def train(mel, frames: np.ndarray, Y: np.ndarray, train_rows: np.ndarray, val_rows: np.ndarray, eval_fn,
          epochs: int = 15, batch: int = 64, lr: float = 2e-3, patience: int = 4, width: int = 16, seed: int = 0,
          init: dict | None = None, log=print) -> tuple[CoughCNN, Normaliser, list]:
    """Y: (N, n_tasks) with NaN = no label. eval_fn(val_logits) -> score to maximise (early stopping).
    init: a saved checkpoint ({"state", "mean", "std"}) to continue training from."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + (1 if init else 0))
    n_tasks = Y.shape[1]
    model = CoughCNN(n_tasks, width)
    if init:
        model.load_state_dict(init["state"])
        norm = Normaliser(init["mean"], init["std"])
    else:
        norm = Normaliser().fit(mel, train_rows)
    lab = ~np.isnan(Y[train_rows])
    pos = np.nansum(Y[train_rows], 0)
    neg = lab.sum(0) - pos
    pos_weight = torch.tensor(np.sqrt(np.maximum(neg, 1) / np.maximum(pos, 1)), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    steps = epochs * int(np.ceil(len(train_rows) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    best, best_state, bad, hist = -1.0, None, 0, []
    if init:
        best = eval_fn(predict(model, mel, val_rows, norm))
        best_state = copy.deepcopy(model.state_dict())
        log(f"    continuing from the saved model: validation score {best:.4f}")
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        order = rng.permutation(train_rows)
        for s in range(0, len(order), batch):
            idx = np.sort(order[s:s + batch])
            x = torch.from_numpy(_augment(norm(mel[idx]), frames[idx], rng))
            y = torch.from_numpy(Y[idx]).float()
            m = ~torch.isnan(y)
            logits = model(x)
            loss_all = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(y), reduction="none",
                                                          pos_weight=pos_weight)
            loss = (loss_all * m).sum() / m.sum().clamp_min(1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        score = eval_fn(predict(model, mel, val_rows, norm))
        hist.append(round(float(score), 4))
        log(f"    epoch {ep:2d}: loss {tot / max(nb, 1):.4f}, validation score {score:.4f} ({time.time() - t0:.0f}s)")
        if score > best:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), norm, hist
