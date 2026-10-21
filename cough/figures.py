"""Plain black-and-white result figures for the README (ROC curves and a calibration plot)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
GREYS = ["black", "dimgray", "gray", "darkgray", "black", "dimgray"]


def roc_figure(curves: dict, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    for i, (name, (y, p)) in enumerate(curves.items()):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, linestyle=STYLES[i % 6], color=GREYS[i % 6], lw=1.6,
                label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], color="lightgray", lw=1)
    ax.axvline(0.1, color="lightgray", lw=0.8, linestyle=":")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def calibration_figure(y: np.ndarray, p: np.ndarray, path: Path, title: str, bins: int = 10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    mp = [p[idx == b].mean() for b in range(bins) if (idx == b).any()]
    my = [y[idx == b].mean() for b in range(bins) if (idx == b).any()]
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    top = max(max(mp), max(my)) * 1.1
    ax.plot([0, top], [0, top], color="lightgray", lw=1)
    ax.plot(mp, my, "o-", color="black", lw=1.4, ms=4)
    ax.set_xlabel("Predicted probability (mean per group of 10%)")
    ax.set_ylabel("Observed share of positives")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
