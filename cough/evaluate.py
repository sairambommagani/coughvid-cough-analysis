"""Metrics for a screening classifier, with uncertainty.

  ROC-AUC          ranking quality, independent of the threshold and of class balance
  PR-AUC           precision/recall trade-off for the rare positive class (compare with the prevalence)
  sens@90spec      share of positives caught when only 10% of negatives are flagged (a screening operating point)
  Brier, ECE       calibration: does a predicted 30% risk mean 30% of such people are positive?
  95% CI           bootstrap over test recordings (1,000 resamples)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def sens_at_spec(y: np.ndarray, p: np.ndarray, spec: float = 0.9) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= 1 - spec + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


def bootstrap(y: np.ndarray, p: np.ndarray, fn=roc_auc_score, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], p[i]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def paired(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, n: int = 1000, seed: int = 0) -> float:
    """Share of bootstrap resamples in which model a has a higher AUC than model b."""
    rng = np.random.default_rng(seed)
    wins, tot = 0, 0
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        wins += roc_auc_score(y[i], pa[i]) > roc_auc_score(y[i], pb[i])
        tot += 1
    return wins / max(tot, 1)


def report(y: np.ndarray, p: np.ndarray, ci: bool = True, threshold: float | None = None) -> dict:
    """p must be a probability (calibrated) for Brier/ECE to be meaningful."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0 or y.min() == y.max():
        return {"n": int(len(y)), "positives": int(y.sum())}
    r = {"n": int(len(y)), "positives": int(y.sum()), "prevalence": round(float(y.mean()), 4),
         "roc_auc": round(float(roc_auc_score(y, p)), 4), "pr_auc": round(float(average_precision_score(y, p)), 4),
         "sens_at_90spec": round(sens_at_spec(y, p), 4), "brier": round(float(brier_score_loss(y, p)), 4),
         "ece": round(ece(y, p), 4)}
    if ci:
        lo, hi = bootstrap(y, p)
        r["roc_auc_95ci"] = [round(lo, 4), round(hi, 4)]
    if threshold is not None:                     # fixed threshold chosen on validation
        flag = p >= threshold
        r["at_val_threshold"] = {"threshold": round(float(threshold), 4),
                                 "sensitivity": round(float(flag[y == 1].mean()), 4),
                                 "specificity": round(float((~flag[y == 0]).mean()), 4),
                                 "flagged_share": round(float(flag.mean()), 4)}
    return r


def threshold_for_spec(y: np.ndarray, p: np.ndarray, spec: float = 0.9) -> float:
    """Lowest score threshold that keeps the specificity >= spec (on validation)."""
    neg = np.sort(p[y == 0])
    if not len(neg):
        return 0.5
    return float(neg[min(len(neg) - 1, int(np.ceil(spec * len(neg))))])
