"""COUGHVID metadata: reading, expert-label aggregation, task labels, pseudo-user groups, splits and EDA.

Files (public_dataset_v3.zip): one audio file per recording (<uuid>.webm or <uuid>.ogg), a <uuid>.json with its
metadata, and metadata_compiled.csv with everything in one table:
  uuid, datetime, cough_detected (probability from the authors' cough classifier), SNR, latitude, longitude,
  age, gender, respiratory_condition, fever_muscle_pain, status (self-reported: healthy / symptomatic / COVID-19),
  status_SSL (labels re-estimated by the authors' semi-supervised model), and for the recordings reviewed by
  pulmonologists, per expert i: quality_i, cough_type_i, dyspnea_i, wheezing_i, stridor_i, choking_i,
  congestion_i, nothing_i, diagnosis_i, severity_i.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .audio import AUDIO_EXT

EXPERT_FIELDS = ["quality", "cough_type", "dyspnea", "wheezing", "stridor", "choking", "congestion", "nothing",
                 "diagnosis", "severity"]
CAT_FIELDS = ["quality", "cough_type", "diagnosis", "severity"]
COUGH_MIN = 0.8          # the authors' recommendation: below 0.8 most files contain no cough
NONCOUGH_MAX = 0.2


def find_metadata(raw: Path) -> Path | None:
    hits = sorted(raw.rglob("metadata_compiled.csv"))
    return hits[0] if hits else None


def audio_index(raw: Path) -> dict[str, Path]:
    idx = {}
    for p in raw.rglob("*"):
        if p.suffix.lower() in AUDIO_EXT and p.is_file():
            idx.setdefault(p.stem, p)
    return idx


def _from_json(raw: Path) -> pd.DataFrame:
    rows = []
    for p in raw.rglob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        r = {"uuid": p.stem}
        for k, v in d.items():
            m = re.match(r"expert_labels_(\d)", k)
            if m and isinstance(v, dict):
                for f, val in v.items():
                    r[f"{f}_{m.group(1)}"] = val
            elif not isinstance(v, (dict, list)):
                r[k] = v
        rows.append(r)
    return pd.DataFrame(rows)


def read_metadata(raw: str | Path) -> pd.DataFrame:
    raw = Path(raw)
    csv = find_metadata(raw)
    df = pd.read_csv(csv, low_memory=False) if csv else _from_json(raw)
    if df.empty:
        raise FileNotFoundError(f"no metadata_compiled.csv or *.json found under {raw}")
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    for c in ["datetime", "cough_detected", "latitude", "longitude", "age", "gender", "respiratory_condition",
              "fever_muscle_pain", "status", "status_SSL", "SNR"]:
        if c not in df.columns:
            df[c] = np.nan
    df["uuid"] = df["uuid"].astype(str)
    idx = audio_index(raw)
    df["path"] = df["uuid"].map(lambda u: str(idx[u]) if u in idx else None)
    df = df[df["path"].notna()].reset_index(drop=True)
    df["cough_detected"] = pd.to_numeric(df["cough_detected"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    for c in ["status", "status_SSL", "gender"]:
        df[c] = df[c].astype("string").str.strip()
    for c in ["respiratory_condition", "fever_muscle_pain"]:
        df[c] = df[c].map(lambda v: _bool(v))
    return add_expert(df)


def _bool(v):
    if isinstance(v, (bool, np.bool_)):
        return float(v)
    if isinstance(v, str):
        return {"true": 1.0, "false": 0.0}.get(v.strip().lower(), np.nan)
    return float(v) if pd.notna(v) else np.nan


def add_expert(df: pd.DataFrame) -> pd.DataFrame:
    """Majority vote of the experts per field (exp_<field>) and the number of experts per recording."""
    ids = sorted({m.group(1) for c in df.columns if (m := re.match(rf"^(?:{'|'.join(EXPERT_FIELDS)})_(\d)$", c))})
    df["n_experts"] = 0
    if not ids:
        for f in EXPERT_FIELDS:
            df[f"exp_{f}"] = pd.NA
        return df
    q = [f"quality_{i}" for i in ids if f"quality_{i}" in df]
    df["n_experts"] = df[q].notna().sum(1) if q else 0
    for f in EXPERT_FIELDS:
        cols = [f"{f}_{i}" for i in ids if f"{f}_{i}" in df]
        if not cols:
            df[f"exp_{f}"] = pd.NA
            continue
        vals = df[cols]
        if f not in CAT_FIELDS:
            vals = vals.apply(lambda s: s.map(_bool))
        df[f"exp_{f}"] = [_majority(r) for r in vals.itertuples(index=False)]
    return df


def _majority(row):
    v = [x for x in row if pd.notna(x)]
    if not v:
        return pd.NA
    (top, n), *rest = Counter(v).most_common()
    if rest and rest[0][1] == n:                          # tie between experts -> no label
        return pd.NA
    return top


def _status_code(s: pd.Series) -> pd.Series:
    return s.str.lower().map({"healthy": 0.0, "symptomatic": 1.0, "covid-19": 2.0, "covid": 2.0})


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    st, ssl = _status_code(df["status"]), _status_code(df["status_SSL"])
    cough = (df["cough_detected"] >= COUGH_MIN).to_numpy(bool)
    df["is_cough"] = cough
    df["y_covid"] = np.where(cough & (st == 2), 1.0, np.where(cough & (st == 0), 0.0, np.nan))
    df["y_covid_ssl"] = np.where(cough & (ssl == 2), 1.0, np.where(cough & (ssl == 0), 0.0, np.nan))
    df["y_symptomatic"] = np.where(cough & (st >= 1), 1.0, np.where(cough & (st == 0), 0.0, np.nan))
    ct = df["exp_cough_type"].astype("string").str.lower()
    wet, dry = (ct == "wet").fillna(False).to_numpy(bool), (ct == "dry").fillna(False).to_numpy(bool)
    df["y_wet"] = np.where(cough & wet, 1.0, np.where(cough & dry, 0.0, np.nan))
    df["y_cough"] = np.where(cough, 1.0, np.where((df["cough_detected"] <= NONCOUGH_MAX).to_numpy(bool), 0.0, np.nan))
    return df


def pseudo_users(df: pd.DataFrame) -> pd.Series:
    """There is no user id. Recordings from (almost) the same place on the same day are put in one group, so a
    person who submitted several coughs cannot end up in both train and test."""
    lat, lon = df["latitude"].round(2), df["longitude"].round(2)
    day = df["datetime"].dt.strftime("%Y-%m-%d")
    key = lat.astype(str) + "_" + lon.astype(str) + "_" + day.astype(str)
    has = lat.notna() & lon.notna() & day.notna()
    return pd.Series(np.where(has, key, df["uuid"]), index=df.index)


def split(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """train / val / test ~ 72 / 14 / 14, stratified by status, grouped by pseudo-user."""
    from sklearn.model_selection import StratifiedGroupKFold
    strat = _status_code(df["status"]).fillna(3).astype(int) + 4 * (df["y_cough"] == 0)
    groups = pseudo_users(df)
    out = np.array(["train"] * len(df), dtype=object)
    folds = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(folds.split(df, strat, groups)):
        if k == 0:
            out[te] = "test"
        elif k == 1:
            out[te] = "val"
        else:
            break
    return pd.Series(out, index=df.index)


def select(df: pd.DataFrame, max_noncough: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Recordings with a cough (cough_detected >= 0.8) + a sample of clear non-coughs (<= 0.2) for the cough check."""
    cough = df[df["cough_detected"] >= COUGH_MIN]
    non = df[df["cough_detected"] <= NONCOUGH_MAX]
    non = non.sample(min(len(non), max_noncough), random_state=seed)
    out = pd.concat([cough, non]).sort_values("uuid").reset_index(drop=True)
    out = add_labels(out)
    out["group"] = pseudo_users(out)
    out["split"] = split(out)
    return out


# ------------------------------------------------------------------ EDA
def _counts(s: pd.Series) -> dict:
    return {str(k): int(v) for k, v in s.value_counts(dropna=False).items()}


def expert_agreement(df: pd.DataFrame) -> dict:
    """Pairwise agreement between experts on the recordings two of them labelled (Cohen's kappa)."""
    from sklearn.metrics import cohen_kappa_score
    ids = sorted({m.group(1) for c in df.columns if (m := re.match(r"^diagnosis_(\d)$", c))})
    out = {}
    for f in ["cough_type", "diagnosis", "severity", "quality"]:
        kap, agree, n = [], [], 0
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                ca, cb = f"{f}_{ids[a]}", f"{f}_{ids[b]}"
                if ca not in df or cb not in df:
                    continue
                both = df[[ca, cb]].dropna()
                if len(both) < 10:
                    continue
                kap.append(cohen_kappa_score(both[ca].astype(str), both[cb].astype(str)))
                agree.append(float((both[ca].astype(str) == both[cb].astype(str)).mean()))
                n += len(both)
        if kap:
            k = float(np.nanmean(kap)) if not np.all(np.isnan(kap)) else None
            out[f] = {"pairs_compared": n, "agreement": round(float(np.mean(agree)), 3),
                      "cohen_kappa": None if k is None else round(k, 3)}
    return out


def eda(full: pd.DataFrame, sel: pd.DataFrame) -> dict:
    cough = full[full["cough_detected"] >= COUGH_MIN]
    st = cough["status"].fillna("not given")
    tab = pd.crosstab(cough["status"].fillna("none"), cough["status_SSL"].fillna("none"))
    e = {
        "recordings": int(len(full)),
        "period": [str(full["datetime"].min())[:10], str(full["datetime"].max())[:10]],
        "cough_detected": {">=0.8 (used)": int((full["cough_detected"] >= COUGH_MIN).sum()),
                           "0.2-0.8 (unclear, not used)": int(full["cough_detected"].between(NONCOUGH_MAX, COUGH_MIN,
                                                                                        inclusive="neither").sum()),
                           "<=0.2 (no cough)": int((full["cough_detected"] <= NONCOUGH_MAX).sum())},
        "status_of_coughs": _counts(st),
        "status_SSL_of_coughs": _counts(cough["status_SSL"].fillna("not given")),
        "status_vs_SSL": {str(i): {str(j): int(tab.loc[i, j]) for j in tab.columns} for i in tab.index},
        "gender": _counts(cough["gender"].fillna("not given")),
        "age": {"median": float(cough["age"].median()) if cough["age"].notna().any() else None,
                "missing_share": round(float(cough["age"].isna().mean()), 3)},
        "fever_muscle_pain_by_status": {str(k): round(float(v), 3) for k, v in
                                        cough.groupby("status")["fever_muscle_pain"].mean().dropna().items()},
        "respiratory_condition_by_status": {str(k): round(float(v), 3) for k, v in
                                            cough.groupby("status")["respiratory_condition"].mean().dropna().items()},
        "expert_labelled": int((full["n_experts"] > 0).sum()),
        "expert_labelled_by_2plus": int((full["n_experts"] >= 2).sum()),
        "expert_diagnosis": _counts(full.loc[full["n_experts"] > 0, "exp_diagnosis"].astype("string").fillna("tie / none")),
        "expert_cough_type": _counts(full.loc[full["n_experts"] > 0, "exp_cough_type"].astype("string").fillna("tie / none")),
        "expert_agreement": expert_agreement(full),
        "selected": {"recordings": int(len(sel)), "by_split": _counts(sel["split"]),
                     "pseudo_users": int(sel["group"].nunique())},
        "task_sizes": {t: {"n": int(sel[t].notna().sum()), "positives": int((sel[t] == 1).sum())}
                       for t in ["y_covid", "y_covid_ssl", "y_symptomatic", "y_wet", "y_cough"]},
    }
    return e
