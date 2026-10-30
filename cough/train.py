"""Step 3: train and evaluate all models (run after cough.prepare, and cough.embed if available).

  python -m cough.train

Tasks (labels in meta.parquet):
  covid        COVID-19 vs healthy, self-reported status (main task)
  symptomatic  COVID-19 or symptomatic vs healthy (general respiratory screening)
  wet          wet vs dry cough, majority vote of the pulmonologists (expert-labelled subset)
  cough        cough vs no cough (quality check before analysis)

Models per task:
  handcrafted  LightGBM on ~100 handcrafted cough features
  panns        logistic regression on PANNs CNN14 embeddings (pretrained on AudioSet)
  cnn          multi-task CNN on log-mel spectrograms (trained from scratch)
  fusion       weighted sum of the models' standardised scores (weights chosen on validation), then Platt
               scaling on validation so the output is a calibrated probability

Validation chooses everything (early stopping, regularisation, fusion weights, calibration, the 90%-specificity
threshold); the test split is used once. Extra analyses for the COVID-19 task: audio vs self-reported symptoms
(metadata), user vs SSL labels, results by subgroup, and a time-based check.
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import cnn as cnnmod
from .evaluate import paired, report, threshold_for_spec

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")

TASKS = {"covid": "y_covid", "symptomatic": "y_symptomatic", "wet": "y_wet", "cough": "y_cough"}
NOT_FEATURES = {"uuid", "ok", "region_start", "region_end"}
META_COLS = ["age", "male", "female", "respiratory_condition", "fever_muscle_pain"]
WEIGHTS = [0.0, 0.5, 1.0, 2.0]


def lgbm(Xtr, ytr, Xva, yva, seed=0):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                           random_state=seed, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def logreg(Xtr, ytr, Xva, yva):
    sc = StandardScaler().fit(Xtr)
    best = None
    for C in [0.001, 0.01, 0.1, 1.0]:
        m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr), ytr)
        auc = roc_auc_score(yva, m.decision_function(sc.transform(Xva)))
        if best is None or auc > best[0]:
            best = (auc, C, m)
    return sc, best[2], best[1]


def fuse(val_scores: dict, yva: np.ndarray):
    """Weights (grid) on z-scored model scores, maximising validation AUC; then Platt scaling."""
    names = list(val_scores)
    stats = {n: (float(np.mean(s)), float(np.std(s) + 1e-9)) for n, s in val_scores.items()}
    z = {n: (val_scores[n] - stats[n][0]) / stats[n][1] for n in names}
    best = None
    for w in itertools.product(WEIGHTS, repeat=len(names)):
        if sum(w) == 0:
            continue
        s = sum(wi * z[n] for wi, n in zip(w, names))
        auc = roc_auc_score(yva, s)
        if best is None or auc > best[0] + 1e-6:
            best = (auc, dict(zip(names, w)))
    w = best[1]
    s = sum(w[n] * z[n] for n in names)
    platt = LogisticRegression(C=1e6).fit(s[:, None], yva)
    return {"weights": w, "stats": stats, "platt": [float(platt.coef_[0, 0]), float(platt.intercept_[0])]}


def apply_fusion(f: dict, scores: dict) -> np.ndarray:
    s = sum(f["weights"][n] * (scores[n] - f["stats"][n][0]) / f["stats"][n][1] for n in f["weights"])
    a, b = f["platt"]
    return 1 / (1 + np.exp(-(a * s + b)))


def platt_only(val_s, yva):
    m = LogisticRegression(C=1e6).fit(val_s[:, None], yva)
    return lambda s: m.predict_proba(np.asarray(s)[:, None])[:, 1]


def metadata_matrix(meta: pd.DataFrame) -> pd.DataFrame:
    g = meta["gender"].astype("string").str.lower()
    return pd.DataFrame({"age": meta["age"].astype(float), "male": (g == "male").astype(float),
                         "female": (g == "female").astype(float),
                         "respiratory_condition": meta["respiratory_condition"].astype(float),
                         "fever_muscle_pain": meta["fever_muscle_pain"].astype(float)})


RECORDING_FEATURES = ["snr_db", "duration", "rms_mean", "rms_std"]      # how it was recorded, not the cough


def file_logger(path: Path):
    """Print and append to a log file immediately (progress can be checked while training runs)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    return log


def run(cache: Path, art: Path, epochs: int = 15, fast: bool = False, cnn_mode: str = "new", log=print) -> dict:
    """cnn_mode: "new" trains the CNN from scratch, "resume" continues from artifacts/cnn.pt (lower learning rate),
    "reuse" loads artifacts/cnn.pt without training."""
    t0 = time.time()
    art.mkdir(parents=True, exist_ok=True)
    if (art / "version.json").exists() or (art / "models.joblib").exists():   # no silent overwrite
        raise FileExistsError(f"{art} already holds a trained model; train into a new version "
                              f"(python -m cough.train --bump minor) instead of overwriting it")
    meta = pd.read_parquet(cache / "meta.parquet")
    feats = pd.read_parquet(cache / "features.parquet")
    mel = np.load(cache / "logmel.npy", mmap_mode="r")
    frames = np.load(cache / "frames.npy")
    has_panns = (cache / "panns_emb.npy").exists()
    E = np.load(cache / "panns_emb.npy").astype(np.float32) if has_panns else None
    tags = np.load(cache / "panns_tags.npy") if has_panns else None
    eda = json.loads((cache / "eda.json").read_text())
    ok = (feats["ok"] == 1).to_numpy()
    fcols = [c for c in feats.columns if c not in NOT_FEATURES]
    X = feats[fcols].astype(np.float32).to_numpy()
    M = metadata_matrix(meta)
    split = meta["split"].to_numpy()
    log(f"{ok.sum():,} recordings, {len(fcols)} handcrafted features, PANNs embeddings: {has_panns}")

    # ---------------- multi-task CNN (one training for all tasks)
    Y = np.stack([meta[c].to_numpy(float) for c in TASKS.values()], 1)
    Y[~ok] = np.nan
    tr_rows = np.flatnonzero(ok & (split == "train") & ~np.isnan(Y).all(1))
    va_rows = np.flatnonzero(ok & (split == "val") & ~np.isnan(Y).all(1))
    yva_all = Y[va_rows]

    def cnn_score(logits):            # early stopping: COVID AUC counts double
        aucs = []
        for k, wgt in zip(range(Y.shape[1]), [2, 1, 1, 1]):
            m = ~np.isnan(yva_all[:, k])
            if m.sum() > 10 and 0 < yva_all[m, k].mean() < 1:
                aucs += [roc_auc_score(yva_all[m, k], logits[m, k])] * wgt
        return float(np.mean(aucs)) if aucs else 0.0

    import torch
    prev = json.loads((art / "config.json").read_text()).get("cnn_history", []) if (art / "config.json").exists() else []
    if cnn_mode in ("resume", "reuse") and (art / "cnn.pt").exists():
        ck = torch.load(art / "cnn.pt", map_location="cpu", weights_only=False)
    else:
        ck, cnn_mode, prev = None, "new", []
    if cnn_mode == "reuse":
        log("CNN: using the saved model (no training)")
        cnn_model = cnnmod.CoughCNN(len(TASKS))
        cnn_model.load_state_dict(ck["state"])
        cnn_model.eval()
        norm, cnn_hist = cnnmod.Normaliser(ck["mean"], ck["std"]), prev
    else:
        log(f"CNN: {'continuing' if ck else 'training'} on {len(tr_rows):,} recordings")
        cnn_model, norm, hist = cnnmod.train(mel, frames, Y, tr_rows, va_rows, cnn_score,
                                             epochs=2 if fast else epochs, lr=5e-4 if ck else 2e-3, init=ck, log=log)
        cnn_hist = prev + hist
        torch.save({"state": cnn_model.state_dict(), "mean": norm.mean, "std": norm.std, "tasks": list(TASKS)},
                   art / "cnn.pt")
    all_rows = np.flatnonzero(ok)
    cnn_logits = np.full((len(meta), len(TASKS)), np.nan, np.float32)
    cnn_logits[all_rows] = cnnmod.predict(cnn_model, mel, all_rows, norm)

    results, models, preds, curves = {}, {}, {}, {}
    for k, (task, col) in enumerate(TASKS.items()):
        y = meta[col].to_numpy(float)
        rows = {s: np.flatnonzero(ok & ~np.isnan(y) & (split == s)) for s in ["train", "val", "test"]}
        if min(len(r) for r in rows.values()) < 10 or len(np.unique(y[rows["val"]])) < 2 \
                or len(np.unique(y[rows["test"]])) < 2:
            log(f"[{task}] not enough labelled data, skipped")
            continue
        tr, va, te = rows["train"], rows["val"], rows["test"]
        log(f"[{task}] train {len(tr):,} / val {len(va):,} / test {len(te):,} (positives in test: {int(y[te].sum())})")
        score = {"val": {}, "test": {}}
        m_hand = lgbm(X[tr], y[tr], X[va], y[va])
        for s, r in (("val", va), ("test", te)):
            score[s]["handcrafted"] = m_hand.predict_proba(X[r], raw_score=True)
        models[task] = {"handcrafted": m_hand}
        if has_panns:
            sc, lr, C = logreg(E[tr], y[tr], E[va], y[va])
            for s, r in (("val", va), ("test", te)):
                score[s]["panns"] = lr.decision_function(sc.transform(E[r]))
            models[task]["panns"] = (sc, lr)
        for s, r in (("val", va), ("test", te)):
            score[s]["cnn"] = cnn_logits[r, k]
        fusion = fuse(score["val"], y[va])
        models[task]["fusion"] = fusion
        p_val = apply_fusion(fusion, score["val"])
        thr = threshold_for_spec(y[va], p_val, 0.9)
        models[task]["threshold_90spec"] = thr
        res, curves[task] = {}, {}
        for name in score["test"]:                                   # each model, calibrated on validation
            cal = platt_only(score["val"][name], y[va])
            res[name] = report(y[te], cal(score["test"][name]))
            curves[task][name] = (y[te], cal(score["test"][name]))
        p_test = apply_fusion(fusion, score["test"])
        curves[task]["fusion"] = (y[te], p_test)
        res["fusion"] = report(y[te], p_test, threshold=thr)
        res["fusion"]["weights"] = fusion["weights"]
        best_single = max((n for n in score["test"]), key=lambda n: res[n]["roc_auc"])
        res["fusion"]["p_better_than_" + best_single] = round(paired(y[te], p_test, score["test"][best_single]), 3)
        results[task] = res
        preds[task] = pd.DataFrame({"row": te, "y": y[te], "p": p_test})
        log("    " + ", ".join(f"{n} {r['roc_auc']:.3f}" for n, r in res.items()) + " (test ROC-AUC)")

        if task == "covid":
            results["covid_extra"] = covid_extra(meta, X, M, y, rows, p_test, p_val, score, fusion, feats, log,
                                                 curves.setdefault("covid_vs_symptoms", {}))
        if task == "cough" and has_panns:                           # independent check: AudioSet "Cough"
            results["cough"]["audioset_cough_class"] = report(y[te], tags[te, 3], ci=False)
            if "exp_quality" in meta:
                q = meta["exp_quality"].astype("string").str.lower()
                nc = np.flatnonzero(ok & (q == "no_cough").fillna(False).to_numpy() & (split == "test"))
                if len(nc):
                    p_nc = apply_fusion(fusion, {"handcrafted": m_hand.predict_proba(X[nc], raw_score=True),
                                                 **({"panns": models[task]["panns"][1].decision_function(
                                                     models[task]["panns"][0].transform(E[nc]))} if has_panns else {}),
                                                 "cnn": cnn_logits[nc, k]})
                    results["cough"]["expert_no_cough_flagged"] = {"n": int(len(nc)),
                                                                    "flagged_as_no_cough": round(float((p_nc < 0.5).mean()), 3)}

    joblib.dump({t: {k: v for k, v in m.items()} for t, m in models.items()}, art / "models.joblib")
    importance = pd.Series(models["covid"]["handcrafted"].booster_.feature_importance("gain"), index=fcols) \
        if "covid" in models else pd.Series(dtype=float)
    cfg = {"features": fcols, "tasks": TASKS, "has_panns": has_panns, "cnn_history": cnn_hist,
           "trained_in_min": round((time.time() - t0) / 60, 1)}
    (art / "config.json").write_text(json.dumps(cfg, indent=2))
    out = {"dataset": "COUGHVID v3", "eda": eda, "results": results,
           "top_features_covid": {k: round(float(v), 1) for k, v in importance.sort_values(ascending=False).head(15).items()}}
    (art / "metrics.json").write_text(json.dumps(out, indent=2, default=str))
    test_rows = np.unique(np.concatenate([p["row"].to_numpy() for p in preds.values()])) if preds else np.array([], int)
    ex = meta.loc[test_rows, ["uuid", "status", "status_SSL", "exp_cough_type", "cough_detected", "path"]].copy()
    for t, p in preds.items():
        ex[f"p_{t}"] = pd.Series(p["p"].to_numpy(), index=p["row"].to_numpy()).reindex(test_rows).to_numpy()
    ex.astype({c: "string" for c in ["uuid", "status", "status_SSL", "exp_cough_type", "path"]}) \
        .to_parquet(art / "test_predictions.parquet", index=False)
    try:
        make_figures(curves, art / "figures")
    except Exception as ex:                                          # matplotlib missing: figures are optional
        log(f"figures skipped: {ex}")
    log(f"done in {(time.time() - t0) / 60:.1f} min -> {art}")
    return out


def make_figures(curves: dict, out: Path):
    from .figures import calibration_figure, roc_figure
    out.mkdir(parents=True, exist_ok=True)
    titles = {"covid": "COVID-19 vs healthy (audio only), test set",
              "symptomatic": "Symptomatic or COVID-19 vs healthy, test set",
              "wet": "Wet vs dry cough (expert labels), test set", "cough": "Cough vs no cough, test set",
              "covid_vs_symptoms": "COVID-19: audio vs self-reported symptoms, test set"}
    for task, c in curves.items():
        if c:
            roc_figure(c, out / f"roc_{task}.png", titles.get(task, task))
    if "covid" in curves:
        y, p = curves["covid"]["fusion"]
        calibration_figure(y, p, out / "calibration_covid.png", "Calibration: COVID-19 fusion model, test set")


def covid_extra(meta, X, M, y, rows, p_test, p_val, score, fusion, feats, log, curves) -> dict:
    tr, va, te = rows["train"], rows["val"], rows["test"]
    out = {}
    fcols = [c for c in feats.columns if c not in NOT_FEATURES]
    # 0) is the model listening to the cough or to the recording conditions? drop SNR / loudness / length
    keep = [i for i, c in enumerate(fcols) if c not in RECORDING_FEATURES]
    m_nq = lgbm(X[tr][:, keep], y[tr], X[va][:, keep], y[va])
    out["handcrafted_without_recording_quality"] = report(
        y[te], platt_only(m_nq.predict_proba(X[va][:, keep], raw_score=True), y[va])(
            m_nq.predict_proba(X[te][:, keep], raw_score=True)))
    out["handcrafted_without_recording_quality"]["removed"] = RECORDING_FEATURES
    snr = feats["snr_db"].to_numpy()
    lab = ~np.isnan(y) & (feats["ok"] == 1).to_numpy()
    out["snr_db_median"] = {"covid": round(float(np.median(snr[lab & (y == 1)])), 1),
                            "healthy": round(float(np.median(snr[lab & (y == 0)])), 1),
                            "snr_alone_roc_auc": round(float(roc_auc_score(y[te], -snr[te])), 4)}
    # 1) what the self-reported symptoms alone achieve, and audio + symptoms
    Mv = M.to_numpy(np.float32)
    m_meta = lgbm(Mv[tr], y[tr], Mv[va], y[va])
    s_meta_va = m_meta.predict_proba(Mv[va], raw_score=True)
    s_meta_te = m_meta.predict_proba(Mv[te], raw_score=True)
    out["metadata_only"] = report(y[te], platt_only(s_meta_va, y[va])(s_meta_te))
    f2 = fuse({**score["val"], "metadata": s_meta_va}, y[va])
    out["audio_plus_metadata"] = report(y[te], apply_fusion(f2, {**score["test"], "metadata": s_meta_te}))
    out["audio_plus_metadata"]["weights"] = f2["weights"]
    curves["audio (fusion)"] = (y[te], p_test)
    curves["self-reported symptoms"] = (y[te], platt_only(s_meta_va, y[va])(s_meta_te))
    curves["audio + symptoms"] = (y[te], apply_fusion(f2, {**score["test"], "metadata": s_meta_te}))
    # 2) noisy labels: train on SSL labels, test on both label sets
    ssl = meta["y_covid_ssl"].to_numpy(float)
    ok = (feats["ok"] == 1).to_numpy()
    sp = meta["split"].to_numpy()
    r_ssl = {s: np.flatnonzero(ok & ~np.isnan(ssl) & (sp == s)) for s in ["train", "val", "test"]}
    if min(len(r) for r in r_ssl.values()) > 20 and len(np.unique(ssl[r_ssl["test"]])) == 2:
        m_ssl = lgbm(X[r_ssl["train"]], ssl[r_ssl["train"]], X[r_ssl["val"]], ssl[r_ssl["val"]])
        m_usr = lgbm(X[tr], y[tr], X[va], y[va])
        out["labels"] = {
            "user_labels_train__user_test": round(roc_auc_score(y[te], m_usr.predict_proba(X[te])[:, 1]), 4),
            "ssl_labels_train__user_test": round(roc_auc_score(y[te], m_ssl.predict_proba(X[te])[:, 1]), 4),
            "user_labels_train__ssl_test": round(roc_auc_score(ssl[r_ssl["test"]], m_usr.predict_proba(X[r_ssl["test"]])[:, 1]), 4),
            "ssl_labels_train__ssl_test": round(roc_auc_score(ssl[r_ssl["test"]], m_ssl.predict_proba(X[r_ssl["test"]])[:, 1]), 4),
            "agreement_user_vs_ssl": round(float((y == ssl)[~np.isnan(y) & ~np.isnan(ssl)].mean()), 4),
        }
    # 3) subgroups (fusion model, test)
    t = meta.iloc[te]
    f = feats.iloc[te]
    groups = {
        "gender": t["gender"].astype("string").fillna("not given"),
        "age": pd.cut(t["age"], [0, 30, 50, 120], labels=["<30", "30-50", "50+"]).astype("string").fillna("not given"),
        "snr": pd.cut(f["snr_db"], [-np.inf, 10, 20, np.inf], labels=["<10 dB", "10-20 dB", ">20 dB"]).astype("string"),
        "coughs_in_recording": np.where(f["n_coughs"] <= 1, "0-1", "2+"),
        "cough_detected": np.where(t["cough_detected"] >= 0.95, ">=0.95", "0.80-0.95"),
    }
    sub = {}
    for g, vals in groups.items():
        sub[g] = {}
        for v in pd.unique(np.asarray(vals)):
            m = np.asarray(vals) == v
            if m.sum() >= 30 and 0 < y[te][m].mean() < 1:
                sub[g][str(v)] = {"n": int(m.sum()), "positives": int(y[te][m].sum()),
                                  "roc_auc": round(float(roc_auc_score(y[te][m], p_test[m])), 4)}
    out["subgroups"] = sub
    # 4) time-based check (handcrafted model): train on the earlier 80% of dates, test on the latest 20%
    dt = pd.to_datetime(meta["datetime"], errors="coerce", utc=True)
    lab = np.flatnonzero(ok & ~np.isnan(y) & dt.notna().to_numpy())
    if len(lab) > 200:
        cut = dt.iloc[lab].quantile(0.8)
        early = lab[(dt.iloc[lab] < cut).to_numpy()]
        late = lab[(dt.iloc[lab] >= cut).to_numpy()]
        rng = np.random.default_rng(0)
        v = rng.random(len(early)) < 0.15
        if len(np.unique(y[late])) == 2 and len(np.unique(y[early[v]])) == 2:
            m_t = lgbm(X[early[~v]], y[early[~v]], X[early[v]], y[early[v]])
            out["time_based"] = {"train_until": str(cut)[:10], "test_from": str(cut)[:10], "n_test": int(len(late)),
                                 "positives": int(y[late].sum()),
                                 "roc_auc": round(float(roc_auc_score(y[late], m_t.predict_proba(X[late])[:, 1])), 4)}
    log("    covid extra: metadata only {:.3f}, audio+metadata {:.3f}, audio without recording-quality features {:.3f}".format(
        out["metadata_only"].get("roc_auc", float("nan")), out["audio_plus_metadata"].get("roc_auc", float("nan")),
        out["handcrafted_without_recording_quality"].get("roc_auc", float("nan"))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--cnn", choices=["new", "resume", "reuse"], default="new",
                    help="new: train the CNN from scratch; resume: continue from the current version's cnn.pt; "
                         "reuse: no CNN training")
    ap.add_argument("--bump", choices=["major", "minor", "patch"], default="minor",
                    help="semantic version step for this training run (see cough/versioning.py): major = new "
                         "architecture/features, minor = retrained (default), patch = fix")
    ap.add_argument("--version", default=None, help="explicit version MAJOR.MINOR.PATCH instead of --bump")
    ap.add_argument("--notes", default="", help="what changed in this version (stored in the model card)")
    a = ap.parse_args()
    from . import versioning
    root = Path(a.artifacts)
    prev_dir = versioning.previous_dir(root)
    parent = versioning.version_info(prev_dir).get("version") if prev_dir else None
    version, vdir = versioning.new_version(root, a.bump, a.version)
    if prev_dir is not None and a.cnn in ("resume", "reuse"):          # start from the current version's CNN
        for f in ("cnn.pt", "config.json"):
            if (prev_dir / f).exists():
                shutil.copyfile(prev_dir / f, vdir / f)          # copy without the read-only flag of the parent
    log = file_logger(vdir / "train_log.txt")
    log(f"--- {time.strftime('%Y-%m-%d %H:%M')} cough.train {vdir.name} --cnn {a.cnn} --epochs {a.epochs}")
    out = run(Path(a.cache), vdir, a.epochs, a.fast, a.cnn, log=log)
    card = versioning.register(root, vdir, version, "explicit" if a.version else a.bump, Path(a.cache),
                               {"epochs": a.epochs, "fast": a.fast, "cnn": a.cnn}, a.notes, parent)
    log(f"registered {card['name']} (parent {parent}); LATEST -> {card['name']}")
    print(json.dumps({t: {m: r.get("roc_auc") for m, r in res.items() if isinstance(r, dict) and "roc_auc" in r}
                      for t, res in out["results"].items() if t != "covid_extra"}, indent=2))


if __name__ == "__main__":
    main()
