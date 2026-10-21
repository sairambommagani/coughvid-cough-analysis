"""End-to-end tests on a small synthetic dataset in the real COUGHVID format (webm/ogg + json + metadata csv)."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cough.audio import ffmpeg_available, segment_coughs
from cough.data import add_expert, add_labels
from cough.evaluate import report, sens_at_spec, threshold_for_spec

needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")


def test_segmentation_finds_bursts():
    sr = 16000
    rng = np.random.default_rng(0)
    x = rng.normal(scale=0.01, size=5 * sr)
    for t in (1.0, 2.5, 4.0):
        n = int(0.3 * sr)
        x[int(t * sr):int(t * sr) + n] += rng.normal(size=n) * np.exp(-np.arange(n) / (0.06 * sr))
    segs, snr = segment_coughs(x / np.abs(x).max(), sr)
    assert len(segs) == 3
    assert snr > 10


def test_expert_majority_and_labels():
    df = pd.DataFrame({"uuid": ["a", "b", "c"], "cough_detected": [0.9, 0.95, 0.1],
                       "status": ["COVID-19", "healthy", "healthy"], "status_SSL": ["COVID-19", None, None],
                       "cough_type_1": ["wet", "dry", None], "cough_type_2": ["wet", "wet", None],
                       "cough_type_3": [None, "dry", None], "quality_1": ["good", "ok", None],
                       "quality_2": ["good", "ok", None], "quality_3": [None, "good", None]})
    df = add_labels(add_expert(df))
    assert list(df["exp_cough_type"].astype(object).fillna("-")) == ["wet", "dry", "-"]
    assert df["n_experts"].tolist() == [2, 3, 0]
    assert df["y_covid"].tolist()[:2] == [1.0, 0.0] and np.isnan(df["y_covid"][2])
    assert df["y_cough"].tolist() == [1.0, 1.0, 0.0]


def test_metrics():
    y = np.array([0] * 90 + [1] * 10)
    p = np.r_[np.linspace(0, 0.5, 90), np.linspace(0.4, 1, 10)]
    r = report(y, p)
    assert 0.9 < r["roc_auc"] <= 1 and r["roc_auc_95ci"][0] <= r["roc_auc"]
    assert sens_at_spec(y, p, 0.9) >= 0.8
    thr = threshold_for_spec(y, p, 0.9)
    assert ((p >= thr)[y == 0]).mean() <= 0.11


@needs_ffmpeg
def test_pipeline_and_api(tmp_path):
    from scripts.make_synthetic import write
    from cough import prepare, train
    raw, cache, art = tmp_path / "raw", tmp_path / "cache", tmp_path / "art"
    write(raw, n=120, seed=1)
    e = prepare.run(raw, cache, workers=2, max_noncough=100, log=lambda *a: None)
    assert e["audio"]["decoded"] == 120 and e["audio"]["failed"] == 0
    out = train.run(cache, art, epochs=2, log=lambda *a: None)
    assert "covid" in out["results"] and "cough" in out["results"]
    assert out["results"]["cough"]["fusion"]["roc_auc"] > 0.8          # non-coughs are easy to tell apart

    os.environ["COUGH_ARTIFACTS"] = str(art)
    import importlib
    import cough.api as api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    with TestClient(api.app) as c:
        ex = c.get("/examples", params={"n": 2}).json()
        r = c.get(f"/examples/{ex[0]['uuid']}/analyze")
        assert r.status_code == 200 and "cough_check" in r.json()
        f = sorted(raw.glob("*.ogg"))[0]
        r = c.post("/analyze", files={"file": (f.name, f.read_bytes(), "audio/ogg")})
        assert r.status_code == 200 and "recording" in r.json()
        assert c.post("/analyze", files={"file": ("x.webm", b"not audio", "audio/webm")}).status_code == 422
