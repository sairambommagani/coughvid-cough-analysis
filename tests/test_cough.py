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
    os.environ["COUGH_MODE"] = "research"                  # research mode: all outputs and the test-set browser
    os.environ["COUGH_AUDIT_DIR"] = str(tmp_path / "audit")
    os.environ["COUGH_DATA_DIR"] = str(tmp_path / "store")
    import importlib
    import cough.api as api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    with TestClient(api.app) as c:
        ex = c.get("/examples", params={"n": 2}).json()
        r = c.get(f"/examples/{ex[0]['uuid']}/analyze")
        assert r.status_code == 200 and "cough_check" in r.json()
        assert c.get(f"/examples/{ex[0]['uuid']}/audio").status_code == 200
        assert "Cough Analysis" in c.get("/").text
        f = sorted(raw.glob("*.ogg"))[0]
        r = c.post("/analyze", files={"file": (f.name, f.read_bytes(), "audio/ogg")})
        assert r.status_code == 200 and "recording" in r.json()
        # not audio: stopped by the format check before the models (415); real audio header but cut off: 422
        assert c.post("/analyze", files={"file": ("x.webm", b"not audio", "audio/webm")}).status_code == 415
        cut = sorted(raw.glob("*.webm"))[0].read_bytes()[:60]
        assert c.post("/analyze", files={"file": ("cut.webm", cut, "audio/webm")}).status_code == 422
        assert r.json()["input"]["format"] == "ogg" and r.json()["model_version"]
        assert "covid_19" in r.json() or not r.json()["cough_check"]["is_cough"]      # research: all outputs

    # ---------------- production mode on a registered model version (SaMD controls, docs/regulatory/07)
    import hashlib
    import shutil
    from cough import audit as auditmod, versioning
    root = tmp_path / "models"
    v, vdir = versioning.new_version(root, "minor")
    shutil.copytree(art, vdir, dirs_exist_ok=True)
    versioning.register(root, vdir, v, "major", cache, {"epochs": 2}, notes="test")
    os.environ.update({"COUGH_ARTIFACTS": str(root), "COUGH_MODE": "production"})
    importlib.reload(api)
    ids = {"X-User-Id": "user-001", "X-Org-Id": "org-001"}
    f = sorted(raw.glob("*.webm"))[0]
    data = f.read_bytes()
    with TestClient(api.app) as c:
        assert c.get("/health").json()["integrity"] == "verified"                               # TC-VER-02
        r = c.post("/analyze", files={"file": (f.name, data, "audio/webm")})
        assert r.status_code == 401                                                             # TC-AUD-02
        r = c.post("/analyze", files={"file": (f.name, data, "audio/webm")}, headers=ids)
        assert r.status_code == 200, r.text
        j = r.json()
        assert "covid_19" not in j and "respiratory_symptoms" not in j                          # TC-INT-01
        assert j["intended_use"].startswith("Detects whether") and "Not a medical device" in j["regulatory_status"]
        assert j["model_version"] == vdir.name and r.headers["X-Request-ID"] == j["request_id"]
        assert c.post("/analyze", files={"file": (f.name, data, "audio/webm")}, data={"latitude": "12.9"},
                      headers=ids).status_code == 400                                           # TC-PHI-02
        assert c.get("/examples").status_code == 404                                            # TC-INT-03
        assert set(c.get("/model-info").json()["results"]) <= {"cough", "wet"}                  # TC-INT-04
        recs = auditmod.find(Path(os.environ["COUGH_AUDIT_DIR"]), j["request_id"])              # TC-AUD-01
        assert len(recs) == 1
        rec = recs[0]
        assert rec["user_id"] == "user-001" and rec["org_id"] == "org-001" and rec["model_version"] == vdir.name
        assert rec["input_sha256"] == hashlib.sha256(data).hexdigest() and rec["output"] == j
        assert rec["model_card_sha256"] == versioning.card_sha256(vdir)
        assert auditmod.verify(Path(os.environ["COUGH_AUDIT_DIR"]))["ok"]                         # TC-AUD-03
        store = Path(os.environ["COUGH_DATA_DIR"])
        line = (store / "features.jsonl").read_text().splitlines()[-1]                          # TC-PHI-01
        assert "latitude" not in line and "longitude" not in line and f.name not in line
        assert not (store / "audio").exists()                                                   # TC-PHI-03
        r = c.delete("/subjects/user-001", headers=ids)                                         # TC-PHI-05
        assert r.status_code == 200 and r.json()["feature_records_deleted"] >= 1
        assert "user-001" not in (store / "features.jsonl").read_text()
    # tampered model file: production refuses to serve it                                      # TC-VER-03
    mj = vdir / "models.joblib"
    mj.chmod(0o666)
    mj.write_bytes(mj.read_bytes() + b"x")
    importlib.reload(api)
    with TestClient(api.app) as c:
        assert c.post("/analyze", files={"file": (f.name, data, "audio/webm")}, headers=ids).status_code == 503
    for k in ("COUGH_MODE", "COUGH_ARTIFACTS"):
        os.environ.pop(k, None)
