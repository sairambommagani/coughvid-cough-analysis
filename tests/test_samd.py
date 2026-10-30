"""Unit tests of the SaMD controls. Test-case ids (TC-...) refer to docs/regulatory/07_verification_test_protocol.md.
The end-to-end production checks on a trained model are in tests/test_cough.py::test_pipeline_and_api."""
import json
import os
import stat
import time
from pathlib import Path

import pytest

from cough import audit, phi, samd, versioning


# ---------------- intended use and output rules
def test_tc_int_02_default_mode_is_production(monkeypatch):
    monkeypatch.delenv("COUGH_MODE", raising=False)
    assert samd.is_production() and samd.allowed_tasks() == ("cough", "wet")
    monkeypatch.setenv("COUGH_MODE", "research")
    assert "covid" in samd.allowed_tasks()
    monkeypatch.setenv("COUGH_MODE", "anything")
    with pytest.raises(ValueError):
        samd.mode()


def test_tc_int_05_labelling(monkeypatch):
    monkeypatch.delenv("COUGH_MODE", raising=False)
    lab = samd.labelling()
    assert "Does not diagnose COVID-19" in lab["intended_use"]
    assert lab["regulatory_status"].startswith("Not a medical device")


def test_tc_rsk_01_quality_gate():
    assert samd.quality_gate(3, 25.0, True) == (True, None)
    assert not samd.quality_gate(1, 25.0, True)[0]            # 0-1 coughs: COVID AUC 0.52 in this group
    assert not samd.quality_gate(3, 6.0, True)[0]             # noisy: SNR shortcut risk
    assert not samd.quality_gate(3, 25.0, False)[0]


def test_tc_rsk_02_indeterminate_band():
    assert samd.cough_type_label(0.10) == "dry"
    assert samd.cough_type_label(0.50) == "indeterminate"
    assert samd.cough_type_label(0.40) == "indeterminate" and samd.cough_type_label(0.60) == "indeterminate"
    assert samd.cough_type_label(0.85) == "wet"


def test_tc_phi_04_forbidden_model_inputs():
    samd.check_model_inputs(["duration", "n_coughs", "snr_db", "mfcc1_mean"])
    for bad in (["duration", "latitude"], ["longitude"], ["lat_bin"], ["age"], ["gender"]):
        with pytest.raises(RuntimeError):
            samd.check_model_inputs(bad)


def test_tc_phi_04_trained_feature_list_is_audio_only():
    """The handcrafted feature names the pipeline produces contain no metadata."""
    import numpy as np
    from cough.features import analyse
    sr = 16000
    rng = np.random.default_rng(0)
    x = rng.normal(scale=0.01, size=4 * sr)
    for t in (0.5, 1.5, 2.5):
        n = int(0.3 * sr)
        x[int(t * sr):int(t * sr) + n] += rng.normal(size=n) * np.exp(-np.arange(n) / (0.06 * sr))
    names = list(analyse((x / np.abs(x).max()).astype(np.float32))["features"])
    samd.check_model_inputs(names)


# ---------------- audit trail
def test_tc_aud_03_hash_chain_detects_tampering(tmp_path):
    log = audit.AuditLog(tmp_path)
    for i in range(5):
        log.write("inference", audit.new_request_id(), user_id="u1", org_id="o1", model_version="v",
                  input_sha256="ab" * 32, output={"i": i})
    assert audit.verify(tmp_path)["ok"] and audit.verify(tmp_path)["records"] == 5
    f = next(tmp_path.glob("audit-*.jsonl"))
    lines = f.read_text().splitlines()
    tampered = lines[:]
    tampered[2] = tampered[2].replace('"i":2', '"i":9')                       # edit an output
    f.write_text("\n".join(tampered) + "\n")
    r = audit.verify(tmp_path)
    assert not r["ok"] and r["first_bad"]["seq"] == 3
    f.write_text("\n".join(lines[:2] + lines[3:]) + "\n")                     # delete a record
    assert not audit.verify(tmp_path)["ok"]
    f.write_text("\n".join(lines) + "\n")
    assert audit.verify(tmp_path)["ok"]


def test_tc_aud_04_append_only_and_chain_survives_restart(tmp_path):
    a = audit.AuditLog(tmp_path)
    a.write("inference", "r1", output={})
    b = audit.AuditLog(tmp_path)                                               # new process, same log
    rec = b.write("inference", "r2", output={})
    assert rec["seq"] == 2 and audit.verify(tmp_path)["ok"]
    assert [x["request_id"] for x in audit.find(tmp_path, "r2")] == ["r2"]


def test_tc_aud_05_seal_makes_old_days_read_only(tmp_path):
    (tmp_path / "audit-20200101.jsonl").write_text("")
    log = audit.AuditLog(tmp_path)
    assert log.seal() == ["audit-20200101.jsonl"]
    assert not (tmp_path / "audit-20200101.jsonl").stat().st_mode & stat.S_IWUSR


def test_tc_aud_02_id_rules():
    assert audit.valid_id("user-001") and audit.valid_id("org:clinic_7")
    assert not audit.valid_id("jane.doe@mail.com") and not audit.valid_id("") and not audit.valid_id(None)
    assert not audit.valid_id("Jane Doe")


# ---------------- PHI
def test_tc_phi_03_audio_not_stored_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("COUGH_AUDIO_RETENTION_DAYS", raising=False)
    assert phi.store_audio(tmp_path, "r1", "org1", b"RIFF....", ".wav") is None
    assert not (tmp_path / "audio").exists()


def test_tc_phi_06_retention_and_purge(tmp_path, monkeypatch):
    monkeypatch.setenv("COUGH_AUDIO_RETENTION_DAYS", "7")
    p = phi.store_audio(tmp_path, "r1", "org1", b"audio", ".wav")
    assert p.exists()
    assert phi.purge_expired(tmp_path, now=time.time() + 6 * 86400) == 0
    assert phi.purge_expired(tmp_path, now=time.time() + 8 * 86400) == 1 and not p.exists()


def test_tc_phi_05_erasure(tmp_path, monkeypatch):
    monkeypatch.setenv("COUGH_AUDIO_RETENTION_DAYS", "30")
    for rid, user in (("r1", "u1"), ("r2", "u2"), ("r3", "u1")):
        phi.store_features(tmp_path, rid, user, "o1", "h" * 64, {"duration": 3.0}, "v")
        phi.store_audio(tmp_path, rid, "o1", b"audio", ".webm")
    r = phi.erase_subject(tmp_path, "u1")
    assert r["feature_records_deleted"] == 2 and r["audio_files_deleted"] == 2
    left = (tmp_path / "features.jsonl").read_text()
    assert "u1" not in left and "u2" in left
    assert sorted(p.name for p in (tmp_path / "audio" / "o1").iterdir()) == ["r2.meta.json", "r2.webm"]


# ---------------- versioned models
def _fake_version(root, part="minor"):
    v, d = versioning.new_version(root, part)
    (d / "config.json").write_text("{}")
    (d / "models.joblib").write_bytes(b"model")
    (d / "metrics.json").write_text(json.dumps({"results": {"cough": {"fusion": {"roc_auc": 0.98}}}}))
    return v, d


def test_tc_ver_01_model_card_has_version_data_hash_metrics(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "meta.parquet").write_bytes(b"meta")
    (cache / "features.parquet").write_bytes(b"feats")
    v, d = _fake_version(tmp_path / "art")
    card = versioning.register(tmp_path / "art", d, v, "minor", cache, {"epochs": 1})
    assert card["version"] == "1.0.0" and d.name.startswith("coughvid_v1.0.0_fusion_")
    assert len(card["data"]["training_data_sha256"]) == 64 and card["metrics_file"] == "metrics.json"
    assert set(card["file_sha256"]) == {"config.json", "models.joblib", "metrics.json"}
    assert "ffmpeg" in card["soup"] and "lightgbm" in card["soup"] and "scikit-learn" in card["soup"]


def test_tc_ver_03_integrity_and_no_overwrite(tmp_path):
    root = tmp_path / "art"
    v, d = _fake_version(root)
    versioning.register(root, d, v, "minor", None, {})
    assert versioning.verify_integrity(d) == []
    assert not (d / "models.joblib").stat().st_mode & stat.S_IWUSR           # read-only after registration
    with pytest.raises(FileExistsError):
        versioning.register(root, d, v, "minor", None, {})                    # never re-registered
    (d / "models.joblib").chmod(0o666)
    (d / "models.joblib").write_bytes(b"changed")
    assert versioning.verify_integrity(d) == ["models.joblib"]


def test_tc_ver_04_train_refuses_to_overwrite(tmp_path):
    pytest.importorskip("torch")
    from cough import train
    (tmp_path / "models.joblib").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        train.run(tmp_path / "cache", tmp_path, epochs=1)


def test_tc_doc_01_regulatory_documents_present():
    docs = Path(__file__).resolve().parent.parent / "docs" / "regulatory"
    names = {p.name for p in docs.glob("*.md")}
    for need in ("01_intended_use.md", "05_risk_management_file.md", "06_soup_list.md",
                 "07_verification_test_protocol.md", "08_traceability_matrix.md"):
        assert need in names, f"missing {need}"
    assert samd.INTENDED_USE in (docs / "01_intended_use.md").read_text(encoding="utf-8")


def test_tc_ver_07_migration_adds_hashes_once(tmp_path):
    root = tmp_path / "art"
    v, d = _fake_version(root)
    versioning.register(root, d, v, "minor", None, {})
    card = json.loads((d / "version.json").read_text())
    card.pop("file_sha256")                                   # simulate a card from before integrity checks
    (d / "version.json").chmod(0o666)
    (d / "version.json").write_text(json.dumps(card))
    assert versioning.verify_integrity(d) == ["model card has no file hashes"]
    out = versioning.add_missing_hashes(root, v)
    assert out["hashes_added"] and versioning.verify_integrity(d) == []
    with pytest.raises(FileExistsError):
        versioning.add_missing_hashes(root, v)                # never twice
