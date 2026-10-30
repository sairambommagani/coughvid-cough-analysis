"""Format matrix tests: every input format real users send x every step (detect, decode, API), plus the files that
must be rejected, plus model versioning.

  python -m pytest -q tests/test_formats.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from cough.audio import SR, ffmpeg_available, load_audio, segment_coughs
from cough.formats import extension_matches, sniff
from cough import versioning

needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")

# generator format -> (detected format, detected codec or None when the container header does not show it)
MATRIX = {
    "webm_opus": ("webm", "opus"),
    "ogg_opus": ("ogg", "opus"),
    "ogg_vorbis": ("ogg", "vorbis"),
    "mp4_aac": ("mp4", None),
    "m4a_aac": ("m4a", None),
    "aac_adts": ("aac", "aac"),
    "mp3": ("mp3", "mp3"),
    "wav_pcm16": ("wav", "pcm"),
    "wav_8k": ("wav", "pcm"),
    "flac": ("flac", "flac"),
}

REJECT = {
    "png": (b"\x89PNG\r\n\x1a\n" + bytes(200), "PNG image"),
    "pdf": (b"%PDF-1.7\n" + b"x" * 200, "PDF document"),
    "zip": (b"PK\x03\x04" + bytes(200), "ZIP"),
    "text": (b"hello, this is not a cough recording at all\n" * 5, "text"),
    "json": (b'{"cough": true}' + b" " * 100, "JSON"),
    "random": (bytes(np.random.default_rng(0).integers(0, 256, 400, dtype=np.uint8).tolist()), "unknown"),
}


@pytest.fixture(scope="module")
def samples(tmp_path_factory):
    """One synthetic cough recording (3 coughs) per format, written by the synthetic generator."""
    if not ffmpeg_available():
        pytest.skip("ffmpeg not installed")
    from scripts.make_synthetic import FORMATS, sample
    d = tmp_path_factory.mktemp("formats")
    return {f: sample(d / f"cough_{f}", f, kind="healthy", seed=3) for f in FORMATS}


def test_matrix_covers_every_generator_format():
    from scripts.make_synthetic import FORMATS
    assert set(MATRIX) == set(FORMATS)


@needs_ffmpeg
@pytest.mark.parametrize("fmt", list(MATRIX))
def test_detect(samples, fmt):
    info = sniff(samples[fmt].read_bytes()[:4096])
    want_fmt, want_codec = MATRIX[fmt]
    assert info["supported"], info
    assert info["format"] == want_fmt
    if want_codec:
        assert info["codec"] == want_codec


@needs_ffmpeg
@pytest.mark.parametrize("fmt", list(MATRIX))
def test_decode_to_model_input(samples, fmt):
    """Every format becomes the same model input: 16 kHz mono, normalised, same length, coughs still found."""
    ref = load_audio(samples["wav_pcm16"])
    x = load_audio(samples[fmt])
    assert x.dtype == np.float32 and x.ndim == 1
    assert abs(np.abs(x).max() - 1.0) < 1e-3                        # peak-normalised
    assert abs(len(x) - len(ref)) < 0.15 * SR                         # codec padding at most ~0.1 s
    segs, _ = segment_coughs(x)
    ref_segs, _ = segment_coughs(ref)
    assert abs(len(segs) - len(ref_segs)) <= 1, (fmt, len(segs), len(ref_segs))


@needs_ffmpeg
def test_wrong_extension_is_detected_and_still_decodes(samples, tmp_path):
    renamed = tmp_path / "actually_mp3.webm"
    renamed.write_bytes(samples["mp3"].read_bytes())
    info = sniff(renamed.read_bytes()[:4096])
    assert info["format"] == "mp3" and extension_matches(renamed.name, "mp3") is False
    good = tmp_path / ("fixed" + info["extension"])
    good.write_bytes(renamed.read_bytes())
    assert len(load_audio(good)) > SR


@pytest.mark.parametrize("name", list(REJECT))
def test_not_audio_is_rejected(name):
    data, why = REJECT[name]
    info = sniff(data)
    assert not info["supported"] and info["format"] is None
    assert why.split()[0].lower() in info["reason"].lower()


def test_empty_and_tiny():
    assert not sniff(b"")["supported"] and not sniff(b"Og")["supported"]


@needs_ffmpeg
def test_truncated_file_fails_to_decode(samples, tmp_path):
    bad = tmp_path / "cut.webm"
    bad.write_bytes(samples["webm_opus"].read_bytes()[:60])            # header only
    assert sniff(bad.read_bytes())["format"] == "webm"
    with pytest.raises(ValueError):
        x = load_audio(bad)
        if len(x) == 0:
            raise ValueError("no audio")


# ------------------------------------------------------------------ API: the same matrix through POST /analyze
class FakeEngine:
    """Stands in for the trained models, so the format layer is tested without training. It still decodes the
    file with the real loader, exactly as the real engine does."""
    version = {"name": "coughvid_v0.0.1_fusion_test", "version": "0.0.1"}
    metrics = {"results": {}}
    card_sha256 = "0" * 64
    integrity = "verified"

    def analyse_file(self, path):
        x = load_audio(path)
        if len(x) < 1600:
            raise ValueError("no audio decoded")
        segs, snr = segment_coughs(x)
        res = {"recording": {"duration_s": round(len(x) / SR, 2), "coughs_found": len(segs), "snr_db": snr},
               "cough_check": {"is_cough": bool(segs), "probability": 0.9}, "model_version": self.version["name"]}
        return res, {"duration": len(x) / SR, "n_coughs": float(len(segs))}


IDS = {"X-User-Id": "user-001", "X-Org-Id": "org-001"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    pytest.importorskip("torch")
    from fastapi.testclient import TestClient
    import cough.api as api
    d = tmp_path_factory.mktemp("api")
    old = api.AUDIT_DIR, api.DATA_DIR
    api.AUDIT_DIR, api.DATA_DIR = d / "audit", d / "store"
    api._cache.pop("a", None)
    api._cache["e"] = FakeEngine()
    with TestClient(api.app, headers=IDS) as c:
        yield c
    api._cache.pop("e", None)
    api._cache.pop("a", None)
    api.AUDIT_DIR, api.DATA_DIR = old


@needs_ffmpeg
@pytest.mark.parametrize("fmt", list(MATRIX))
@pytest.mark.parametrize("declared", ["right", "wrong_name", "no_name_type"])
def test_api_accepts_every_format(client, samples, fmt, declared):
    f = samples[fmt]
    name, ctype = {"right": (f.name, "audio/*"), "wrong_name": ("recording.webm", "audio/webm"),
                   "no_name_type": ("blob", "application/octet-stream")}[declared]
    r = client.post("/analyze", files={"file": (name, f.read_bytes(), ctype)})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["input"]["format"] == MATRIX[fmt][0]
    assert j["input"]["converted_to"] == "16 kHz mono PCM"
    assert j["recording"]["coughs_found"] >= 2
    assert j["model_version"].startswith("coughvid_v")
    if declared == "wrong_name" and MATRIX[fmt][0] != "webm":
        assert j["input"]["extension_matches"] is False


@pytest.mark.parametrize("name", list(REJECT))
def test_api_rejects_non_audio(client, name):
    r = client.post("/analyze", files={"file": (f"x.{name}", REJECT[name][0], "audio/webm")})
    assert r.status_code == 415 and "Supported audio formats" in r.json()["detail"]


def test_api_empty_and_too_large(client):
    assert client.post("/analyze", files={"file": ("x.wav", b"", "audio/wav")}).status_code == 400
    big = b"RIFF" + bytes(4) + b"WAVE" + bytes(20_000_001)
    assert client.post("/analyze", files={"file": ("x.wav", big, "audio/wav")}).status_code == 413


@needs_ffmpeg
def test_api_truncated_is_422(client, samples):
    data = samples["webm_opus"].read_bytes()[:60]
    r = client.post("/analyze", files={"file": ("cut.webm", data, "audio/webm")})
    assert r.status_code == 422


def test_api_lists_formats(client):
    j = client.get("/formats").json()
    assert {"webm", "ogg", "mp4", "m4a", "wav", "mp3"} <= set(j["supported"])


# ------------------------------------------------------------------ versioning
def test_semver_and_names():
    assert versioning.bump(None, "minor") == "1.0.0"
    assert versioning.bump("1.4.2", "major") == "2.0.0"
    assert versioning.bump("1.4.2", "minor") == "1.5.0"
    assert versioning.bump("1.4.2", "patch") == "1.4.3"
    assert versioning.version_name("1.1.0", "20261028") == "coughvid_v1.1.0_fusion_20261028"
    with pytest.raises(ValueError):
        versioning.parse("1.1")


def test_registry_latest_and_rollback(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "config.json").write_text("{}")                             # old flat layout, trained before versioning
    (root / "metrics.json").write_text(json.dumps({"results": {"covid": {"fusion": {"roc_auc": 0.6}}}}))
    assert versioning.resolve_artifacts(root) == root
    card = versioning.adopt_legacy(root)
    assert card["version"] == "1.0.0" and versioning.resolve_artifacts(root).name == card["name"]
    for part, want in (("minor", "1.1.0"), ("patch", "1.1.1"), ("major", "2.0.0")):
        v, d = versioning.new_version(root, part)
        assert v == want and d.name.startswith(f"coughvid_v{want}_fusion_")
        (d / "config.json").write_text("{}")
        (d / "metrics.json").write_text(json.dumps({"results": {"covid": {"fusion": {"roc_auc": 0.7}}}}))
        c = versioning.register(root, d, v, part, None, {"epochs": 1}, notes=f"test {part}", parent=None)
        assert (d / "version.json").exists() and c["test_roc_auc"]["covid"]["fusion"] == 0.7
    reg = versioning.load_registry(root)
    assert [v["version"] for v in reg["versions"]] == ["1.0.0", "1.1.0", "1.1.1", "2.0.0"]
    assert versioning.resolve_artifacts(root).name.startswith("coughvid_v2.0.0_")
    monkeypatch.setenv("COUGH_MODEL_VERSION", "1.1.0")                 # rollback
    assert versioning.resolve_artifacts(root).name.startswith("coughvid_v1.1.0_")
    with pytest.raises(ValueError):
        versioning.new_version(root, version="1.1.0")                   # no duplicate versions


def test_crashed_run_and_unadopted_legacy(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "config.json").write_text("{}")                             # flat model, never adopted
    v, d = versioning.new_version(root, "minor")
    assert v == "1.1.0"                                                  # legacy was adopted as 1.0.0 first
    assert versioning.load_registry(root)["versions"][0]["version"] == "1.0.0"
    v2, d2 = versioning.new_version(root, "minor")                       # previous run crashed before register
    assert v2 == "1.1.0" and d2.exists()


# ------------------------------------------------------------------ audit and decoder limits through the API
@needs_ffmpeg
def test_tc_inp_05_decoder_time_limit(samples, monkeypatch):
    import cough.audio as audio
    monkeypatch.setattr(audio, "FFMPEG_TIMEOUT_S", 0.000001)
    with pytest.raises(ValueError, match="took longer"):
        audio.load_audio(samples["webm_opus"])


@needs_ffmpeg
def test_tc_aud_06_rejected_requests_are_audited(client):
    import cough.api as api
    from cough import audit as auditmod
    r = client.post("/analyze", files={"file": ("x.png", REJECT["png"][0], "image/png")})
    assert r.status_code == 415
    recs = auditmod.find(api.AUDIT_DIR, r.json()["request_id"])
    assert len(recs) == 1 and recs[0]["event"] == "rejected" and recs[0]["status"] == 415
    assert recs[0]["input_sha256"] and recs[0]["user_id"] == "user-001"
    assert auditmod.verify(api.AUDIT_DIR)["ok"]


@needs_ffmpeg
def test_tc_aud_07_no_result_without_audit_record(client, samples, monkeypatch):
    import cough.api as api

    def broken(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(api.audit(), "write", broken)
    f = samples["wav_pcm16"]
    r = client.post("/analyze", files={"file": (f.name, f.read_bytes(), "audio/wav")})
    assert r.status_code == 503 and "cough_check" not in r.json()
