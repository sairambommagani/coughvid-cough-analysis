# 06 SOUP list (software of unknown provenance)

IEC 62304: 5.3.3 (functional and performance requirements of SOUP), 5.3.4 (hardware and software required by
SOUP), 7.1.2 / 7.1.3 (hazards from SOUP, known anomalies), 8.1.2 (identification). The CDSCO software guidance
also expects an SBOM with vulnerability monitoring: this list plus `requirements.lock.txt` is that SBOM.

Exact versions: `python scripts/soup_versions.py` writes 06a_soup_versions.md and `requirements.lock.txt` from the
environment that passed the release tests; every model card also records the SOUP versions it was trained with.
Minimum versions are in `requirements.txt`.

## 1. SOUP items

| Id | SOUP | Supplier | Licence | Used by | Purpose / requirement on it (5.3.3) | Risk if it fails (7.1.2) | Risk level | Verification |
|---|---|---|---|---|---|---|---|---|
| S-01 | ffmpeg (binary; from PATH or bundled by imageio-ffmpeg) | FFmpeg project | LGPL-2.1+ / GPL (build dependent; licence review before distribution) | SI-4 | decode WebM/Ogg/MP4/WAV/MP3/AAC/FLAC/AMR to 16 kHz mono f32 PCM; must decode all supported formats and fail with a non-zero exit on corrupt input | wrong samples -> wrong output; parser vulnerability (untrusted input); hang | High | TC-FMT-02 (10 formats decode to the same signal), TC-FMT-06 (truncated file fails), 30 s limit, format check first |
| S-02 | imageio-ffmpeg | imageio | BSD-2 | SI-4 | locate a working ffmpeg binary | no decoder -> service unavailable (fails safe) | Low | test suite needs ffmpeg |
| S-03 | PyTorch | Linux Foundation / Meta | BSD-3 | SI-5 | CNN inference on CPU, deterministic in eval mode; load `cnn.pt` | wrong CNN logits -> wrong fused score | Medium | integration test scores, model-file hash (RC-12) |
| S-04 | LightGBM | Microsoft | MIT | SI-5 | gradient-boosted model inference and SHAP contributions | wrong score or wrong reasons | Medium | integration tests, test-set metrics reproduced from the model card |
| S-05 | scikit-learn | scikit-learn developers | BSD-3 | SI-5, SI-10 | logistic regression, scaling, metrics | wrong PANNs-branch score, wrong metrics | Medium | metric unit tests (`test_metrics`) |
| S-06 | PANNs CNN14 (code + checkpoint `Cnn14_mAP=0.431.pth`) | Kong et al. (Zenodo record 3987831) | code MIT; checkpoint licence to be confirmed before commercial use | SI-5 | 2048-d audio embedding and AudioSet tags | wrong embedding -> wrong score; checkpoint missing -> PANNs dropped from fusion (handled) | Medium | integration test; fusion without PANNs tested |
| S-07 | NumPy | NumPy developers | BSD-3 | all | arrays, signal arithmetic | wrong features | Medium | unit tests (segmentation, features) |
| S-08 | pandas, pyarrow | pandas / Apache | BSD-3 / Apache-2.0 | SI-10, SI-1 | metadata, parquet files | training data errors (offline) | Low | data unit tests |
| S-09 | librosa, soundfile | librosa / soundfile developers | ISC / BSD-3 | SI-4 | log-mel, MFCC, spectral features | wrong features | Medium | feature tests, audio-only feature check (TC-PHI-04) |
| S-10 | SciPy | SciPy developers | BSD-3 | SI-4, SI-10 | signal and statistics | wrong features | Low | unit tests |
| S-11 | joblib | joblib developers | BSD-3 | SI-5, SI-8 | load the model bundle | wrong / tampered model loaded | Medium | model-file hash before load (RC-12) |
| S-12 | FastAPI, Starlette, Uvicorn, python-multipart | FastAPI / Encode / Kludex | MIT / BSD-3 / Apache-2.0 | SI-1 | HTTP server, routing, multipart parsing | request mixed up, wrong status | Medium | API tests (TC-FMT-04..08, TC-AUD-*) |
| S-13 | Python 3.11 runtime | Python Software Foundation | PSF | all | interpreter | any | Medium | test suite on the release environment |
| S-14 | Browser MediaRecorder API (client) | browser vendors | - | SI-9 | record Opus or MP4 | wrong format named | Low | server re-detects the format (RC-08) |
| S-15 | matplotlib, pytest, httpx | various | PSF-based / MIT / BSD-3 | development only | figures, tests | none at run time | Low | - |

Hardware and software required by SOUP (5.3.4): x86-64 or ARM64 CPU, 4 cores, 8 GB RAM, about 2 GB disk for
models and SOUP; Windows 10+ or Linux; no GPU required.

## 2. Known anomalies (7.1.3)

Before each release the published issue trackers and vulnerability databases (NVD / GitHub advisories) are checked
for the exact versions in 06a, with the result recorded below. Each anomaly is judged: does it affect the functions
we use (section 1), and can it lead to a hazardous situation?

| Date checked | SOUP | Anomaly / advisory | Affects our use? | Action |
|---|---|---|---|---|
| (at release) | ffmpeg | decoder memory-safety advisories are regular | yes: we parse untrusted audio | keep ffmpeg current; format check + size limit + separate process + timeout (RC-08, RC-09) |
| (at release) | PyTorch | `torch.load` of untrusted files can execute code | only if the model file were replaced | model files hash-verified before loading and read-only (RC-12); only our own files are loaded |
| (at release) | joblib / pickle | loading untrusted pickles executes code | same as above | same as above |

## 3. SOUP monitoring

Monthly and before every release: check new versions and advisories for every item above, update 06a and the lock
file, rerun the full test suite, and record the result in the table in section 2.
