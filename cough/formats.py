"""Input format check: the layer in front of the models.

Phones, browsers and regions produce different audio formats (Chrome/Firefox record WebM or Ogg with Opus, Safari and
iPhones record MP4/M4A with AAC, many Android apps and messengers send Ogg/Opus, 3GP or AMR, people upload MP3 or WAV).
Before anything reaches the models, an upload is checked here:

  1. detect   the real format from the file's first bytes ("magic numbers"), not from the file name or the
              Content-Type the client claims (both are often wrong: an MP3 renamed to .webm is still an MP3)
  2. decide   supported -> continue; unknown / not audio (text, image, PDF, zip...) -> rejected with a clear message
  3. convert  every supported format is decoded by ffmpeg to the one format the models were trained on:
              16 kHz mono float PCM (cough.audio.load_audio). A file that looks like audio but cannot be decoded
              (truncated, corrupted) is rejected as well.

  sniff(first_bytes) -> {"format": "webm", "codec": "opus", "mime": "audio/webm", "supported": True, ...}
"""
from __future__ import annotations

import re

# format -> (extension used for the temporary file, mime type, description)
SUPPORTED = {
    "wav": (".wav", "audio/wav", "WAV (PCM)"),
    "mp3": (".mp3", "audio/mpeg", "MP3"),
    "webm": (".webm", "audio/webm", "WebM (Opus/Vorbis) - Chrome, Firefox, Edge recorder"),
    "ogg": (".ogg", "audio/ogg", "Ogg (Opus/Vorbis/FLAC) - Firefox, Android, WhatsApp-style voice notes"),
    "mp4": (".mp4", "audio/mp4", "MP4 (AAC) - Safari / iPhone recorder"),
    "m4a": (".m4a", "audio/mp4", "M4A (AAC/ALAC) - iPhone voice memos"),
    "3gp": (".3gp", "audio/3gpp", "3GP (AMR/AAC) - older Android phones"),
    "aac": (".aac", "audio/aac", "AAC (ADTS stream)"),
    "flac": (".flac", "audio/flac", "FLAC"),
    "amr": (".amr", "audio/amr", "AMR - phone voice recordings"),
    "aiff": (".aiff", "audio/aiff", "AIFF"),
    "mkv": (".mkv", "audio/x-matroska", "Matroska audio"),
}
NOT_AUDIO = [(b"\x89PNG", "PNG image"), (b"\xff\xd8\xff", "JPEG image"), (b"GIF8", "GIF image"),
             (b"%PDF", "PDF document"), (b"PK\x03\x04", "ZIP archive / Office document"),
             (b"{", "JSON / text"), (b"<", "HTML / XML text")]


def _mp3_frame(b: bytes) -> bool:
    """MPEG audio frame header: 11 sync bits, layer III, valid bitrate and sample-rate fields."""
    return (len(b) >= 4 and b[0] == 0xFF and (b[1] & 0xE0) == 0xE0 and ((b[1] >> 1) & 0x03) == 0x01
            and (b[2] >> 4) not in (0x0, 0xF) and ((b[2] >> 2) & 0x03) != 0x03)


def _adts(b: bytes) -> bool:
    """AAC ADTS header: 12 sync bits, layer 00."""
    return len(b) >= 2 and b[0] == 0xFF and (b[1] & 0xF6) == 0xF0


def sniff(head: bytes) -> dict:
    """Detect the container and (when visible in the first bytes) the codec. Pass the first ~4 KB of the file."""
    b = head or b""
    fmt, codec = None, None
    if len(b) < 4:
        return result(None, None, reason="empty or too small to be audio")
    if b[:4] == b"RIFF" and b[8:12] == b"WAVE":
        fmt = "wav"
        codec = {1: "pcm", 3: "pcm_float", 0x11: "ima_adpcm", 0x55: "mp3", 0xFFFE: "extensible"}.get(
            int.from_bytes(b[20:22], "little"), "other") if len(b) >= 22 else None
    elif b[:3] == b"ID3" or _mp3_frame(b):
        fmt, codec = "mp3", "mp3"
    elif b[:4] == b"OggS":
        fmt = "ogg"
        codec = "opus" if b"OpusHead" in b else "vorbis" if b"\x01vorbis" in b else "flac" if b"\x7fFLAC" in b else None
    elif b[:4] == b"\x1a\x45\xdf\xa3":                               # EBML: WebM / Matroska
        fmt = "webm" if b"webm" in b[:64] else "mkv"
        codec = "opus" if b"A_OPUS" in b else "vorbis" if b"A_VORBIS" in b else "aac" if b"A_AAC" in b else None
    elif b[4:8] == b"ftyp":                                          # ISO base media: MP4 / M4A / 3GP / MOV
        brand = b[8:12].decode("latin-1").strip().lower()
        fmt = "m4a" if brand.startswith("m4a") else "3gp" if brand.startswith("3g") else "mp4"
        codec = "aac" if b"mp4a" in b else "alac" if b"alac" in b else None
        if brand == "qt":
            fmt = "mp4"
    elif b[:4] == b"fLaC":
        fmt, codec = "flac", "flac"
    elif b[:5] == b"#!AMR":
        fmt, codec = "amr", "amr_wb" if b[:8] == b"#!AMR-WB" else "amr_nb"
    elif b[:4] == b"FORM" and b[8:12] in (b"AIFF", b"AIFC"):
        fmt, codec = "aiff", "pcm"
    elif _adts(b):
        fmt, codec = "aac", "aac"
    if fmt:
        return result(fmt, codec)
    for magic, name in NOT_AUDIO:
        if b.lstrip()[: len(magic)] == magic:
            return result(None, None, reason=f"this is a {name}, not an audio recording")
    if all(32 <= c < 127 or c in (9, 10, 13) for c in b[:256]):
        return result(None, None, reason="this is a text file, not an audio recording")
    return result(None, None, reason="unknown file type")


def result(fmt, codec, reason: str | None = None) -> dict:
    ok = fmt in SUPPORTED
    return {"format": fmt, "codec": codec, "supported": ok, "mime": SUPPORTED[fmt][1] if ok else None,
            "extension": SUPPORTED[fmt][0] if ok else None,
            "reason": None if ok else (reason or f"format '{fmt}' is not supported")}


def supported_list() -> str:
    return ", ".join(sorted(SUPPORTED))


def extension_matches(filename: str | None, fmt: str | None) -> bool | None:
    """Does the uploaded file name agree with the detected format? (None when there is no extension.)"""
    m = re.search(r"\.([A-Za-z0-9]+)$", filename or "")
    if not m or not fmt:
        return None
    ext = m.group(1).lower()
    same = {"m4a": {"m4a", "mp4", "aac"}, "mp4": {"mp4", "m4a", "mov"}, "ogg": {"ogg", "oga", "opus"},
            "webm": {"webm", "weba"}, "aiff": {"aiff", "aif"}, "3gp": {"3gp", "3gpp"}}
    return ext in same.get(fmt, {fmt})
