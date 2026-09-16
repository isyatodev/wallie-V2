"""TTS MP3→PCM auto-decode (tts/decode.py + tts/openai_tts.py).

When a gateway ignores ``response_format: "pcm"`` and answers MP3/OGG/WAV,
the OpenAI-compatible TTS client buffers the payload and decodes it via
ffmpeg. Without ffmpeg the sentence fails LOUD with an actionable error —
never static on the speakers.
"""
from __future__ import annotations

import types

import httpx
import pytest

from tts.decode import decode_to_pcm16, find_ffmpeg, sniff_compressed
from tts.openai_tts import OpenAICompatibleTTS


# ---------------------------------------------------------------------------
# Sniffer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("head,kind", [
    (b"ID3\x04\x00\x00", "mp3"),                       # ID3 tag
    (b"\xff\xfb\x90\x44", "mp3"),                      # sync frame, layer III
    (b"RIFF\x24\x00\x00\x00WAVE", "wav"),              # WAV container
    (b"OggS\x00\x02", "ogg"),
    (b'{"error":"nope"}', "json-or-xml"),              # JSON error with 200
    (b"<?xml ", "json-or-xml"),
])
def test_sniff_detects_compressed(head, kind):
    assert sniff_compressed(head) == kind


@pytest.mark.parametrize("head", [
    b"",                        # empty
    b"\x00\x00\x01\x00\x02\x00\x03\x00",   # plausible PCM16
    b"\xff\x7f\xfe\x7f\xfd\x7f",          # loud PCM (0x7F & 0xE0 ≠ 0xE0 → no sync word)
    b"\x01\x02",                            # too short to judge
])
def test_sniff_leaves_pcm_alone(head):
    assert sniff_compressed(head) is None


# ---------------------------------------------------------------------------
# ffmpeg discovery
# ---------------------------------------------------------------------------

def test_find_ffmpeg_env_override(monkeypatch):
    monkeypatch.setenv("WALLIE_FFMPEG", "C:/tools/ffmpeg.exe")
    assert find_ffmpeg() == "C:/tools/ffmpeg.exe"


def test_find_ffmpeg_path_lookup(monkeypatch):
    monkeypatch.setenv("WALLIE_FFMPEG", "")
    monkeypatch.setattr("tts.decode.shutil.which", lambda name: "C:/bin/ffmpeg.exe" if name == "ffmpeg" else None)
    assert find_ffmpeg() == "C:/bin/ffmpeg.exe"


def test_find_ffmpeg_imageio_wheel(monkeypatch):
    monkeypatch.setenv("WALLIE_FFMPEG", "")
    monkeypatch.setattr("tts.decode.shutil.which", lambda name: None)
    fake = types.ModuleType("imageio_ffmpeg")
    fake.get_ffmpeg_exe = lambda: "C:/wheels/ffmpeg.exe"
    monkeypatch.setitem(__import__("sys").modules, "imageio_ffmpeg", fake)
    assert find_ffmpeg() == "C:/wheels/ffmpeg.exe"


def test_find_ffmpeg_absent(monkeypatch):
    monkeypatch.setenv("WALLIE_FFMPEG", "")
    monkeypatch.setattr("tts.decode.shutil.which", lambda name: None)
    import sys
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)  # import → ImportError
    assert find_ffmpeg() is None


# ---------------------------------------------------------------------------
# decode_to_pcm16 (subprocess mocked — no real ffmpeg on CI machines)
# ---------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, out: bytes = b"", err: bytes = b"", rc: int = 0):
        self._out, self._err, self._rc = out, err, rc
        self.returncode = rc
        self.killed = False

    async def communicate(self, data: bytes):
        self.stdin_data = data
        return self._out, self._err

    def kill(self):
        self.killed = True

    async def wait(self):
        return self._rc


def _patch_subprocess(monkeypatch, proc: _FakeProc) -> list[list]:
    """Swap the asyncio module seen by tts.decode for a fake with PIPE consts."""
    calls: list[list] = []

    async def fake_exec(*cmd, **kwargs):
        calls.append(list(cmd))
        return proc

    fake_asyncio = types.SimpleNamespace(
        create_subprocess_exec=fake_exec,
        subprocess=types.SimpleNamespace(PIPE=-1),
    )
    monkeypatch.setattr("tts.decode.asyncio", fake_asyncio)
    return calls


@pytest.mark.asyncio
async def test_decode_invokes_ffmpeg_with_s16le_args(monkeypatch):
    proc = _FakeProc(out=b"\x01\x00\x02\x00\x03\x00")
    calls = _patch_subprocess(monkeypatch, proc)

    out = await decode_to_pcm16(b"MP3DATA", "ffmpeg.exe", 24000)
    assert out == b"\x01\x00\x02\x00\x03\x00"
    assert proc.stdin_data == b"MP3DATA"
    cmd = calls[0]
    assert cmd[0] == "ffmpeg.exe"
    assert "-f" in cmd and cmd[cmd.index("-f") + 1] == "s16le"
    assert "-ar" in cmd and cmd[cmd.index("-ar") + 1] == "24000"
    assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "1"


@pytest.mark.asyncio
async def test_decode_trims_odd_final_byte(monkeypatch):
    _patch_subprocess(monkeypatch, _FakeProc(out=b"\x01\x00\x02\x00\x02"))
    out = await decode_to_pcm16(b"MP3DATA", "ffmpeg.exe", 24000)
    assert len(out) % 2 == 0


@pytest.mark.asyncio
async def test_decode_failure_raises_with_stderr_tail(monkeypatch):
    _patch_subprocess(monkeypatch, _FakeProc(err=b"line1\nInvalid data found\n", rc=1))
    with pytest.raises(RuntimeError, match="Invalid data found"):
        await decode_to_pcm16(b"garbage", "ffmpeg.exe", 24000)


# ---------------------------------------------------------------------------
# Client orchestration: sniff on first chunk, then stream / buffer+decode
# ---------------------------------------------------------------------------

def _client() -> OpenAICompatibleTTS:
    return OpenAICompatibleTTS(
        api_key="k", base_url="https://x/v1", model="tts-1", voice="alloy"
    )


@pytest.mark.asyncio
async def test_client_pcm_streams_unchanged():
    """Genuine raw PCM keeps the zero-buffer streaming path."""
    pcm = bytes(range(256)) * 4

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=pcm)

    t = _client()
    t._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chunks = [c async for c in t.synthesize("hello")]
    assert chunks == [pcm]
    await t.aclose()


@pytest.mark.asyncio
async def test_client_mp3_without_ffmpeg_fails_actionable(monkeypatch):
    mp3 = b"ID3\x04\x00\x00\x00" + b"\x00" * 64

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=mp3)

    monkeypatch.setattr("tts.openai_tts.find_ffmpeg", lambda: None)
    t = _client()
    t._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(TTSError := __import__("tts.base", fromlist=["TTSError"]).TTSError, match="ffmpeg"):
        [c async for c in t.synthesize("hello")]
    await t.aclose()


@pytest.mark.asyncio
async def test_client_mp3_with_ffmpeg_decodes_whole_payload(monkeypatch):
    """All chunks (first sniffed + the rest) reach the decoder as one blob."""
    mp3 = b"\xff\xfb\x90\x44" + b"frame" * 8
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        # Streamed in pieces, like a real gateway (async iterator: the client is async).
        async def pieces():
            yield mp3[:3]
            yield mp3[3:12]
            yield mp3[12:]
        return httpx.Response(200, content=pieces())

    async def fake_decode(data: bytes, ffmpeg: str, sample_rate: int) -> bytes:
        seen["blob"], seen["ffmpeg"], seen["rate"] = data, ffmpeg, sample_rate
        return b"\x00\x00\x01\x00"

    monkeypatch.setattr("tts.openai_tts.find_ffmpeg", lambda: "fake-ffmpeg")
    monkeypatch.setattr("tts.openai_tts.decode_to_pcm16", fake_decode)
    t = _client()
    t._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chunks = [c async for c in t.synthesize("hello")]
    assert chunks == [b"\x00\x00\x01\x00"]
    assert seen["blob"] == mp3          # complete payload, in order
    assert seen["ffmpeg"] == "fake-ffmpeg"
    assert seen["rate"] == t.sample_rate
    await t.aclose()


@pytest.mark.asyncio
async def test_client_wav_without_ffmpeg_names_the_format(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 20)

    monkeypatch.setattr("tts.openai_tts.find_ffmpeg", lambda: None)
    t = _client()
    t._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(Exception, match="wav"):
        [c async for c in t.synthesize("hello")]
    await t.aclose()


# ---------------------------------------------------------------------------
# End-to-end with a REAL mp3 (skipped on machines without ffmpeg)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_client_decodes_real_mp3_end_to_end():
    """ffmpeg encodes a sine → the client sniffs + decodes it back to PCM."""
    import asyncio

    ff = find_ffmpeg()
    if not ff:
        pytest.skip("ffmpeg not available on this machine")

    proc = await asyncio.create_subprocess_exec(
        ff, "-v", "error", "-f", "lavfi", "-i",
        "sine=frequency=440:duration=0.3:sample_rate=24000",
        "-f", "mp3", "-ac", "1", "-ar", "24000", "-b:a", "64k", "pipe:1",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    mp3, err = await proc.communicate()
    assert proc.returncode == 0, err
    assert sniff_compressed(mp3[:12]) == "mp3"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=mp3)  # gateway ignored response_format=pcm

    t = _client()
    t._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    pcm = b"".join([c async for c in t.synthesize("hello")])
    await t.aclose()
    assert len(pcm) > 2000                       # 0.3s @ 24kHz ≈ 14k bytes
    assert len(pcm) % 2 == 0
    # 440 Hz sine has real amplitude, not digital silence.
    assert max(abs(int.from_bytes(pcm[i:i + 2], "little", signed=True)) for i in range(0, 4000, 2)) > 1000
