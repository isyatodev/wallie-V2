"""TTS output-device selection (Voice page "Output device" dropdown).

Covers: the backend resolver that turns the saved spec into a concrete
device index, the /api/audio-devices route the dropdown is fed from
(host-API name + default flag), graceful behavior when PortAudio
itself is unavailable, and the AudioPlayer drained-signal race that
cut the voice test short.
"""
from __future__ import annotations

import asyncio

import pytest


# ---------------------------------------------------------------------------
# resolve_output_device
# ---------------------------------------------------------------------------

def test_resolve_output_device_passthrough():
    from audio.player import resolve_output_device

    assert resolve_output_device(None) is None
    assert resolve_output_device("") is None
    assert resolve_output_device(3) == 3
    assert resolve_output_device("3") == 3  # numeric string = index


def test_resolve_output_device_name_matching_prefers_wasapi(monkeypatch):
    """A bare name matches across host APIs; WASAPI wins when present."""
    import sounddevice as sd

    from audio.player import resolve_output_device

    devices = [
        {"name": "CABLE Input (USB Audio CODEC)", "hostapi": 0, "max_output_channels": 2},
        {"name": "CABLE Input (USB Audio CODEC)", "hostapi": 1, "max_output_channels": 2},
        {"name": "Speakers (Realtek)", "hostapi": 1, "max_output_channels": 2},
    ]
    monkeypatch.setattr(sd, "query_devices", lambda: devices)
    monkeypatch.setattr(sd, "query_hostapis", lambda: [{"name": "MME"}, {"name": "WASAPI"}])

    assert resolve_output_device("CABLE Input") == 1  # WASAPI instance


def test_resolve_output_device_falls_back_when_no_wasapi(monkeypatch):
    import sounddevice as sd

    from audio.player import resolve_output_device

    devices = [{"name": "Speakers (Realtek)", "hostapi": 0, "max_output_channels": 2}]
    monkeypatch.setattr(sd, "query_devices", lambda: devices)
    monkeypatch.setattr(sd, "query_hostapis", lambda: [{"name": "MME"}])

    assert resolve_output_device("speakers") == 0  # case-insensitive substring


def test_resolve_output_device_unknown_name_passes_through(monkeypatch):
    import sounddevice as sd

    from audio.player import resolve_output_device

    monkeypatch.setattr(sd, "query_devices", lambda: [])
    monkeypatch.setattr(sd, "query_hostapis", lambda: [])

    assert resolve_output_device("Ghost Device") == "Ghost Device"


# ---------------------------------------------------------------------------
# GET /api/audio-devices
# ---------------------------------------------------------------------------

def _devices_app(tmp_path, monkeypatch) -> "TestClient":
    from fastapi.testclient import TestClient

    import config
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(config, "STATE_FILE", tmp_path / "state.json")
    from dashboard.server import DashboardState, _build_app
    return TestClient(_build_app(DashboardState(), None, pin=""))


def _patch_sd(monkeypatch, *, devices, hostapis, default_out=None, boom=False, default_device=None):
    import sounddevice as sd

    if boom:
        monkeypatch.setattr(sd, "query_devices", lambda: (_ for _ in ()).throw(RuntimeError("PortAudio died")))
    else:
        monkeypatch.setattr(sd, "query_devices", lambda: devices)
    monkeypatch.setattr(sd, "query_hostapis", lambda: hostapis)
    monkeypatch.setattr(
        sd, "default",
        type("D", (), {"device": default_device, "output": None})(),
        raising=False,
    )


def test_audio_devices_route_merges_duplicate_apis_and_flags_default(tmp_path, monkeypatch):
    """One row per endpoint: the MME row's truncated name folds into the full
    one, the shown API is the one the player picks (WASAPI), and the system
    default is flagged."""
    client = _devices_app(tmp_path, monkeypatch)
    _patch_sd(
        monkeypatch,
        devices=[
            {"name": "Mic Array", "max_output_channels": 0, "hostapi": 0},  # input-only → hidden
            {"name": "Alto-falantes (VB-Audio Voiceme", "max_output_channels": 2, "hostapi": 0},
            {"name": "Speakers", "max_output_channels": 2, "hostapi": 1,
             "default_samplerate": 48000.0},
            {"name": "CABLE Input", "max_output_channels": 2, "hostapi": 1,
             "default_samplerate": 48000.0},
            {"name": "Alto-falantes (VB-Audio Voicemeeter VAIO)", "max_output_channels": 2,
             "hostapi": 1, "default_samplerate": 48000.0},
        ],
        hostapis=[{"name": "MME"}, {"name": "Windows WASAPI"}],
        default_device=[0, 3],
    )

    r = client.get("/api/audio-devices")
    assert r.status_code == 200
    data = r.json()
    assert [d["name"] for d in data] == [
        "Alto-falantes (VB-Audio Voicemeeter VAIO)", "CABLE Input", "Speakers",
    ]  # alphabetical
    speakers = next(d for d in data if d["name"] == "Speakers")
    assert speakers["api"] == "WASAPI"          # "Windows " prefix stripped
    assert speakers["index"] == 2
    assert speakers["is_default"] is False
    cable = next(d for d in data if d["name"] == "CABLE Input")
    assert cable["index"] == 3
    assert cable["is_default"] is True          # default_device pair [0, 3]
    vaio = next(d for d in data if "Voicemeeter VAIO" in d["name"])
    assert vaio["api"] == "WASAPI"              # merged row reports the best API
    assert vaio["is_default"] is False


def test_audio_devices_route_empty_when_portaudio_unavailable(tmp_path, monkeypatch):
    """No audio subsystem → [] (HTTP 200), so the UI can show a hint, not an error."""
    client = _devices_app(tmp_path, monkeypatch)
    _patch_sd(monkeypatch, devices=[], hostapis=[], boom=True)

    r = client.get("/api/audio-devices")
    assert r.status_code == 200
    assert r.json() == []


def test_audio_devices_route_tolerates_odd_hostapi_rows(tmp_path, monkeypatch):
    """A malformed hostapis row must not break the listing (api is cosmetic)."""
    client = _devices_app(tmp_path, monkeypatch)
    _patch_sd(
        monkeypatch,
        devices=[{"name": "Speakers", "max_output_channels": 2, "hostapi": 9}],
        hostapis=[{"name": "MME"}],  # index 9 does not exist
    )

    r = client.get("/api/audio-devices")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["api"] == ""
    assert data[0]["is_default"] is False


# ---------------------------------------------------------------------------
# "test output" beep (play_test_beep + POST /api/test/audio-output)
# ---------------------------------------------------------------------------

def test_play_test_beep_sounds_on_resolved_device(monkeypatch):
    """The beep must be non-silent and go to the RESOLVED device (by name)."""
    import numpy as np
    import sounddevice as sd

    from audio.player import play_test_beep

    seen: dict = {}

    def _fake_play(data, samplerate, *, device=None, blocking=False):
        seen["data"] = data
        seen["sr"] = samplerate
        seen["device"] = device
        seen["blocking"] = blocking

    monkeypatch.setattr(sd, "play", _fake_play)
    monkeypatch.setattr(
        sd, "query_devices",
        lambda idx=None: {"name": "CABLE Input (VB-Audio Virtual Cable)"}
        if idx is not None else [
            {"name": "CABLE Input (VB-Audio Virtual Cable)", "hostapi": 0, "max_output_channels": 2},
        ],
    )

    out = play_test_beep("CABLE Input (VB-Audio Virtual Cable)")

    assert seen["blocking"] is True
    assert seen["sr"] == 44100
    assert seen["device"] == 0  # resolved by name via resolve_output_device
    assert out["device"] == "CABLE Input (VB-Audio Virtual Cable)"
    data = np.asarray(seen["data"])
    assert data.dtype == np.float32
    assert float(np.abs(data).max()) > 0.1      # audible, not silence
    assert len(data) == int(44100 * 0.25)       # ~0.25 s


def test_play_test_beep_failure_surfaces_reason(monkeypatch):
    import sounddevice as sd

    from audio.player import play_test_beep

    def _boom(*a, **k):
        raise Exception("device unplugged")

    monkeypatch.setattr(sd, "play", _boom)
    monkeypatch.setattr(sd, "query_devices", lambda idx=None: [] if idx is None else {"name": "x"})
    monkeypatch.setattr(sd, "query_hostapis", lambda: [])

    with pytest.raises(RuntimeError, match="device unplugged"):
        play_test_beep("Ghost Device")


def test_test_output_route_beeps_and_reports(tmp_path, monkeypatch):
    client = _devices_app(tmp_path, monkeypatch)
    import sounddevice as sd

    seen: dict = {}

    def _fake_play(data, samplerate, *, device=None, blocking=False):
        seen["device"] = device

    monkeypatch.setattr(sd, "play", _fake_play)
    monkeypatch.setattr(
        sd, "query_devices",
        lambda idx=None: {"name": "Speakers"} if idx is None
        else {"name": "Speakers"},
    )
    monkeypatch.setattr(sd, "query_hostapis", lambda: [])

    r = client.post("/api/test/audio-output", json={"device": ""})
    assert r.status_code == 200
    assert r.json()["device"] == "Speakers"


def test_test_output_route_failure_is_400_with_reason(tmp_path, monkeypatch):
    client = _devices_app(tmp_path, monkeypatch)
    import sounddevice as sd

    def _boom(*a, **k):
        raise RuntimeError("device busy")

    monkeypatch.setattr(sd, "play", _boom)
    monkeypatch.setattr(
        sd, "query_devices",
        lambda idx=None: {"name": "Speakers"} if idx is not None
        else [{"name": "Speakers", "hostapi": 0, "max_output_channels": 2}],
    )
    monkeypatch.setattr(sd, "query_hostapis", lambda: [])

    r = client.post("/api/test/audio-output", json={"device": "Speakers"}
    )
    assert r.status_code == 400
    assert "device busy" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Hearing loopback devices (GET /api/loopback-devices + named capture)
# ---------------------------------------------------------------------------

def _patch_soundcard(monkeypatch, *, speakers, default=None, mic_for=None):
    """Fake the `soundcard` module and rebind it where hearing.capture imported it."""
    import types

    import hearing.capture as cap_mod

    fake = types.ModuleType("soundcard")

    class _Spk:
        def __init__(self, name):
            self.name = name

    fake.all_speakers = lambda: [_Spk(n) for n in speakers]
    fake.default_speaker = lambda: _Spk(default) if default else None
    fake.get_microphone = lambda name, include_loopback=False: mic_for.get(name)
    monkeypatch.setattr(cap_mod, "sc", fake)
    return fake


def test_loopback_devices_route_lists_speakers_with_default(tmp_path, monkeypatch):
    client = _devices_app(tmp_path, monkeypatch)
    _patch_soundcard(
        monkeypatch,
        speakers=["Speakers (Realtek)", "CABLE Input (VB-Audio Virtual Cable)", "Speakers (Realtek)"],
        default="CABLE Input (VB-Audio Virtual Cable)",
    )

    r = client.get("/api/loopback-devices")
    assert r.status_code == 200
    data = r.json()
    assert [d["name"] for d in data] == ["Speakers (Realtek)", "CABLE Input (VB-Audio Virtual Cable)"]
    assert data[0]["is_default"] is False
    assert data[1]["is_default"] is True


def test_loopback_devices_route_empty_without_soundcard(tmp_path, monkeypatch):
    """soundcard missing → [] with HTTP 200 (UI hint, not an error)."""
    import sys
    import types

    client = _devices_app(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "soundcard", types.ModuleType("soundcard"))
    import hearing.capture as cap_mod
    monkeypatch.setattr(cap_mod, "sc", None)

    r = client.get("/api/loopback-devices")
    assert r.status_code == 200
    assert r.json() == []


def test_system_capture_uses_named_loopback_device(monkeypatch):
    """SystemAudioCapture(device=...) must open a loopback mic by that name."""
    import time

    import hearing.capture as cap_mod

    opened: dict = {}
    _patch_soundcard(monkeypatch, speakers=["A", "B"], default="A", mic_for={})

    class _Rec:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def record(self, numframes):
            import numpy as np
            time.sleep(0.2)
            return np.zeros((numframes, 1), dtype="float32")

    class _LoopMic:
        def recorder(self, **kw):
            opened["recorder"] = kw
            return _Rec()

    cap_mod.sc.get_microphone = lambda name, include_loopback=False: (
        _LoopMic() if name == "B" else None)

    cap = cap_mod.SystemAudioCapture(samplerate=16000, device="B")
    cap.open()
    time.sleep(0.5)
    cap.close()
    assert opened.get("recorder") is not None  # named device opened (not default 'A')


def test_system_capture_named_device_missing_logs_and_ends(tmp_path, monkeypatch):
    """A device name that matches nothing must NOT fall back silently to the
    default speaker — the thread logs and ends (the ring stays empty)."""
    import time

    import hearing.capture as cap_mod

    _patch_soundcard(monkeypatch, speakers=["A"], default="A", mic_for={})
    cap_mod.sc.get_microphone = lambda name, include_loopback=False: None

    cap = cap_mod.SystemAudioCapture(samplerate=16000, device="Ghost Device")
    cap.open()
    import time
    time.sleep(0.4)
    audio = cap.latest(1.0)
    cap.close()
    assert audio.size > 0  # ring exists
    assert float(abs(audio).max()) == 0.0  # ...but nothing was captured


# ---------------------------------------------------------------------------
# POST /api/device-check (stale saved device warning)
# ---------------------------------------------------------------------------

def test_device_check_reports_missing_saved_devices(tmp_path, monkeypatch):
    """A saved output/loopback name matching nothing is reported as gone."""
    client = _devices_app(tmp_path, monkeypatch)
    _patch_sd(
        monkeypatch,
        devices=[{"name": "Speakers (Realtek)", "max_output_channels": 2, "hostapi": 0}],
        hostapis=[{"name": "MME"}],
    )
    _patch_soundcard(monkeypatch, speakers=["Speakers (Realtek)"], default="Speakers (Realtek)")

    r = client.post("/api/device-check",
                    json={"tts_output": "Ghost Headset Pro", "loopback": "CABLE Input (VB-Audio)"})
    assert r.status_code == 200
    data = r.json()
    assert data["tts_output"] == {"exists": False, "name": ""}
    assert data["loopback"] == {"exists": False, "is_default": False}


def test_device_check_accepts_defaults_and_case_insensitive_names(tmp_path, monkeypatch):
    client = _devices_app(tmp_path, monkeypatch)
    _patch_sd(
        monkeypatch,
        devices=[{"name": "Speakers (Realtek)", "max_output_channels": 2, "hostapi": 0}],
        hostapis=[{"name": "MME"}],
    )
    _patch_soundcard(monkeypatch, speakers=["Speakers (Realtek)"], default="Speakers (Realtek)")

    r = client.post("/api/device-check", json={"tts_output": "", "loopback": "SPEAKERS"})
    assert r.status_code == 200
    data = r.json()
    assert data["tts_output"] == {"exists": True, "name": ""}   # "" = system default
    assert data["loopback"] == {"exists": True, "is_default": True}  # substring, case-insensitive


def test_device_check_never_500s_when_audio_stack_dead(tmp_path, monkeypatch):
    """The warning is best-effort: a dead PortAudio/soundcard must not fail
    the save flow — resolve_output_device passes the unknown name through,
    sd dies on it (exists=False) and soundcard enumeration degrades likewise."""
    client = _devices_app(tmp_path, monkeypatch)
    _patch_sd(monkeypatch, devices=[], hostapis=[], boom=True)
    import types
    import hearing.capture as cap_mod
    fake = types.ModuleType("soundcard")
    fake.all_speakers = lambda: (_ for _ in ()).throw(RuntimeError("no audio service"))
    fake.default_speaker = lambda: (_ for _ in ()).throw(RuntimeError("no audio service"))
    monkeypatch.setattr(cap_mod, "sc", fake)

    r = client.post("/api/device-check", json={"tts_output": "Speakers", "loopback": "Speakers"})
    assert r.status_code == 200
    data = r.json()
    assert data["tts_output"]["exists"] is False
    assert data["loopback"]["exists"] is False


# ---------------------------------------------------------------------------
# Multi-instance fallback: one endpoint, several host-API instances. The
# preferred (WASAPI) one can be enumerated but unroutable (-9996) — the beep
# and the preflight probe must try the other instances of the SAME endpoint.
# ---------------------------------------------------------------------------

_DEV_ROWS = [
    {"name": "Ghost Endpoint", "hostapi": 0, "max_output_channels": 2,
     "default_samplerate": 48000.0},   # idx 0: WASAPI — refuses to OPEN (-9996)
    {"name": "Ghost Endpoint", "hostapi": 1, "max_output_channels": 2,
     "default_samplerate": 48000.0},   # idx 1: DirectSound — opens, dies at START
    {"name": "Healthy Speaker", "hostapi": 0, "max_output_channels": 2,
     "default_samplerate": 48000.0},   # idx 2: unrelated healthy device
]
_APIS = [{"name": "Windows WASAPI"}, {"name": "Windows DirectSound"}]


def _patch_multi_instance(monkeypatch, *, mode="sibling_ok"):
    """mode: 'sibling_ok'  → idx0 fails open, idx1 opens+starts fine
             'all_dead'    → idx0 fails open, idx1 opens then fails start"""
    import sounddevice as sd

    monkeypatch.setattr(
        sd, "query_devices",
        lambda idx=None: _DEV_ROWS[idx] if idx is not None else _DEV_ROWS)
    monkeypatch.setattr(sd, "query_hostapis", lambda *a: _APIS)
    monkeypatch.setattr(sd, "default",
                        type("D", (), {"device": [None, 2], "output": None})(),
                        raising=False)

    opened: list = []

    class _FakeStream:
        def __init__(self, **kw):
            self.device = kw.get("device")

        def start(self):
            if self.device == 0:
                raise Exception("Error opening OutputStream: Invalid device [PaErrorCode -9996]")
            if mode == "all_dead" and self.device == 1:
                raise Exception("Error starting stream: Unanticipated host error [PaErrorCode -9999]")
            opened.append(self.device)

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(sd, "RawOutputStream", lambda **kw: _FakeStream(**kw))
    return opened


def test_play_test_beep_falls_back_to_sibling_instance(monkeypatch):
    """WASAPI instance refuses to open (-9996) → the beep plays on the
    DirectSound sibling of the SAME endpoint and reports that instance."""
    import sounddevice as sd

    from audio.player import play_test_beep

    opened = _patch_multi_instance(monkeypatch)
    played: dict = {}

    def _fake_play(data, samplerate, *, device=None, blocking=False):
        if device == 0:  # the WASAPI instance refuses, like the real -9996
            raise Exception("Error opening OutputStream: Invalid device [PaErrorCode -9996]")
        played["device"] = device

    monkeypatch.setattr(sd, "play", _fake_play)

    out = play_test_beep("Ghost Endpoint")
    assert played["device"] == 1                       # sibling instance used
    assert out["index"] == 1 and out["device"] == "Ghost Endpoint"


def test_play_test_beep_all_instances_dead_actionable_message(monkeypatch):
    """Every instance dead → RuntimeError whose text tells the user what to do
    (re-pick the output), keeping the original PaErrorCode."""
    import sounddevice as sd

    from audio.player import play_test_beep

    _patch_multi_instance(monkeypatch, mode="all_dead")

    def _boom(data, samplerate, *, device=None, blocking=False):
        raise Exception("Error opening OutputStream: Invalid device [PaErrorCode -9996]")

    monkeypatch.setattr(sd, "play", _boom)

    with pytest.raises(RuntimeError, match="re-pick the output") as ei:
        play_test_beep("Ghost Endpoint")
    assert "-9996" in str(ei.value)                    # original reason kept


def test_probe_output_device_tries_all_instances_and_starts(monkeypatch):
    """The preflight probe must open AND start (a dead endpoint's DirectSound
    instance opens fine and only fails at start) and succeed via a sibling."""
    from audio.player import probe_output_device

    opened = _patch_multi_instance(monkeypatch)
    res = probe_output_device("Ghost Endpoint", sample_rate=24000)
    assert res["ok"] is True and res["index"] == 0
    assert res["fallback_index"] == 1                  # succeeded on the sibling
    assert opened == [1]


def test_probe_output_device_all_dead_reports_reason(monkeypatch):
    """All instances dead → ok: False with the first error, for preflight."""
    from audio.player import probe_output_device

    _patch_multi_instance(monkeypatch, mode="all_dead")
    res = probe_output_device("Ghost Endpoint", sample_rate=24000)
    assert res["ok"] is False
    assert "-9996" in res["reason"]


def test_probe_output_device_default_spec_is_ok_without_probing(monkeypatch):
    """Empty spec = system default → ok without touching the audio stack."""
    from audio.player import probe_output_device

    opened = _patch_multi_instance(monkeypatch, mode="all_dead")
    assert probe_output_device("") == {"ok": True, "index": None, "fallback_index": None}
    assert opened == []                                # nothing was opened


# ---------------------------------------------------------------------------
# AudioPlayer drained signal — wait_drained() race regression
#
# The voice test (dashboard /api/test/voice, preview path) does
# start() → write(all) → wait_drained() → close(). The drained event starts
# SET, and the audio callback re-sets it whenever it sees an empty buffer —
# via a DEFERRED call_soon_threadsafe. A callback that ran just before the
# write re-set the event AFTER the write cleared it, so wait_drained()
# returned with the fresh audio still queued and close() killed the stream:
# ElevenLabs streams arrive in one burst, so its test was cut to <2s.
#
# These tests drive _callback() directly (no real device, never started),
# yielding once after each pump so the deferred loop callbacks run.
# ---------------------------------------------------------------------------


def _pump(p, frames: int = 960) -> None:
    """Run one audio callback by hand, like PortAudio would."""
    out = bytearray(frames * p._channels * 2)
    p._callback(out, frames, None, 0)


def _fresh_player():
    from audio.player import AudioPlayer

    p = AudioPlayer()  # no start(): no stream, callbacks invoked manually
    return p


@pytest.mark.asyncio
async def test_wait_drained_ignores_stale_drain_callbacks():
    """Callbacks that saw an empty buffer BEFORE a write must not signal
    drained after it — the race that truncated the ElevenLabs voice test."""
    p = _fresh_player()
    p._loop = asyncio.get_running_loop()  # what start() would set

    _pump(p)                                    # callback sees an empty buffer
    p._buf.extend(b"\x00\x01" * 960)            # one block appears
    _pump(p)                                    # drains it; buffer empty again
    await asyncio.sleep(0)                      # deferred re-sets fire
    assert p._finished_event.is_set()

    await p.write(b"\x00\x01" * (960 * 10))     # fresh audio lands NOW
    assert not p._finished_event.is_set()

    # The stale deferred sets (scheduled before the write) must NOT wake it.
    await asyncio.sleep(0.05)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(p.wait_drained(), timeout=0.25)
    assert p.seconds_queued() > 0               # audio is still queued

    # Draining the real audio DOES signal, and the wait completes.
    while p.seconds_queued() > 0:
        _pump(p)
        await asyncio.sleep(0)
    await asyncio.wait_for(p.wait_drained(), timeout=1.0)


@pytest.mark.asyncio
async def test_wait_drained_returns_immediately_when_idle():
    """Idle player (nothing ever written): wait_drained is instant."""
    p = _fresh_player()
    p._loop = asyncio.get_running_loop()

    await asyncio.wait_for(p.wait_drained(), timeout=0.1)  # must not block


@pytest.mark.asyncio
async def test_write_rearms_drained_wait_after_completion():
    """write → drained → write again: the second wait must block until the
    new audio drains (the orchestrator's wait_drained-then-write pattern)."""
    p = _fresh_player()
    p._loop = asyncio.get_running_loop()

    await p.write(b"\x00\x01" * 1920)
    assert not p._finished_event.is_set()
    while p.seconds_queued() > 0:
        _pump(p)
        await asyncio.sleep(0)
    await asyncio.wait_for(p.wait_drained(), timeout=1.0)

    await p.write(b"\x00\x01" * 1920)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(p.wait_drained(), timeout=0.15)
    while p.seconds_queued() > 0:
        _pump(p)
        await asyncio.sleep(0)
    await asyncio.wait_for(p.wait_drained(), timeout=1.0)
