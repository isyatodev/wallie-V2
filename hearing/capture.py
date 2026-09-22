"""System-audio capture via WASAPI loopback (soundcard), continuous + lag-free.

A background thread drains the loopback device non-stop into a rolling ring buffer,
so the buffer never overflows while transcription is running. The processing loop
just grabs the most RECENT `window` seconds whenever it wants — always current audio,
no "data discontinuity", no stale/echoed audio. This is Wallie's ear.
"""
from __future__ import annotations

import threading

import numpy as np

try:
    import soundcard as sc
except Exception:  # pragma: no cover - optional dep
    sc = None


def list_loopback_devices() -> list[dict]:
    """Speakers/endpoints the WASAPI loopback can listen through.

    Mirrors the dashboard's output-device list, but names come from soundcard
    itself — the exact strings get_microphone() accepts. Runs on FastAPI
    worker threads, so COM is initialized HERE before any soundcard call:
    when PortAudio was imported first, a fresh thread's enumeration dies with
    0x800401f0 (CO_E_NOTINITIALIZED) without this. Returns [] when soundcard
    is missing or the audio stack is unavailable (logged, not silent).
    """
    if sc is None:
        return []
    _co_done = False
    try:
        from ctypes import windll
        hr = windll.ole32.CoInitialize(None)
        _co_done = hr in (0, 1)  # S_OK / S_FALSE both need a matching CoUninitialize
    except Exception:  # noqa: BLE001 — non-Windows or no COM; soundcard may still work
        _co_done = False
    try:
        default_name = str(sc.default_speaker().name) if sc.default_speaker() else ""
        out = []
        seen: set[str] = set()
        for s in sc.all_speakers():
            name = str(s.name)
            if not name or name in seen:  # identical endpoints → get_microphone picks the first anyway
                continue
            seen.add(name)
            out.append({"name": name, "is_default": name == default_name})
        return out
    except Exception as e:  # no audio service / devices vanished
        from loguru import logger
        logger.warning(f"hearing: loopback device enumeration failed: {e}")
        return []
    finally:
        if _co_done:
            try:
                from ctypes import windll
                windll.ole32.CoUninitialize()
            except Exception:  # noqa: BLE001
                pass


def resolve_loopback_device(name: str) -> dict:
    """Does a saved loopback device name still exist? (dashboard stale-device check)

    Hearing saves the device by NAME; a name that matches nothing means the
    capture thread would end immediately (no fallback to the default, by
    design). Matching mirrors _capture_loop's own fallback: case-insensitive
    substring over the endpoints. An empty name = the system default, always
    valid. Returns {"exists": bool, "is_default": bool}; exists=False when
    soundcard is missing or enumeration fails (logged). Runs on FastAPI
    worker threads, so COM is initialized here first, exactly like
    list_loopback_devices above.
    """
    if sc is None:
        return {"exists": False, "is_default": False}
    _co_done = False
    try:
        from ctypes import windll
        hr = windll.ole32.CoInitialize(None)
        _co_done = hr in (0, 1)  # S_OK / S_FALSE both need a matching CoUninitialize
    except Exception:  # noqa: BLE001 — non-Windows or no COM; soundcard may still work
        _co_done = False
    try:
        wanted = (name or "").strip().lower()
        if not wanted:
            return {"exists": True, "is_default": True}  # "" = system default
        default_name = str(sc.default_speaker().name) if sc.default_speaker() else ""
        for s in sc.all_speakers() or []:
            if wanted in str(s.name).strip().lower():
                return {"exists": True, "is_default": str(s.name) == default_name}
        return {"exists": False, "is_default": False}
    except Exception as e:  # no audio service / devices vanished
        from loguru import logger
        logger.warning(f"hearing: loopback device check failed: {e}")
        return {"exists": False, "is_default": False}
    finally:
        if _co_done:
            try:
                from ctypes import windll
                windll.ole32.CoUninitialize()
            except Exception:  # noqa: BLE001
                pass


class SystemAudioCapture:
    """Continuous loopback capture into a rolling buffer (thread-backed)."""

    def __init__(self, samplerate: int = 16000, channels: int = 1, buffer_sec: float = 10.0,
                 device: str = "") -> None:
        if sc is None:
            raise RuntimeError(
                "soundcard not installed — hearing needs it. Install: pip install soundcard"
            )
        self._sr = samplerate
        self._ch = channels
        self._device_name = (device or "").strip()
        self._ring = np.zeros(int(samplerate * buffer_sec), dtype="float32")
        self._lock = threading.Lock()
        self._thread: "threading.Thread | None" = None
        self._running = False

    def open(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="audio-capture", daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        # soundcard's WASAPI/MediaFoundation backend uses COM, which must be initialized
        # on THIS thread. The dashboard starts us from a worker thread where it isn't,
        # so without this soundcard raises 0x800401f0 (CO_E_NOTINITIALIZED) and the ear dies.
        _co_done = False
        try:
            from ctypes import windll
            hr = windll.ole32.CoInitialize(None)
            _co_done = hr in (0, 1)  # S_OK / S_FALSE both need a matching CoUninitialize
        except Exception:  # noqa: BLE001 — non-Windows or no COM; soundcard may still work
            _co_done = False
        try:
            if self._device_name:
                mic = sc.get_microphone(self._device_name, include_loopback=True)
                if mic is None:
                    # Fallback: case-insensitive substring over the endpoints,
                    # then give up with a logged reason (thread ends cleanly).
                    cand = [s for s in (sc.all_speakers() or [])
                            if self._device_name.lower() in str(s.name).lower()]
                    mic = cand[0] if cand else None
                if mic is None:
                    from loguru import logger
                    logger.error(f"hearing: loopback device not found: {self._device_name!r}")
                    return
            else:
                spk = sc.default_speaker()
                mic = sc.get_microphone(str(spk.name), include_loopback=True)
            chunk = max(1, int(self._sr * 0.25))  # drain in 250 ms slices
            with mic.recorder(samplerate=self._sr, channels=self._ch, blocksize=chunk) as rec:
                while self._running:
                    try:
                        data = rec.record(numframes=chunk)
                    except Exception:
                        break
                    data = data.flatten().astype("float32")
                    n = data.shape[0]
                    if n == 0:
                        continue
                    with self._lock:
                        if n >= self._ring.shape[0]:
                            self._ring[:] = data[-self._ring.shape[0]:]
                        else:
                            self._ring[:-n] = self._ring[n:]
                            self._ring[-n:] = data
        finally:
            if _co_done:
                try:
                    from ctypes import windll
                    windll.ole32.CoUninitialize()
                except Exception:  # noqa: BLE001
                    pass

    def latest(self, seconds: float) -> "np.ndarray":
        """Most recent `seconds` of audio from the ring (always current)."""
        k = int(self._sr * seconds)
        with self._lock:
            return self._ring[-k:].copy()

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    @property
    def sample_rate(self) -> int:
        return self._sr
