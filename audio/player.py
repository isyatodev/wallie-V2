"""Low-latency PCM audio player with alignment-safe writes."""
from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import sounddevice as sd
from loguru import logger


def resolve_output_device(spec: Optional[int | str]) -> Optional[int | str]:
    """Turn a device spec into a concrete output-device index.

    A bare name like 'CABLE Input' matches the same endpoint across multiple host
    APIs (MME/DirectSound/WASAPI), which makes sounddevice raise on ambiguity. We
    resolve by name ourselves and prefer WASAPI (lowest latency, matches the rest
    of the pipeline). Indices/None/empty pass straight through.
    """
    if spec is None or spec == "":
        return None
    try:
        return int(spec)  # already an index (or numeric string)
    except (ValueError, TypeError):
        pass
    name = str(spec).lower()
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception:  # noqa: BLE001 — let sounddevice handle it downstream
        return spec
    wasapi = next((i for i, h in enumerate(hostapis) if "wasapi" in h["name"].lower()), None)
    matches = [i for i, d in enumerate(devices)
               if name in d["name"].lower() and d["max_output_channels"] > 0]
    if not matches:
        return spec
    for i in matches:  # prefer the WASAPI instance when present
        if wasapi is not None and devices[i]["hostapi"] == wasapi:
            return i
    return matches[0]


def _output_candidates(spec: Optional[int | str], resolved: Optional[int | str]) -> list:
    """Instances to try, in order: the resolved one first, then every other
    instance of the same endpoint (one per host API). Windows exposes each
    endpoint on up to four APIs and virtual-cable instances are sometimes
    unroutable on one but fine on another — callers fall back down this list.
    Only meaningful when the spec is a NAME; indices/default have no aliases."""
    candidates = [resolved]
    s = str(spec or "").strip()
    if s and not s.lstrip("-").isdigit():
        try:
            nm = s.lower()
            candidates.extend(
                i for i, d in enumerate(sd.query_devices())
                if i != resolved and nm in str(d.get("name", "")).lower()
                and d.get("max_output_channels", 0) > 0
            )
        except Exception:  # noqa: BLE001 — enumeration is best-effort
            pass
    return candidates


def play_test_beep(device: Optional[int | str] = None) -> dict:
    """Play a short tone through the given output device (resolver-aware).

    Used by the dashboard's "test output" button next to the device picker,
    so the user can confirm the TTS destination without a real utterance.

    A bare NAME can match several instances of the same endpoint (one per
    host API). If the preferred instance (WASAPI) refuses to open — virtual
    cables managed by Voicemeeter are often enumerated but unroutable — the
    other instances are tried before giving up, and the failure message says
    what to do instead of surfacing a bare PaErrorCode.
    """
    idx = resolve_output_device(device)
    name = ""
    sr = 44100
    try:
        dev = sd.query_devices(idx)
        name = str(dev["name"])
        # WASAPI/WDM-KS endpoints often reject anything but their default
        # rate (VB-Cable: PaErrorCode -9997 at 44100). Play at the device's
        # own rate instead of forcing one.
        sr = int(round(dev.get("default_samplerate") or sr))
    except Exception:  # noqa: BLE001 — name/rate fall back to defaults
        pass
    dur = 0.25
    candidates = _output_candidates(device, idx)

    def _tone(rate: int):
        t = np.linspace(0.0, dur, int(rate * dur), endpoint=False)
        # 880 Hz sine with ~20 ms fade in/out so the beep doesn't click.
        env = np.minimum(1.0, np.minimum(t / 0.02, (dur - t) / 0.02))
        return (0.25 * np.sin(2 * np.pi * 880.0 * t) * env).astype("float32")

    last_err: Optional[Exception] = None
    for cand in candidates:
        try:
            rate = sr
            try:
                rate = int(round(sd.query_devices(cand).get("default_samplerate") or sr))
            except Exception:  # noqa: BLE001 — keep the resolved rate
                pass
            sd.play(_tone(rate), rate, device=cand, blocking=True)
            if cand != idx:
                logger.warning(f"audio: test beep: device {idx} refused to open; "
                               f"played on alternate instance {cand} instead")
            return {"device": name or ("system default" if cand is None else str(cand)),
                    "index": cand, "seconds": dur}
        except Exception as e:  # try the next instance of the same endpoint
            last_err = e

    # Total failure. When the spec resolved to a REAL endpoint (Windows lists
    # it), the useful message is 'it exists but won't open — re-pick it',
    # regardless of which PaErrorCode each instance died with (-9996 open,
    # -9999/-2004287478 DirectSound start, -9985 unavailable…).
    resolved_to_real = any(isinstance(c, int) for c in candidates)
    msg = f"could not play on output {device!r}: {last_err}"
    if resolved_to_real:
        msg += (" — the endpoint is listed by Windows but refuses to open "
                "(typical of virtual cables under Voicemeeter); re-pick the "
                "output on the Voice page and save")
    raise RuntimeError(msg) from last_err


def probe_output_device(spec: Optional[int | str], *, sample_rate: int = 24000,
                        channels: int = 1) -> dict:
    """Preflight probe: can the saved TTS output actually OPEN?

    Enumeration alone lies — Windows lists endpoints that refuse to open
    (PaErrorCode -9996), e.g. virtual cables managed by Voicemeeter, and the
    session would run mute. Mirrors AudioPlayer.start()'s parameters (rate,
    channels, WASAPI auto-convert) and candidate order (preferred instance
    first, then every other instance of the same endpoint) WITHOUT playing:
    a silent stream is opened, STARTED and immediately closed — some broken
    instances (DirectSound of a dead endpoint) open fine and only fail at
    start, so the probe does both. Nothing is audible: no audio is ever fed.
    """
    resolved = resolve_output_device(spec)
    if resolved is None:
        return {"ok": True, "index": None, "fallback_index": None}  # system default
    candidates = _output_candidates(spec, resolved)
    first_err: Optional[Exception] = None
    for i, cand in enumerate(candidates):
        try:
            extra = None
            try:
                if isinstance(cand, int):
                    ha = sd.query_hostapis(sd.query_devices(cand)["hostapi"])["name"]
                    if "wasapi" in ha.lower():
                        extra = sd.WasapiSettings(auto_convert=True)
            except Exception:  # noqa: BLE001 — fall back to no extra settings
                extra = None
            # Matches AudioPlayer.start(); never fed audio → inaudible.
            stream = sd.RawOutputStream(samplerate=sample_rate, channels=channels,
                                        dtype="int16", device=cand,
                                        extra_settings=extra,
                                        callback=lambda outdata, frames, ti, st: None)
            stream.start()
            stream.stop()
            stream.close()
            return {"ok": True, "index": resolved,
                    "fallback_index": (cand if i > 0 else None)}
        except Exception as e:  # try the next instance of the same endpoint
            if first_err is None:
                first_err = e
    return {"ok": False, "index": resolved,
            "reason": str(first_err) if first_err else "no output device",
            "fallback_index": None}


def check_output_device(spec: Optional[int | str]) -> dict:
    """Does the saved TTS output spec still exist? (dashboard stale-device check)

    Mirrors what real playback would do: run resolve_output_device and see if
    the result maps to a live output. Empty/None = system default, always
    valid. Returns {"exists": bool, "name": <concrete device name, "" when
    unknown>}.
    """
    resolved = resolve_output_device(spec)
    if resolved is None:
        return {"exists": True, "name": ""}  # system default
    try:
        dev = sd.query_devices(resolved)
        return {"exists": True, "name": str(dev.get("name", ""))}
    except Exception:  # noqa: BLE001 — index out of range / PortAudio dead
        return {"exists": False, "name": ""}


class AudioPlayer:
    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        channels: int = 1,
        blocksize: int = 960,  # ~20ms at 48kHz, ~40ms at 24kHz
        device: Optional[int | str] = None,
    ) -> None:
        self._sr = sample_rate
        self._channels = channels
        self._blocksize = blocksize
        self._device = resolve_output_device(device)

        self._buf = bytearray()
        self._lock = threading.Lock()
        self._finished_event = asyncio.Event()
        self._finished_event.set()
        self._pending_odd: Optional[int] = None
        self._device_spec = device   # original spec, for messages/candidates
        self._drain_gen = 0  # bumped by every write(); guards the drained signal

        self._stream: Optional[sd.RawOutputStream] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_write_ts: float = 0.0  # when audio was last queued (for hearing self-mute)

    # ----- lifecycle -----
    def _open_stream(self, device) -> tuple[sd.RawOutputStream, bool]:
        """Open (and start) the output stream on one concrete instance.
        Returns (stream, used_wasapi_auto_convert)."""
        extra = None
        try:
            if isinstance(device, int):
                ha = sd.query_hostapis(sd.query_devices(device)["hostapi"])["name"]
                if "wasapi" in ha.lower():
                    extra = sd.WasapiSettings(auto_convert=True)
        except Exception:  # noqa: BLE001 — fall back to no extra settings
            extra = None
        stream = sd.RawOutputStream(
            samplerate=self._sr,
            channels=self._channels,
            dtype="int16",
            blocksize=self._blocksize,
            device=device,
            callback=self._callback,
            extra_settings=extra,
        )
        stream.start()
        return stream, extra is not None

    def start(self) -> None:
        if self._stream is not None:
            return
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()
        # WASAPI shared mode demands the stream rate match the device's mix format
        # (e.g. VB-CABLE is fixed at 48 kHz) and rejects our 24 kHz TTS otherwise.
        # Enable auto-convert on WASAPI devices so PortAudio resamples for us.
        # Some instances of the named endpoint refuse to open entirely
        # (PaErrorCode -9996, e.g. virtual cables under Voicemeeter) while
        # other instances of the SAME endpoint work — fall down the candidate
        # list before failing. This is not a silent device change: every
        # candidate routes to the endpoint the user picked.
        last_err: Optional[Exception] = None
        for i, cand in enumerate(_output_candidates(self._device_spec, self._device)):
            try:
                self._stream, auto = self._open_stream(cand)
                if i > 0:
                    logger.warning(
                        f"audio: output {self._device_spec!r} refused to open on "
                        f"instance {self._device!r}; using alternate instance {cand} "
                        f"of the same endpoint")
                logger.info(f"audio: playing at {self._sr} Hz, {self._channels}ch, "
                            f"block={self._blocksize}{' (WASAPI auto-convert)' if auto else ''}")
                return
            except Exception as e:  # try the next instance of the same endpoint
                last_err = e
        raise RuntimeError(
            f"could not open output {self._device_spec!r}: {last_err} — the endpoint "
            "is listed by Windows but refuses to open (typical of virtual cables "
            "under Voicemeeter); re-pick the output on the Voice page and save") from last_err

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning(f"audio stream close: {e}")
            self._stream = None

    # ----- writing -----
    async def write(self, pcm: bytes) -> None:
        if not pcm:
            return
        if self._pending_odd is not None:
            pcm = bytes((self._pending_odd,)) + pcm
            self._pending_odd = None
        if len(pcm) & 1:
            self._pending_odd = pcm[-1]
            pcm = pcm[:-1]
        if not pcm:
            return
        with self._lock:
            self._buf.extend(pcm)
            self._drain_gen += 1  # invalidates any "empty buffer" seen before this write
        self._last_write_ts = time.time()
        self._finished_event.clear()
        await asyncio.sleep(0)

    def speaking_recently(self, window: float) -> bool:
        """True if Wallie is currently outputting audio or did within `window` seconds.
        Used by hearing to skip windows that contain Wallie's own voice (no self-echo)."""
        return self.seconds_queued() > 0.02 or (time.time() - self._last_write_ts) < window

    def boundary(self) -> None:
        if self._pending_odd is not None:
            logger.debug(f"audio: dropping leftover odd byte at sentence boundary")
            self._pending_odd = None

    def reset(self) -> None:
        with self._lock:
            dropped = len(self._buf)
            self._buf.clear()
        self._pending_odd = None
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._finished_event.set)
            except RuntimeError:
                pass
        logger.info(f"audio: hard reset, dropped {dropped} bytes")

    def interrupt(self) -> None:
        with self._lock:
            dropped = len(self._buf)
            self._buf.clear()
        self._pending_odd = None
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._finished_event.set)
        if dropped:
            logger.info(f"audio: interrupt, dropped {dropped} bytes")

    async def wait_drained(self) -> None:
        await self._finished_event.wait()

    def seconds_queued(self) -> float:
        with self._lock:
            return len(self._buf) / (self._sr * self._channels * 2)

    # ----- audio callback -----
    def _callback(self, outdata, frames: int, time_info, status) -> None:  # noqa: ARG002
        if status:
            logger.debug(f"audio status: {status}")
        needed = frames * self._channels * 2
        with self._lock:
            available = len(self._buf)
            take = min(needed, available)
            if take:
                chunk = bytes(self._buf[:take])
                del self._buf[:take]
            else:
                chunk = b""
            empty_after = len(self._buf) == 0
            gen = self._drain_gen

        if take < needed:
            chunk = chunk + b"\x00" * (needed - take)
        outdata[: len(chunk)] = chunk

        if empty_after and self._loop is not None:
            # "Drained" only if nothing was written since we observed the empty
            # buffer. A write landing between that observation and this deferred
            # set bumps _drain_gen, and the callback that drains the NEW audio
            # signals instead. Without this guard, a callback that ran just
            # before a write re-set the event after the write cleared it, and
            # wait_drained() returned with the fresh audio still queued — the
            # voice test got cut off after ~2s (ElevenLabs streams arrive in
            # one burst, hitting the race almost every time).
            def _signal_drained(g: int = gen) -> None:
                with self._lock:
                    if self._drain_gen == g:
                        self._finished_event.set()

            try:
                self._loop.call_soon_threadsafe(_signal_drained)
            except RuntimeError:
                pass  # event loop closed mid-playback


def list_output_devices() -> list[dict]:
    """Enumerate playable outputs for the dashboard's device dropdown.

    One row per device NAME. Windows exposes every endpoint on up to four
    host APIs (MME/DirectSound/WASAPI/WDM-KS) — and MME truncates names to
    31 chars ("Alto-falantes (VB-Audio Voiceme") — so rows sharing a full
    name are merged into one entry whose `api` says what the player will
    actually pick (WASAPI, via resolve_output_device). Returns [] instead
    of raising when PortAudio itself is dead (missing drivers, no audio
    service) — the UI shows "no devices" rather than 500.
    """
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        default_out = None
        dev = getattr(sd.default, "device", None)
        if dev is not None:
            default_out = dev[1]  # (input, output) pair
    except Exception:  # noqa: BLE001 — PortAudio uninitialized / device vanished
        return []

    def _api(i: int) -> str:
        try:
            return str(hostapis[devices[i].get("hostapi", 0)]["name"]).replace("Windows ", "")
        except Exception:  # noqa: BLE001 — cosmetic only
            return ""

    rows = [
        {
            "index": i,
            "name": d.get("name", ""),
            "api": _api(i),
            "default_samplerate": d.get("default_samplerate", 0),
            "is_default": i == default_out,
        }
        for i, d in enumerate(devices)
        if d.get("max_output_channels", 0) > 0
    ]

    # Merge multi-API duplicates. MME truncates names to exactly 31 chars, so
    # a 31-char name that prefixes a longer one is the same physical endpoint.
    full_names = {r["name"] for r in rows}

    def _canonical(name: str) -> str:
        if len(name) == 31:
            for full in full_names:
                if full != name and full.startswith(name):
                    return full
        return name

    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(_canonical(r["name"]), []).append(r)

    def _rank(api: str) -> int:
        return {"wasapi": 3, "directsound": 2, "mme": 1}.get(api.lower(), 0)

    out = []
    for name, rs in groups.items():
        best = max(rs, key=lambda r: (_rank(r["api"]), -r["index"]))
        out.append(
            {
                "index": best["index"],
                "name": name,
                "api": best["api"],
                "default_samplerate": best["default_samplerate"],
                "is_default": any(r["is_default"] for r in rs),
            }
        )
    out.sort(key=lambda d: d["name"].lower())
    return out
