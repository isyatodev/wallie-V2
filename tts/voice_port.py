"""Voice port — carry a cloned cloud voice over to a local engine.

A saved voice's ``.json`` (the Voice Lab's own backup, a Fish Audio model dump
or an ElevenLabs voice dump) carries the voice's **name and id, not its
weights**: neither provider hands the model over, and Piper/Kokoro cannot learn
a voice from an id. What *can* cross the border is **audio**, so a port is a
real, staged pipeline instead of a format shuffle:

* **inspect** — read the ``.json`` and pull out every voice it names, with the
  reason any of them cannot be ported (missing id, missing API key, already
  local).
* **capture** — the cloud voice reads a fixed corpus; each take is trimmed and
  stored as a WAV. This is the only step that costs provider credits.
* **package** — the takes are formatted for the target engine:
  - Piper: a training dataset (22.05 kHz mono WAVs + ``metadata.csv``) plus a
    README with what the training run needs. Piper has no zero-shot cloning, so
    the weights have to be trained from this folder.
  - Kokoro: a 24 kHz reference corpus. Kokoro voices are learned style
    embeddings — it cannot take audio to make a new voice either.
* **attach** — once the user has a model produced from that package (a Piper
  ``.onnx`` + ``.onnx.json`` pair, a Kokoro ``.pt`` voicepack), it is dropped
  into ``voices/`` and registered in the local voice list, so the ported voice
  shows up in the Voice page pickers and the Voice Lab library.

Nothing here touches the active profile or the orchestrator: the routes compose
these pieces, which keeps the filesystem layout and the provider HTTP shapes
unit-testable.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from config import Secrets

from . import voice_lab

# Captured audio is always stored at this rate; the target's own rate is applied
# when the package is written (Piper trains at 22.05 kHz, Kokoro runs at 24 kHz).
CAPTURE_SR = 24000

# Engines a port can be prepared for. ``model_kind`` is what the attach step
# expects, and ``name_hint`` shows the naming rule the engine's own loader
# enforces (Piper: <lang_REGION>-<voice>-<quality>; Kokoro: <lang><gender>_<name>).
TARGETS: list[dict[str, Any]] = [
    {
        "id": "piper",
        "label": "Piper — local, CPU, free",
        "sample_rate": 22050,
        "model_kind": ".onnx + .onnx.json",
        "name_hint": "pt_BR-luna-medium",
        "produces": "training dataset (wav 22.05 kHz + metadata.csv)",
        "learns_from_audio": False,
    },
    {
        "id": "kokoro",
        "label": "Kokoro — local, CPU/GPU, free",
        "sample_rate": 24000,
        "model_kind": ".pt voicepack",
        "name_hint": "pf_luna",
        "produces": "reference corpus (wav 24 kHz) for a fine-tune",
        "learns_from_audio": False,
    },
]

PORT_DIR_NAME = "ported"
MANIFEST_FILE = "manifest.json"

# Providers whose voices can be ported: they expose a TTS endpoint we can drive
# to capture the corpus. A piper/kokoro preset is already local, and nothing
# else in the library has a voice id we could speak with.
PORTER_PROVIDERS: dict[str, str] = {"fish": "fish", "elevenlabs": "elevenlabs"}

# How much audio a port can capture in one run. The corpus is cycled when the
# requested minutes exceed it, which is normal for TTS datasets.
MIN_MINUTES = 1.0
MAX_MINUTES = 12.0
AVG_CLIP_SECONDS = 8.0
MAX_CLIPS = 90
MIN_CLIPS = 3

# Trimmed clips keep this much margin around the speech, so a hard cut never
# clips the first or last phoneme.
TRIM_THRESHOLD = 220          # int16 amplitude treated as silence
TRIM_MARGIN_SEC = 0.05

# ---------------------------------------------------------------------------
# Corpus — broad phoneme coverage, no music, no markup, no list formatting
# ---------------------------------------------------------------------------
CORPUS: dict[str, dict[str, Any]] = {
    "pt": {
        "label": "pt-BR",
        "espeak": "pt-br",
        "lines": [
            "Bom dia, eu sou a Luna e hoje eu vou conversar com você sobre muitas coisas.",
            "O caminho para a praia estava cheio de gente, mas ninguém parecia com pressa.",
            "Ela comprou três pães, duas maçãs e um pouco de queijo naquela padaria antiga.",
            "Quando a chuva começou, nós corremos para debaixo da marquise da esquina.",
            "O pequeno príncipe olhou para o céu e perguntou onde ficava o planeta dele.",
            "Minha avó sempre dizia que paciência é o melhor remédio para quase tudo.",
            "Os pássaros cantavam tão alto que dava para ouvi-los da janela do quarto.",
            "Você já experimentou aquele bolo de milho que a vizinha faz no domingo?",
            "Amanhã eu preciso acordar cedo, arrumar a mochila e pegar o ônibus das sete.",
            "Ha uma diferença enorme entre ouvir alguém e realmente escutar o que diz.",
            "O menino girava a chave na mão, sem coragem de abrir a porta da oficina.",
            "Nós vamos caminhar até a praça, comprar um sorvete e voltar devagar.",
            "A música tocava baixinho enquanto o vento balançava as cortinas da sala.",
            "Ele explicou o problema com tanta calma que todos entenderam de primeira.",
            "Quinta-feira à noite é o melhor momento para organizar a semana seguinte.",
            "A cidade acordou coberta de neblina e os prédios pareciam flutuar.",
            "Se você quiser, eu preparo um chá e a gente conversa com calma sobre isso.",
            "O cachorro correu pelo quintal, latiu para a bicicleta e voltou satisfeito.",
            "Aprendi a fazer arroz soltinho depois de errar muitas e muitas vezes.",
            "Ninguém esperava que a exposição de fotografia ficasse tão bonita.",
            "O relógio da igreja marcava meio-dia e trinta quando o trem apitou.",
            "Ela escreveu uma carta longa, dobrou com cuidado e guardou no envelope azul.",
            "O professor pediu silêncio, respirou fundo e repetiu a explicação inteira.",
            "Faz tanto calor hoje que até as formigas procuram sombra no concreto.",
            "Meu vizinho coleciona discos antigos e conhece a história de cada um deles.",
            "A velha ponte de madeira range sempre que alguém atravessa correndo.",
            "Nós rimos muito daquela história, mesmo já tendo ouvido ela várias vezes.",
            "O projeto ficou pronto no último minuto, mas funcionou exatamente como deveria.",
            "Crianças brincavam de bola na rua enquanto o sol descia atrás das casas.",
            "Amanhã cedo eu vou regar as plantas, lavar a louça e começar tudo de novo.",
        ],
    },
    "en": {
        "label": "en-US",
        "espeak": "en-us",
        "lines": [
            "Good morning, I am the assistant and I would like to talk with you for a while.",
            "The road to the harbour was crowded, yet nobody seemed to be in a hurry.",
            "She bought three loaves of bread, two apples and a little cheese at the old bakery.",
            "When the rain started, we ran underneath the awning on the corner of the street.",
            "My grandmother always said that patience is the best medicine for almost everything.",
            "The birds were singing so loudly that we could hear them through the bedroom window.",
            "Have you ever tried the corn cake that the neighbour bakes on Sunday afternoons?",
            "Tomorrow I need to wake up early, pack my bag and catch the seven o'clock bus.",
            "There is a huge difference between hearing someone and actually listening to them.",
            "The boy turned the key over in his hand, without the courage to open the workshop door.",
            "We will walk to the square, buy some ice cream and take the long way back.",
            "The music played softly while the wind moved the curtains in the living room.",
            "He explained the problem so calmly that everyone understood it on the first try.",
            "Thursday evening is the best moment to organise the rest of the week.",
            "The city woke up under a thick fog and the buildings seemed to float.",
            "If you want, I will make some tea and we can talk about this properly.",
            "The dog ran around the garden, barked at the bicycle and came back satisfied.",
            "I learned to cook fluffy rice after getting it wrong many, many times.",
            "Nobody expected the photography exhibition to turn out so beautiful.",
            "The church clock read half past noon when the train sounded its whistle.",
            "She wrote a long letter, folded it carefully and put it in the blue envelope.",
            "The teacher asked for silence, took a deep breath and repeated the whole explanation.",
            "It is so hot today that even the ants look for shade on the concrete.",
            "My neighbour collects old records and knows the story behind every single one.",
            "The old wooden bridge creaks whenever somebody crosses it at a run.",
            "We laughed a lot at that story, even though we had heard it several times before.",
            "The project was finished at the last minute, but it worked exactly as it should.",
            "Children were playing ball in the street while the sun set behind the houses.",
            "Early tomorrow I will water the plants, wash the dishes and start all over again.",
            "A quiet voice can still be heard clearly when the room finally goes silent.",
        ],
    },
}

DEFAULT_LANG = "pt"


class VoicePortError(RuntimeError):
    """A failure worth showing the user verbatim (bad file, missing key…)."""


@dataclass
class PortSource:
    """One voice found in a ``.json``, ready to be captured (or explained)."""

    name: str
    provider: str = ""
    voice_id: str = ""
    notes: str = ""
    source: str = ""
    reason: str = ""            # why it cannot be ported ("" = it can)

    @property
    def supported(self) -> bool:
        return not self.reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "voice_id": self.voice_id,
            "notes": self.notes,
            "source": self.source,
            "reason": self.reason,
            "supported": self.supported,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slug(text: str, fallback: str = "voice") -> str:
    """Filesystem-safe stem for a voice name (kept recognisable, not hashed)."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text or "").strip()).strip("-._")
    return s[:60] or fallback


def ported_dir() -> Path:
    """Where packages live — under the same ``voices/`` tree the engines use."""
    from . import piper_setup
    return piper_setup.voices_dir() / PORT_DIR_NAME


def target_spec(target: str) -> dict[str, Any]:
    wanted = (target or "").strip().lower()
    for spec in TARGETS:
        if spec["id"] == wanted:
            return dict(spec)
    known = ", ".join(t["id"] for t in TARGETS)
    raise VoicePortError(f"unknown local engine {target!r} — use one of {known}")


def lang_spec(lang: str) -> dict[str, Any]:
    key = (lang or DEFAULT_LANG).strip().lower()[:2]
    if key not in CORPUS:
        known = ", ".join(CORPUS)
        raise VoicePortError(f"unknown corpus language {lang!r} — use one of {known}")
    return {"id": key, **CORPUS[key]}


# ---------------------------------------------------------------------------
# Reading the .json
# ---------------------------------------------------------------------------

def _reason_for(provider: str, voice_id: str) -> str:
    p = (provider or "").strip().lower()
    if p in ("piper", "kokoro"):
        return f"already local — a {p} voice needs no port"
    if p not in PORTER_PROVIDERS:
        return f"{p or 'without a provider'} has no voice we can speak with"
    if not (voice_id or "").strip():
        return "no voice id in the file — re-export the voice with its id"
    return ""


def _source_from_raw(raw: dict[str, Any]) -> PortSource | None:
    """One dict → PortSource, tolerating the Wallie, Fish and ElevenLabs shapes.

    Wallie backup / library: ``{name, provider, voice_id}``.
    Fish model dump:         ``{_id|id, title|name}`` (``state``/``visibility``
                             are ignored — they describe the account, not audio).
    ElevenLabs voice dump:   ``{voice_id, name}`` (``labels``/``samples`` ignored).

    A dict with neither a name nor an id is not a voice (a settings dump, a
    wrapper object) — it is skipped rather than turned into a nameless port.
    """
    name = str(
        raw.get("name") or raw.get("title") or raw.get("voice_name") or ""
    ).strip()
    provider = str(
        raw.get("provider") or raw.get("service") or raw.get("platform") or ""
    ).strip().lower()
    voice_id = str(
        raw.get("voice_id") or raw.get("reference_id") or raw.get("_id") or raw.get("id") or ""
    ).strip()
    # Fish dumps name the provider implicitly (a model dump has a ``train_mode``
    # and a ``_id``); ElevenLabs dumps carry ``voice_id`` without a provider.
    if not provider:
        if "train_mode" in raw or "visibility" in raw:
            provider = "fish"
        elif raw.get("voice_id") or raw.get("labels") or raw.get("category"):
            provider = "elevenlabs"
    if not name and not voice_id:
        return None
    return PortSource(
        name=name or voice_id[:8],
        provider=provider,
        voice_id=voice_id,
        notes=str(raw.get("notes") or raw.get("description") or "")[:400],
        source=str(raw.get("source") or ""),
        reason=_reason_for(provider, voice_id),
    )


def parse_sources(payload: Any, *, text: str = "") -> list[PortSource]:
    """Every voice named in an uploaded ``.json`` (or a saved-voice dict).

    Accepts what users actually have on disk: a Wallie backup (``presets``), a
    library file (same shape, no ``format``), a bare list, one preset, a Fish
    model dump or an ElevenLabs voice dump. Unknown shapes raise a message that
    names the formats instead of a KeyError.
    """
    items: Iterable[Any]
    if isinstance(payload, dict):
        for key in ("presets", "voices", "items", "data", "models"):
            if isinstance(payload.get(key), list):
                items = payload[key]
                break
        else:
            items = [payload]
    elif isinstance(payload, list):
        items = payload
    else:
        raise VoicePortError(
            "that file is not a voice export — expected a Wallie backup, "
            "a Fish model dump or an ElevenLabs voice dump"
        )

    found: list[PortSource] = []
    seen: set[tuple[str, str]] = set()
    for i, raw in enumerate(items):
        if not isinstance(raw, dict):
            continue
        src = _source_from_raw(raw)
        if src is None:
            continue
        key = (src.provider, src.voice_id or src.name.lower())
        if key in seen:
            continue
        seen.add(key)
        found.append(src)
    if not found:
        if text and len(text.strip()) < 3:
            raise VoicePortError("that file is empty — pick the voice .json")
        raise VoicePortError(
            "no voice found in that file — expected a Wallie backup, a Fish "
            "model dump or an ElevenLabs voice dump"
        )
    return found[:50]


def sources_from_library(profile: str) -> list[PortSource]:
    """Saved voices of one profile, in the same shape an uploaded file gives."""
    out: list[PortSource] = []
    for preset in voice_lab.load_library(profile):
        out.append(PortSource(
            name=preset.name,
            provider=(preset.provider or "").strip().lower(),
            voice_id=preset.voice_id,
            notes=preset.notes,
            source=preset.source,
            reason=_reason_for(preset.provider, preset.voice_id),
        ))
    return out


# ---------------------------------------------------------------------------
# Preflight — everything the UI can say BEFORE a single credit is spent
# ---------------------------------------------------------------------------

def _key_for(provider: str, secrets: Secrets) -> str:
    if provider == "fish":
        return (getattr(secrets, "fish_api_key", "") or "").strip()
    if provider == "elevenlabs":
        return (getattr(secrets, "elevenlabs_api_key", "") or "").strip()
    return ""


def _key_hint(provider: str) -> str:
    return {
        "fish": "FISH_API_KEY",
        "elevenlabs": "ELEVENLABS_API_KEY",
    }.get(provider, "an API key")


def clips_for(minutes: float, lang: str) -> int:
    """How many corpus lines cover ``minutes`` of speech (cycled if needed)."""
    lang_spec(lang)                       # validates the language up front
    try:
        wanted = float(minutes)
    except (TypeError, ValueError):
        wanted = MIN_MINUTES
    wanted = max(MIN_MINUTES, min(MAX_MINUTES, wanted))
    count = int(round(wanted * 60.0 / AVG_CLIP_SECONDS))
    return max(MIN_CLIPS, min(MAX_CLIPS, count))


def corpus_lines(lang: str, clips: int) -> list[str]:
    """The first ``clips`` lines of the corpus, cycled when it runs out."""
    lines = list(lang_spec(lang)["lines"])
    return [lines[i % len(lines)] for i in range(max(1, clips))]


def preflight(
    *,
    provider: str,
    voice_id: str,
    target: str,
    seconds: float,
    lang: str,
    secrets: Secrets,
) -> dict[str, Any]:
    """Validate a port request and say what it will (and will not) do.

    Returns the plan the UI shows before the button is pressed: the engine spec,
    the clips that will be captured and the honest warning about what still
    needs to happen outside the dashboard.
    """
    spec = target_spec(target)
    corpus = lang_spec(lang)
    provider = (provider or "").strip().lower()
    voice_id = (voice_id or "").strip()
    reason = _reason_for(provider, voice_id)
    if reason:
        raise VoicePortError(reason)
    if not _key_for(provider, secrets):
        raise VoicePortError(
            f"porting needs the {_key_hint(provider)} — add it on the API Keys page "
            f"(capturing the corpus speaks with your {provider} voice)"
        )
    clips = clips_for(seconds / 60.0, corpus["id"])
    warnings = [
        "the .json carries the voice's id, not its weights — this captures audio, "
        "it does not download the model",
    ]
    if spec["id"] == "piper":
        warnings.append(
            "Piper has no zero-shot cloning: the package is a training dataset, and "
            "the .onnx has to be trained from it — the Train panel below does that "
            "in the dashboard (progress, log, cancel)"
        )
    else:
        warnings.append(
            "Kokoro voices are learned style embeddings — the package is a 24 kHz "
            "reference corpus for a fine-tune, not a ready voicepack"
        )
    return {
        "ok": True,
        "target": spec,
        "lang": {"id": corpus["id"], "label": corpus["label"], "espeak": corpus["espeak"]},
        "provider": provider,
        "voice_id": voice_id,
        "clips": clips,
        "est_seconds": round(clips * AVG_CLIP_SECONDS, 1),
        "capture_sample_rate": CAPTURE_SR,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Capture — the cloud voice reads the corpus
# ---------------------------------------------------------------------------

def _tts_for(provider: str, voice_id: str, secrets: Secrets) -> Any:
    """A streaming TTS provider for the ported voice (monkeypatched in tests)."""
    if provider == "fish":
        from .fish import FishTTS
        return FishTTS(
            api_key=secrets.fish_api_key,
            voice_id=voice_id,
            sample_rate=CAPTURE_SR,
        )
    if provider == "elevenlabs":
        from .elevenlabs import ElevenLabsTTS
        return ElevenLabsTTS(
            api_key=secrets.elevenlabs_api_key,
            voice_id=voice_id,
            sample_rate=CAPTURE_SR,
        )
    raise VoicePortError(f"{provider or 'that provider'} cannot be captured")


def trim_silence(pcm: bytes, sample_rate: int = CAPTURE_SR) -> bytes:
    """Drop leading/trailing near-silence, keeping a small margin.

    Training data wants the speech, not the two seconds of room tone a cloud
    TTS pads around every take. Pure-stdlib on purpose: the port must work with
    the base requirements, where numpy is present but ``soundfile`` is not.
    """
    frames = len(pcm) // 2
    if frames <= 0:
        return b""
    import array

    samples = array.array("h")
    samples.frombytes(pcm[: frames * 2])
    if array.array("h", [1]).tobytes() != b"\x01\x00":     # big-endian host
        samples.byteswap()

    start = 0
    while start < frames and abs(samples[start]) < TRIM_THRESHOLD:
        start += 1
    end = frames - 1
    while end >= start and abs(samples[end]) < TRIM_THRESHOLD:
        end -= 1
    if end <= start:                       # everything was silence
        return b""
    margin = int(sample_rate * TRIM_MARGIN_SEC)
    start = max(0, start - margin)
    end = min(frames - 1, end + margin)
    body = samples[start: end + 1]
    if array.array("h", [1]).tobytes() != b"\x01\x00":
        body.byteswap()
    return body.tobytes()


def pcm_seconds(pcm: bytes, sample_rate: int = CAPTURE_SR) -> float:
    return round(len(pcm) / 2.0 / max(1, sample_rate), 2)


def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resample of mono PCM16 (no scipy/soundfile).

    The captured corpus is 24 kHz and Piper trains at 22.05 kHz. Interpolation
    is not a studio-grade resampler, but it preserves the spectral envelope that
    matters for training, and it keeps the dependency list to numpy.
    """
    if src_rate == dst_rate or not pcm:
        return pcm
    import numpy as np

    frames = len(pcm) // 2
    if frames <= 1:
        return pcm
    src = np.frombuffer(pcm[: frames * 2], dtype="<i2").astype(np.float32)
    out_len = max(1, int(round(frames * float(dst_rate) / float(src_rate))))
    idx = np.linspace(0.0, float(frames - 1), out_len, dtype=np.float64)
    out = np.interp(idx, np.arange(frames, dtype=np.float64), src)
    return np.clip(np.round(out), -32768, 32767).astype("<i2").tobytes()


async def capture(
    *,
    provider: str,
    voice_id: str,
    secrets: Secrets,
    clips: int,
    lang: str,
) -> list[dict[str, Any]]:
    """Speak the corpus with the cloud voice, one trimmed WAV per line.

    A provider failure aborts the run — a half-captured corpus is not a package,
    and silently keeping 4 of 12 clips would build a dataset that cannot be
    trusted. The caller gets the provider's own message.
    """
    tts = _tts_for(provider, voice_id, secrets)
    takes: list[dict[str, Any]] = []
    try:
        for i, text in enumerate(corpus_lines(lang, clips), 1):
            pcm = b"".join([chunk async for chunk in tts.synthesize(text) if chunk])
            pcm = trim_silence(pcm, CAPTURE_SR)
            if not pcm:
                raise VoicePortError(
                    f"clip {i} came back silent — the voice may be muted or the id wrong"
                )
            takes.append({
                "index": i,
                "text": text,
                "pcm": pcm,
                "seconds": pcm_seconds(pcm, CAPTURE_SR),
            })
    finally:
        closer = getattr(tts, "aclose", None)
        if closer is not None:
            try:
                await closer()
            except Exception:      # noqa: BLE001 - closing must never mask the capture error
                pass
    return takes


# ---------------------------------------------------------------------------
# Package — the target engine's own layout
# ---------------------------------------------------------------------------

def _write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(voice_lab.pcm16_bytes_to_wav(pcm, sample_rate=sample_rate, channels=1))


def _piper_readme(spec: dict[str, Any], lang: dict[str, Any], clips: int, seconds: float) -> str:
    return f"""# Piper package — {spec['name_hint']}

Captured with Wallie's Voice Lab: {clips} clips, {seconds}s of speech, from a
cloned {lang['label']} voice.

## What is here

```
dataset/wav/NNNN.wav    {spec['sample_rate']} Hz, mono, 16-bit PCM
dataset/metadata.csv    <file>|<text>, one line per clip (Piper's own format)
corpus/                 the untrimmed-rate originals (24 kHz) and corpus.txt
manifest.json           source voice, counts, rates — read by the attach step
```

## The last mile: training

Piper cannot clone a voice from an id, so the weights have to be trained from
this folder. The dashboard does that for you — **Voice Lab → Train the Piper
voice** — running the trainer as a child process with a live log, epoch
progress and a stop button:

* It needs `piper-tts[train]` (torch + lightning) in the interpreter you point
  it at; the panel offers a one-click install and accepts a separate venv.
* It detects `piper1-gpl` (`python -m piper.train fit`) or the legacy
  `piper_train` workflow, and normalises `metadata.csv` for the newer trainer
  (which wants `0001.wav|text` next to `--data.audio_dir`, not `wav/0001.wav`).
* A stopped run (cancel, crash, dashboard closed) can be **resumed from its
  last checkpoint** — the finished epochs are kept instead of retrained.
* espeak voice: `{lang['espeak']}`; it is also recorded in `manifest.json`.
* A `.ckpt` checkpoint fine-tunes far better than training from scratch.
* {clips} clips ({seconds}s) is enough to hear the voice. A voice that has to
  live on stream wants 10–30 minutes — re-run the port with more minutes (the
  corpus is cycled, which is normal for TTS datasets).

The run writes `train/run-*/train.log`, its checkpoints and — on success —
`<name>.onnx` + `.onnx.json`. Attach that pair from the same panel (or with the
`⤒ attach model` strip) and the voice lands in `voices/`.

Docs: https://github.com/OHF-Voice/piper1-gpl (training / `piper.train`).

## Then attach it

Bring the produced `.onnx` **and** its `.onnx.json` back to the Voice Lab's
"Convert to a local engine" strip and press *attach*. The pair is copied into
`voices/` under the name you give it ({spec['name_hint']} — Piper's loader wants
`<lang_REGION>-<voice>-<quality>`) and registered as a saved Piper voice.
"""


def _kokoro_readme(spec: dict[str, Any], lang: dict[str, Any], clips: int, seconds: float) -> str:
    return f"""# Kokoro package — {spec['name_hint']}

Captured with Wallie's Voice Lab: {clips} clips, {seconds}s of speech, from a
cloned {lang['label']} voice.

## What is here

```
corpus/NNNN.wav     {spec['sample_rate']} Hz, mono, 16-bit PCM (Kokoro's rate)
corpus.txt          the line read in each clip
manifest.json       source voice, counts, rates — read by the attach step
```

## Why this is a corpus, not a voice

A Kokoro voice is a **learned style embedding** (`voices/<id>.pt`); the engine
has no zero-shot path that turns a reference clip into a voice. So there is
nothing to convert here — the audio is the honest bridge, and it is what a
fine-tune consumes.

## Then attach it

After a fine-tune produces a `.pt` voicepack, drop it in the Voice Lab's
"Convert to a local engine" strip and press *attach*. It is copied to
`voices/kokoro/<name>.pt` (the name must look like `{spec['name_hint']}`) and
registered as a saved Kokoro voice, so the Voice page picker lists it.
"""


def _clear_dir(path: Path) -> None:
    """Start a package from scratch, never deleting anything outside it."""
    root = ported_dir().resolve()
    target = path.resolve()
    if root not in target.parents:
        raise VoicePortError("refusing to write outside the ported voices folder")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_package(
    *,
    target: str,
    name: str,
    takes: Sequence[dict[str, Any]],
    source: PortSource,
    lang: str,
    now: str = "",
) -> dict[str, Any]:
    """Write the engine-specific package for already-captured takes."""
    spec = target_spec(target)
    corpus = lang_spec(lang)
    now = now or _now_iso()
    pkg = ported_dir() / f"{slug(name)}-{spec['id']}"
    _clear_dir(pkg)

    # The originals are kept at the capture rate for every target, so a package
    # can be re-used for the other engine without spending a single credit.
    for take in takes:
        _write_wav(pkg / "corpus" / f"{take['index']:04d}.wav", take["pcm"], CAPTURE_SR)
    (pkg / "corpus.txt").write_text(
        "\n".join(t["text"] for t in takes) + "\n", encoding="utf-8"
    )

    seconds = round(sum(float(t["seconds"]) for t in takes), 2)
    files: list[str] = [f"corpus/{len(takes)} clips", "corpus.txt"]

    if spec["id"] == "piper":
        rate = int(spec["sample_rate"])
        dataset = pkg / "dataset"
        dataset.mkdir(parents=True, exist_ok=True)
        rows: list[str] = []
        for take in takes:
            rel = f"wav/{take['index']:04d}.wav"
            _write_wav(dataset / rel, resample_pcm16(take["pcm"], CAPTURE_SR, rate), rate)
            # One line per clip; a pipe inside the text would split the row.
            rows.append(f"{rel}|{take['text'].replace('|', ',')}")
        (dataset / "metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        (pkg / "README.md").write_text(
            _piper_readme(spec, corpus, len(takes), seconds), encoding="utf-8"
        )
        files += [f"dataset/wav/{len(takes)} clips @ {rate} Hz", "dataset/metadata.csv", "README.md"]
        next_step = (
            f"dataset ready ({clips_note(len(takes), seconds)}) — press Train below to run "
            "the Piper trainer here, then attach the .onnx it exports"
        )
    else:
        rate = int(spec["sample_rate"])
        for take in takes:
            if rate != CAPTURE_SR:
                _write_wav(
                    pkg / "corpus" / f"{take['index']:04d}.wav",
                    resample_pcm16(take["pcm"], CAPTURE_SR, rate),
                    rate,
                )
        (pkg / "README.md").write_text(
            _kokoro_readme(spec, corpus, len(takes), seconds), encoding="utf-8"
        )
        files += [f"corpus {len(takes)} clips @ {rate} Hz", "README.md"]
        next_step = (
            f"corpus ready ({clips_note(len(takes), seconds)}) — Kokoro cannot learn a voice "
            "from audio, so fine-tune a .pt voicepack and attach it here"
        )

    manifest = {
        "format": "wallie.port",
        "version": 1,
        "created_at": now,
        "target": spec,
        "source": source.to_dict(),
        "name": name,
        "lang": {"id": corpus["id"], "label": corpus["label"], "espeak": corpus["espeak"]},
        "clips": len(takes),
        "seconds": seconds,
        "capture_sample_rate": CAPTURE_SR,
        "clips_meta": [
            {"index": t["index"], "text": t["text"], "seconds": t["seconds"]} for t in takes
        ],
    }
    (pkg / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    files.append(MANIFEST_FILE)

    return {
        "ok": True,
        "target": spec["id"],
        "name": name,
        "package": str(pkg),
        "clips": len(takes),
        "seconds": seconds,
        "sample_rate": int(spec["sample_rate"]),
        "capture_sample_rate": CAPTURE_SR,
        "files": files,
        "next": next_step,
        "attach_name": spec["name_hint"],
        "attach_kind": spec["model_kind"],
        "warning": (
            "captured audio, not the model — "
            + ("train the .onnx from this dataset" if spec["id"] == "piper"
               else "fine-tune the .pt from this corpus")
        ),
    }


def clips_note(clips: int, seconds: float) -> str:
    return f"{clips} clips · {seconds}s"


def read_manifest(package: str | Path) -> dict[str, Any]:
    """Manifest of a package the UI is pointing at (never trusts a raw path)."""
    path = Path(package)
    root = ported_dir().resolve()
    try:
        resolved = path.resolve()
    except OSError as e:
        raise VoicePortError(f"bad package path: {e}") from e
    if root not in resolved.parents:
        raise VoicePortError("that folder is not one of this profile's ported packages")
    manifest = resolved / MANIFEST_FILE
    if not manifest.is_file():
        raise VoicePortError("that folder has no manifest.json — re-run the conversion")
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise VoicePortError(f"could not read the package manifest: {e}") from e


async def convert(
    *,
    profile: str,
    provider: str,
    voice_id: str,
    name: str,
    target: str,
    minutes: float,
    lang: str,
    secrets: Secrets,
    now: str = "",
) -> dict[str, Any]:
    """Capture + package in one call — what ``POST /api/voices/port`` runs."""
    spec = target_spec(target)
    corpus = lang_spec(lang)
    clips = clips_for(minutes, corpus["id"])
    plan = preflight(
        provider=provider,
        voice_id=voice_id,
        target=target,
        seconds=clips * AVG_CLIP_SECONDS,
        lang=corpus["id"],
        secrets=secrets,
    )
    source = PortSource(
        name=name or voice_id,
        provider=provider.strip().lower(),
        voice_id=voice_id.strip(),
        source=f"port:{provider.strip().lower()}",
    )
    takes = await capture(
        provider=source.provider,
        voice_id=source.voice_id,
        secrets=secrets,
        clips=clips,
        lang=corpus["id"],
    )
    report = build_package(
        target=spec["id"],
        name=name or voice_id,
        takes=takes,
        source=source,
        lang=corpus["id"],
        now=now,
    )
    report.update({
        "plan": plan,
        "profile": profile,
        "audition": {
            "url": "data:audio/wav;base64," + _b64(takes[0]["pcm"]) if takes else "",
            "text": takes[0]["text"] if takes else "",
            "seconds": takes[0]["seconds"] if takes else 0,
        },
        "library": [p.to_dict() for p in voice_lab.load_library(profile)],
    })
    return report


def _b64(blob: bytes) -> str:
    import base64
    return base64.b64encode(blob).decode("ascii")


# ---------------------------------------------------------------------------
# Attach — put the trained model where the engine looks for it
# ---------------------------------------------------------------------------

def _piper_target_dir() -> Path:
    from . import piper_setup
    return piper_setup.voices_dir()


def _kokoro_target_dir() -> Path:
    from . import kokoro_setup
    return kokoro_setup.voices_dir()


def attach(
    *,
    profile: str,
    target: str,
    files: Sequence[tuple[str, bytes]],
    name: str = "",
    display_name: str = "",
    source_provider: str = "",
) -> dict[str, Any]:
    """Install a model produced from a port and register it as a local voice.

    ``target`` decides both the file layout and the naming rule the engine's own
    loader enforces, so a misnamed file is refused here with the rule spelled
    out instead of failing later inside ``PiperTTS``.
    """
    spec = target_spec(target)
    blobs = [(fn or f"file{i}", blob) for i, (fn, blob) in enumerate(files, 1)]
    blobs = [(fn, blob) for fn, blob in blobs if blob]
    if not blobs:
        raise VoicePortError("attach the model file(s) first")

    if spec["id"] == "piper":
        return _attach_piper(profile, blobs, name, display_name, source_provider)
    return _attach_kokoro(profile, blobs, name, display_name, source_provider)


def _two_suffix_groups(blobs: Sequence[tuple[str, bytes]]) -> tuple[tuple[str, bytes] | None, tuple[str, bytes] | None]:
    model = next(((fn, b) for fn, b in blobs if fn.lower().endswith(".onnx")), None)
    config = next(
        ((fn, b) for fn, b in blobs if fn.lower().endswith(".onnx.json")),
        None,
    )
    return model, config


def _attach_piper(
    profile: str,
    blobs: Sequence[tuple[str, bytes]],
    name: str,
    display_name: str,
    source_provider: str,
) -> dict[str, Any]:
    from . import piper_setup
    from . import local_voices

    model, config = _two_suffix_groups(blobs)
    if model is None:
        raise VoicePortError("a Piper voice needs the .onnx file (and its .onnx.json)")
    if config is None:
        raise VoicePortError(
            "the .onnx.json config must come with the model — Piper loads both"
        )
    stem = (name or Path(model[0]).stem.replace(".onnx", "")).strip()
    if not piper_setup.is_valid_voice_name(stem):
        raise VoicePortError(
            f"“{stem}” is not a Piper voice name — use <lang_REGION>-<voice>-<quality>, "
            f"e.g. {TARGETS[0]['name_hint']}"
        )
    directory = _piper_target_dir()
    directory.mkdir(parents=True, exist_ok=True)
    onnx_path = directory / f"{stem}.onnx"
    cfg_path = directory / f"{stem}.onnx.json"
    onnx_path.write_bytes(model[1])
    cfg_path.write_bytes(config[1])

    label = (display_name or stem).strip()[:60]
    voice = local_voices.add_local(profile, "piper", {
        "name": label,
        "voice_id": str(onnx_path),
        "notes": f"ported from {source_provider or 'a cloud voice'} · {stem}",
        "source": f"port:{source_provider or 'cloud'}",
    })
    return {
        "ok": True,
        "target": "piper",
        "name": label,
        "path": str(onnx_path),
        "files": [str(onnx_path), str(cfg_path)],
        "voice": voice.to_dict(),
    }


def _attach_kokoro(
    profile: str,
    blobs: Sequence[tuple[str, bytes]],
    name: str,
    display_name: str,
    source_provider: str,
) -> dict[str, Any]:
    from . import kokoro_setup
    from . import local_voices

    pack = next(((fn, b) for fn, b in blobs if fn.lower().endswith(".pt")), None)
    if pack is None:
        raise VoicePortError("a Kokoro voice is a .pt voicepack — attach that file")
    stem = (name or Path(pack[0]).stem).strip()
    if not kokoro_setup.VOICE_NAME_RE.match(stem):
        raise VoicePortError(
            f"“{stem}” is not a Kokoro voice id — use <lang><gender>_<name>, "
            f"e.g. {TARGETS[1]['name_hint']}"
        )
    directory = _kokoro_target_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.pt"
    path.write_bytes(pack[1])

    label = (display_name or stem).strip()[:60]
    # The engine resolves a Kokoro voice BY ID (voices/kokoro/<id>.pt), so the id
    # is the stem — not the path — or the picker would hand it a filename it
    # cannot validate.
    voice = local_voices.add_local(profile, "kokoro", {
        "name": label,
        "voice_id": stem,
        "notes": f"ported from {source_provider or 'a cloud voice'} · {stem}",
        "source": f"port:{source_provider or 'cloud'}",
    })
    return {
        "ok": True,
        "target": "kokoro",
        "name": label,
        "path": str(path),
        "files": [str(path)],
        "voice": voice.to_dict(),
    }


# ---------------------------------------------------------------------------
# Everything the UI needs to render the panel in one call
# ---------------------------------------------------------------------------

def panel_payload(profile: str) -> dict[str, Any]:
    """Targets, languages and the saved voices that can be ported."""
    from . import piper_setup, kokoro_setup

    def _installed(module: str) -> bool:
        try:
            from importlib import util
            return util.find_spec(module) is not None
        except (ImportError, ValueError):
            return False

    return {
        "targets": [
            {
                **spec,
                "installed": _installed(spec["id"]),
                "name_rule": (
                    "<lang_REGION>-<voice>-<quality>" if spec["id"] == "piper"
                    else "<lang><gender>_<name>"
                ),
            }
            for spec in TARGETS
        ],
        "languages": [
            {"id": key, "label": spec["label"], "lines": len(spec["lines"])}
            for key, spec in CORPUS.items()
        ],
        "default_lang": DEFAULT_LANG,
        "min_minutes": MIN_MINUTES,
        "max_minutes": MAX_MINUTES,
        "ported_dir": str(ported_dir()),
        "piper_dir": str(piper_setup.voices_dir()),
        "kokoro_dir": str(kokoro_setup.voices_dir()),
        "library": [s.to_dict() for s in sources_from_library(profile)],
    }
