> **This is a fork.** It is `isyatodev/wallie-V2`, a personal fork of
> **[Alradyin/wallie-V2](https://github.com/Alradyin/wallie-V2)** by Alradyin — the original
> project, its design, and the vast majority of the code in this repo are theirs. Credit for the
> architecture, the streamer core, the vision/hearing/play pipelines and the original dashboard
> belongs upstream. If you are looking for the canonical project, go there.
>
> This fork keeps the upstream engine and adds dashboard-level work on top: a **Voice Lab** with
> provider voice cloning, **any-model vision testing**, a fuller **Kokoro** voice picker
> (including pt-BR), and per-profile themes. See [What this fork adds](#what-this-fork-adds) —
> everything else in this document describes behavior inherited from upstream.

<p align="center">
  <h1 align="center">Wallie <small>(fork)</small></h1>
  <p align="center"><strong>The open-source AI streamer that sees, hears &amp; reacts — and actually feels alive.</strong></p>
</p>

<p align="center">
  <a href="https://github.com/isyatodev/wallie-V2/archive/refs/heads/main.zip"><img alt="Download (fork)" src="https://img.shields.io/badge/Download_ZIP_(fork)-2ea44f?style=for-the-badge&logo=github&logoColor=white" /></a>
  &nbsp;
  <a href="https://github.com/Alradyin/wallie-V2"><img alt="Upstream project" src="https://img.shields.io/badge/Upstream-Alradyin%2Fwallie--V2-555?style=for-the-badge&logo=github&logoColor=white" /></a>
  &nbsp;
  <a href="docs/guide.md"><img alt="Docs" src="https://img.shields.io/badge/Docs-555?style=for-the-badge" /></a>
</p>

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg" />
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-brightgreen.svg" />
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg" />
</p>

<p align="center">
  <a href="#what-this-fork-adds">This fork</a> &nbsp;·&nbsp;
  <a href="#quick-start">Quick Start</a> &nbsp;·&nbsp;
  <a href="#the-voice-lab">Voice Lab</a> &nbsp;·&nbsp;
  <a href="#vision-and-hearing">Vision &amp; Hearing</a> &nbsp;·&nbsp;
  <a href="#architecture">Architecture</a> &nbsp;·&nbsp;
  <a href="#roadmap">Roadmap</a>
</p>

---

Every AI streamer you've seen has the same problem.

It says two sentences. Pauses. Says two more. Pauses. Describes your YouTube homepage like a screen reader. Forgets what it said thirty seconds ago. Ends every thought with "what do you guys think?" into a chat that doesn't exist. It's not a show — it's a tech demo on loop.

**Wallie is different.** It's an AI that actually streams — develops thoughts across minutes, reacts to your screen like a person who's been using that computer all day, remembers what it covered an hour ago, takes opinions and sticks with them, drifts between topics the way real conversations drift, and shuts up when there's nothing worth saying.

You design the streamer. Wallie runs the show.

```
   persona  ──┐
   topics   ──┤
   chat     ──┤
   vision   ──┼──►  Wallie  ──►  Voice (TTS)  ──►  OBS / virtual cable  ──►  your stream
   hearing  ──┤             ──►  Live2D avatar (VTube Studio)
   schedule ──┘
```

Pick the personality. Pick the voice. Pick the LLM. Pick the platform. Everything is swappable, everything is configurable from the browser, and the whole thing runs on your machine with your keys.

---

## What this fork adds

Everything below is fork-specific. If a feature is not listed here, assume it came from
[upstream](https://github.com/Alradyin/wallie-V2) and is documented in the sections that follow.

| Area | What's here | Where |
|---|---|---|
| **Voice Lab** | A second voice page: build a **cloned voice** on ElevenLabs or Fish Audio from reference audio, plus a **per-profile library of saved voices** (provider + voice id + tuning) you can recall in one click | Dashboard → **Voice Lab** |
| **Voice cloning** | Upload up to 5 clips **or record straight from the default mic** (WAV is wrapped with stdlib only — no extra dependency), then create the voice at the provider without leaving the dashboard | Dashboard → **Voice Lab** |
| **Voice A/B** | Read the *same line* with two voice configurations and hear them back to back (`play A → B`) or side by side. The audio is returned to the page — it never touches the live output device, and nothing is saved | Dashboard → **Voice Lab** |
| **Voice library portability** | **⇪ copy** one voice, a ticked selection or the whole library into another profile; **⇩ pull** the same way back; **⤓ export / ⤒ import** the library as one small `.json` file you can keep in git, mail to someone or restore after a reinstall | Dashboard → **Voice Lab** |
| **Voice mirroring** | **⇄ mirror** makes two profiles hold the same saved voices, **both ways**, showing the diff first. Same-named voices that differ are left alone by default — or you pick which side wins — and any overwrite asks before replacing | Dashboard → **Voice Lab** |
| **Voice partner badge** | Mark a profile as another one's **voice partner** and the profile picker shows a **⇄ badge** whenever the two libraries drift apart (with the counts in the tooltip); click it to land in the Voice Lab with the diff already loaded. A **⇄** also marks the profile in the dropdown itself, and switching to a partnered profile syncs the missing voices both ways, saying what it copied | Dashboard → **Voice Lab** |
| **Local voices in the library** | Piper/Kokoro voices saved on the **Voice** page live in the same per-profile library as the Voice Lab presets, so they show up in the A/B and travel with every copy, pull, backup and mirror | Dashboard → **Voice** / **Voice Lab** |
| **Piper voice catalogue** | Browse the official `rhasspy/piper-voices` catalogue **inside the dashboard** — search by name, filter by language, see speakers/quality/size per voice — and download any of them with a live progress bar. No CLI needed | Dashboard → **Voice** |
| **Piper tuning** | Speed (`length_scale`), **Expressiveness** (`noise_scale`) and **Pacing variety** (`noise_w`) as sliders, so a local voice can be tuned to taste instead of taking Piper's defaults | Dashboard → **Voice** |
| **Kokoro languages** | A real language picker instead of US/UK only — including **Portuguese (Brazil)** (`pf_dora` / `pm_alex` / `pm_santa`), plus Spanish, French, Hindi, Italian, Japanese and Mandarin. Voice ids stay in sync with the language, and the voice field autocompletes per language | Dashboard → **Voice** |
| **Vision test with any model** | The Vision test strip can probe **any provider + model** (and base URL) *without saving it* — the escape hatch when a provider retires a model mid-stream | Dashboard → **Vision** |
| **Current Gemini defaults** | New profiles and the setup wizard now point at `gemini-3.8-flash`; the deprecated `gemini-2.0-flash` entry is gone and the restricted 2.5 family is labelled as such | Dashboard → **Engine** / setup wizard |
| **Per-profile themes** | Dashboard accent themes (cyan / amber / rose) saved in the profile, so each persona can look different | Top bar |
| **Engagement gate** | Optional "only speak when addressed": answer when someone talks to Wallie, otherwise stay quiet (with optional no-LLM acknowledgements) | Dashboard → **Chat** |
| **Audio device routing** | Pick the TTS output device and the hearing loopback by name, with a test beep and stale-device warnings when a saved device disappears | Dashboard → **Voice** / **Hearing** |
| **Provider-block healing** | Deleting a provider block blanks any config reference pointing at it (and dangling refs are healed on load) instead of silently falling back | Dashboard → **API Keys** |

### Upstream features inherited by this fork

Play mode (Minecraft), vision with the SKIP escape hatch, hearing with local Whisper + music
analysis, the Live2D avatar, LivePix/Streamlabs donations, the caption overlay, long-term and
per-person memory, mood/attention engines and the multi-provider architecture are all upstream
work — see [Features](#features) and [Architecture](#architecture).

---

## The Voice Lab

The fork's biggest addition. Dashboard → **Voice Lab** has two halves:

**1. Saved voices (presets).** A preset snapshots the Voice page's current settings — provider,
voice id, speed, stability, the Kokoro language, and so on — under a name. Presets are stored per
profile in `profiles/<profile>.voices.json`, so "Wallie BR" can be a Kokoro pt-BR voice while
"Wallie EN" is a cloned ElevenLabs voice, and each profile remembers its own. **use** applies a
preset and saves; **▶ test** applies it *and* synthesizes a sample line through the real TTS
pipeline so you hear it before going live. The output device is deliberately excluded from a
preset — it routes audio, it isn't part of a voice.

**2. Clone a voice.** Creates a *real* voice on your own provider account:

| Provider | Endpoint used | Key needed | Notes |
|---|---|---|---|
| **ElevenLabs** | `POST /v1/voices/add` (instant voice clone) | `ELEVENLABS_API_KEY` | ~1 minute of clean speech is enough |
| **Fish Audio** | `POST /model` (private voice model) | `FISH_API_KEY` | Returns a model id you use as the voice id |

Provide reference audio by **dropping in files** (wav / mp3 / m4a / flac / ogg, up to 5 files,
under ~10 MB each) or by **recording from the default microphone** right in the page — the clip is
wrapped into a WAV with the Python stdlib, so no `soundfile`/`ffmpeg` install is required. You can
audition every clip in the browser before cloning. On success the new voice id is saved into the
library above, ready to apply.

**3. A/B compare.** Pick two voices — a saved voice, or "current Voice page settings" for whatever the
Voice page shows right now, including unsaved edits — type one line, hit **synthesize both**. Both
clips come back to the page as playable WAVs (with per-side provider/voice/bytes/seconds metadata),
so you can hit **play A → B** to hear them back to back, or replay either one. Nothing is routed
through the output device — comparing a voice must not talk over the stream — and nothing is saved.
If one side fails (provider plan limits, a bad voice id), its own message is shown on its card and the
clip that *did* synthesize is kept, since it was already billed.

**4. Moving voices around.** The library is per profile, so there are four ways to move voices —
between profiles on this machine, or out of it entirely. Tick rows with the checkboxes on the left
(**select all** / **clear** sit next to the list) to act on several voices at once:

| Move | What it does |
|---|---|
| **⇪ / ⇪ copy** | pushes voices **into** another profile: a single row's **⇪**, the ticked selection, or the whole list |
| **⇩ pull** | the inverse — reads *another* profile's library, you tick what you want, and it lands here |
| **⤓ / ⤒ `.json`** | exports the library (or just the ticked voices) as a portable file and merges one back in |
| **⇄ mirror** | makes this profile and another one hold the same voices, **both ways**, after showing the diff |

Beside the ⇄ button, **☆ set as voice partner** records one of those profiles as this
profile's *partner* — the one it is meant to stay identical to. That only stores the choice
(nothing is copied) and switches on **automatic syncing**: whenever you switch to a profile
that has a partner and the two libraries differ, the voices each side is missing are copied
both ways as part of the switch, with a short toast saying what moved. The policy is always
*leave differing voices alone* — you can pick a winner with ⇄ mirror, but the server never does
it for you — and a library too full to take them reports the refusal instead of failing the
switch. From then on the **profile picker in the top bar shows a ⇄ badge** while the two
libraries differ, with the counts in its tooltip, and a ⇄ next to that profile's name in the
dropdown. **Click the badge** and the Voice Lab opens with the partner already chosen as the
mirror target, the diff loaded and the ⇄ mirror button waiting — the shortcut when you want to
see exactly what differs before evening them out. A partner that was since deleted is reported as *partner missing* rather than
silently forgotten, and the same button — now **★ voice partner** — clears the partnership
(and stops the auto-sync).

The export is a small, plain JSON file — good for backups, for versioning next to the profile, or
for handing a voice to someone else:

```json
{
  "format": "wallie.voices",
  "version": 1,
  "exported_at": "2026-10-10T01:00:00+00:00",
  "source_profile": "default",
  "count": 1,
  "presets": [
    {
      "name": "Wallie BR",
      "provider": "kokoro",
      "voice_id": "pf_dora",
      "notes": "",
      "source": "local:kokoro",
      "created_at": "2026-10-09T23:58:14+00:00",
      "tts": { "kokoro_lang_code": "p" }
    }
  ]
}
```

What every move has in common:

- **Upsert by name.** A voice with the same name (case-insensitive) is *replaced*, never duplicated,
  so re-importing the same file or mirroring twice is a harmless no-op.
- **Overwriting asks first.** If that name already exists **with different settings**, the server
  refuses with a `409` carrying the exact diff — `Wallie BR: kokoro : pf_dora → elevenlabs : v-9` —
  and the dashboard asks before replacing anything; declining writes nothing. Mirroring is explicit
  instead: the diff lists the clashing voices and you pick *leave them alone* (the default), *keep
  this profile's version* or *use the other profile's version*.
- **No pointless writes.** Voices that are already identical are left untouched, and a move that
  would overflow a profile's 60-voice cap is refused *before* anything is written.
- **A bad file fails loudly, not halfway.** An entry with no name is skipped, but a file that is not
  a voice backup, is empty, is over ~2 MB or was written by a newer version is refused with a
  message that says why.
- **Local voices stay local to their engine.** A Piper preset carries only `piper_*` tuning, a
  Kokoro one only `kokoro_*`, and neither can bring in the other engine's settings — nor the output
  device, which routes audio and is not part of a voice.

A few honest notes:

- Cloning is **provider-side**. The audio is uploaded to ElevenLabs/Fish Audio under *your* account
  and the provider's terms apply to whatever you clone. Only clone voices you have the rights to use.
- **Piper and Kokoro cannot be cloned into** — they're local, fixed-voice models. The page says so
  instead of failing silently. Kokoro's language picker is the closest equivalent for local voices.
- Malformed or too-short samples are rejected by the provider; the provider's own message is shown
  so you can fix it rather than guess.
- Piper/Kokoro voices saved on the **Voice** page live in this same library — that is what makes them
  appear in the A/B and travel with every copy, pull, backup and mirror. (A pre-unification
  `profiles/<profile>.localvoices.json` is imported into it once and then removed.)

---

## Vision and hearing

### Testing vision with a different model

The Vision page's test strip (below the model settings) grabs one frame and sends it to a vision
model. It has three override boxes — **provider**, **model** and (for Ollama / OpenAI-compatible
endpoints) **base URL**:

- Leave them blank → the test uses the **saved config**, exactly like a live session (a dedicated
  Vision block when one resolves, otherwise the main engine).
- Fill any of them → the test builds a **throwaway provider** for that one request. Nothing is
  written to the profile, so you can try candidate models without committing.

This is the recovery path when a provider retires a model. The API answers something like
*"This model models/gemini-2.5-flash is no longer available to new users. Please update your code
to use models/gemini-3.8-flash"* — type that model into the strip, confirm it works, then move it
into **Engine → Model** (or a dedicated Vision block) for real. The response footer shows which
provider:model answered and whether it was an override.

### Vision, hearing and Play (upstream)

- **Vision** — first-person ownership of what's on screen, a `SKIP` escape hatch when there's
  nothing specific to name, scroll/typing/app-switch activity adaptation, and an attention engine
  that decides between a deep reaction, a glance, a tangent, or deliberate silence.
- **Hearing** — WASAPI loopback capture plus local Whisper (or a remote OpenAI-compatible STT
  block), fused with vision into a single reaction, with a self-echo guard so Wallie never reacts
  to its own TTS.
- **Play (Minecraft)** — a planning brain plus [Baritone](https://github.com/cabaletta/baritone)
  and a custom Fabric mod, so the commentary is grounded in the agent's real inventory, health and
  goal instead of guessing from a frame.

---

## Quick start

### Just double-click `start.bat`

No Python knowledge, no terminal. Download → double-click → done.

1. **Download the fork as a ZIP** ([`isyatodev/wallie-V2`](https://github.com/isyatodev/wallie-V2/archive/refs/heads/main.zip)) and unzip it — or, to get upstream instead, [Alradyin/wallie-V2](https://github.com/Alradyin/wallie-V2/archive/refs/heads/main.zip). Or clone:
   ```bash
   git clone https://github.com/isyatodev/wallie-V2.git
   # add upstream so you can pull their engine fixes:
   cd wallie-V2 && git remote add upstream https://github.com/Alradyin/wallie-V2.git
   ```
2. **Double-click `start.bat`.** First run installs everything it needs, then the dashboard opens at `http://127.0.0.1:8765`.
3. **Paste your API key** in the dashboard → pick a model → hit **Start**.

> **macOS / Linux:** run `./start.sh` instead (`chmod +x start.sh` once).

### Pick your budget

| Path | LLM | TTS | Cost/hour | Quality |
|---|---|---|---|---|
| **Free** | Gemini 3.8 Flash | Piper (local) | $0 | Good for testing |
| **Free, better voice** | Gemini 3.8 Flash | Kokoro (local, incl. **pt-BR**) | $0 | Best free voice |
| **Cheap & fast** | Groq (Llama 4 Scout) | Fish Audio | ~$1.50 | Best balance |
| **Premium** | Claude Sonnet | ElevenLabs | ~$6.50 | Best quality + vision |

The setup wizard's **free** path now configures `gemini-3.8-flash` — the 2.5 family is restricted to
accounts that already used it, and new keys get a `404` telling you to move to 3.8.

<details>
<summary><strong>Kokoro (free local TTS) setup</strong></summary>

**On Windows the one-click setup already does this.** `start.bat` installs Kokoro and
pre-downloads the model + voice before the dashboard ever opens, so the free local voice is ready
on the first run (~500 MB; set `WALLIE_SKIP_KOKORO=1` to skip it, or pre-download another language
with `WALLIE_KOKORO_LANG=p WALLIE_KOKORO_VOICE=pf_dora`). It closes by *verifying* it — a line is
really synthesised and **played through your configured output device**, using the same provider
and player a session builds — and reports the verdict, so "setup complete" never hides a voice
that doesn't work or one that would come out of the wrong speakers. The manual steps below are for
macOS/Linux, for a language the automatic step didn't pre-load (the model itself is shared, so a
second language is only a few MB), or if that step failed for any reason.

> **Python 3.10–3.12 required.** Kokoro's multilingual wheels (`kokoro>=0.9.4`, needed for
> Portuguese and the other non-English voices) are published only for `>=3.10,<3.13` — there is no
> 3.13/3.14 build, so `pip` reports *"No matching distribution found"* on a newer interpreter. If
> your venv is 3.13+, install Python 3.12, recreate the venv (`py -3.12 -m venv .venv`), reinstall
> `requirements.txt`, then run the helper below. `python scripts/install_kokoro.py` checks this and
> tells you exactly what to do.

The project ships a helper that installs the versions it expects **and** pre-downloads the model +
voice, so the first line is never stalled by a HuggingFace fetch:

```bash
# 1. install kokoro (>= 0.9.4) + soundfile (+ a language extra; PyTorch comes along)
python scripts/install_kokoro.py --install

# 2. download the model + a voice, e.g. Portuguese (Brazil) pf_dora
python scripts/install_kokoro.py --lang p --voice pf_dora
```

Manual equivalent: `pip install "kokoro>=0.9.4" soundfile`. Then in the dashboard: **Voice** →
provider `kokoro` → pick a **Language** (the voice list follows it) → pick a **Voice**. No API key,
no cloud calls. For **Portuguese (Brazil)** choose the `Portuguese (Brazil)` language — the voice id
becomes `pf_dora` / `pm_alex` / `pm_santa`. The **Voice** tab and the first-run **Setup** wizard both
carry this same guide in a collapsible panel.

Non-English languages need a reasonably recent `kokoro` (≥ 0.9.4). Japanese and Mandarin may also
need their `misaki` extra (`python scripts/install_kokoro.py --lang j --install`) and espeak-ng,
depending on your platform. If a language refuses to load, the Kokoro error in the server log says
exactly which package is missing.
</details>

<details>
<summary><strong>Piper (free TTS) setup</strong></summary>

```bash
# In your wallie directory, with the venv active:
pip install piper-tts onnxruntime
python scripts/download_piper_voice.py en_US-amy-medium
```

Or skip the terminal entirely: on the **Voice** page with `piper` selected, **⬇ One-click install**
installs the runtime and the **Download a voice** box — click **browse catalogue** — lists every
published voice (search by name, filter by language, with speakers/quality/size) and downloads the
one you pick with a progress bar. Files land in `voices/` either way — the CLI stays as the headless
alternative.

Then in the dashboard: **Voice** → provider `piper`, path `voices/en_US-amy-medium.onnx` — and tune it
right there: **Speed** (`piper_length_scale`), **Expressiveness** (`piper_noise_scale`) and **Pacing
variety** (`piper_noise_w`). Everything you set is part of a saved voice, so it comes back with one
click from the library in the Voice Lab.
</details>

<details>
<summary><strong>Vision setup (screen reactions)</strong></summary>

Requires a vision-capable LLM.

1. **Engine** → toggle **Vision capable** ON (or configure a dedicated VISION block on **API Keys**).
2. Use a model that accepts images: `gemini-3.8-flash`, `claude-sonnet-4-6`, `gpt-4o`, or `llama-4-scout` on Groq.
3. **Vision** section → toggle ON, adjust the frame interval and sensitivity.
4. Hit **📸 Capture screen + ask LLM** in the test strip. If you want to try a different model first, put
   it in the strip's override boxes — that probe is not saved.
</details>

<details>
<summary><strong>Chat platform setup</strong></summary>

**Twitch:** Add an OAuth token from [twitchtokengenerator.com](https://twitchtokengenerator.com) (chat:read scope). Or leave it empty for anonymous read-only.

**YouTube:** Drop `client_secret.json` in `scripts/`. First run opens a browser for Google OAuth consent.

**Kick:** Just enter the channel slug. No auth needed (public Pusher WebSocket).
</details>

<details>
<summary><strong>VTube Studio avatar</strong></summary>

1. Run VTube Studio with the API enabled (Settings → API → Enable).
2. In the dashboard: **Avatar** → toggle ON.
3. First connect triggers a plugin approval popup in VTS — click Allow.
4. Expression slots are **auto-mapped** from your model's hotkeys on connect; override manually if needed.
5. Adjust lipsync gain/ceiling for your voice. Viseme lipsync (spectral mouth shape) is on by default — disable it if your model has no `ParamMouthForm`.
6. Blink, body motion and mood-reactive behaviour are on by default — tweak per feature.

Works with any Live2D model that has standard parameters (`MouthOpen`, `EyeOpenLeft/Right`,
`FaceAngleX/Y/Z`, `BodyAngleX/Y/Z`).
</details>

<details>
<summary><strong>OBS output routing</strong></summary>

Wallie plays audio through the output device you pick in **Voice → Output device**.

**Windows:** Install [VB-CABLE](https://vb-audio.com/Cable/), pick `CABLE Input` as the output device in the dashboard. In OBS: Audio Input Capture → CABLE Output.

**macOS:** Use [BlackHole](https://existential.audio/blackhole/) the same way.

**Linux:** PipeWire / PulseAudio loopback (`pw-loopback`).
</details>

---

## Features

### Bring your own everything

Six LLM providers, five TTS engines, three chat platforms, two donation platforms. Mix and match per profile.

| LLM | TTS | Chat | Donations | Avatar |
|---|---|---|---|---|
| OpenAI | Fish Audio | Twitch | LivePix (webhook) | VTube Studio (Live2D) |
| Anthropic (Claude) | ElevenLabs | YouTube | Streamlabs (Socket API) | — |
| Google (Gemini) | Piper (local, free) | Kick | — | — |
| Groq | Kokoro (local, free · multilingual) | — | — | — |
| OpenRouter | **OpenAI-Compatible (speech)** | — | — | — |
| Ollama (local, free) | — | — | — | — |
| **OpenAI-Compatible (any endpoint)** | — | — | — | — |

Beyond that, **Vision** and **Hearing (STT)** can each run on their own OpenAI-compatible block —
see [Dedicated vision & STT blocks](#dedicated-vision--stt-blocks) — and Wallie ships a
self-clearing **caption overlay** for OBS: [Captions](#captions-browser-source-overlay).

Swap providers without changing code. Run a fully offline stream with Ollama + Piper/Kokoro, or go premium with Claude + ElevenLabs — and build the voice you actually want in the **Voice Lab**.

### Full persona design

Not just a name and a system prompt. You design a *character*:

- **Identity** — name, handle, pronouns, age range, origin, archetype, backstory
- **Voice** — energy level (chill → unhinged), humor style (pick multiple: ironic, deadpan, absurd, observational, roast...), profanity level, formality
- **Flavor** — catchphrases, running gags, banned words, favorite topics, taboo topics
- **Opinions** — strong opinions toggle, admit uncertainty, break the fourth wall
- **Extra notes** — free text for anything the structured fields miss

Save multiple personas. Switch between them with a dropdown. Themes are per profile too.

### Hours-long sessions without decay

The single biggest technical challenge. Most AI chatbots break down after 20 minutes because the context window fills up.

Wallie's approach:

- **Rolling summarizer** — every ~14 segments, a background LLM call compresses older history into tight bullet notes
- **Session notes** — that compressed memory is injected into every system prompt, so the streamer knows what it already said
- **Cross-session memory** — key facts and viewer interactions persist across streams
- **Per-person memories** — with Speaker ID on, facts learned from an enrolled voice (e.g. the owner) are bound to that person ("about") and resurface naturally in the prompt the next time they talk. Works for manually added memories too (optional "about person" field in the Memory section); auto-consolidation never merges memories of different people.
- **Auto-consolidation** — when the long-term store passes the configured threshold (default 300), the memory model merges groups of related old entries into single general summaries, so the store keeps growing in *depth* instead of only in *size*.
- **Dedupe engine** — paraphrase-aware similarity check (bigram + trigram Jaccard) catches the model repeating itself in different words

### Organic pacing

Real streamers don't talk at a constant rate.

- **Mood engine** — arousal, valence, focus and talkativity drift slowly over the stream. High arousal = faster, warmer output. Low talkativity = silence beats. Mood feeds the avatar for reactive animation.
- **Attention engine** — vision events pass through a probabilistic decision layer. DEEP reactions, quick GLANCEs, personal TANGENTs, deliberate IGNOREs and SILENCE beats, with streak fatigue so it never reacts the same way twice in a row.
- **Engagement gate** (this fork) — optional: only answer when actually addressed, so Wallie doesn't monologue over a quiet channel.
- **Silence beats** — the streamer holds natural pauses when the mood says to.
- **Pipeline overlap** — the next segment starts generating while the current audio is still playing.

### Vision that isn't narration

Screen reactions are the hardest part to get right:

- **First-person ownership** — gaming: "I just got bodied by that boss", not "the character is fighting a boss". Never third-person, never narration.
- **SKIP escape hatch** — if there's nothing specific to name, the model outputs `SKIP` and stays quiet.
- **Activity adaptation** — detects scrolling, typing, app-switching and video playback, and adapts. Typing → ignore. App switch → react. Rapid browsing → wait until things settle.
- **Scene memory** — remembers what it last said about the current screen; a dedupe threshold around 0.65 catches paraphrased repetition.

### Hearing — it reacts to what it hears, not just what it sees

Wallie captures system audio (WASAPI loopback) and reacts live, fused with vision into a single reaction — sight and sound, one voice.

- **Speech & lyrics** — local speech-to-text (faster-whisper) transcribes videos, voice chat and lyrics. Uses your GPU when available.
- **Real music understanding** — pure-numpy DSP, no extra ML weight: major/minor key → mood, tempo and beat strength, instrumentation texture, production quality. A sad song reads as melancholy; a beat drop reads as energy.
- **Multimodal fusion** — one coherent reaction, never two competing ones.
- **No self-echo** — content-based guard so Wallie never reacts to its own TTS bleeding through the loopback.
- **Pure-hearing mode** — run it with vision off and it reacts to audio the way it reacts to a screen.

### Wallie plays games — not just watches them

**Play mode** hands Wallie the controls. First game: Minecraft, survival, live and unscripted.

- **Reliable autonomy** — a planning brain picks high-level goals; a deterministic skill library (crafting, smelting, tool/armour sets, mining to ore depths, food & wool, beds, combat, pickup) does the actual work, so even a small free model plays competently.
- **Human-like camera** — a custom Fabric mod eases every turn at full FPS.
- **Grounded commentary** — reactions come from the agent's real game state, fused through the same single pipeline as vision and hearing.
- **One-click install** — Fabric + mods straight from the dashboard's Play section.

### Live2D avatar with emotion

Six animation layers over a single WebSocket, so the avatar feels like a person rather than a puppet.

- **Viseme lip sync** — spectral analysis of the PCM stream estimates mouth shape in real time; front vowels spread the mouth, back vowels round it, RMS drives jaw openness, with attack/release envelope, noise-floor gating and a speaking smile.
- **Blink** — ~3.8s interval with variation and occasional double-blinks; blink rate adapts to mood.
- **Body motion + idle motion** — slow torso sway so the avatar breathes, plus head sway and eye darts that speed up as focus drops.
- **11 expression slots** — happy, surprised, laughing, angry, sad, thinking, smug, eyeroll, confused, hype, deadpan — driven by keyword regex and auto-mapped from your VTS hotkeys.
- **Mood-reactive** — arousal/valence/focus feed directly into sway amplitude, brows and eye-dart frequency.

### The dashboard

Everything is configured from the browser. No YAML files. No terminal commands after setup.

```
┌──────────────────────────────────────────────────────────────┐
│ ◤ WALLIE   Profile: marlow ▾  ⊕ ⎘ 🗑      ● LIVE  Stop  ‹  │
├──────────┬───────────────────────────────┬───────────────────┤
│          │                               │   LIVE STATUS     │
│ Identity │  Section editor               │ ┌───────────────┐ │
│ Personal.│                               │ │ Current topic  │ │
│ Voice    │  (each section has its own    │ │ Now saying...  │ │
│ VoiceLab │   editor with live controls)  │ │ Open threads   │ │
│ Topics   │                               │ │ Recent angles  │ │
│ Vision   │  Test ▶ monologue / chat / vis│ │ Session memory │ │
│ Chat     │  Output preview + ▶ speak this│ │ Mood state     │ │
│ Engine   │                               │ │ WebSocket log  │ │
│ API Keys │                               │ └───────────────┘ │
└──────────┴───────────────────────────────┴───────────────────┘
```

Three columns: navigation, section editor, live status. Test any configuration change instantly with the preview buttons before going live.

<p align="center">
  <img src="docs/images/dashboard-identity.png" alt="Dashboard — Identity" width="100%" />
  <br /><br />
  <img src="docs/images/dashboard-personality.png" alt="Dashboard — Personality" width="100%" />
</p>

---

## AI Providers — OpenAI-Compatible

Beyond the built-in providers, Wallie speaks the **generic OpenAI chat/completions API**: any endpoint that implements it works — no per-provider code.

Configure (dashboard → **Engine**, or the profile YAML):

| Setting | Where | Example |
|---|---|---|
| Provider | Engine → Provider | `openai_compatible` |
| Base URL | Engine → Base URL | `https://api.openai.com/v1` · `https://openrouter.ai/api/v1` · `https://api.groq.com/openai/v1` · your own gateway |
| Model | Engine → Model | exact model id your endpoint exposes |
| API key | **API Keys → OpenAI-Compatible (generic)** or `.env` | `OPENAI_COMPATIBLE_API_KEY=…` |

The Base URL must include the version path (`…/v1`). Streaming, temperature/top-p, max tokens,
penalties, timeout and retries all work as with the named providers. Vision works the same way as
everywhere else: tick **Model supports vision** only if your model accepts images — Wallie refuses
to attach screenshots to a text-only model and disables the vision loop instead of shipping it a
broken request.

Only the OpenAI-compatible surface is used — no provider-specific extensions are sent (prompt-cache
breakpoints, for example, remain an OpenRouter-only feature).

### Dedicated vision & STT blocks

Some hosts split models across different providers — a Qwen-VL endpoint for vision while the brain
stays on Claude, a hosted Whisper for hearing while the brain runs locally. Each subsystem therefore
has its own OpenAI-compatible block:

| Subsystem | Config (dashboard) | API key (.env) | Endpoint shape |
|---|---|---|---|
| **Vision** | Engine → Vision model source → *Separate block*, then Vision → Vision model | `OPENAI_COMPATIBLE_VISION_API_KEY` | `POST {base}/chat/completions` (images in, text out) |
| **TTS** | Voice → Provider → *OpenAI-Compatible* | `OPENAI_COMPATIBLE_TTS_API_KEY` | `POST {base}/audio/speech` (PCM16 out) |
| **STT (Hearing)** | Hearing → STT engine → *OpenAI-Compatible (remote)* | `OPENAI_COMPATIBLE_STT_API_KEY` | `POST {base}/audio/transcriptions` |

Vision routing: when the dedicated block is configured, every vision-intent segment runs on it;
monologue/chat/outro stay on the main LLM. If the dedicated block is missing or fails to build,
vision falls back to the main LLM with a logged warning — the stream never dies over config.

Remote STT swaps the local Whisper model for an HTTP call — same hallucination guards, VAD path and
self-echo filter, without the VRAM cost. The local model stays selected whenever the engine is left
on **Local Whisper**.

## Captions — browser source overlay

Wallie's speech can be rendered as live captions on stream via a self-contained web page:

1. **Captions** → toggle **Enable caption overlay** → Save.
2. Start the orchestrator. The dashboard shows the overlay URL (default `http://127.0.0.1:8765/captions`).
3. In OBS: Sources → **+** → **Browser** → paste the URL (width 1280, height 200 works well).

How it behaves:

- Sentences stream in **as they are TTS'd**, sentence by sentence, over Server-Sent Events.
- **The caption box is emptied when the TTS segment ends** (after an optional hold, 0s default) — an old message never lingers on screen.
- Style knobs live in the dashboard: font size, box opacity, max lines, lowercase, persona-name prefix. The page is transparent, so it composites cleanly over any scene.
- The **Push to overlay** test button exercises the real bridge — the line should appear and then vanish ~2.5s later.

## Donations

Two platforms are supported, and both feed the **same** pipeline as chat — no second AI path:

```
LivePix (webhook) ──┐
                    ├──► DonationEvent (normalized) ──► orchestrator queue
Streamlabs (socket)─┘         (as highlight chat)            │
                                                        LLM → TTS → avatar
```

A donation replies in Wallie's voice through the existing TTS, thanks the donor by name and reacts
to their message. The LLM sees donations as their own turn type (`[DONATION — donor — amount]` +
source + message), never as indistinguishable chat text. The streamer's own messages are tagged
`[STREAMER]` (from the Twitch broadcaster badge) and viewer messages `[VIEWER]`.

**LivePix** ([docs](https://docs.livepix.gg))
1. Create an app in your LivePix account settings → get `client_id` / `client_secret` → put them in `.env` (`LIVEPIX_CLIENT_ID`, `LIVEPIX_CLIENT_SECRET`) or API Keys.
2. Optionally set `LIVEPIX_USER_ID` — webhook payloads for other accounts are then rejected.
3. Enable **LivePix** in dashboard → Donations. The webhook is mounted on the dashboard app at `livepix_webhook_path` (default `/webhooks/livepix`). The dashboard must be reachable from the internet: run with `DASHBOARD_HOST=0.0.0.0` behind a tunnel/reverse proxy, then register `http://<host>:<port>/webhooks/livepix` as a LivePix webhook.
4. Flow: webhook validates the payload → dedupes by the real LivePix resource id → answers `HTTP 200` immediately → enrichment (`GET /v2/messages/{id}`) and queuing happen in the background.

**Streamlabs** ([docs](https://dev.streamlabs.com/docs/socket-api))
1. Streamlabs Dashboard → Settings → API Settings → API Tokens → copy the **Socket API Token** → `STREAMLABS_SOCKET_TOKEN` in `.env`.
2. Or authorize OAuth with `donations.read` + `socket.token` scopes and set `STREAMLABS_ACCESS_TOKEN`; the socket token is fetched from `/socket/token` automatically.
3. Enable **Streamlabs** in dashboard → Donations. Wallie connects to `sockets.streamlabs.com`, auto-reconnects with jittered exponential backoff and keeps exactly one live connection.
4. Only `type === "donation"` events are processed. Follows/subs/bits/raids are ignored by design.

**Testing without money (mock mode)**

- Dashboard → Donations → **Test strip**: inject a fake LivePix/Streamlabs donation into the real queue, POST a LivePix-shaped payload through the actual webhook route, or verify your Streamlabs token reaches a live socket.
- CLI with the dashboard running:

```
python scripts/test_donations.py livepix     --donor Maria --amount 10 --message "manda salve"
python scripts/test_donations.py streamlabs  --donor Joao  --amount 5  --message "gg"
python scripts/test_donations.py both
```

## Architecture

```
  Screen ────► Vision ───┐
  (mss + pHash)          │
                         ▼
  Chat ─────────►  Orchestrator  ◄──── Persona + Topics + Mood
  (YT/Twitch/Kick)      ▲
                        │
  LivePix (webhook) ────┤
                        │        donation events enter the SAME queue
  Streamlabs (socket) ──┘        as highlight chat — one pipeline only
                         │  intent → system prompt + user message
                         ▼
                   LLM (streaming)     ← 6 providers (incl. any OpenAI-compatible endpoint)
                         │
                         │  token stream
                         ▼
                  SentenceStreamer      ← splits tokens into TTS-ready sentences
                         │
                         ▼
                   TTS pipeline        ← Fish, ElevenLabs, Piper, Kokoro, any speech gateway
                         │
                         │  PCM16 audio
                         ▼
                    AudioPlayer ──────► VTube Studio (viseme lipsync + blink + body + expressions)
                         │                    ▲
                         │              Mood Engine (arousal/valence/focus)
                         │
                         ▼
                  speakers / OBS / virtual cable
```

**One pipeline. One conversation history. No competing buffers.** This is the defining design
choice. Early prototypes had parallel generation paths and went insane — the streamer would repeat
itself, contradict itself, and lose all continuity. Everything goes through one orchestrator, one
set of messages, one output path.

**Intent priority:** highlight chat (barge in) → vision event → ordinary chat → monologue.
Higher-priority intents preempt lower ones. **Donations ride the highlight path** — they preempt
vision/monologue exactly like bits/superchats, never waiting, never overlapping speech.

**Continuity machinery:** rolling summary of older turns, open-thread tracker, phrase cooldown,
theme tracker.

### Project structure

```
wallie-v2/
├── wallie.py              # entrypoint
├── config.py              # pydantic models, profile management
├── core/
│   ├── orchestrator.py    # the single pipeline
│   ├── persona.py         # prompt engineering
│   ├── context.py         # conversation history + rolling summary
│   ├── attention.py       # vision reaction decisions
│   ├── mood.py            # slow-evolving emotional state
│   └── memory_store.py    # cross-session persistent memory
├── llm/                   # LLM provider adapters (incl. generic OpenAI-compatible)
├── tts/
│   ├── voice_lab.py       # ← fork: voice presets + provider-side cloning
│   └── ...                # TTS adapters (fish, elevenlabs, piper, kokoro, any gateway)
├── audio/                 # sounddevice player with alignment safety
├── vision/                # screen capture + change detection + activity classification
├── hearing/               # system-audio loopback + local/remote STT + music analysis
├── captions/              # TTS → SSE caption bridge (self-clearing OBS overlay)
├── chat/                  # YouTube, Twitch, Kick monitors
├── donations/             # LivePix + Streamlabs → normalized DonationEvent → orchestrator queue
├── avatar/                # VTube Studio WebSocket client
├── dashboard/             # FastAPI + Alpine.js (no build step)
├── profiles/              # saved persona profiles (YAML) + per-profile voice libraries
└── scripts/               # setup utilities
```

---

## Provider compatibility

### LLM

| Provider | Streaming | Vision | Notes |
|---|---|---|---|
| **Groq** | ✅ | ✅ (Llama-4) | Fastest inference. Free tier. |
| **OpenAI** | ✅ | ✅ (GPT-4o) | Strong vision. Premium pricing. |
| **OpenRouter** | ✅ | ✅ (varies) | One key, many models. |
| **Anthropic** | ✅ | ✅ (Claude 4) | Best character/IP recognition. |
| **Gemini** | ✅ | ✅ (`gemini-3.8-flash`) | Free tier. The 2.5 family is restricted to existing users — new keys get a 404 pointing at 3.8. |
| **Ollama** | ✅ | ✅ (llava, etc.) | Fully local. No API key. |

### TTS

| Provider | Streaming | Cloning | Cost | Notes |
|---|---|---|---|---|
| **Fish Audio** | ✅ | ✅ **via Voice Lab** | ~$15/M chars | Creates a private model from your clips |
| **ElevenLabs** | ✅ | ✅ **via Voice Lab** | ~$30/M chars | Instant voice cloning |
| **Kokoro** | ✅ (local) | ❌ | $0 | 8 languages incl. **pt-BR**; pick voice + language in Voice — saved voices land in the Voice Lab library |
| **Piper** | ✅ (local) | ❌ | $0 | Fixed voices, fastest local option. One-click runtime install, the whole official catalogue downloadable from the dashboard, plus speed/expressiveness/pacing sliders |
| **OpenAI-Compatible (speech)** | ✅ | ❌ | any gateway | Depends on the gateway |

---

## Security

- API keys are stored in `.env` with restricted permissions (`chmod 600` on POSIX)
- The dashboard never exposes raw keys — only masked previews (`sk-•••xyz`)
- The dashboard binds to `127.0.0.1` only — not accessible from the network (set `DASHBOARD_PIN` to put a PIN in front of it)
- Atomic writes prevent `.env` corruption
- Provider error messages are scrubbed of key-shaped strings before display
- Allowed env variables are hard-coded — the UI cannot write arbitrary keys

**Voice cloning uploads audio to the provider you choose.** ElevenLabs/Fish Audio receive your
clips and create the voice under your account; their terms and privacy policies apply to that
audio. Reference clips are *not* stored by Wallie — only the returned voice id lands in
`profiles/<profile>.voices.json`.

**A voice backup exported from the dashboard carries no secrets.** The `.json` file holds names,
provider/voice ids and tuning knobs only — no API keys and no audio — so it is safe to version or
hand to someone else.

**Do not expose the dashboard to a public network without a reverse proxy with authentication.**

---

## Troubleshooting

<details>
<summary><strong>Vision test fails with 404 / "model is no longer available"</strong></summary>

The provider retired that model. Put a current one — the API's message usually names it, e.g.
`gemini-3.8-flash` — into the Vision test strip's **provider/model** boxes (this is *not* saved),
confirm it answers, then set it for real in **Engine → Model** (or in the dedicated Vision block).
See [Testing vision with a different model](#testing-vision-with-a-different-model).
</details>

<details>
<summary><strong>Audio becomes static</strong></summary>

Hit the **reset audio** button in the dashboard top bar. If it recurs, check the logs — usually a TTS provider returning non-PCM data (detected and aborted automatically in most cases).
</details>

<details>
<summary><strong>Kokoro fails to load a language (e.g. pt-BR)</strong></summary>

Non-English Kokoro sets need a recent `kokoro` (≥ 0.9.4). Japanese/Mandarin may also need
`pip install "misaki[ja]"` / `"misaki[zh]"` and espeak-ng. The server log shows the exact missing
package. Voice ids must belong to the selected language (`pf_dora` for pt-BR, not `af_heart`) —
switching the language in the dashboard swaps the voice automatically.
</details>

<details>
<summary><strong>Voice clone rejected by the provider</strong></summary>

The provider's own message is shown in the Voice Lab, usually about sample quality: too short,
too noisy, music in the background, or too many files. Re-record 30–60s of clean speech (the mic
button writes a WAV for you) and retry. "No API key" means the key for that provider is missing on
the **API Keys** page.
</details>

<details>
<summary><strong>AI describes generic UI ("I see a YouTube page")</strong></summary>

The SKIP escape hatch depends on the model. Smaller models are worse at following it. Upgrade to
Claude Sonnet or GPT-4o for better vision, or set commentary density to `sparse`.
</details>

<details>
<summary><strong>AI keeps asking questions to chat</strong></summary>

The question throttle is automatic (it forces statements after a question-ending). If it persists, raise frequency penalty to 0.5–0.7 in Engine settings, or turn on the engagement gate in the Chat section.
</details>

<details>
<summary><strong>Avatar mouth doesn't move</strong></summary>

Check that the `MouthOpen` parameter name matches your model (some use `ParamMouthOpen` or `MouthOpenY`). Override it in Avatar → Parameter mapping.
</details>

<details>
<summary><strong>Avatar mouth moves but the shape looks wrong</strong></summary>

Viseme lip sync drives `ParamMouthForm` for mouth shape (wide ↔ round). If your model uses a different parameter name, update it in Avatar → Parameters → Mouth form. If your model has no mouth-form parameter, disable "Spectral mouth shape" and lipsync falls back to volume-only.
</details>

<details>
<summary><strong>Avatar doesn't blink / expressions don't fire</strong></summary>

Blink needs `EyeOpenLeft` / `EyeOpenRight` (some models use `ParamEyeLOpen` / `ParamEyeROpen`); the
parameter names are overridable. Expression slots must match VTS hotkey names or IDs — use
"Discover hotkeys from VTS" to see what's available, then map or set `expr_happy`, `expr_sad`, etc.
</details>

<details>
<summary><strong>TTS returns 401</strong></summary>

Verify the key on the **API Keys** page — the masked preview should match your provider dashboard. Hit **test** to confirm.
</details>

<details>
<summary><strong>Copying / pulling / importing a voice asks about a name that already exists</strong></summary>

That is the overwrite guard, not an error. The destination profile already has a saved voice with
that name **and different settings**, so the dashboard shows the diff (`name: current → incoming`)
and asks before replacing it. Accept to overwrite, cancel to leave everything as it was — nothing is
written until you answer. Mirroring never asks: its diff lists the clashing voices and you choose
which side wins (or turn the difference into a no-op with *leave them alone*).
</details>

<details>
<summary><strong>The Piper catalogue won't load, or a voice download fails</strong></summary>

The catalogue is fetched live from `huggingface.co/rhasspy/piper-voices` (a few hundred KB, cached
in-process for an hour), so it needs internet the first time — **refresh** retries. A failed
download keeps its last progress and its message in the log under the box; the usual causes are no
network, a proxy, or a wrong name. Names are `lang_REGION-voice-quality`
(`pt_BR-faber-medium`), or just pick from the catalogue instead of typing.
</details>

---

## Roadmap

Fork-specific:

- **More cloning backends** — add any provider with a create-a-voice API behind the same Voice Lab interface
- **Voice blending for local engines** — average Kokoro voice embeddings to make in-between local voices (no cloud, no cloning)
- **Blind A/B** — hide which clip is which until you pick, so the choice stops being biased by the label
- **Dashboard i18n** — start with pt-BR, since the fork's audience is largely Brazilian

Inherited upstream roadmap, still open here:

- **Streaming avatar backend (HeyGen)** — realistic avatars as an alternative to Live2D
- **Docker image** — `docker run wallie` with a volume for config
- **Cost meter** — running spend tally in the live drawer
- **OBS WebSocket integration** — scene switching tied to stream events
- **More donation platforms** — Pix/Mercado Pago, YouTube Super Chat via Streamlabs, Kick

If any of these matter to you, open an issue — or better, a PR.

---

## Contributing

Issues and PRs are welcome here, but note the fork relationship: upstream-owned code lives in
[Alradyin/wallie-V2](https://github.com/Alradyin/wallie-V2). If a change is about the streamer
engine rather than the fork's dashboard work, please upstream it there too — and consider sending
it there *first*, so this fork stays a thin layer. Fixes that are about the Voice Lab, the vision
test, the theme picker or the Kokoro picker belong here.

Ground rules inherited from upstream:

- **Code is English-only.** Comments, logs, variables — all English. UI can be localized.
- **Single pipeline, single history.** Don't add parallel generation paths. That road has been walked.
- **Lazy imports for optional deps.** Groq users shouldn't need `anthropic` installed.
- Before submitting: run `python -m pytest tests/` (covers the config, provider-resolution and dashboard-route layers) and `node --check dashboard/static/app.js` — the dashboard has no build step, so that syntax check is the whole "compile".

---

## License

MIT. See [LICENSE](LICENSE). This fork keeps the upstream license and attribution.

### Bundled / third-party software (Minecraft Play)

Wallie's own code is MIT. The Minecraft Play mode uses several third-party mods, which are
installed (and, for convenience, bundled under `dist/mods/`) and remain under their own licenses:

| Mod | License | Source |
|---|---|---|
| **Baritone** (pathfinding/mining) | LGPL-3.0 | https://github.com/cabaletta/baritone |
| **Meteor Client** (loads Baritone) | GPL-3.0 | https://github.com/MeteorDevelopment/meteor-client |
| **Fabric API** | Apache-2.0 | https://github.com/FabricMC/fabric |
| **Fabric Loader** | Apache-2.0 | https://github.com/FabricMC/fabric-loader |
| smooth-camera mod (Wallie's own) | MIT | this repo / upstream |

These mods are used unmodified and distributed as separate jars (mere aggregation); they are the
property of their respective authors.

---

<p align="center">
  <strong>Wallie is built for people who want their AI streamer to feel like a real show, not a tech demo.</strong>
  <br />
  Upstream engine by <a href="https://github.com/Alradyin">Alradyin</a>. Fork work by <a href="https://github.com/isyatodev">isyatodev</a>.
</p>
