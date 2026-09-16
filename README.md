[![Featured on Product Hunt](https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1158674&theme=light)](https://www.producthunt.com/products/wallie-v2)

https://github.com/user-attachments/assets/bb4404e5-9138-4288-a428-345486b86f7e

<p align="center">
  <h1 align="center">Wallie</h1>
  <p align="center"><strong>The open-source AI streamer that sees, hears &amp; reacts — and actually feels alive.</strong></p>
</p>

<p align="center">
  <a href="https://github.com/Alradyin/wallie-V2/archive/refs/heads/main.zip"><img alt="Download" src="https://img.shields.io/badge/Download_ZIP-2ea44f?style=for-the-badge&logo=github&logoColor=white" /></a>
  &nbsp;
  <a href="https://spontaneous-dodol-cabb7d.netlify.app/"><img alt="Website" src="https://img.shields.io/badge/Website-6c8cff?style=for-the-badge" /></a>
  &nbsp;
  <a href="https://github.com/Alradyin/wallie-V2/blob/main/docs/guide.md"><img alt="Docs" src="https://img.shields.io/badge/Docs-555?style=for-the-badge" /></a>
</p>

<p align="center">
  <a href="https://github.com/Alradyin/wallie-V2/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Alradyin/wallie-V2?style=social" /></a>
  &nbsp;&nbsp;
  <img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg" />
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-brightgreen.svg" />
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg" />
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &nbsp;·&nbsp;
  <a href="#features">Features</a> &nbsp;·&nbsp;
  <a href="#the-dashboard">Dashboard</a> &nbsp;·&nbsp;
  <a href="#architecture">Architecture</a> &nbsp;·&nbsp;
  <a href="#roadmap">Roadmap</a>
</p>

> ⭐ **If you think an AI that watches _and hears_ your screen should exist — give it a star.** Two seconds, and it genuinely helps a solo dev keep shipping.

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

## 🎮 New in v2.0 — Wallie can now **PLAY**

Wallie doesn't just watch your screen anymore — **it picks up the game and plays it.** Right now that's Minecraft: it gathers, crafts, builds, fights, and explores entirely on its own, narrating every moment in character. Smooth, human-looking camera. No human at the keyboard, no script — a real playthrough with a personality.

https://www.youtube.com/shorts/NEfsmppOXJk

And here's the part that makes it believable: **the commentary is grounded in what Wallie is _actually_ doing** — its real inventory, health, and current goal — not guessed from a blurry frame. So it says *"heading down to mine diamonds"* because it is, and brings up *"that one time I dug straight into a lava pit"* because that's its run.

- **It actually plays** — a planning brain sets the goal; [Baritone](https://github.com/cabaletta/baritone) handles pathfinding & mining, and a custom Fabric mod handles crafting, combat, item pickup, and a buttery-smooth camera that looks human, not robotic.
- **Grounded commentary** — the streamer core stays in sync with the playing brain, so what it _says_ matches what it _does_.
- **One-click setup** — the dashboard installs Fabric + every mod for you (and backs up your current ones first).
- **Any persona, any model** — Play is just another mode. Vision, Hearing, and Play each toggle on/off independently.

> Flip on **Play** in the dashboard, pick a goal, and go — or jump straight in:
> ```bash
> python scripts/run_wallie_live.py
> ```

---

## Why another AI streamer?

Because the existing ones are toys.

They work for a 2-minute demo and fall apart on a real stream. Here's what actually goes wrong after 10 minutes:

| The problem | What usually happens | What Wallie does |
|---|---|---|
| **Repetition** | Says "that's interesting" every 30 seconds | Dedupe engine + phrase cooldown + rolling summary that tracks everything already said |
| **Short memory** | Forgets the topic from 5 minutes ago | Rolling summarizer compresses old turns into bullet notes, injected into every prompt |
| **Robotic vision** | "I can see a YouTube homepage with several videos" | First-person ownership + SKIP escape hatch — narrates nothing, reacts to what matters |
| **Question loops** | Every segment ends with "what do you think, chat?" | Question detector throttles after one; next prompt forces a statement ending |
| **No personality** | Generic helpful assistant voice | Full persona system: energy, humor style, catchphrases, backstory, opinions, taboo topics |
| **Choppy pacing** | 2 sentences → long pause → 2 sentences | Pipeline overlap + longer thought development eliminates dead air |
| **Broken threads** | Sets up a story, never delivers it | Open-thread tracker forces the next segment to pay off teases and answer its own questions |
| **Topic whiplash** | Jumps from AI to cooking to space with no bridge | Association-based topic drift with configurable style (rigid / natural / freeform) |

These aren't edge cases. They're the default behavior of most AI streamer projects. Wallie was built by hitting every one of these problems and refusing to ship until they were fixed.

---

## Features

### Bring your own everything

Six LLM providers. Three TTS engines. Three chat platforms. Mix and match per profile.

| LLM | TTS | Chat | Donations | Avatar |
|---|---|---|---|---|
| OpenAI | Fish Audio | Twitch | LivePix (webhook) | VTube Studio (Live2D) |
| Anthropic (Claude) | ElevenLabs | YouTube | Streamlabs (Socket API) | — |
| Google (Gemini) | Piper (local, free) | Kick | — | — |
| Groq | Kokoro (local, free) | — | — | — |
| OpenRouter | **OpenAI-Compatible (speech)** | — | — | — |
| Ollama (local, free) | — | — | — | — |
| **OpenAI-Compatible (any endpoint)** | — | — | — | — |

Beyond that, **Vision** and **Hearing (STT)** can each run on their own OpenAI-compatible
block — see [Dedicated vision & STT blocks](#dedicated-vision--stt-blocks) — and Wallie ships a
self-clearing **caption overlay** for OBS: [Captions](#captions-browser-source-overlay).

Swap providers without changing code. Run a fully offline stream with Ollama + Piper, or go premium with Claude + ElevenLabs.

### Full persona design

Not just a name and a system prompt. You design a *character*:

- **Identity** — name, handle, pronouns, age range, origin, archetype, backstory
- **Voice** — energy level (chill → unhinged), humor style (pick multiple: ironic, deadpan, absurd, observational, roast...), profanity level, formality
- **Flavor** — catchphrases, running gags, banned words, favorite topics, taboo topics
- **Opinions** — strong opinions toggle, admit uncertainty, break the fourth wall
- **Extra notes** — free text for anything the structured fields miss

Save multiple personas. Switch between them with a dropdown.

### Hours-long sessions without decay

The single biggest technical challenge. Most AI chatbots break down after 20 minutes because the context window fills up.

Wallie's approach:
- **Rolling summarizer** — every ~14 segments, a background LLM call compresses older history into tight bullet notes
- **Session notes** — that compressed memory is injected into every system prompt, so the streamer knows what it already said
- **Cross-session memory** — key facts and viewer interactions persist across streams
- **Per-person memories** — with Speaker ID on, facts learned from an enrolled voice (e.g. the owner) are bound to that person ("about") and resurface naturally in the prompt the next time they talk — remembered history, not recited notes. Works for manually added memories too (optional "about person" field in the Memory section); auto-consolidation never merges memories of different people.
- **Auto-consolidation** — when the long-term store passes the configured threshold (default 300), the memory model merges groups of related old entries into single general summaries, so the store keeps growing in *depth* instead of only in *size*. Summaries inherit the group's importance (sum of hits) and original age, are excluded from future merge passes, and never replace high-traffic facts the AI keeps re-learning.
- **Dedupe engine** — paraphrase-aware similarity check (bigram + trigram Jaccard) catches the model repeating itself in different words

### Organic pacing

Real streamers don't talk at a constant rate. They get excited, they pause to think, they go on tangents, they settle into a flow.

- **Mood engine** — arousal, valence, focus, and talkativity drift slowly over the stream. High arousal = faster, warmer output. Low talkativity = silence beats. Mood state feeds into the avatar for reactive animation.
- **Attention engine** — vision events are filtered through a probabilistic decision layer. Not every screen change gets a deep reaction. Some get a glance. Some get ignored. Some trigger a personal tangent.
- **Silence beats** — the streamer holds natural pauses when the mood says to. Dead air handled by design, not by accident.
- **Pipeline overlap** — next segment starts generating while current audio is still playing. No awkward 3-second gaps between thoughts.

### Vision that isn't narration


https://github.com/user-attachments/assets/d824cffa-bd75-4002-a58f-1c6af33ac590


Screen reactions are the hardest part to get right. Wallie's approach:

- **First-person ownership** — Wallie owns whatever is on screen. Gaming: "I just got bodied by that boss", not "the character is fighting a boss". Browsing: "let me pull this up", not "the user is browsing". Never third-person, never narration.
- **SKIP escape hatch** — if there's nothing specific to name, the model outputs `SKIP` and stays quiet. No more narrating generic UIs.
- **Activity adaptation** — detects scrolling, typing, app-switching, video playback and adjusts reactions accordingly. Typing → ignore. App switch → react. Rapid browsing → wait until user settles. Each activity type gets context-aware prompting.
- **Attention engine** — not every screen change gets the same treatment. DEEP reactions (22%), quick GLANCEs (28%), personal TANGENTs (5%), deliberate IGNOREs (27%), and SILENCE beats (18%). Streak fatigue prevents reacting the same way twice in a row.
- **Scene memory** — remembers what it last said about the current screen. Dedupe threshold at 0.65 with per-sentence comparison catches paraphrased repetition.

### Hearing — it reacts to what it hears, not just what it sees

Wallie captures your system audio (WASAPI loopback) and reacts to it live, fused with vision into a single reaction — sight and sound, one voice. Just like you organize your senses, not two narrators talking over each other.

- **Speech & lyrics** — local speech-to-text (faster-whisper) transcribes videos, voice chat, and song lyrics. Auto-uses your GPU if available, clean CPU fallback if not.
- **Real music understanding** — pure-numpy DSP, no extra ML weight. It reads **major/minor key → mood**, tempo + beat strength, instrumentation texture (bass-heavy / acoustic / airy / lo-fi), and production quality (crisp / muddy / harsh). A sad song reads as melancholy; a beat drop reads as energy. It'll call a track a banger or roast a muddy mix.
- **Multimodal fusion** — what it hears is fused with what it sees into one coherent reaction, never two competing ones.
- **No self-echo** — content-based guard so Wallie never reacts to its own TTS bleeding back through the loopback.
- **Pure-hearing mode** — run it with vision off and it reacts to audio the way it reacts to a screen: stays quiet, listens, then reacts to what's playing.

Enable it in the dashboard's Hearing section. Needs two extra deps: `pip install soundcard faster-whisper`.

### Wallie plays games — not just watches them

**Play mode** hands Wallie the controls. First game: Minecraft, survival, live and unscripted.

- **Reliable autonomy** — a planning brain picks high-level goals; a deterministic skill library (crafting, smelting, full tool/armour sets, mining to ore depths, hunting for food & wool, beds, combat with weapon selection, dropped-item pickup) does the actual work — so even a small/free model plays competently instead of flailing.
- **Human-like camera** — a custom Fabric mod eases every turn at full FPS. No snapping, no teleport-cam — it reads like a person playing.
- **Grounded commentary** — reactions come from the agent's real game state (inventory, health, threats, current goal), fused through the same single pipeline as vision and hearing. No screen-guessing, no hallucinated "desert temples".
- **One-click install** — Fabric + all mods straight from the dashboard's Play section. Vision/Hearing/Play toggle independently, on any persona.

### Live2D avatar with emotion

Not just a mouth that opens and closes. Six animation layers run simultaneously over a single WebSocket, so the avatar feels like a person — not a puppet.

- **Viseme lip sync** — spectral analysis of PCM audio estimates mouth shape in real-time. Front vowels (A/E/I) spread the mouth wide, back vowels (O/U) round it. Combined with RMS amplitude for jaw openness, attack/release envelope, noise floor gating, and a speaking smile baseline. The result: the avatar's mouth actually forms different shapes per sound, not just open/close.
- **Blink** — periodic natural eye blinks (~3.8s interval) with random variation and occasional double-blinks. Blink rate adapts to mood: sleepy = more blinks, alert = fewer
- **Body motion** — slow torso sway on BodyAngleX/Y/Z, lower amplitude and longer period than head movement, so the avatar breathes even when silent
- **Idle motion** — head sway and eye darts when not speaking. Eye-dart frequency increases when the streamer's mood focus drops (scattered attention)
- **11 expression slots** — happy, surprised, laughing, angry, sad, thinking, smug, eyeroll, confused, hype, deadpan
- **Keyword-driven emotions** — sentence content automatically triggers matching expressions (21 regex patterns, first match wins)
- **Expression auto-mapping** — on connect, Wallie discovers VTS hotkeys and matches them to empty expression slots by name. No manual configuration needed for models with standard hotkey names
- **Mood-reactive avatar** — the MoodEngine's arousal/valence/focus feed directly into the avatar. Arousal scales idle and body sway amplitude (0.5×–1.6×). Valence shifts brow position and adds a resting smile when the streamer is in a good mood. Focus modulates eye-dart frequency
- **Event cues** — super chat → hype face, screen change → surprised look + upward glance, thinking pause → thinking expression, chat message → head turn toward "the chat"

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
│ Avatar   │   editor with live controls)  │ │ Open threads   │ │
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

## Quick start

### Just double-click `start.bat`

No Python knowledge, no terminal, no setup steps. Download → double-click → done.

1. **[Download the ZIP](https://github.com/Alradyin/wallie-V2/archive/refs/heads/main.zip)** and unzip it (or `git clone https://github.com/Alradyin/wallie-V2.git`).
2. **Double-click `start.bat`.** First run installs everything it needs, then the dashboard opens at `http://127.0.0.1:8765`.
3. **Paste your API key** in the dashboard → pick a model → hit **Start**.

That's the whole setup. Everything else is configured in the browser.

> **macOS / Linux:** run `./start.sh` instead (`chmod +x start.sh` once).

### Pick your budget

| Path | LLM | TTS | Cost/hour | Quality |
|---|---|---|---|---|
| **Free** | Gemini 2.5 Flash | Piper (local) | $0 | Good for testing |
| **Cheap & fast** | Groq (Llama 3.3 70B) | Fish Audio | ~$1.50 | Best balance |
| **Premium** | Claude Sonnet | ElevenLabs | ~$6.50 | Best quality + vision |

All configured from the dashboard. Add your API keys → pick provider → pick model → Start.

<details>
<summary><strong>Piper (free TTS) setup</strong></summary>

Piper runs locally — no API key needed. One extra terminal command to download a voice:

```bash
# In your wallie directory, with venv active:
pip install piper-tts onnxruntime
python scripts/download_piper_voice.py en_US-amy-medium
```

Then in the dashboard: Voice → provider: `piper`, path: `voices/en_US-amy-medium.onnx`.
</details>

<details>
<summary><strong>Vision setup (screen reactions)</strong></summary>

Requires a vision-capable LLM. In the dashboard:

1. **Engine** → toggle "Vision capable" ON
2. Use a vision model: `claude-sonnet-4-5`, `gpt-4o`, `gemini-2.5-pro`, or `llama-4-scout` on Groq
3. **Vision** section → toggle ON, adjust frame interval and sensitivity

The SKIP escape hatch is always active — no configuration needed.
</details>

<details>
<summary><strong>Chat platform setup</strong></summary>

**Twitch:** Add OAuth token from [twitchtokengenerator.com](https://twitchtokengenerator.com) (chat:read scope). Or leave empty for anonymous read-only.

**YouTube:** Drop `client_secret.json` in `scripts/`. First run opens browser for Google OAuth consent.

**Kick:** Just enter the channel slug. No auth needed (public Pusher WebSocket).
</details>

<details>
<summary><strong>VTube Studio avatar</strong></summary>

1. Run VTube Studio with API enabled (Settings → API → Enable)
2. In the dashboard: Avatar → toggle ON
3. First connect triggers a plugin approval popup in VTS — click Allow
4. Expression slots are **auto-mapped** from your model's hotkeys on connect. Override manually if needed
5. Adjust lipsync gain/ceiling for your voice. Viseme lip sync (spectral mouth shape) is enabled by default — disable it in Avatar config if your model doesn't have a `ParamMouthForm` parameter
6. Blink, body motion, and mood-reactive behaviour are enabled by default — tweak or disable per-feature in config

Works with any Live2D model that has standard parameters (`MouthOpen`, `EyeOpenLeft/Right`, `FaceAngleX/Y/Z`, `BodyAngleX/Y/Z`).
</details>

<details>
<summary><strong>OBS output routing</strong></summary>

Wallie outputs audio through your system's default audio device. To route it into OBS:

**Windows:** Install [VB-CABLE](https://vb-audio.com/Cable/). Set CABLE Input as default playback. In OBS: Audio Input Capture → CABLE Output.

**macOS:** Use [BlackHole](https://existential.audio/blackhole/) the same way.

**Linux:** PipeWire / PulseAudio loopback (`pw-loopback`).
</details>

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

The Base URL must include the version path (`…/v1`). Streaming, temperature/top-p, max tokens, penalties, timeout and retries all work as with the named providers. Vision works the same way as everywhere else: tick **Model supports vision** only if your model accepts images — Wallie refuses to attach screenshots to a text-only model and disables the vision loop instead of shipping it a broken request.

Only the OpenAI-compatible surface is used — no provider-specific extensions are sent (prompt-cache breakpoints, for example, remain an OpenRouter-only feature).

### Dedicated vision & STT blocks

Some hosts split models across different providers — a Qwen-VL endpoint for vision while the
brain stays on Claude, a hosted Whisper for hearing while the brain runs locally. Each subsystem
therefore has its own OpenAI-compatible block:

| Subsystem | Config (dashboard) | API key (.env) | Endpoint shape |
|---|---|---|---|
| **Vision** | Engine → Vision model source → *Separate block*, then Vision → Vision model | `OPENAI_COMPATIBLE_VISION_API_KEY` | `POST {base}/chat/completions` (images in, text out) |
| **TTS** | Voice → Provider → *OpenAI-Compatible* | `OPENAI_COMPATIBLE_TTS_API_KEY` | `POST {base}/audio/speech` (PCM16 out) |
| **STT (Hearing)** | Hearing → STT engine → *OpenAI-Compatible (remote)* | `OPENAI_COMPATIBLE_STT_API_KEY` | `POST {base}/audio/transcriptions` |

Vision routing: when the dedicated block is configured, every vision-intent segment runs on it;
monologue/chat/outro stay on the main LLM. If the dedicated block is missing or fails to build,
vision falls back to the main LLM with a logged warning — the stream never dies over config.

Remote STT swaps the local Whisper model for an HTTP call — same hallucination guards, VAD path
and self-echo filter, without the VRAM cost. The local model stays selected whenever the engine
is left on **Local Whisper**.

## Captions — browser source overlay

Wallie's speech can be rendered as live captions on stream via a self-contained web page:

1. **Captions** → toggle **Enable caption overlay** → Save.
2. Start the orchestrator. The dashboard shows the overlay URL (default
   `http://127.0.0.1:8765/captions`).
3. In OBS: Sources → **+** → **Browser** → paste the URL (width 1280, height 200 works well).

How it behaves:

- Sentences stream in **as they are TTS'd**, sentence-by-sentence, over Server-Sent Events.
- **The caption box is emptied when the TTS segment ends** (after an optional hold, 0s default)
  — an old message never lingers on screen. This is the whole point: captions track the voice,
  then disappear.
- Style knobs live in the dashboard: font size, box opacity, max lines, lowercase, persona-name
  prefix. The page is transparent, so it composites cleanly over any scene.
- The **Push to overlay** test button exercises the real bridge — you should see the line appear
  and then vanish ~2.5s later.

## Donations

Two platforms are supported out of the box, and both feed the **same** pipeline as chat — no second AI path:

```
LivePix (webhook) ──┐
                    ├──► DonationEvent (normalized) ──► orchestrator queue
Streamlabs (socket)─┘         (as highlight chat)            │
                                                        LLM → TTS → avatar
```

A donation replies in Wallie's voice through the existing TTS, thanks the donor by name, and reacts to their message. The LLM sees donations as their own turn type (`[DONATION — donor — amount]` + source + message), never as indistinguishable chat text. The streamer's own messages are tagged `[STREAMER]` (detected from the Twitch broadcaster badge) and viewer messages `[VIEWER]`.

**LivePix** ([docs](https://docs.livepix.gg))
1. Create an app in your LivePix account settings → get `client_id` / `client_secret` → put them in `.env` (`LIVEPIX_CLIENT_ID`, `LIVEPIX_CLIENT_SECRET`) or API Keys.
2. Optionally set `LIVEPIX_USER_ID` — webhook payloads for other accounts are then rejected.
3. Enable **LivePix** in dashboard → Donations. The webhook is mounted on the dashboard app at `livepix_webhook_path` (default `/webhooks/livepix`). The dashboard must be reachable from the internet: run with `DASHBOARD_HOST=0.0.0.0` behind a tunnel/reverse proxy, then register `http://<host>:<port>/webhooks/livepix` as a LivePix webhook (or let `LIVEPIX_ACCESS_TOKEN` + enrichment client create it via the API).
4. Flow: webhook validates the payload → dedupes by the real LivePix resource id → answers `HTTP 200` immediately → enrichment (`GET /v2/messages/{id}`) and queuing happen in the background. The webhook never waits for the LLM/TTS. LivePix's docs document no signature header; validation is structural + the optional userId allowlist.

**Streamlabs** ([docs](https://dev.streamlabs.com/docs/socket-api))
1. Easiest: Streamlabs Dashboard → Settings → API Settings → API Tokens → copy the **Socket API Token** → `STREAMLABS_SOCKET_TOKEN` in `.env`.
2. Or authorize OAuth with `donations.read` + `socket.token` scopes and set `STREAMLABS_ACCESS_TOKEN`; the socket token is fetched from `/socket/token` automatically.
3. Enable **Streamlabs** in dashboard → Donations. Wallie connects to `sockets.streamlabs.com`, auto-reconnects with jittered exponential backoff, keeps exactly one live connection, and cleans up on shutdown.
4. Only `type === "donation"` events are processed (the `message` field is an array and every item is normalized). Follows/subs/bits/raids are ignored by design. Dedupe uses Streamlabs' own `_id` / `event_id`.

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
                   TTS pipeline        ← 3 providers, parallel pre-fire
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

**One pipeline. One conversation history. No competing buffers.** This is the defining design choice. Early prototypes had parallel generation paths and went insane — the streamer would repeat itself, contradict itself, and lose all continuity. Everything goes through one orchestrator, one set of messages, one output path.

**Intent priority:** highlight chat (barge in) → vision event → ordinary chat → monologue. Higher-priority intents preempt lower ones. **Donations ride the highlight path** — they preempt vision/monologue exactly like bits/superchats, never waiting, never overlapping speech.

**Continuity machinery:**
- Rolling summary of older turns → injected into system prompt
- Open-thread tracker → forces the next segment to resolve dangling questions and teases
- Phrase cooldown → prevents catchphrase spam
- Theme tracker → blocks repeated angles

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
├── tts/                   # TTS provider adapters (incl. OpenAI-compatible speech)
├── audio/                 # sounddevice player with alignment safety
├── vision/                # screen capture + change detection + activity classification
├── hearing/               # system-audio loopback + local/remote STT + music analysis
├── captions/              # TTS → SSE caption bridge (self-clearing OBS overlay)
├── chat/                  # YouTube, Twitch, Kick monitors
├── donations/             # LivePix + Streamlabs → normalized DonationEvent → orchestrator queue
├── avatar/                # VTube Studio WebSocket client
├── dashboard/             # FastAPI + Alpine.js (no build step)
├── profiles/              # saved persona profiles (YAML)
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
| **Gemini** | ✅ | ✅ (2.5 family) | Free tier (50 req/min). |
| **Ollama** | ✅ | ✅ (llava, etc.) | Fully local. No API key. |

### TTS

| Provider | Streaming | Cloning | Cost |
|---|---|---|---|
| **Fish Audio** | ✅ | ✅ | ~$15/M chars |
| **ElevenLabs** | ✅ | ✅ | ~$30/M chars |
| **Piper** | ✅ (local) | ❌ | $0 |
| **OpenAI-Compatible (speech)** | ✅ | ❌ | any speech gateway |

---

## Security

- API keys stored in `.env` with restricted permissions (`chmod 600` on POSIX)
- Dashboard never exposes raw keys — only masked previews (`sk-•••xyz`)
- Dashboard binds to `127.0.0.1` only — not accessible from the network
- Atomic writes prevent `.env` corruption
- Provider error messages scrubbed of key-shaped strings before display
- Allowed env variables are hard-coded — the UI cannot write arbitrary keys

**Do not expose the dashboard to a public network without a reverse proxy with authentication.**

---

## Troubleshooting

<details>
<summary><strong>Audio becomes static</strong></summary>

Hit the **reset audio** button in the dashboard top bar. If it recurs, check logs — usually a TTS provider returning non-PCM data (detected and aborted automatically in most cases).
</details>

<details>
<summary><strong>AI describes generic UI ("I see a YouTube page")</strong></summary>

The SKIP escape hatch depends on the model. Smaller models (Llama-4 Scout, Gemini Flash) are worse at following it. Upgrade to Claude Sonnet or GPT-4o for better vision, or set commentary density to `sparse`.
</details>

<details>
<summary><strong>AI keeps asking questions to chat</strong></summary>

The question throttle is automatic (forces statements after one question-ending). If it persists, raise frequency penalty to 0.5–0.7 in Engine settings.
</details>

<details>
<summary><strong>Avatar mouth doesn't move</strong></summary>

Check that the `MouthOpen` parameter name matches your model (some use `ParamMouthOpen` or `MouthOpenY`). Override in Avatar → Parameter mapping.
</details>

<details>
<summary><strong>Avatar mouth moves but shape looks wrong</strong></summary>

Viseme lip sync drives `ParamMouthForm` for mouth shape (wide ↔ round). If your model uses a different parameter name, update it in Avatar → Parameters → Mouth form. If your model doesn't support mouth form at all, disable "Spectral mouth shape" in the Viseme section — lipsync will fall back to volume-only (open/close).
</details>

<details>
<summary><strong>Avatar doesn't blink</strong></summary>

Check that your model has `EyeOpenLeft` / `EyeOpenRight` parameters (some use `ParamEyeLOpen` / `ParamEyeROpen`). Override the parameter names in Avatar config. Blink can be disabled with `enable_blink: false`.
</details>

<details>
<summary><strong>Expressions don't fire</strong></summary>

Expression slots need to match VTS hotkey names or IDs. If auto-mapping didn't find your hotkeys (naming mismatch), map them manually in the dashboard or set `expr_happy`, `expr_sad`, etc. in your profile. Use the dashboard's "Discover hotkeys from VTS" to see available names.
</details>

<details>
<summary><strong>TTS returns 401</strong></summary>

Verify the key in API Keys — the masked preview should match your provider dashboard. Hit "test" to confirm.
</details>

---

## Roadmap

What's coming next:

- **Streaming avatar backend (HeyGen)** — realistic-looking avatars as an alternative to Live2D
- **First-run wizard** — guided setup that walks you through provider choice → key entry → first stream
- **Docker image** — `docker run wallie` with a volume for config
- **Cost meter** — running spend tally in the live drawer
- **OBS WebSocket integration** — scene switching tied to stream events
- **Voice cloning UI** — upload reference audio, create a voice from the dashboard
- **More donation platforms** — Pix/Mercado Pago, YouTube Super Chat via Streamlabs, Kick — all plug into the same DonationEvent normalizer without touching the orchestrator

If any of these matter to you — open an issue, or better yet, a PR.

---

## Contributing

Issues and PRs welcome. Ground rules:

- **Code is English-only.** Comments, logs, variables — all English. UI can be localized.
- **Single pipeline, single history.** Don't add parallel generation paths. That road has been walked.
- **Lazy imports for optional deps.** Groq users shouldn't need `anthropic` installed.
- Run `python scripts/_import_check.py` before submitting.

---

## License

MIT. See [LICENSE](LICENSE).

### Bundled / third-party software (Minecraft Play)

Wallie's own code is MIT. The Minecraft Play mode uses several third-party mods, which are
installed (and, for convenience, bundled under `dist/mods/`) and remain under their own licenses:

| Mod | License | Source |
|---|---|---|
| **Baritone** (pathfinding/mining) | LGPL-3.0 | https://github.com/cabaletta/baritone |
| **Meteor Client** (loads Baritone) | GPL-3.0 | https://github.com/MeteorDevelopment/meteor-client |
| **Fabric API** | Apache-2.0 | https://github.com/FabricMC/fabric |
| **Fabric Loader** | Apache-2.0 | https://github.com/FabricMC/fabric-loader |
| smooth-camera mod (Wallie's own) | MIT | this repo |

These mods are used unmodified and distributed as separate jars (mere aggregation); they are the
property of their respective authors. Each is freely available at the source links above.

---

<p align="center">
  <strong>Wallie is built for people who want their AI streamer to feel like a real show, not a tech demo.</strong>
  <br />
  Star the repo. Try the $0 path. Tell us what broke.
</p>
