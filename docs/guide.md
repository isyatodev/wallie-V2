# Wallie — Complete Guide

Everything you need to set up, configure, and run Wallie. If something isn't covered here, open an issue.

---

## Table of Contents

1. [Installation](#installation)
2. [Getting API Keys](#getting-api-keys)
3. [Cost Breakdown — What Will This Cost Me?](#cost-breakdown)
4. [First Stream — Start to Finish](#first-stream)
5. [Choosing an LLM Provider](#choosing-an-llm-provider)
6. [Choosing a TTS Provider](#choosing-a-tts-provider)
7. [The Voice Lab](#the-voice-lab)
8. [Designing Your Persona](#designing-your-persona)
9. [Vision — Screen Reactions](#vision)
10. [Chat Integration](#chat-integration)
11. [Avatar — VTube Studio Setup](#avatar)
12. [OBS Setup — Getting Audio Into Your Stream](#obs-setup)
13. [Remote Control — Dashboard From Your Phone](#remote-control)
14. [Engine Tuning](#engine-tuning)
15. [Session Management](#session-management)
16. [Profiles](#profiles)
17. [Environment Variables Reference](#environment-variables)
18. [Troubleshooting](#troubleshooting)
19. [FAQ](#faq)

---

## Installation

### Windows — just double-click `start.bat`

[Download the ZIP](https://github.com/Alradyin/wallie-V2/archive/refs/heads/main.zip) and unzip it (or `git clone https://github.com/Alradyin/wallie-V2.git`), then **double-click `start.bat`**.

That's it. The first run installs everything it needs (Python included, if missing), sets up the free local voice (**Kokoro**, English US by default — pt-BR and 7 more languages included), then opens the dashboard at `http://127.0.0.1:8765`. Paste your API key there and hit Start.

Want to skip the voice download, or pre-load a different language? Set the setup variables below (`WALLIE_SKIP_KOKORO=1`, `WALLIE_KOKORO_LANG=p WALLIE_KOKORO_VOICE=pf_dora`) and double-click `install.bat`.

### macOS / Linux

```bash
git clone https://github.com/Alradyin/wallie-V2.git
cd wallie-V2
chmod +x start.sh
./start.sh
```

### Requirements

Nothing to install by hand — `start.bat` handles it. For reference:

- **Python 3.11+** (auto-installed on Windows if missing)
- No GPU required — everything runs on CPU + external APIs
- ~200 MB disk space (excluding Python/venv and the local voice)
- The free local voice the Windows setup pre-installs (Kokoro) adds ~500 MB — skip it with `WALLIE_SKIP_KOKORO=1` if that matters
- Internet connection (unless using Piper + Ollama for a fully offline setup)

### Manual Install (if the scripts don't work)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python wallie.py
```

---

## Getting API Keys

You need at least one LLM key and one TTS key (or use free options). Here's where to get each one:

### LLM Providers

| Provider | Where to Get the Key | Free Tier? |
|---|---|---|
| **Groq** | [console.groq.com](https://console.groq.com) → API Keys | Yes — generous free tier |
| **OpenAI** | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | No — pay-as-you-go |
| **Anthropic** | [console.anthropic.com](https://console.anthropic.com) → API Keys | $5 free credit on signup |
| **Google Gemini** | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Yes — 50 req/min free |
| **OpenRouter** | [openrouter.ai/keys](https://openrouter.ai/keys) | Some free models |
| **Ollama** | [ollama.com](https://ollama.com) — install locally | Fully free (local) |

**Steps for any cloud provider:**
1. Create an account on the provider's website
2. Navigate to API Keys / Settings
3. Generate a new key — copy it immediately (you won't see it again)
4. In the Wallie dashboard: go to **API Keys** → paste the key → click Save
5. Hit **Test** to verify it works

### TTS Providers

| Provider | Where to Get the Key | Free Tier? |
|---|---|---|
| **Fish Audio** | [fish.audio](https://fish.audio) → Settings → API Keys | Small free credit |
| **ElevenLabs** | [elevenlabs.io](https://elevenlabs.io) → Profile → API Key | 10k chars/month free |
| **Piper** | No key needed — runs locally | Fully free |

**For Fish Audio or ElevenLabs:**
1. Sign up → get your API key
2. Browse their voice library and find a voice you like
3. Copy the **Voice ID** (usually a long string or UUID)
4. In Wallie dashboard: **Voice** → set provider → paste voice ID

**For Piper (free, local):**
```bash
# In your wallie directory, with venv active:
pip install piper-tts onnxruntime
python scripts/download_piper_voice.py en_US-amy-medium
```
Then in the dashboard: **Voice** → provider: `piper` → path: `voices/en_US-amy-medium.onnx`

You can also skip the terminal entirely: on the **Voice** page with `piper` selected, **⬇ One-click install** installs the runtime, and the **Download a voice** box → **browse catalogue** lists every published voice — search by name, filter by language, see speakers/quality/size — and downloads the one you pick with a live progress bar. Files land in `voices/` either way; the CLI stays as the headless alternative, and Piper's own speed/expressiveness/pacing sliders are on the same page.

Available voices: [rhasspy.github.io/piper-samples](https://rhasspy.github.io/piper-samples/)

---

## Cost Breakdown

Real numbers based on a 1-hour stream with moderate activity.

### The $0 Path

| Component | Choice | Cost |
|---|---|---|
| LLM | Gemini 2.5 Flash (free tier) | $0 |
| TTS | Piper (local) | $0 |
| **Total** | | **$0** |

Limitations: Piper voices sound more robotic. Gemini free tier has rate limits (50 req/min — usually fine for streaming). Vision works but character recognition is weaker.

### The Budget Path (~$1.50/hr)

| Component | Choice | Cost/hr |
|---|---|---|
| LLM | Groq — Llama 4 Scout (free) | $0 |
| TTS | Fish Audio | ~$1.50 |
| **Total** | | **~$1.50** |

Best balance. Groq is fast with a free tier. Fish Audio voices sound natural. Vision works with Llama 4 Scout.

### The Quality Path (~$6.50/hr)

| Component | Choice | Cost/hr |
|---|---|---|
| LLM | Anthropic — Claude Sonnet 4.5 | ~$4.50 |
| TTS | ElevenLabs | ~$2.00 |
| **Total** | | **~$6.50** |

Best quality. Claude has the strongest vision (recognizes game characters, IP, brands). ElevenLabs has the most natural voices.

### What Drives Cost?

- **LLM tokens**: Every reaction, monologue, and chat reply costs tokens. Vision costs more because images are attached. A typical 1-hour stream uses 300k-600k tokens.
- **TTS characters**: Every word Wallie speaks gets synthesized. A talkative hour = ~15k-25k characters.
- **Silence helps**: Wallie's attention engine deliberately skips ~45% of screen events. This isn't just for naturalness — it directly saves money.

---

## First Stream

Step-by-step from zero to live:

1. **Install** (see above) — dashboard opens at `http://127.0.0.1:8765`

2. **Add API keys** — go to **API Keys** tab, add at least:
   - One LLM key (Groq is easiest — free, fast)
   - One TTS key (Fish Audio or ElevenLabs), OR set up Piper

3. **Pick your engine** — go to **Engine** tab:
   - Select your LLM provider (e.g., Groq)
   - Select a model from the dropdown
   - Vision-capable models are marked with `· vision`

4. **Set up voice** — go to **Voice** tab:
   - Pick your TTS provider
   - Enter your Voice ID
   - Type a test sentence and click **Test voice** to verify

5. **Design your persona** — go to **Identity** and **Personality** tabs:
   - Give your streamer a name, handle, backstory
   - Set energy level, humor style, catchphrases
   - Click **Test monologue** to hear a sample

6. **Set up audio routing** — see [OBS Setup](#obs-setup)

7. **Optional: Enable vision** — go to **Vision** tab:
   - Toggle ON, select your monitor
   - Make sure your engine has a vision-capable model selected
   - Click **Capture screen + ask LLM** to test

8. **Hit Start** — top-right corner of the dashboard. Wallie begins streaming.

---

## Choosing an LLM Provider

### Comparison

| Provider | Speed | Vision Quality | Character Recognition | Cost | Best For |
|---|---|---|---|---|---|
| **Groq** | Fastest | Good (Llama 4) | Moderate | Free | Budget streams, testing |
| **Gemini** | Fast | Good | Good | Free | Zero-cost streams |
| **OpenAI** | Fast | Excellent | Excellent | $$$ | Best vision accuracy |
| **Anthropic** | Medium | Excellent | Best | $$$ | Best character/IP recognition |
| **OpenRouter** | Varies | Varies | Varies | Varies | Access to many models with one key |
| **Ollama** | Local | Varies | Varies | Free | Fully offline, privacy |

### Which Models Support Vision?

In the dashboard, vision-capable models are labeled with `· vision` in the dropdown. If you enable vision, make sure to select one of these models AND toggle "Model supports vision" ON in Engine settings.

### Recommendations

- **Just testing?** → Groq + Llama 3.3 70B (no vision, but free and fast)
- **Want vision on a budget?** → Groq + Llama 4 Scout (free, decent vision)
- **Want the best vision?** → Anthropic + Claude Sonnet 4.5 (recognizes game characters, logos, memes)
- **Want to go fully offline?** → Ollama + llama3.2-vision

### Ollama Setup (Local LLM)

1. Install Ollama from [ollama.com](https://ollama.com)
2. Pull a model: `ollama pull llama3.2` (or `llama3.2-vision` for vision)
3. In the dashboard: Engine → provider: `ollama` → model: `llama3.2`
4. No API key needed — Ollama runs on your machine
5. **Note**: Quality depends on your hardware. 8B models need ~6 GB RAM, 70B needs ~40 GB.

---

## Choosing a TTS Provider

### Comparison

| Provider | Voice Quality | Latency | Voice Cloning | Cost |
|---|---|---|---|---|
| **Fish Audio** | Natural | Low (~200ms) | Yes (upload samples) | ~$15/M chars |
| **ElevenLabs** | Most natural | Low (~300ms) | Yes (best in class) | ~$30/M chars |
| **Piper** | Robotic but clear | Instant (local) | No | Free |
| **Kokoro** | Natural (local) | Instant (local) | No | Free |

### Recommendations

- **Budget** → Fish Audio (half the price of ElevenLabs, nearly as good)
- **Best quality** → ElevenLabs (most natural, best cloning)
- **Free** → Piper (sounds robotic but perfectly functional; install it and grab any catalogued voice without leaving the dashboard)
- **Free, better voice** → Kokoro (local and natural, also speaks pt-BR — the Windows setup installs and pre-downloads it for you; see the local voice setup variables above)

### Finding a Voice ID

**Fish Audio:**
1. Go to [fish.audio](https://fish.audio)
2. Browse voices or search
3. Click a voice → the URL contains the ID (e.g., `fish.audio/voice/4ee64f72...`)
4. Copy that ID into the dashboard

**ElevenLabs:**
1. Go to [elevenlabs.io/voice-library](https://elevenlabs.io/voice-library)
2. Find a voice → click "Use"
3. Go to your Voices page → click the voice → copy Voice ID
4. Or use the API tab for the ID

### Voice Tuning

In the Voice section of the dashboard:

- **Stability** (ElevenLabs): Lower = more expressive but less consistent. Start at 0.45.
- **Similarity Boost** (ElevenLabs): Higher = closer to the original voice. 0.75 is good default.
- **Sample Rate**: 24000 Hz is the default and works for all providers.
- **Latency Mode** (Fish Audio): `balanced` is recommended. `normal` is slower but slightly better quality.
- **Language / Speed** (Kokoro): the language picker decides which voice ids are valid (`Portuguese (Brazil)` → `pf_dora` / `pm_alex` / `pm_santa`); speed stretches or compresses delivery. No API key needed.
- **Speed / Expressiveness / Pacing variety** (Piper): `piper_length_scale` (1.0 = natural, lower talks faster), `piper_noise_scale` (0.667 default) and `piper_noise_w` (0.8 default). Anything you set here is part of a saved voice and comes back with one click from the Voice Lab library.

---

## The Voice Lab

The **Voice Lab** is a second voice page (dashboard → **Voice Lab**). Where the **Voice** page configures *how the streamer sounds right now*, the Voice Lab is where you keep, create and move voices.

### Saved voices (the library)

A **saved voice** — a preset — snapshots the Voice page's current settings under a name: provider, voice id, speed, stability, the Kokoro language, the Piper tuning, and so on. The one thing a preset never carries is the **output device** — that routes audio, it isn't part of a voice.

Presets live in a per-profile library, `profiles/<profile>.voices.json`, so "Wallie BR" can be a Kokoro pt-BR voice in one profile and "Wallie EN" a cloned ElevenLabs voice in another. Each profile keeps its own list, up to **60 voices**.

Two buttons apply a preset:

| Button | What it does |
|---|---|
| **use** | applies the preset to the Voice page and saves |
| **▶ test** | applies it *and* synthesizes a sample through the real TTS pipeline, so you hear it before going live |

**Piper and Kokoro voices saved on the Voice page live in this same library.** That is deliberate: it means a local voice shows up in the A/B compare and travels with every copy, pull, backup and mirror. (An older `profiles/<profile>.localvoices.json` from before the two stores were merged is imported into the library once and then removed — nothing is lost.)

### Creating a voice (cloning)

The Voice Lab can create a *real* voice on your own provider account:

| Provider | Endpoint used | Key needed | Notes |
|---|---|---|---|
| **ElevenLabs** | `POST /v1/voices/add` (instant voice clone) | `ELEVENLABS_API_KEY` | ~1 minute of clean speech is enough |
| **Fish Audio** | `POST /model` (private voice model) | `FISH_API_KEY` | Returns a model id you use as the voice id |

Give it reference audio by **dropping in files** (wav / mp3 / m4a / flac / ogg, up to 5, under ~10 MB each) or by **recording straight from the default microphone** in the page — the clip is wrapped into a WAV with the Python standard library, so nothing extra needs installing. Audition each clip in the browser, then create the voice; on success its id is saved into the library above, ready to apply.

**Piper and Kokoro cannot be cloned into** — they are local fixed-voice models, and the page says so instead of failing silently.

### A/B compare

Pick two voices — a saved voice, or *current Voice page settings* (including unsaved edits) — type one line and hit **synthesize both**. Both clips come back as playable WAVs, so you can hit **play A → B** to hear them back to back, or replay either one. The audio is returned to the page and never touches the live output device (comparing a voice must not talk over the stream), and nothing is saved. If one side fails — plan limits, a bad voice id — only that side's card shows an error, and the clip that did synthesize is kept.

### Moving voices around

The library is per profile, so there are four ways to move voices — between profiles on this machine, or out of it entirely. Tick rows with the checkboxes on the left (**select all** / **clear** sit next to the list) to act on several at once:

| Move | What it does |
|---|---|
| **⇪ copy** | pushes voices **into** another profile: a single row's **⇪**, the ticked selection, or the whole list |
| **⇩ pull** | the inverse — reads *another* profile's library, you tick what you want, and it lands here |
| **⤓ / ⤒ `.json`** | exports the library (or just the ticked voices) as a portable file and merges one back in |
| **⇄ mirror** | makes this profile and another hold the same voices, **both ways**, after showing the diff |

Beside the ⇄ button, **☆ set as voice partner** records one of those profiles as this profile's *partner* — the one it is meant to stay identical to. That only stores the choice and switches on **automatic syncing**: whenever you switch to a profile that has a partner and the two libraries differ, the voices each side is missing are copied both ways as part of the switch, and a toast tells you what moved. The policy is always *leave differing voices alone* — a name that exists on both sides with different settings is never overwritten automatically (use ⇄ mirror to pick a winner); and a library too full to take the partner's voices reports the refusal instead of failing the switch. From then on the **profile picker in the top bar shows a ⇄ badge** while the two libraries differ (the tooltip carries the counts: how many to copy, how many to pull, how many differ), plus a ⇄ next to that profile's name in the dropdown. Mirroring brings them back in sync and the badge goes back to *⇄ in sync*; **clicking the badge** opens the Voice Lab with the partner already chosen as the mirror target, the diff loaded and the ⇄ mirror button waiting. The same button — now **★ voice partner** — clears the partnership (and stops the auto-sync). A partner that was since deleted shows as *⇄ partner missing* instead of being silently forgotten.

An export is a small, plain JSON file you can keep in git, mail to someone, or restore after a reinstall:

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

The backup carries **no secrets** — only names, provider/voice ids and tuning knobs, no keys and no audio — so it is safe to version or hand to someone else.

What every move has in common:

- **Upsert by name.** A voice with the same name (case-insensitive) is *replaced*, never duplicated, so re-importing the same file or mirroring twice is a harmless no-op.
- **Overwriting asks first.** If that name already exists **with different settings**, the server refuses with a `409` carrying the exact diff — `Wallie BR: kokoro : pf_dora → elevenlabs : v-9` — and the dashboard asks before replacing anything; declining writes nothing. Mirroring is explicit instead: the diff lists the clashing voices and you choose *leave them alone* (the default), *keep this profile's version* or *use the other profile's version*.
- **No pointless writes.** Voices that are already identical are left untouched, and a move that would overflow the 60-voice cap is refused *before* anything is written.
- **A bad file fails loudly, not halfway.** An entry with no name is skipped, but a file that isn't a voice backup, is empty, is over ~2 MB or was written by a newer version is refused with a message that says why.
- **Local voices stay local to their engine.** A Piper preset carries only `piper_*` tuning, a Kokoro one only `kokoro_*`; neither can bring in the other engine's settings.

A few honest notes:

- **Cloning is provider-side.** The audio is uploaded to ElevenLabs / Fish Audio under *your* account and their terms apply to whatever you clone — only clone voices you have the rights to use.
- Malformed or too-short samples are rejected by the provider, and the provider's own message is shown so you can fix it rather than guess.
- The **Voice** page is still where you change the live voice; the Voice Lab is where you save, create and move voices.

---

## Designing Your Persona

The persona system is what makes Wallie different from a chatbot. You're not just setting a system prompt — you're designing a character.

### Identity Tab

| Field | What It Does | Example |
|---|---|---|
| **Name** | The streamer's display name | "Marlow" |
| **Handle** | Social media handle | "@marlow_lol" |
| **Language** | Primary language for output | "en" |
| **Pronouns** | Used in third-person references | "he/him" |
| **Age range** | Affects maturity of references | "late 20s" |
| **Origin** | Where they're "from" | "Brooklyn basement" |
| **Archetype** | One-line character summary | "variety streamer · tech cynic" |
| **Backstory** | Deeper character background | Free text |

### Personality Tab

**Energy levels:**
- `chill` — laid-back, slower delivery, dry humor
- `warm` — conversational, moderate energy
- `hyped` — excited, faster, more reactive
- `unhinged` — chaotic, unpredictable, maximum energy

**Humor styles** (pick multiple):
- `ironic` — says the opposite of what they mean
- `deadpan` — delivers jokes with zero emotion
- `absurd` — random, surreal connections
- `observational` — comments on everyday oddities
- `self_deprecating` — makes fun of themselves
- `roast` — teases chat and content
- `wholesome` — genuine warmth
- `chaotic` — unpredictable, rapid shifts

**Catchphrases:** Short phrases your AI uses periodically. Don't add too many — 3-5 is ideal. The system automatically cooldowns catchphrases to prevent spam.

**Banned words:** Words the AI will never use. Good for removing generic filler ("synergy", "leverage", "iconic").

**Extra style notes:** Free text for anything the structured fields can't capture. Examples:
- "Pauses with 'okay' or 'right', never 'um'"
- "Argues with himself out loud"
- "Mentions his cat Pixel about once every 10 minutes"

### Tips for Good Personas

1. **Be specific.** "Sarcastic gamer" is boring. "Ex-speedrunner who quit after getting trolled by a 12-year-old in Dark Souls" is interesting.
2. **Give them opinions.** Toggle "Strong opinions" ON. Wishy-washy AI is boring AI.
3. **Use banned words aggressively.** Every word you ban forces the AI to be more creative.
4. **Test before streaming.** Use the Test buttons to generate monologues and vision reactions. Iterate until it sounds right.
5. **Keep catchphrases short.** "Let's go" works. "Well, that's certainly an interesting development in the ongoing saga of..." doesn't.

---

## Vision

Vision lets Wallie react to what's on your screen — games, videos, websites, whatever.

### Setup

1. **Engine**: Select a vision-capable model (marked `· vision` in the dropdown)
2. **Engine**: Toggle "Model supports vision" ON
3. **Vision** tab: Toggle "Enable screen vision" ON
4. **Monitor index**: `1` = primary monitor, `2` = secondary, `0` = all monitors combined
5. **Test**: Click "Capture screen + ask LLM" to verify the pipeline

### Key Settings

| Setting | Default | What It Does |
|---|---|---|
| **Frame interval** | 3.0s | How often to capture the screen. Lower = more responsive, more expensive. |
| **Change threshold** | 8 | How much the screen must change to trigger a reaction. Higher = less sensitive. |
| **Max edge (px)** | 768 | Downscale frames before sending. Lower = cheaper and faster. 768 is a good balance. |
| **Startup delay** | 5s | Wait before starting vision after hitting Start. Gives you time to switch away from the dashboard. |
| **First-person framing** | ON | "I just got bodied" vs "the character died". Keep this ON. |
| **Commentary density** | balanced | `sparse` = reacts less often, `balanced` = normal, `dense` = reacts to almost everything |

### How Vision Decisions Work

Not every screen change gets the same reaction. The attention engine decides:

| Reaction Type | Probability | What Happens |
|---|---|---|
| **DEEP** | 22% | Full multi-sentence reaction with opinions |
| **GLANCE** | 28% | Quick 1-2 sentence comment |
| **TANGENT** | 5% | Screen triggers a personal thought/memory |
| **IGNORE** | 27% | Deliberately skips — nothing worth saying |
| **SILENCE** | 18% | Stays quiet for a natural pause |

This prevents the "screen reader" problem where the AI narrates everything. Real streamers don't react to every pixel change.

### Vision and Cost

Vision is the most expensive feature because each reaction sends a full image to the LLM. Tips to reduce cost:

- Set commentary density to `sparse`
- Increase the change threshold (less reactions trigger)
- Lower max edge px (smaller images cost less)
- Use Groq + Llama 4 Scout (free vision) instead of Claude/GPT-4o

### Multi-Monitor Setup

If you have two monitors:
- Put your game/content on **Monitor 1**
- Put the Wallie dashboard on **Monitor 2**
- Set `monitor_index: 1` in Vision settings

This way Wallie only sees your content, never the dashboard.

---

## Chat Integration

Wallie can read and respond to live chat from Twitch, YouTube, and Kick.

### Twitch

1. Get an OAuth token from [twitchtokengenerator.com](https://twitchtokengenerator.com) (select `chat:read` scope)
2. In the dashboard **API Keys**: paste the token as `TWITCH_OAUTH_TOKEN`
3. Set `TWITCH_CHANNEL` to your channel name (lowercase)
4. **Chat** tab → toggle Twitch ON

> You can skip the OAuth token for anonymous read-only access. Set just the channel name.

### YouTube

1. Create a project in [Google Cloud Console](https://console.cloud.google.com)
2. Enable YouTube Data API v3
3. Create OAuth credentials → download as `client_secret.json`
4. Put the file in `scripts/client_secret.json`
5. First run opens a browser for Google OAuth consent — authorize it
6. Set `YOUTUBE_LIVE_CHAT_ID` in API Keys (find this in YouTube Studio → your live stream)
7. **Chat** tab → toggle YouTube ON

### Kick

1. Just enter your channel slug (the part after kick.com/) as `KICK_CHANNEL`
2. No auth needed — Kick uses public WebSockets
3. **Chat** tab → toggle Kick ON

### Chat Settings

| Setting | Default | What It Does |
|---|---|---|
| **Reply probability** | 35% | Chance of replying to a regular chat message. Not every message gets a response. |
| **Min reply interval** | 8s | Minimum time between chat replies. Prevents spam. |

- **Highlighted messages** (subs, donos, bits) always get a reply and trigger a hype expression on the avatar.
- Chat replies are in-character — Wallie responds as the persona, not as a support bot.

---

## Avatar

Wallie drives a Live2D avatar via VTube Studio with six animation layers running simultaneously.

### Setup

1. Install [VTube Studio](https://store.steampowered.com/app/1325860/VTube_Studio/) and load your Live2D model
2. In VTS: Settings → API → Enable
3. In Wallie dashboard: **Avatar** → toggle ON
4. First connect triggers a plugin approval popup in VTS — click **Allow**
5. Expression slots are auto-mapped from your model's hotkeys on connect

### What Wallie Controls

| Layer | Parameter | What It Does |
|---|---|---|
| **Viseme lip sync** | MouthOpen + ParamMouthForm | Drives mouth open/close AND mouth shape (round vs wide) from audio spectrum |
| **Smile** | MouthSmile | Speaking smile + mood-driven resting smile |
| **Blink** | EyeOpenLeft/Right | Natural eye blinks with double-blink variation |
| **Head idle** | FaceAngleX/Y | Slow head sway when not speaking |
| **Eye darts** | EyeLeftX/Y | Random small saccades every few seconds |
| **Body motion** | BodyAngleX/Y/Z | Slow torso sway independent of head |
| **Expressions** | Hotkeys | 11 emotion slots triggered by sentence content |
| **Brows** | Brows | Position shifts based on mood valence |

### Lip Sync Tuning

| Setting | Default | What It Does |
|---|---|---|
| **Gain** | 4.0 | Amplify the audio signal. Higher = more mouth movement. |
| **Ceiling** | 0.85 | Maximum mouth opening (1.0 = full open). |
| **Floor** | 0.02 | Noise threshold below which mouth stays closed. |
| **Attack** | 0.65 | How fast the mouth opens. 1.0 = instant. |
| **Release** | 0.30 | How fast the mouth closes. Lower = smoother. |
| **Speaking smile** | 0.15 | Subtle smile while talking. 0 = flat. |

### Viseme (Spectral Mouth Shape)

Enabled by default. Uses audio frequency analysis to determine mouth shape:
- Front vowels (A, E, I) → wide mouth shape
- Back vowels (O, U) → round mouth shape

| Setting | Default | What It Does |
|---|---|---|
| **Enable** | ON | Toggle spectral analysis on/off |
| **Smoothing** | 0.35 | How fast mouth shape follows audio. Higher = snappier. |
| **Mouth form parameter** | ParamMouthForm | The VTS parameter name for mouth shape |

If your model doesn't have `ParamMouthForm`, disable viseme. Lip sync will fall back to volume-only (open/close), which still works fine.

### Expression Mapping

Wallie has 11 expression slots:

`happy`, `surprised`, `laughing`, `angry`, `sad`, `thinking`, `smug`, `eyeroll`, `confused`, `hype`, `deadpan`

**Auto-mapping**: On connect, Wallie scans your model's hotkeys and matches them to empty slots by name. If your hotkey is called "happy_face", it'll auto-assign to the `happy` slot.

**Manual mapping**: If auto-mapping doesn't find your hotkeys, enter the exact hotkey name in each slot manually.

**How expressions trigger:**
1. **Per-sentence keywords**: "haha" → laughing, "let's go" → hype, "wait what" → surprised
2. **Per-intent**: Chat highlight → hype, screen change → surprised, thinking pause → thinking

### Custom Parameter Names

Different Live2D models use different parameter names. If your model doesn't respond, check the parameter mapping:

| Wallie Default | Common Alternatives |
|---|---|
| `MouthOpen` | `ParamMouthOpenY`, `MouthOpenY` |
| `EyeOpenLeft` | `ParamEyeLOpen` |
| `EyeOpenRight` | `ParamEyeROpen` |
| `FaceAngleX` | `ParamAngleX` |
| `Brows` | `ParamBrowLY`, `BrowLeftY` |

Override any of these in Avatar → Parameters.

---

## OBS Setup

Wallie outputs audio to your system's default playback device. To get it into OBS, you need a virtual audio cable.

### Windows

1. Install [VB-CABLE](https://vb-audio.com/Cable/) (free)
2. Set **CABLE Input** as your default playback device (Windows Sound Settings)
3. In OBS: Add an **Audio Input Capture** source → select **CABLE Output**
4. Wallie's audio now flows directly into OBS

### macOS

1. Install [BlackHole](https://existential.audio/blackhole/) (free, open-source)
2. Set BlackHole as default output
3. In OBS: Add Audio Input Capture → select BlackHole

### Linux

```bash
# PipeWire
pw-loopback --capture-props='media.class=Audio/Sink' &

# PulseAudio
pactl load-module module-null-sink sink_name=wallie
pactl set-default-sink wallie
# In OBS: Audio Input Capture → "Monitor of wallie"
```

### Hearing Your Own Audio

If you route audio to a virtual cable, you won't hear it through your speakers anymore. To monitor:

- **Windows**: In VB-CABLE control panel, set "Cable Output" to also play through your speakers
- **OBS**: Right-click the audio source → Advanced Audio Properties → set Monitoring to "Monitor and Output"

---

## Remote Control

The dashboard is accessible from any device on your local network — phone, tablet, another PC.

### How It Works

By default, the dashboard binds to `0.0.0.0:8765` (all network interfaces). When accessed from another device:

1. A **PIN** is auto-generated and printed in the terminal
2. Open `http://<your-pc-ip>:8765` on your phone
3. Enter the PIN on the login page
4. Full dashboard access — start/stop, config, live monitoring

### Finding Your PC's IP

The terminal shows it on startup: `dashboard: access from your phone/tablet: http://192.168.x.x:8765`

Or manually:
- **Windows**: `ipconfig` → look for IPv4 Address
- **macOS/Linux**: `ifconfig` or `ip addr`

### Setting a Permanent PIN

Add to your `.env` file:
```
DASHBOARD_PIN=1234
```

Without this, a new random PIN is generated every time Wallie starts.

### Why Use Remote Control?

If you stream on the same machine that runs Wallie, the dashboard browser tab is visible to vision capture. Solutions:

1. **Phone/tablet** (recommended) — control from another device, nothing on screen
2. **Second monitor** — put dashboard on monitor 2, set `monitor_index: 1`
3. **Startup delay** — 5-second delay before vision starts (gives you time to switch windows)

---

## Engine Tuning

Fine-tune LLM behavior in the **Engine** tab.

| Setting | Default | What It Does |
|---|---|---|
| **Temperature** | 0.85 | Higher = more creative/random. Lower = more focused. 0.8-0.9 is good for streaming. |
| **Top-p** | 0.95 | Nucleus sampling. Lower = more focused. Usually leave at 0.95. |
| **Max tokens** | 350 | Maximum response length. 300-500 is good for streaming. |
| **Presence penalty** | 0.3 | Penalizes topics already discussed. Higher = more topic variety. |
| **Frequency penalty** | 0.4 | Penalizes word repetition. 0.3-0.5 prevents catchphrase spam. |

### When to Adjust

- **AI keeps repeating itself** → Increase frequency penalty to 0.5-0.7
- **AI is too random/incoherent** → Lower temperature to 0.7
- **AI responses are too long** → Lower max tokens to 200-300
- **AI sticks to one topic** → Increase presence penalty to 0.5
- **AI is too generic/safe** → Increase temperature to 0.9-1.0

---

## Session Management

### Session Duration

Set in the dashboard under **Engine → Session duration (min)**. Set to `0` for unlimited.

When a duration is set:
- A countdown appears in the live drawer
- Near the end, Wallie performs an outro (wraps up, says goodbye)
- The outro duration is configurable (default: 30s)

### Breaks

Wallie takes periodic breaks to feel more natural:

| Setting | Default | What It Does |
|---|---|---|
| **Enable breaks** | ON | Periodic silent pauses |
| **Break every (min)** | 8 | Average time between breaks |
| **Break jitter** | 0.35 | Randomization factor (±35%) |
| **Break length** | 4-12s | Random duration per break |

### Memory

**Within a session:**
- Rolling summarizer compresses older history every ~14 segments
- Dedupe engine catches repeated phrases and paraphrases
- Theme tracker prevents the same angle twice

**Across sessions:**
- Key facts and viewer interactions persist in memory files
- Session notes from previous streams are injected into the next one

---

## Profiles

Profiles let you save and switch between different streamer configurations.

### Creating Profiles

- **New profile**: Click ＋ in the top bar → enter a name
- **Clone profile**: Click ⎘ to duplicate the current profile
- **Switch**: Use the dropdown in the top bar
- **Delete**: Click 🗑 (can't delete the last profile)

### What's Saved in a Profile

Everything: persona, LLM config, TTS config, vision settings, chat settings, avatar config, engine tuning. API keys are NOT part of profiles — they're stored in `.env` and shared across all profiles.
The **voice library** (`profiles/<profile>.voices.json`) belongs to the profile too — see [The Voice Lab](#the-voice-lab) to copy, export or mirror voices between profiles.

### Use Cases

- One profile for gaming streams (vision ON, energetic persona)
- One profile for chill/music streams (vision OFF, laid-back persona)
- One profile for testing (cheap model, fast iteration)

---

## Environment Variables

All set via the dashboard **API Keys** tab or manually in `.env`:

### LLM Keys

| Variable | Provider |
|---|---|
| `GROQ_API_KEY` | Groq |
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GEMINI_API_KEY` | Google Gemini |
| `OPENROUTER_API_KEY` | OpenRouter |

### TTS Keys

| Variable | Provider |
|---|---|
| `FISH_API_KEY` | Fish Audio |
| `ELEVENLABS_API_KEY` | ElevenLabs |

### Chat Keys

| Variable | What |
|---|---|
| `TWITCH_OAUTH_TOKEN` | Twitch chat OAuth token |
| `TWITCH_CHANNEL` | Twitch channel name (lowercase) |
| `TWITCH_NICK` | Bot username for Twitch |
| `KICK_CHANNEL` | Kick channel slug |
| `YOUTUBE_LIVE_CHAT_ID` | YouTube live chat ID |

### Dashboard

| Variable | Default | What |
|---|---|---|
| `DASHBOARD_HOST` | `0.0.0.0` | Bind address. Set to `127.0.0.1` to disable remote access. |
| `DASHBOARD_PORT` | `8765` | Dashboard port |
| `DASHBOARD_PIN` | (auto) | PIN for remote access. Auto-generated if not set. |

### Local Voice Setup (Windows)

Read by `install.bat` / the Kokoro installer — set them in the shell you run the setup from (not in `.env`):

| Variable | Default | What |
|---|---|---|
| `WALLIE_SKIP_KOKORO` | (unset) | Set to `1` to skip installing and pre-downloading the free local voice. |
| `WALLIE_KOKORO_LANG` | `a` | Kokoro language the setup pre-downloads — `p` for Portuguese (Brazil). |
| `WALLIE_KOKORO_VOICE` | (language default) | Voice id to pre-download, e.g. `pf_dora`. |

---

## Troubleshooting

### Voice

**Copying / pulling / importing a voice asks about a name that already exists:**
That's the overwrite guard, not an error. The destination already has a saved voice with that name *and different settings*, so the dashboard shows the diff (`name: current → incoming`) and asks before replacing it. Accept to overwrite, cancel to leave everything as it was — nothing is written until you answer. Mirroring never asks: its diff lists the clashing voices and you choose which side wins (or make the difference a no-op).

**The Piper catalogue won't load, or a voice download fails:**
The catalogue is fetched live from `huggingface.co/rhasspy/piper-voices` (a few hundred KB, cached in-process for an hour), so it needs internet the first time — **refresh** retries. A failed download keeps its last progress and its message in the log under the box; the usual causes are no network, a proxy, or a wrong name. Names are `lang_REGION-voice-quality` (`pt_BR-faber-medium`), or just pick from the catalogue instead of typing.

### Audio

**Audio is static/crackling:**
Click **reset audio** in the top bar. If it recurs, check the terminal logs — usually a TTS provider returning non-PCM data.

**No audio at all:**
- Check that your TTS provider and voice ID are set correctly
- Click **Test voice** in the Voice tab with a sample sentence
- Make sure your default playback device is correct

### Vision

**AI describes generic UI ("I see a YouTube page"):**
The SKIP mechanism depends on the model. Smaller models (Llama 3.1 8B, Gemini Flash) are worse at following it. Use Claude Sonnet or GPT-4o for better vision, or set commentary density to `sparse`.

**AI sees the dashboard on startup:**
Increase the startup delay in Vision settings (default: 5s). Or control Wallie from your phone.

**AI speaks in third person during games ("the character is fighting"):**
First-person framing should be ON in Vision settings. If the model still does it, use a stronger model (Claude Sonnet is best at following first-person instructions).

### Avatar

**Mouth doesn't move:**
Check that `MouthOpen` parameter name matches your model. Some use `ParamMouthOpenY` or `MouthOpenY`. Override in Avatar → Parameters.

**Mouth moves but shape looks wrong:**
Viseme drives `ParamMouthForm`. If your model uses a different name, update it. If your model doesn't support mouth form, disable "Spectral mouth shape" — lip sync falls back to volume-only.

**Expressions don't fire:**
Expression slots need to match VTS hotkey names. Use "Discover hotkeys from VTS" in the dashboard to see available names, then map them manually.

**Can't connect to VTS:**
- Make sure VTS is running with API enabled (Settings → API → Enable)
- Default port is 8001 — check if another app is using it
- Firewall might be blocking the connection

### LLM

**TTS returns 401:**
API key is invalid or expired. Verify in API Keys → Test.

**Responses are too slow:**
- Switch to Groq (fastest inference)
- Lower max tokens
- If using Ollama, a smaller model (8B) is much faster than 70B

**AI keeps asking chat "what do you think?":**
The question throttle is automatic. If it persists, raise frequency penalty to 0.5-0.7.

---

## FAQ

**Q: Can I run Wallie on a cloud server?**
Yes. Set `DASHBOARD_HOST=0.0.0.0` and `DASHBOARD_PIN` in `.env`. Vision capture works if the server has a display (or use a virtual framebuffer like Xvfb on Linux). TTS audio goes to the server's audio device — route it via virtual cables to your streaming software.

**Q: Can I use multiple LLM providers at the same time?**
No. One LLM provider and one TTS provider per profile. But you can switch profiles instantly.

**Q: Does Wallie work with non-English languages?**
Yes. Set the language in Identity → Language. The persona system, prompts, and TTS all support multiple languages. The quality depends on the LLM and TTS provider's support for that language.

**Q: How much RAM/CPU does Wallie use?**
Minimal — ~100-200 MB RAM. All heavy lifting (LLM, TTS) happens on external APIs. The only CPU-intensive part is vision capture (screen grabbing + pHash comparison), which is lightweight. If using Piper (local TTS) or Ollama (local LLM), resource usage increases significantly.

**Q: Can I use a custom Live2D model?**
Yes. Any Live2D model loaded in VTube Studio works. Wallie uses standard VTS parameters — if your model has non-standard parameter names, override them in Avatar → Parameters.

**Q: Can I stream to multiple platforms simultaneously?**
Wallie outputs audio to one audio device. Your streaming software (OBS/Streamlabs) handles multi-platform output. Chat integration supports Twitch, YouTube, and Kick simultaneously — enable all three in the Chat tab.

**Q: Is my API key safe?**
Yes. Keys are stored in `.env` with restricted permissions. The dashboard never exposes raw keys — only masked previews (`sk-•••xyz`). The dashboard binds to localhost by default with PIN protection for remote access.
A voice backup exported from the dashboard carries no secrets either — only names, provider/voice ids and tuning, never keys and never audio.

**Q: How do I update Wallie?**
```bash
cd wallie-V2
git pull
pip install -r requirements.txt  # in case dependencies changed
```
Your profiles, API keys, and settings are preserved.

**Q: Can viewers interact with Wallie beyond chat?**
Currently, chat messages are the interaction channel. Highlighted messages (subs, donations, bits) get priority responses and trigger hype expressions. Custom interaction modes (polls, channel points) are on the roadmap.

**Q: What happens if the API goes down mid-stream?**
Wallie handles provider errors gracefully. TTS failures are caught and logged — the streamer goes silent for that sentence and continues. LLM failures cause a short pause before retrying. Avatar errors never interrupt audio. Nothing crashes.
