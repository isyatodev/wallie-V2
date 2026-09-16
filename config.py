"""Runtime configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROFILES_DIR = BASE_DIR / "profiles"
STATE_FILE = BASE_DIR / ".wallie_state.json"


# -------------------------------------------------------------------
# Secrets
# -------------------------------------------------------------------
class Secrets(BaseModel):
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    groq_api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    openrouter_api_key: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    anthropic_api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    fish_api_key: str = Field(default_factory=lambda: os.getenv("FISH_API_KEY", ""))
    elevenlabs_api_key: str = Field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))

    youtube_api_key: str = Field(default_factory=lambda: os.getenv("YOUTUBE_API_KEY", ""))
    youtube_client_secret_file: str = Field(
        default_factory=lambda: os.getenv("YOUTUBE_CLIENT_SECRET_FILE", "scripts/client_secret.json")
    )
    youtube_live_chat_id: str = Field(default_factory=lambda: os.getenv("YOUTUBE_LIVE_CHAT_ID", ""))

    twitch_oauth_token: str = Field(default_factory=lambda: os.getenv("TWITCH_OAUTH_TOKEN", ""))
    twitch_channel: str = Field(default_factory=lambda: os.getenv("TWITCH_CHANNEL", ""))
    twitch_nick: str = Field(default_factory=lambda: os.getenv("TWITCH_NICK", ""))

    kick_channel: str = Field(default_factory=lambda: os.getenv("KICK_CHANNEL", ""))

    # Generic OpenAI-compatible endpoint (any provider exposing the chat/completions API).
    openai_compatible_base_url: str = Field(
        default_factory=lambda: os.getenv("OPENAI_COMPATIBLE_BASE_URL", "")
    )
    openai_compatible_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_COMPATIBLE_API_KEY", "")
    )

    # Dedicated OpenAI-compatible endpoints per subsystem. Each can point at a
    # different provider (e.g. qwen-vl for vision, an OpenAI-compatible speech
    # gateway for TTS/STT) — the generic LLM block stays untouched.
    openai_compatible_vision_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_COMPATIBLE_VISION_API_KEY", "")
    )
    openai_compatible_tts_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_COMPATIBLE_TTS_API_KEY", "")
    )
    openai_compatible_stt_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_COMPATIBLE_STT_API_KEY", "")
    )
    # Dedicated memory-analysis model (small/fast LLM that decides what to
    # remember; e.g. a cheap llama-3.1-8b while the brain runs on Claude).
    openai_compatible_memory_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_COMPATIBLE_MEMORY_API_KEY", "")
    )
    # Dedicated thought-generator model (spontaneous topics/questions).
    openai_compatible_thoughts_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_COMPATIBLE_THOUGHTS_API_KEY", "")
    )

    # LivePix donations (OAuth2 client_credentials + webhooks)
    livepix_client_id: str = Field(default_factory=lambda: os.getenv("LIVEPIX_CLIENT_ID", ""))
    livepix_client_secret: str = Field(default_factory=lambda: os.getenv("LIVEPIX_CLIENT_SECRET", ""))
    livepix_access_token: str = Field(default_factory=lambda: os.getenv("LIVEPIX_ACCESS_TOKEN", ""))
    # LivePix account id — when set, webhook payloads whose `userId` differs are rejected.
    livepix_user_id: str = Field(default_factory=lambda: os.getenv("LIVEPIX_USER_ID", ""))

    # Streamlabs donations (Socket API; token from /socket/token or the dashboard socket token)
    streamlabs_access_token: str = Field(default_factory=lambda: os.getenv("STREAMLABS_ACCESS_TOKEN", ""))
    streamlabs_socket_token: str = Field(default_factory=lambda: os.getenv("STREAMLABS_SOCKET_TOKEN", ""))


# -------------------------------------------------------------------
# Persona
# -------------------------------------------------------------------
Profanity = Literal["none", "mild", "heavy"]
Formality = Literal["street", "casual", "formal"]
SentenceLength = Literal["short", "medium", "mixed"]
HumorStyle = Literal[
    "ironic", "deadpan", "absurd", "observational",
    "self_deprecating", "roast", "wholesome", "chaotic",
]
Energy = Literal["chill", "warm", "hyped", "unhinged"]


# -------------------------------------------------------------------
# Dynamic provider blocks (API Keys page)
# -------------------------------------------------------------------
ProviderCategory = Literal["llm", "vision", "tts", "stt", "memory", "thoughts"]


class ProviderBlock(BaseModel):
    """A user-defined OpenAI-compatible endpoint block. Blocks are created and
    named on the API Keys page; each declares which task (category) it serves
    plus its endpoint URL and model. API keys live in .env as
    PROVIDER_<id>_API_KEY — never in the profile."""
    id: str = ""                                # stable slug; names the PROVIDER_<id>_API_KEY env entry
    name: str = "New provider"                  # display name (editable)
    category: ProviderCategory = "llm"          # which subsystem this block serves
    base_url: str = ""                          # endpoint incl. version path, e.g. https://host/v1
    model: str = ""


class SpeakerIDConfig(BaseModel):
    """Voice-print speaker identification for the hearing loop. Distinguishes the
    OWNER's voice from other people in a voice chat: enrolled prints live on
    device (profiles/<name>.speakers.json) and heard speech is labeled OWNER /
    OTHER before it reaches the prompt."""
    enabled: bool = False
    # Cosine similarity above this = same voice. Raise for fewer false OWNER
    # labels in noisy rooms (typical range 0.55-0.85).
    threshold: float = 0.68
    # If heard speech matches no enrolled print (max similarity below this),
    # the line is labeled UNK(OWN) instead of OTHER — keeps the persona from
    # confidently misattributing strangers.
    unknown_threshold: float = 0.45
    # While enabled, non-owner voices are saved (audio only, on device) so the
    # owner can enroll additional people later from the Voice page.
    collect_other_voices: bool = False


class PersonaConfig(BaseModel):
    name: str = "Wallie"
    handle: str = "@wallie"
    language: str = "en"
    pronouns: str = "they/them"
    age_range: str = "early 20s"
    origin: str = "somewhere online"
    archetype: str = "variety streamer"
    backstory: str = (
        "A chronically online streamer who has seen every weird corner of the internet "
        "and is mildly amused by all of it."
    )

    energy: Energy = "warm"
    humor_style: list[HumorStyle] = Field(default_factory=lambda: ["ironic", "observational"])
    profanity: Profanity = "mild"
    formality: Formality = "casual"
    sentence_length: SentenceLength = "short"
    catchphrases: list[str] = Field(default_factory=list)
    running_gags: list[str] = Field(default_factory=list)
    banned_words: list[str] = Field(default_factory=list)
    extra_style_notes: str = ""

    strong_opinions: bool = True
    admit_uncertainty: bool = True
    break_fourth_wall: bool = False  # Can reference being on a stream
    favorite_topics: list[str] = Field(default_factory=list)
    taboo_topics: list[str] = Field(default_factory=list)

    address_style: Literal["by_name", "generic", "crowd"] = "by_name"
    reply_length: Literal["snappy", "medium", "longer"] = "snappy"
    react_to_highlights_hype: bool = True

    vision_first_person: bool = True
    vision_commentary_density: Literal["sparse", "balanced", "dense"] = "balanced"
    vision_interests: list[str] = Field(default_factory=list)
    vision_boring_signals: list[str] = Field(default_factory=lambda: [
        "google homepage", "new tab", "loading", "blank page",
        "desktop", "login screen", "settings",
    ])

    entertainer_mode: bool = True
    audience_hook_rate: float = 0.30
    anecdote_seeds: list[str] = Field(default_factory=list)
    personal_beat_rate: float = 0.35

    # --- Conversational / companion mode (e.g. VRChat) ---
    # When true, Wallie talks WITH people in a real-time back-and-forth instead of
    # hosting a show: short turns, replies to what the other person actually said,
    # no audience/monologue framing.
    conversational: bool = False
    reveal_ai: bool = False          # openly an AI; owns it, can joke about it
    plug_url: str = ""               # soft self-plug (e.g. github) when someone's curious
    plug_rate: float = 0.15          # 0 = never bring it up unprompted; higher = more readily

    # --- Feeling alive: intent + memory ---
    # An ongoing goal/objective the character pursues this session (a "through-line").
    # Gives the stream a spine instead of pure moment-to-moment reactions. Rotates so
    # the session has chapters. Empty = no explicit goal.
    session_goals: list[str] = Field(default_factory=list)
    goal_rotate_sec: float = 600.0   # advance to the next goal every N sec (0 = never rotate)
    # Encourage natural callbacks to earlier moments/takes from this session.
    enable_callbacks: bool = True
    # When conversational + vision is on: Wallie may START conversations on its own —
    # if it sees someone it could talk to, it opens with a greeting/line instead of
    # only ever waiting to be spoken to.
    initiates: bool = False


# -------------------------------------------------------------------
# Other subsystems
# -------------------------------------------------------------------
class LLMConfig(BaseModel):
    provider: Literal[
        "openai", "groq", "openrouter", "anthropic", "gemini", "ollama",
        "openai_compatible",
    ] = "groq"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.85
    top_p: float = 0.95
    max_tokens: int = 350
    presence_penalty: float = 0.3
    frequency_penalty: float = 0.4
    vision_capable: bool = False
    allow_vision_skip: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_keep_alive: str = "5m"
    # Generic OpenAI-compatible endpoint (base URL must include the version path,
    # e.g. https://example.com/v1). API key lives in Secrets (.env).
    openai_compatible_base_url: str = ""
    openai_compatible_timeout: float = 25.0
    # Optional: name a specific provider block (API Keys page) for the brain.
    # Empty = first block with category "llm" wins, else the fields above.
    provider_ref: str = ""
    # --- Dedicated VISION block ---
    # When `vision_provider` is set, vision turns use this separate
    # OpenAI-compatible endpoint instead of the main LLM (e.g. qwen-vl on a
    # vision-only gateway while the brain stays on Claude). Empty = reuse the
    # main LLM for vision (previous behavior).
    vision_provider: Literal["main", "openai_compatible"] = "main"
    # Optional: name a specific provider block (category "vision") for vision turns.
    vision_provider_ref: str = ""
    vision_model: str = ""
    vision_openai_compatible_base_url: str = ""
    vision_openai_compatible_timeout: float = 30.0
    vision_max_tokens: int = 200


class TTSConfig(BaseModel):
    provider: Literal["fish", "elevenlabs", "piper", "kokoro", "openai_compatible"] = "fish"
    voice_id: str = ""
    sample_rate: int = 24000
    # Output device for Wallie's voice. "" = system default. Accepts a device index
    # or a name substring, e.g. "CABLE Input" to route TTS into VRChat via VB-CABLE.
    output_device: str = ""
    el_model_id: str = "eleven_turbo_v2_5"
    el_stability: float = 0.45
    el_similarity_boost: float = 0.75
    el_style: float = 0.0
    fish_latency_mode: Literal["normal", "balanced"] = "balanced"
    # Server-side text buffer before audio generation. Lower = faster time-to-first-audio
    # (snappier reactions), higher = smoother prosody. Range 100-300; 100 favors latency.
    fish_chunk_length: int = 100
    piper_model_path: str = ""
    piper_length_scale: float = 1.0
    # Kokoro — local, high-quality neural TTS (free, runs on CPU/GPU). voice e.g.
    # af_heart / am_adam / bf_emma; lang_code 'a'=US English, 'b'=UK. speed 0.5-2.0.
    kokoro_voice: str = "af_heart"
    kokoro_lang_code: str = "a"
    kokoro_speed: float = 1.0
    # OpenAI-compatible speech endpoint (POST {base}/audio/speech, PCM output).
    # Works with OpenAI itself, self-hosted gateways (e.g. faster-whisper servers
    # exposing the same API), and most aggregators. Key lives in Secrets (.env).
    openai_compatible_base_url: str = ""
    openai_compatible_model: str = ""
    openai_compatible_timeout: float = 30.0
    # Optional: name a specific provider block (category "tts").
    provider_ref: str = ""
    openai_compatible_voice: str = "alloy"
    openai_compatible_speed: float = 1.0
    openai_compatible_pcm_sample_rate: int = 24000


class VisionConfig(BaseModel):
    enabled: bool = False
    source: Literal["monitor"] = "monitor"
    monitor_index: int = 1
    interval_sec: float = 3.0
    min_change_threshold: int = 8
    max_edge_px: int = 768
    llm_max_edge_px: int = 512
    llm_jpeg_quality: int = 50
    scene_change_threshold: int = 20
    min_emit_interval_sec: float = 8.0
    max_frame_age_sec: float = 5.0
    idle_variance_threshold: float = 15.0
    enrich_monologue: bool = False
    enrich_probability: float = 0.08

    organic_vision: bool = False
    organicity: float = 0.75
    never_interrupt_speech: bool = True
    min_vision_react_interval_sec: float = 20.0
    micro_change_threshold: int = 4
    idle_check_interval_sec: float = 45.0
    min_engagement_for_react: float = 0.35
    startup_delay_sec: float = 5.0
    # AttentionEngine reaction weights — how often a vision event becomes each kind of
    # reaction. Lower deep/glance + higher ignore/silence = talks LESS (good for games
    # where the whole screen changes constantly). Defaults match the engine's baseline.
    react_deep_base: float = 0.22
    react_glance_base: float = 0.28
    react_tangent_base: float = 0.05
    react_ignore_base: float = 0.27
    react_silence_base: float = 0.18
    # Scales the "fill the silence" fallback timer. >1 waits longer before narrating
    # into quiet stretches (less ambient chatter); <1 fills dead air sooner.
    silence_fallback_scale: float = 1.0
    # How strongly the app_switch/media "active content" reaction boost applies.
    # 1.0 = full boost (right for browsing, where switching apps is a real event).
    # Lower it for FULLSCREEN GAMES, where every camera move looks like an app_switch
    # and the boost makes the streamer over-talk. 0.0 = treat it like normal navigation.
    active_content_boost: float = 1.0


class HearingConfig(BaseModel):
    """Wallie's ears — captures system audio (game/video/music/voice) via loopback,
    transcribes speech, and feeds it to the orchestrator to fuse with vision."""
    enabled: bool = False
    window_sec: float = 5.0          # length of each capture+transcribe window
    model_size: str = "small"        # faster-whisper model (tiny/base/small/medium)
    language: str = ""               # "" = auto-detect; or "en", "tr", ...
    silence_threshold: float = 0.006  # RMS below this = silence, skipped entirely
    sound_event_threshold: float = 0.06  # loud non-speech still emits a "sound" event
    max_context_age_sec: float = 12.0    # how long a heard line stays relevant for fusion
    reply_gate_sec: float = 6.0          # min seconds between spoken reactions to heard audio
                                         # (lower it for snappy two-way conversation, e.g. 1.5)
    # --- Conversation latency / accuracy (two-way mode) ---
    # VAD segmentation: instead of fixed `window_sec` slices, detect when the person
    # starts and STOPS talking and transcribe the whole utterance the moment they pause.
    # Replies land right after they finish (low latency) and full sentences aren't cut.
    vad_segmentation: bool = False
    poll_interval_sec: float = 0.3       # how often to check for voice activity
    end_silence_sec: float = 0.6         # trailing silence that marks end-of-utterance
    max_utterance_sec: float = 12.0      # safety cap on a single utterance
    low_latency: bool = False            # fewer decode retries — faster, slightly less robust
    # --- Robustness (accents, background TV, messy mics) ---
    speech_only: bool = False            # dialogue only — never react to music / non-speech sound
    beam_size: int = 5                   # Whisper beam width (higher = more accurate on accents/noise)
    denoise: bool = False                # spectral noise reduction before STT (needs `noisereduce`)
    # --- Speaker ID (voice prints, owner vs others) ---
    # Same field can be fed from the dynamic provider system later; today the
    # embedding runs 100% locally (numpy), no endpoint involved.
    speaker_id: SpeakerIDConfig = Field(default_factory=SpeakerIDConfig)

    # --- Remote STT via an OpenAI-compatible endpoint (POST {base}/audio/transcriptions).
    # Empty engine = local faster-whisper (previous behavior). Remote STT avoids the
    # local Whisper VRAM/CPU cost entirely — useful when the same GPU runs the game.
    engine: Literal["", "openai_compatible"] = ""
    openai_compatible_base_url: str = ""
    openai_compatible_model: str = "whisper-1"
    openai_compatible_timeout: float = 20.0
    # Optional: name a specific provider block (category "stt").
    provider_ref: str = ""
    openai_compatible_prompt: str = ""   # optional vocabulary/term hint for the transcription API


class ChatConfig(BaseModel):
    youtube_enabled: bool = False
    twitch_enabled: bool = False
    kick_enabled: bool = False
    reply_probability: float = 0.35
    min_reply_interval_sec: float = 8.0
    max_message_age_sec: float = 45.0


class DonationsConfig(BaseModel):
    """Donation platforms (LivePix / Streamlabs). Events fan into the SAME
    orchestrator queue as chat highlights — no separate pipeline."""
    livepix_enabled: bool = False
    livepix_webhook_path: str = "/webhooks/livepix"
    livepix_enrich: bool = True       # fetch full message details from the LivePix API
    livepix_verify_user_id: bool = True  # reject payloads whose userId != LIVEPIX_USER_ID
    streamlabs_enabled: bool = False
    streamlabs_url: str = "https://sockets.streamlabs.com"
    streamlabs_reconnect_max_sec: float = 60.0
    donation_reply_probability: float = 1.0
    donation_cooldown_sec: float = 0.0


class TopicConfig(BaseModel):
    mode: Literal["list", "ai_picks"] = "ai_picks"
    topics: list[str] = Field(default_factory=lambda: [
        "Artificial intelligence and the future",
        "Strange decisions from tech companies",
        "Absurd observations from everyday life",
    ])
    switch_min_sec: float = 90.0
    switch_chance: float = 0.15
    drift_style: Literal["rigid", "natural", "freeform"] = "natural"


class OrchestratorConfig(BaseModel):
    segment_target_sec: float = 12.0
    dedupe_window: int = 8
    dedupe_threshold: float = 0.65
    prebuffer: bool = True
    max_words_per_sentence: int = 22

    segment_sentences_min: int = 3
    segment_sentences_max: int = 6
    max_audio_lookahead_sec: float = 8.0

    session_duration_min: float = 0.0
    outro_seconds: float = 30.0

    recent_verbatim_turns: int = 24
    summarize_every_n: int = 14
    max_messages: int = 200
    max_chars: int = 60000
    # Cap how many recent turns are actually SENT to the LLM each call (0 = no cap).
    # The rolling summary already carries older context, so trimming the verbatim
    # tail cuts prompt prefill → lower latency, with minimal continuity loss.
    llm_history_messages: int = 0
    # Auto-highlight: while streaming, flag Wallie's most clip-worthy moments to a
    # per-session JSONL in highlights/ so you can cut Shorts without scrubbing.
    auto_highlight: bool = False
    highlight_threshold: float = 0.55

    organic_enabled: bool = True
    silence_beat_min_sec: float = 2.0
    silence_beat_max_sec: float = 5.5
    silence_beat_ceiling: float = 0.35
    min_inter_segment_gap_sec: float = 0.35
    breathing_gap_max_sec: float = 2.5
    post_vision_silence_sec: float = 3.0

    enable_breaks: bool = True
    break_every_min: float = 8.0
    break_every_jitter: float = 0.35
    break_min_sec: float = 4.0
    break_max_sec: float = 12.0


class AvatarConfig(BaseModel):
    enabled: bool = False
    vts_host: str = "127.0.0.1"
    vts_port: int = 8001

    param_mouth_open:  str = "MouthOpen"
    param_mouth_smile: str = "MouthSmile"
    param_mouth_form:  str = "ParamMouthForm"
    param_face_x:      str = "FaceAngleX"
    param_face_y:      str = "FaceAngleY"
    param_face_z:      str = "FaceAngleZ"
    param_eye_x:       str = "EyeLeftX"
    param_eye_y:       str = "EyeLeftY"
    param_brows:       str = "Brows"

    lipsync_gain:    float = 4.0
    lipsync_ceiling: float = 0.85
    lipsync_floor:   float = 0.02
    lipsync_attack:  float = 0.65
    lipsync_release: float = 0.30
    speaking_smile:  float = 0.15

    enable_viseme_lipsync: bool = True
    viseme_smoothing:      float = 0.35

    enable_idle_motion: bool = True
    idle_sway_amplitude: float = 4.0
    idle_sway_period_sec: float = 6.0
    enable_eye_darts: bool = True
    eye_dart_interval_sec: float = 4.5

    expr_happy:      str = ""
    expr_surprised:  str = ""
    expr_laughing:   str = ""
    expr_angry:      str = ""
    expr_sad:        str = ""
    expr_thinking:   str = ""
    expr_smug:       str = ""
    expr_eyeroll:    str = ""
    expr_confused:   str = ""
    expr_hype:       str = ""
    expr_deadpan:    str = ""

    enable_blink:          bool = True
    param_eye_open_left:   str = "EyeOpenLeft"
    param_eye_open_right:  str = "EyeOpenRight"
    blink_interval_sec:    float = 3.8
    blink_hold_sec:        float = 0.045
    double_blink_chance:   float = 0.15

    enable_body_motion:    bool = True
    param_body_x:          str = "BodyAngleX"
    param_body_y:          str = "BodyAngleY"
    param_body_z:          str = "BodyAngleZ"
    body_sway_amplitude:   float = 2.5
    body_sway_period_sec:  float = 9.0

    enable_mood_link:      bool = True
    mood_idle_min_scale:   float = 0.5
    mood_idle_max_scale:   float = 1.6
    mood_brow_min:         float = -0.4
    mood_brow_max:         float = 0.3
    mood_smile_max:        float = 0.20

    auto_map_expressions:  bool = True


class PlayConfig(BaseModel):
    """Minecraft Play mode — Wallie actually PLAYS the game (agent brain), and the streamer
    commentary is grounded in what the agent is really doing instead of guessing from the screen.
    When disabled, Wallie runs in standard vision mode (reacts to whatever is on screen)."""
    enabled: bool = False
    game: str = "minecraft"
    goal: str = (
        "Build a thriving Minecraft empire LIVE for an audience: gather and stockpile resources, "
        "craft full armour and tool sets, build varied good-looking structures, fight, explore and "
        "take on varied adventures. Progress toward the Ender Dragon over time, but keep the JOURNEY "
        "entertaining — this is a show, NOT a speedrun."
    )
    talk_from_agent: bool = True      # commentary uses the agent's real actions/state, not the frame
    hide_chat: bool = True            # hide in-game chat + Baritone commands on stream
    avoid_water: bool = True          # keep Baritone out of water (open-ground play)


class MemoryConfig(BaseModel):
    """AI memory — durable facts the character keeps across sessions, plus a
    short-term tier with TTL. An optional dedicated extraction model (any
    OpenAI-compatible endpoint) decides WHAT to remember after each segment;
    without it a cheap heuristic extractor fills the gap."""
    enabled: bool = False
    # Extraction engine: "main" reuses the brain LLM; "openai_compatible" uses
    # the dedicated memory block below (cheaper/faster); "off" disables capture.
    extractor: Literal["main", "openai_compatible", "off"] = "openai_compatible"
    # Optional: name a specific provider block (category "memory").
    provider_ref: str = ""
    openai_compatible_base_url: str = ""      # e.g. http://localhost:11434/v1
    model: str = ""                           # e.g. llama-3.1-8b-instant
    timeout: float = 12.0
    max_memories: int = 500                   # cap on the store (entries kept per tier)
    short_term_ttl_sec: float = 24 * 3600.0   # default TTL for short-term entries
    promote_hits: int = 3                     # short-term seen N times -> promoted
    prompt_max_chars: int = 1200              # prompt budget for the memory block
    janitor_interval_sec: float = 300.0
    # AUTO-CONSOLIDATION: when long-term memories pass this count, the memory
    # model merges groups of related old entries into single general ones.
    # 0 disables. Runs before the hard max_memories cap so nothing is lost.
    consolidate_threshold: int = 300
    consolidate_batch: int = 40               # how many old entries per pass


class RandomThoughtsConfig(BaseModel):
    """Spontaneous thoughts: on a customizable timer (or full random mode) the
    character brings up a topic/question on its own — adds life to quiet stretches."""
    enabled: bool = False
    # "fixed" = every interval_sec (± jitter); "random" = draws the next thought
    # from a range so the timing itself feels organic.
    schedule: Literal["fixed", "random"] = "fixed"
    interval_sec: float = 180.0               # fixed: thought every N sec
    min_interval_sec: float = 90.0            # random: earliest next thought
    max_interval_sec: float = 420.0           # random: latest next thought
    jitter: float = 0.35                      # fixed: ±35% on the interval
    # "context" = themed on the current topic/what just happened; "random" =
    # unprompted curiosities/questions; "mix" alternates.
    style: Literal["context", "random", "mix"] = "mix"
    # Which LLM turns the chosen style into an actual seed line: "main" reuses
    # the brain, "openai_compatible" a cheap dedicated block, "off" = use only
    # the seed_topics pool / plain topic nudges.
    generator: Literal["main", "openai_compatible", "off"] = "main"
    # Optional: name a specific provider block (category "thoughts").
    provider_ref: str = ""
    openai_compatible_base_url: str = ""      # e.g. http://localhost:11434/v1
    model: str = ""                           # e.g. llama-3.1-8b-instant
    timeout: float = 12.0
    # Optional manual seed pool — e.g. interview questions the character asks,
    # recurring bits, or topics you want surfaced periodically. Empty = AI pick.
    seed_topics: list[str] = Field(default_factory=list)
    only_when_quiet_sec: float = 20.0         # skip if Wallie spoke more recently than this
    max_per_hour: int = 0                     # rate limit; 0 = unlimited
    min_segments_between: int = 3             # at least N spoken segments between thoughts
    # MEMORY CALLBACKS: when a thought fires, instead of inventing a new topic
    # it may surface an OLD memory from the durable store — e.g. "you know what
    # just reminded me of…". Fully organic recall.
    memory_callback_chance: float = 0.35      # 0 = never; 1 = every thought is a callback
    memory_callback_kinds: list[str] = Field(
        default_factory=lambda: ["long_term", "short_term"]
    )


class CaptionsConfig(BaseModel):
    """Caption overlay — Wallie's TTS output rendered as a browser-source page
    (OBS/Streamlabs). The caption box auto-clears when a segment ends so a
    stale message never lingers on stream."""
    enabled: bool = False
    path: str = "/captions"            # overlay page served by the dashboard
    language: str = ""                  # optional lang hint for the overlay (css/font choice)
    max_lines: int = 3                  # lines kept on screen before oldest drops
    clear_delay_sec: float = 0.0        # extra hold after speech before clearing
    font_size: int = 40
    background_opacity: float = 0.55    # 0 = transparent box, 1 = solid
    lowercase: bool = False             # karaoke-style lowercase captions
    show_avatar_name: bool = False      # prefix lines with the persona name


class AppConfig(BaseModel):
    profile_name: str = "default"
    # --- Dynamic provider blocks (managed on the API Keys page) ---
    # Named OpenAI-compatible endpoints; subsystems point at one by name.
    providers: list[ProviderBlock] = Field(default_factory=list[ProviderBlock])
    # Speaker identification (owner vs others in voice chat).
    speaker_id: SpeakerIDConfig = Field(default_factory=SpeakerIDConfig)

    persona: PersonaConfig = Field(default_factory=PersonaConfig)

    @model_validator(mode="after")
    def _ensure_provider_ids(self) -> "AppConfig":
        """Every provider block needs a stable unique id (it names the .env
        key). Assign slugs from id/name when missing or duplicated so saves
        from any client (dashboard, plain PUT /api/config) just work."""
        seen: set[str] = set()
        for i, p in enumerate(self.providers):
            base = (p.id or p.name or f"provider-{i + 1}").strip()
            base = "".join(c if c.isalnum() else "_" for c in base.lower()).strip("_")[:40] \
                or f"provider-{i + 1}"
            cand, n = base, 2
            while cand in seen:
                cand = f"{base}_{n}"   # underscore survives re-slugging (ids stay stable)
                n += 1
            p.id = cand
            seen.add(cand)
        return self
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    hearing: HearingConfig = Field(default_factory=HearingConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    donations: DonationsConfig = Field(default_factory=DonationsConfig)
    topics: TopicConfig = Field(default_factory=TopicConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    avatar: AvatarConfig = Field(default_factory=AvatarConfig)
    play: PlayConfig = Field(default_factory=PlayConfig)
    captions: CaptionsConfig = Field(default_factory=CaptionsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    random_thoughts: RandomThoughtsConfig = Field(default_factory=RandomThoughtsConfig)


# -------------------------------------------------------------------
# Profiles (multi-persona support)
# -------------------------------------------------------------------
@dataclass
class Runtime:
    config: AppConfig
    secrets: Secrets
    base_dir: Path = BASE_DIR


def _ensure_dirs() -> None:
    PROFILES_DIR.mkdir(exist_ok=True)


def _active_profile_name() -> str:
    _ensure_dirs()
    if STATE_FILE.exists():
        try:
            import json
            return json.loads(STATE_FILE.read_text(encoding="utf-8")).get("active", "default")
        except Exception:
            pass
    return "default"


def _set_active_profile_name(name: str) -> None:
    import json
    STATE_FILE.write_text(json.dumps({"active": name}), encoding="utf-8")


def _profile_path(name: str) -> Path:
    safe = "".join(c for c in name if c.isalnum() or c in "-_") or "default"
    return PROFILES_DIR / f"{safe}.yaml"


def list_profiles() -> list[str]:
    _ensure_dirs()
    return sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))


def load_profile(name: Optional[str] = None) -> AppConfig:
    name = name or _active_profile_name()
    path = _profile_path(name)
    if not path.exists():
        cfg = AppConfig(profile_name=name)
        save_profile(cfg, name)
        _set_active_profile_name(name)
        return cfg
    try:
        import yaml  # type: ignore
        data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data["profile_name"] = name
        return AppConfig(**data)
    except ModuleNotFoundError:
        import json
        data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        data["profile_name"] = name
        return AppConfig(**data)


def save_profile(cfg: AppConfig, name: Optional[str] = None) -> None:
    _ensure_dirs()
    name = name or cfg.profile_name or "default"
    cfg = cfg.model_copy(update={"profile_name": name})
    path = _profile_path(name)
    data = cfg.model_dump()
    try:
        import yaml  # type: ignore
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except ModuleNotFoundError:
        import json
        path.with_suffix(".json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def activate_profile(name: str) -> AppConfig:
    cfg = load_profile(name)
    _set_active_profile_name(name)
    return cfg


def delete_profile(name: str) -> bool:
    path = _profile_path(name)
    if path.exists():
        path.unlink()
        if _active_profile_name() == name:
            remaining = list_profiles()
            _set_active_profile_name(remaining[0] if remaining else "default")
        return True
    return False


def clone_profile(src: str, dst: str) -> AppConfig:
    cfg = load_profile(src)
    save_profile(cfg, dst)
    return load_profile(dst)


def get_runtime() -> Runtime:
    return Runtime(config=load_profile(), secrets=Secrets())


def load_config(*_args, **_kwargs) -> AppConfig:
    return load_profile()


def save_config(cfg: AppConfig, *_args, **_kwargs) -> None:
    save_profile(cfg)
