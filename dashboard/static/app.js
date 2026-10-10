// =====================================================================
// Wallie dashboard — Alpine component
// =====================================================================

// JS port of config.provider_slug (Python) — the two MUST stay in sync.
// ASCII-normalizes ids ("Visão" → "visao") so the browser and the server
// always agree on the PROVIDER_<SLUG>_API_KEY name.
function _providerSlugJs(pid) {
  // NFKD, not NFD: the Python side normalizes compatibility forms too (a
  // ligature id like "Ofﬁce" must slug to "office_provider" on BOTH sides).
  const d = String(pid || "").normalize("NFKD").replace(/\p{Diacritic}/gu, "");
  return (d.toLowerCase().replace(/[^a-z0-9_]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 40)) || "provider";
}
function _providerKeyEnvJs(pid) {
  return "PROVIDER_" + _providerSlugJs(pid).toUpperCase() + "_API_KEY";
}

const SECTIONS = [
  { id: "identity",    label: "Identity",     ico: "🪪" },
  { id: "personality", label: "Personality",  ico: "🎭" },
  { id: "voice",       label: "Voice",        ico: "🎙" },
  { id: "voicelab",    label: "Voice Lab",    ico: "🧪" },
  { id: "topics",      label: "Topics",       ico: "📝" },
  { id: "vision",      label: "Vision",       ico: "👁" },
  { id: "play",        label: "Play (MC)",    ico: "🎮" },
  { id: "hearing",     label: "Hearing",      ico: "🎧" },
  { id: "chat",        label: "Chat",         ico: "💬" },
  { id: "donations",   label: "Donations",    ico: "💸" },
  { id: "captions",    label: "Captions",     ico: "💬" },
  { id: "memory",      label: "Memory",        ico: "🧠" },
  { id: "avatar",      label: "Avatar",       ico: "🎴" },
  { id: "engine",      label: "Engine",       ico: "🧠" },
  { id: "secrets",     label: "API Keys",     ico: "🔑" },
];

const HUMOR_OPTIONS = [
  "ironic", "deadpan", "absurd", "observational",
  "self_deprecating", "roast", "wholesome", "chaotic",
];

// Dashboard accent themes (body[data-theme]) — kept in sync with the option
// list in index.html's theme picker and AppConfig.dashboard_theme.
const THEME_OPTIONS = ["cyan", "amber", "rose"];

// Kokoro v1.0 languages. The voice id's FIRST letter is the lang_code the
// pipeline must be initialised with, so the Voice section keeps the two in
// sync (kokoroLangs / kokoroVoiceOptions / onKokoroLangChange).
const KOKORO_LANGS = [
  { code: "a", label: "English (US)" },
  { code: "b", label: "English (UK)" },
  { code: "p", label: "Portuguese (Brazil)" },
  { code: "e", label: "Spanish" },
  { code: "f", label: "French" },
  { code: "h", label: "Hindi" },
  { code: "i", label: "Italian" },
  { code: "j", label: "Japanese" },
  { code: "z", label: "Chinese (Mandarin)" },
];

const KOKORO_VOICES = {
  a: ["af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck"],
  b: ["bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"],
  p: ["pf_dora", "pm_alex", "pm_santa"],
  e: ["ef_dora", "em_alex", "em_santa"],
  f: ["ff_siwis"],
  h: ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
  i: ["if_sara", "im_nicola"],
  j: ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
  z: ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"],
};

// Voice Lab A/B: the line both voices read when the box is left empty. Long
// enough to expose prosody, short enough not to burn TTS credits.
const AB_DEFAULT_TEXT = "Testing this voice — one, two, three. Is this the one you want?";

const EMOTION_SLOTS = [
  "happy", "surprised", "laughing", "angry", "sad",
  "thinking", "smug", "eyeroll", "confused", "hype", "deadpan",
];

const MODEL_OPTIONS = {
  groq: [
    { id: "meta-llama/llama-4-maverick-17b-128e-instruct", label: "Llama 4 Maverick 17B", vision: true },
    { id: "meta-llama/llama-4-scout-17b-16e-instruct", label: "Llama 4 Scout 17B", vision: true },
    { id: "llama-3.3-70b-versatile", label: "Llama 3.3 70B", vision: false },
    { id: "llama-3.1-8b-instant", label: "Llama 3.1 8B (fast)", vision: false },
    { id: "gemma2-9b-it", label: "Gemma 2 9B", vision: false },
    { id: "mixtral-8x7b-32768", label: "Mixtral 8x7B", vision: false },
  ],
  openai: [
    { id: "gpt-4.1", label: "GPT-4.1", vision: true },
    { id: "gpt-4.1-mini", label: "GPT-4.1 Mini", vision: true },
    { id: "gpt-4.1-nano", label: "GPT-4.1 Nano", vision: true },
    { id: "gpt-4o", label: "GPT-4o", vision: true },
    { id: "gpt-4o-mini", label: "GPT-4o Mini", vision: true },
    { id: "o3-mini", label: "o3 Mini", vision: false },
  ],
  openrouter: [
    { id: "anthropic/claude-sonnet-4-6", label: "Claude Sonnet 4.6", vision: true },
    { id: "anthropic/claude-haiku-4-5", label: "Claude Haiku 4.5", vision: true },
    { id: "openai/gpt-4o", label: "GPT-4o", vision: true },
    { id: "openai/gpt-4.1", label: "GPT-4.1", vision: true },
    { id: "google/gemini-2.5-pro", label: "Gemini 2.5 Pro", vision: true },
    { id: "google/gemini-2.5-flash", label: "Gemini 2.5 Flash", vision: true },
    { id: "meta-llama/llama-3.3-70b-instruct", label: "Llama 3.3 70B", vision: false },
  ],
  anthropic: [
    { id: "claude-sonnet-4-6", label: "Claude Sonnet 4.6", vision: true },
    { id: "claude-opus-4-6", label: "Claude Opus 4.6", vision: true },
    { id: "claude-haiku-4-5", label: "Claude Haiku 4.5", vision: true },
  ],
  gemini: [
    { id: "gemini-3.8-flash", label: "Gemini 3.8 Flash", vision: true },
    // The 2.5 family is restricted to accounts that already used it — new keys
    // get a 404 telling them to move to gemini-3.8-flash. Kept so existing
    // setups can still pick them; the Vision test strip can probe either.
    { id: "gemini-2.5-pro", label: "Gemini 2.5 Pro (restricted)", vision: true },
    { id: "gemini-2.5-flash", label: "Gemini 2.5 Flash (restricted)", vision: true },
  ],
  ollama: [],
};

// First-run wizard: budget path -> provider config + required keys.
const WIZARD_PATHS = {
  free:    { llm: "gemini",    model: "gemini-3.8-flash",                            tts: "piper",      keys: ["GEMINI_API_KEY"] },
  cheap:   { llm: "groq",      model: "meta-llama/llama-4-scout-17b-16e-instruct",   tts: "fish",       keys: ["GROQ_API_KEY", "FISH_API_KEY"] },
  premium: { llm: "anthropic", model: "claude-sonnet-4-6",                           tts: "elevenlabs", keys: ["ANTHROPIC_API_KEY", "ELEVENLABS_API_KEY"] },
};
const WIZARD_KEYMETA = {
  GEMINI_API_KEY:     { env: "GEMINI_API_KEY",     provider: "gemini",     label: "Google AI Studio (Gemini)", url: "https://aistudio.google.com/apikey",                 free: true },
  GROQ_API_KEY:       { env: "GROQ_API_KEY",       provider: "groq",       label: "Groq",                      url: "https://console.groq.com/keys",                      free: true },
  FISH_API_KEY:       { env: "FISH_API_KEY",       provider: "fish",       label: "Fish Audio (voice)",        url: "https://fish.audio/go-api/",                         free: false },
  ANTHROPIC_API_KEY:  { env: "ANTHROPIC_API_KEY",  provider: "anthropic",  label: "Anthropic (Claude)",        url: "https://console.anthropic.com/settings/keys",        free: false },
  ELEVENLABS_API_KEY: { env: "ELEVENLABS_API_KEY", provider: "elevenlabs", label: "ElevenLabs (voice)",        url: "https://elevenlabs.io/app/settings/api-keys",        free: false },
};

function emptyCfg() {
  return {
    profile_name: "default",
    dashboard_theme: "cyan",
    persona: {
      name: "", handle: "", language: "en", pronouns: "", age_range: "", origin: "", archetype: "",
      backstory: "",
      energy: "warm", humor_style: ["ironic", "observational"],
      profanity: "mild", formality: "casual", sentence_length: "short",
      catchphrases: [], running_gags: [], banned_words: [],
      extra_style_notes: "",
      strong_opinions: true, admit_uncertainty: true, break_fourth_wall: false,
      favorite_topics: [], taboo_topics: [],
      address_style: "by_name", reply_length: "snappy", react_to_highlights_hype: true, require_engagement: false, acknowledge_rate: 0.25,
      vision_first_person: true, vision_commentary_density: "balanced",
    },
    llm: { provider: "groq", model: "", temperature: 0.85, top_p: 0.95, max_tokens: 500, presence_penalty: 0.3, frequency_penalty: 0.4, vision_capable: false, ollama_base_url: "http://localhost:11434", ollama_keep_alive: "5m", vision_provider: "main", vision_model: "", vision_provider_ref: "", vision_openai_compatible_base_url: "", vision_openai_compatible_timeout: 30, vision_max_tokens: 200, provider_ref: "", openai_compatible_base_url: "", openai_compatible_timeout: 25 },
    tts: { provider: "fish", voice_id: "", sample_rate: 24000, el_model_id: "eleven_turbo_v2_5", el_stability: 0.45, el_similarity_boost: 0.75, el_style: 0.0, el_optimize_streaming_latency: 3, fish_latency_mode: "balanced", fish_chunk_length: 100, piper_model_path: "", piper_length_scale: 1.0, piper_noise_scale: 0.667, piper_noise_w: 0.8, kokoro_voice: "af_heart", kokoro_lang_code: "a", kokoro_speed: 1.0, openai_compatible_base_url: "", openai_compatible_model: "", openai_compatible_timeout: 30, openai_compatible_voice: "alloy", openai_compatible_speed: 1.0, openai_compatible_pcm_sample_rate: 24000, provider_ref: "", output_device: "" },
    vision: { enabled: false, source: "monitor", monitor_index: 1, interval_sec: 3.0, min_change_threshold: 8, max_edge_px: 768, startup_delay_sec: 5 },
    play: { enabled: false, game: "minecraft", goal: "Build a thriving Minecraft empire LIVE for an audience — gather, craft full gear, build, fight and explore. Make the journey entertaining, not a speedrun.", talk_from_agent: true, hide_chat: true, avoid_water: true },
    hearing: { enabled: false, window_sec: 5.0, model_size: "small", language: "", silence_threshold: 0.006, sound_event_threshold: 0.06, max_context_age_sec: 12.0, engine: "", openai_compatible_base_url: "", openai_compatible_model: "whisper-1", openai_compatible_timeout: 20, openai_compatible_prompt: "", provider_ref: "", loopback_device: "", speaker_id: { enabled: false, threshold: 0.68, unknown_threshold: 0.45, collect_other_voices: false } },
    speaker_id: { enabled: false, threshold: 0.68, unknown_threshold: 0.45, collect_other_voices: false },
    providers: [],
    chat: { youtube_enabled: false, twitch_enabled: false, kick_enabled: false, reply_probability: 0.35, min_reply_interval_sec: 8.0, max_message_age_sec: 45.0 },
    memory: { enabled: false, extractor: "openai_compatible", openai_compatible_base_url: "", model: "", timeout: 12.0, max_memories: 500, short_term_ttl_sec: 86400.0, promote_hits: 3, prompt_max_chars: 1200, janitor_interval_sec: 300.0, provider_ref: "", consolidate_threshold: 300, consolidate_batch: 40 },
    random_thoughts: { enabled: false, schedule: "fixed", interval_sec: 180.0, min_interval_sec: 90.0, max_interval_sec: 420.0, jitter: 0.35, style: "mix", generator: "main", openai_compatible_base_url: "", model: "", timeout: 12.0, seed_topics: [], seed_topics_text: "", memory_callback_chance: 0.35, memory_callback_kinds: ["long_term", "short_term"], provider_ref: "", only_when_quiet_sec: 20.0 },
    donations: { livepix_enabled: false, livepix_webhook_path: "/webhooks/livepix", livepix_enrich: true, livepix_verify_user_id: true, streamlabs_enabled: false, streamlabs_url: "https://sockets.streamlabs.com", streamlabs_reconnect_max_sec: 60.0, donation_reply_probability: 1.0, donation_cooldown_sec: 0.0 },
    topics: { mode: "ai_picks", topics: [], switch_min_sec: 90, switch_chance: 0.15 },
    orchestrator: {
      segment_target_sec: 12,
      dedupe_window: 8,
      dedupe_threshold: 0.78,
      prebuffer: true,
      session_duration_min: 0,
      outro_seconds: 30,
      recent_verbatim_turns: 24,
      summarize_every_n: 14,
      max_messages: 200,
      max_chars: 60000,
    },
    avatar: {
      enabled: false,
      vts_host: "127.0.0.1",
      vts_port: 8001,
      param_mouth_open: "MouthOpen",
      param_mouth_smile: "MouthSmile",
      param_face_x: "FaceAngleX",
      param_face_y: "FaceAngleY",
      param_face_z: "FaceAngleZ",
      param_eye_x: "EyeLeftX",
      param_eye_y: "EyeLeftY",
      param_brows: "Brows",
      lipsync_gain: 4.0,
      lipsync_ceiling: 0.85,
      lipsync_floor: 0.02,
      lipsync_attack: 0.65,
      lipsync_release: 0.30,
      speaking_smile: 0.15,
      param_mouth_form: "ParamMouthForm",
      enable_viseme_lipsync: true,
      viseme_smoothing: 0.35,
      enable_idle_motion: true,
      idle_sway_amplitude: 4.0,
      idle_sway_period_sec: 6.0,
      enable_eye_darts: true,
      eye_dart_interval_sec: 4.5,
      expr_happy: "", expr_surprised: "", expr_laughing: "",
      expr_angry: "", expr_sad: "", expr_thinking: "",
      expr_smug: "", expr_eyeroll: "", expr_confused: "",
      expr_hype: "", expr_deadpan: "",
    },
    captions: {
      enabled: false,
      path: "/captions",
      language: "",
      max_lines: 3,
      clear_delay_sec: 0.0,
      font_size: 40,
      background_opacity: 0.55,
      lowercase: false,
      show_avatar_name: false,
    },
  };
}

function formatHMS(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  seconds = Math.max(0, Math.round(seconds));
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

function app() {
  return {
    sections: SECTIONS,
    humorOptions: HUMOR_OPTIONS,
    section: "identity",

    cfg: emptyCfg(),
    profiles: [],
    activeProfile: "default",

    running: false,
    status: {},
    startBusy: false,
    startError: "",
    // Pre-start checklist (GET /api/preflight): shown when Start finds errors,
    // or on demand via the ⚠ badge. dismissed = "start anyway" this page load.
    preflight: [],
    preflightOpen: false,
    preflightLoading: false,
    preflightDismissed: false,
    logs: [],
    playLog: "",
    playBusy: false,
    // Kokoro one-click install badge (Voice tab + setup wizard panels).
    kokoro: { installed: false, kokoro_version: "", missing: [], installer_present: true, python_ok: true, python_version: "", python_range: "3.10–3.12", can_install: true, running: false, log: "" },
    kokoroBusy: false,
    _kokoroPoll: null,
    // Piper one-click install / voice download badge (Voice tab).
    piper: { installed: false, piper_tts_version: "", voices: [], voice_count: 0, voices_dir: "", installer_present: true, can_install: true, job: { running: false, kind: "", voice: "", returncode: null, log: "" } },
    piperBusy: false,
    _piperPoll: null,
    piperDownloadVoice: "",
    piperCatalogPanel: false,
    piperCatalog: { voices: [], loading: false, msg: "", query: "", lang: "" },
    // One saved-voice list per local engine (Piper / Kokoro), keyed by provider.
    localVoices: { piper: [], kokoro: [] },
    localVoiceName: { piper: "", kokoro: "" },
    localVoiceBusy: "",
    localVoiceMsg: "",
    _nextLogId: 1,
    _ws: null,

    // Durable memory (long-term store) UI state.
    ltm: { long_term: [], short_term: [], tags: [], stats: {} },
    ltmDraft: { text: "", tag: "", kind: "long_term", about: "" },
    ltmEditId: null,
    ltmEdit: { text: "", tag: "", kind: "long_term", about: "" },
    ltmQuery: "",
    ltmFilter: "all",
    memoryBusy: false,
    memoryMsg: "",
    // Live feed of facts the AI captured this session (newest first).
    memoryFeed: [],
    memoryUndone: {},      // id -> true after the user undid that capture
    _nextMemorySeq: 1,

    drawerOpen: true,
    saveMsg: "",
    testing: false,
    captionsStatus: { enabled: false, clients: 0, url: null },
    captionsTestText: "",
    captionsTestMsg: "",
    testResult: "",
    donationTest: { donor: "TestDonor", amount: 10, message: "manda o salve", msg: "" },
    voiceTestText: "",
    voiceTestMsg: "",
    // Voice Lab — saved voices (per profile) + provider-side cloning.
    voiceLab: { presets: [], providers: [] },
    voiceLabClone: { provider: "elevenlabs", name: "", description: "", samples: [] },
    voiceLabBusy: "",
    voiceLabMsg: "",
    voiceLabCloneMsg: "",
    voiceLabSaveName: "",
    voiceLabRecordSeconds: 8,
    voiceLabRecording: false,
    // Copy the whole saved-voice library (incl. local Piper/Kokoro voices) to
    // another profile so a character's voices travel with it.
    voiceExport: { target: "", msg: "" },
    // Backup file: download the saved voices as a .json, or merge one back in.
    voiceBackup: { msg: "" },
    // Names of the saved voices ticked in the list — the ⇪ copy selected /
    // ⤓ selected .json buttons act on these.
    voicePick: [],
    // Pull — the inverse of ⇪: bring voices FROM another profile's library into
    // this one. ``presets`` is that profile's list (read-only peek), ``names``
    // the voices ticked to bring over.
    voicePull: { source: "", presets: [], names: [], loading: false, msg: "" },
    // Mirror — two-way sync with another profile: ``plan`` is the diff the server
    // computed, ``policy`` what to do with shared names that differ.
    voiceMirror: { other: "", policy: "skip", plan: null, loading: false, msg: "" },
    // Voice partner — the profile this one should stay in sync with. Filled
    // from /api/voices/partners; drives the ⇄ badge in the profile picker.
    voicePartners: {},
    // A/B: two voices, one line, audio returned to this page only.
    voiceAb: { a: "", b: "", text: "", msg: "", busy: false },
    abClips: { a: null, b: null },
    abErrors: { a: "", b: "" },
    abDefaultText: AB_DEFAULT_TEXT,
    hearingTestBusy: false,
    hearingTestMsg: "",
    hearingTestResult: "",
    hearingSource: "",
    avatarTestExpr: "",
    avatarTestMsg: "",
    visionTestResult: "",
    visionTestMeta: "",
    // Vision-test overrides (blank = use the saved config). Lets a candidate
    // model be probed without committing it to the profile.
    visionTestProvider: "",
    visionTestModel: "",
    visionTestUrl: "",
    emotionSlots: EMOTION_SLOTS,
    avatarStatus: { enabled: false, connected: false },
    avatarHotkeys: [],
    avatarHotkeysFetched: false,
    // Secrets UI state. Raw values live ONLY inside `secretEdits`, keyed by env
    // name; cleared the moment we save / cancel so they don't linger in memory.
    secrets: [],
    // Dynamic provider blocks (API Keys page)
    providers: [],
    providerCategories: ["llm", "vision", "tts", "stt", "memory", "thoughts"],
    providerBusy: false,      providerMsg: "",
      healedMsg: "",
      partnerSyncMsg: "",      // what switching profile did to the voice libraries
      _partnerSyncTimer: null,
    // Persisted-load failure: when /api/config can't be read the UI would
    // otherwise show factory defaults as if they were the user's settings,
    // and a save in that state would OVERWRITE the profile with defaults.
    loadFailed: false,
    loadErrorMsg: "",
    // Voice-print speaker ID
    speakersInfo: { speakers: [], clips: [], enrolling: 0, active: false },
    enrollName: "",
    enrollReplace: false,
    enrollPending: 0,
    clipNames: {},
    voiceBusy: false,
    voiceMsg: "",
    // Per-voice prompt instruction ("this is my mom — treat her warmly")
    noteEditing: null,   // name of the speaker being edited
    noteDraft: "",
    secretEdits: {},     // env -> draft value (only set while editing)
    secretShow: {},      // env -> whether the input is unmasked while editing
    secretMsg: {},       // env -> "saved" / "tested ✓" / error text
    secretBusy: {},      // env -> bool (test in flight)

    // TTS voice discovery (Voice page, openai_compatible provider)
    ttsVoices: [],
    ttsVoicesEndpoint: "",
    ttsVoicesSource: "",
    ttsVoicesBusy: false,
    ttsVoicesMsg: "",
    // Output-device discovery (Voice page): where TTS audio is played.
    audioDevices: [],
    audioDevicesBusy: false,
    audioDevicesMsg: "",
    audioTestBusy: false,
    audioTestMsg: "",
    // Loopback devices for HEARING (what Wallie listens through) — separate
    // from the TTS output device above.
    hearingLoopbacks: [],
    hearingLoopTestBusy: false,
    hearingLoopTestMsg: "",
    // Saved device names that no longer exist on the system (checked right
    // after every save — empty strings mean no warning is showing).
    deviceWarn: { tts_output: "", loopback: "" },

    // Vision model discovery (Vision page, openai_compatible provider)
    vlModels: [],
    vlModelsEndpoint: "",
    vlModelsSource: "",
    vlModelsBusy: false,
    vlModelsMsg: "",

    // First-run setup wizard (additive — reuses config/secrets/start APIs, breaks nothing).
    wizard: { open: false, step: 1, path: "", busy: false, pickedKokoro: false, keyDrafts: {}, keyMsg: {}, keyBusy: {} },

    async init() {
      await this.loadProfiles();
      await this.loadConfig();
      this.loadAudioDevices(); // so the Output-device dropdown shows the saved pick
      this.loadHearingLoopbacks(); // so the Hearing loopback dropdown shows the saved pick
      this.checkSavedDevices(); // warn if a saved device name no longer exists
      await this.refreshStatus();
      await this.loadSecrets();
      await this.loadProviders();
      await this.loadSpeakers();
      await this.loadLtm();
      await this.loadVoiceLibrary();
      await this.loadLocalVoices();
      // Kokoro badge: probe now, and keep polling if an install is mid-flight.
      this.loadKokoroStatus().then(s => { if (s && s.running) this._pollKokoroInstall(); });
      // Piper badge: same treatment (install + voice downloads share one job).
      this.loadPiperStatus().then(s => { if (s && s.job && s.job.running) this._pollPiperJob(); });
      this.wizardMaybeOpen();
      this.connectWs();
      setInterval(() => { this.refreshStatus(); this.loadSpeakers(); }, 2000);
      setInterval(() => this.refreshAvatarStatus(), 3000);
      setInterval(() => this.refreshCaptions(), 3000);
      setInterval(() => this.loadLtm(), 5000);
    },

    // ----- TTS voice discovery (Voice page) -----
    async loadTtsVoices() {
      if (this.ttsVoicesBusy) return;
      this.ttsVoicesBusy = true;
      this.ttsVoicesMsg = "";
      try {
        const r = await fetch("/api/tts/voices", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ provider_ref: this.cfg.tts.provider_ref || "" }),
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
        this.ttsVoices = data.voices || [];
        this.ttsVoicesEndpoint = data.endpoint || "";
        this.ttsVoicesSource = data.source || "";
        this.ttsVoicesMsg = this.ttsVoices.length
          ? `${this.ttsVoices.length} voices`
          : "endpoint reachable but no voices listed — type the name manually";
      } catch (e) {
        this.ttsVoices = [];
        this.ttsVoicesMsg = "✗ " + (e.message || e);
      } finally {
        this.ttsVoicesBusy = false;
      }
    },

    // ----- Output-device discovery (Voice page) -----
    async loadAudioDevices() {
      if (this.audioDevicesBusy) return;
      this.audioDevicesBusy = true;
      this.audioDevicesMsg = "";
      try {
        const r = await fetch("/api/audio-devices");
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
        // Backend already merges the same endpoint across host APIs (MME
        // truncates names to 31 chars) and flags the system default.
        this.audioDevices = Array.isArray(data) ? data.filter(d => d && d.name) : [];
        // One-time migration: an old saved index becomes its device name, so
        // the dropdown matches and the value survives device renumbering.
        const cur = String(this.cfg.tts.output_device || "");
        if (cur && /^\d+$/.test(cur)) {
          const hit = (Array.isArray(data) ? data : []).find(d => String(d.index) === cur);
          if (hit) this.cfg.tts.output_device = hit.name;
        }
        this.audioDevicesMsg = this.audioDevices.length
          ? `${this.audioDevices.length} output device${this.audioDevices.length === 1 ? "" : "s"} found`
          : "no output devices found — check your audio drivers";
      } catch (e) {
        this.audioDevices = [];
        this.audioDevicesMsg = "✗ " + (e.message || e);
      } finally {
        this.audioDevicesBusy = false;
      }
    },

    async testAudioOutput() {
      if (this.audioTestBusy) return;
      this.audioTestBusy = true;
      this.audioTestMsg = "";
      try {
        const r = await fetch("/api/test/audio-output", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ device: this.cfg.tts.output_device || "" }),
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
        this.audioTestMsg = "✓ beep on " + (data.device || "default output");
      } catch (e) {
        this.audioTestMsg = "✗ " + (e.message || e);
      } finally {
        this.audioTestBusy = false;
        setTimeout(() => (this.audioTestMsg = ""), 6000);
      }
    },

    // ----- Hearing loopback device (Hearing page) -----
    async loadHearingLoopbacks() {
      try {
        const r = await fetch("/api/loopback-devices");
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
        this.hearingLoopbacks = Array.isArray(data) ? data : [];
      } catch (e) {
        this.hearingLoopbacks = [];
        this.hearingLoopTestMsg = "✗ " + (e.message || e);
      }
    },

    async testHearingLoopback() {
      if (this.hearingLoopTestBusy) return;
      this.hearingLoopTestBusy = true;
      this.hearingLoopTestMsg = "… listening (3s)";
      try {
        const r = await fetch("/api/test/hearing", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ seconds: 3, source: "system", device: this.cfg.hearing.loopback_device || "" }),
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
        this.hearingLoopTestMsg = "✓ heard: " + (data.text || "(only sound/silence)");
      } catch (e) {
        this.hearingLoopTestMsg = "✗ " + (e.message || e);
      } finally {
        this.hearingLoopTestBusy = false;
        setTimeout(() => (this.hearingLoopTestMsg = ""), 8000);
      }
    },

    async loadVisionModels() {
      if (this.vlModelsBusy) return;
      this.vlModelsBusy = true;
      this.vlModelsMsg = "";
      try {
        const r = await fetch("/api/vision/models", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ provider_ref: this.cfg.llm.vision_provider_ref || "" }),
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
        this.vlModels = data.models || [];
        this.vlModelsEndpoint = data.endpoint || "";
        this.vlModelsSource = data.source || "";
        this.vlModelsMsg = this.vlModels.length
          ? `${this.vlModels.length} models`
          : "endpoint reachable but no models listed — type the name manually";
      } catch (e) {
        this.vlModels = [];
        this.vlModelsMsg = "✗ " + (e.message || e);
      } finally {
        this.vlModelsBusy = false;
      }
    },

    // ----- dynamic provider blocks (API Keys page) -----
    async loadProviders() {
      try {
        const r = await fetch("/api/providers");
        if (r.ok) {
          const data = await r.json();
          this.providers = data.providers || [];
          this.providerCategories = data.categories || [];
        }
      } catch (e) { console.warn("loadProviders:", e); }
    },

    async refreshProviders() {
      // Same as loadProviders but updates the in-place state the cfg dropdowns
      // read; called after save so block ids/refs stay fresh everywhere.
      await this.loadProviders();
    },

    providerAdd() {
      // Local draft; persisted by providerSaveAll (one PUT, ids assigned server-side).
      this.providers.push({ id: "", name: "New provider", category: "llm", base_url: "", model: "" });
    },

    providerConfirmDelete(p) {
      // Unsaved draft (no id yet): just drop the row, nothing to clean on the server.
      if (!p.id) { this.providers.splice(this.providers.indexOf(p), 1); return; }
      const label = p.name || p.id;
      const slugEnv = _providerKeyEnvJs(p.id);
      // Pre-compute which subsystem settings reference this block, so the
      // confirm dialog can warn up-front (server re-checks on delete).
      const refs = this.clearProviderRefsLocal(p.id);
      let msg = `Delete provider "${label}"?\n\nIts API key (${slugEnv}) will also be removed from .env.`;
      if (refs.length) msg += `\n\nAlso cleared (were pointing at it): ${refs.join(", ")}.`;
      if (confirm(msg)) {
        this.providerDelete(p.id, refs);
      }
    },

    clearProviderRefsLocal(id) {
      // Mirrors config.PROVIDER_REF_FIELDS — must stay in sync.
      const REF_FIELDS = [
        ["llm", "provider_ref"], ["llm", "vision_provider_ref"],
        ["tts", "provider_ref"], ["hearing", "provider_ref"],
        ["memory", "provider_ref"], ["random_thoughts", "provider_ref"],
      ];
      const out = [];
      for (const [sec, field] of REF_FIELDS) {
        if (this.cfg?.[sec]?.[field] === id) out.push(`${sec}.${field}`);
      }
      return out;
    },

    async providerDelete(id, localRefs) {
      this.providerBusy = true;
      try {
        const r = await fetch(`/api/providers/${encodeURIComponent(id)}`, { method: "DELETE" });
        if (!r.ok) throw new Error((await r.json()).detail || `HTTP ${r.status}`);
        const data = await r.json().catch(() => ({}));
        this.providers = this.providers.filter(x => x.id !== id);
        const refs = (data.cleared_refs?.length ? data.cleared_refs : (localRefs || []));
        if (refs.length) {
          this.providerMsg = `deleted — also cleared: ${refs.join(", ")}`;
          await this.loadSecrets();
          await this.loadConfig();
          setTimeout(() => (this.providerMsg = ""), 6000);   // leave time to read the warning
        } else {
          this.providerMsg = "deleted";
          await this.loadSecrets();   // orphan key row disappears from the list
          await this.loadConfig();
          setTimeout(() => (this.providerMsg = ""), 2500);
        }
      } catch (e) {
        this.providerMsg = e.message || "delete failed";
      } finally {
        this.providerBusy = false;
      }
    },

    async providerSaveAll() {
      this.providerBusy = true;
      try {
        const r = await fetch("/api/providers", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(this.providers),
        });
        if (!r.ok) throw new Error((await r.json()).detail || `HTTP ${r.status}`);
        const data = await r.json();
        this.providers = data.providers || [];
        // Rows removed in this save may have cleared dangling refs server-side.
        const refs = data.cleared_refs || [];
        this.providerMsg = refs.length ? `saved — also cleared: ${refs.join(", ")}` : "saved";
        await this.loadSecrets();   // new blocks immediately get their key field
        await this.loadConfig();    // re-sync cfg (dropdowns read block ids)
      } catch (e) {
        this.providerMsg = e.message || "fail";
      } finally {
        this.providerBusy = false;
        // Ref-cleanup notices need longer on screen than a plain "saved".
        const ttl = String(this.providerMsg || "").includes("also cleared") ? 6000 : 1800;
        setTimeout(() => (this.providerMsg = ""), ttl);
      }
    },

    async providerTest(p) {
      // Persist first so the server knows the block (id assignment) before probing.
      await this.providerSaveAll();
      this.providerBusy = true;
      try {
        const r = await fetch(`/api/providers/${encodeURIComponent(p.id)}/test`, { method: "POST" });
        const data = await r.json();
        this.providerMsg = data.ok ? (data.preview ? `ok: ${data.preview}` : (data.note || "ok")) : (data.error || "fail");
      } catch (e) {
        this.providerMsg = e.message || "fail";
      } finally {
        this.providerBusy = false;
        setTimeout(() => this.providerMsg = "", 4000);
      }
    },

    staleRef(ref, category) {
      // A provider_ref is stale when it points to a block id that no longer
      // exists (deleted/renamed). The runtime silently falls back — this warns
      // so the user notices. Returns "" when everything is fine.
      if (!ref || this.providers.some(p => p.id === ref)) return "";
      return `points to deleted block “${ref}” — using fallback`;
    },

    suggestedRef(category) {
      // Mirror of the runtime fallback chain: first block of the category,
      // else the block named "default", else "" (legacy fields take over).
      const first = this.providers.find(p => p.category === category);
      if (first) return first.id;
      const def = this.providers.find(p => p.id === "default");
      return def ? def.id : "";
    },

    applySuggestedRef(refPath, category) {
      // refPath like "cfg.llm.vision_provider_ref"; category is passed
      // explicitly (hearing selects stt blocks — not parseable from the path).
      const sug = this.suggestedRef(category);
      if (!sug) return;
      const parts = refPath.split(".");
      let obj = this;
      for (const k of parts.slice(0, -1)) obj = obj[k];
      obj[parts[parts.length - 1]] = sug;
      this.save();
    },

    providerKeyEnv(p) {
      return _providerKeyEnvJs(p.id || p.name || "provider");
    },

    // ----- voice-print speaker ID -----
    async loadSpeakers() {
      try {
        const r = await fetch("/api/speakers");
        if (r.ok) this.speakersInfo = await r.json();
      } catch (e) { console.warn("loadSpeakers:", e); }
    },

    async enrollStart() {
      this.voiceBusy = true;
      try {
        const r = await fetch("/api/speakers/enroll/start", { method: "POST" });
        if (!r.ok) throw new Error((await r.json()).detail || `HTTP ${r.status}`);
        this.enrollPending = 0;
        await this.loadSpeakers();
      } catch (e) {
        this.voiceMsg = e.message;
        setTimeout(() => this.voiceMsg = "", 3000);
      } finally { this.voiceBusy = false; }
    },

    async enrollFinish() {
      this.voiceBusy = true;
      try {
        const name = (this.enrollName || "").trim();
        if (!name) throw new Error("type a name first");
        const r = await fetch("/api/speakers/enroll/finish", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, replace: this.enrollReplace === true }),
        });
        if (!r.ok) throw new Error((await r.json()).detail || `HTTP ${r.status}`);
        this.enrollPending = 0;
        this.enrollName = "";
        this.voiceMsg = "voice print saved";
        await this.loadSpeakers();
      } catch (e) {
        this.voiceMsg = e.message;
      } finally {
        this.voiceBusy = false;
        setTimeout(() => this.voiceMsg = "", 3000);
      }
    },

    async enrollCancel() {
      this.voiceBusy = true;
      try {
        await fetch("/api/speakers/enroll/cancel", { method: "POST" });
        this.enrollPending = 0;
        await this.loadSpeakers();
      } finally { this.voiceBusy = false; }
    },

    noteEdit(s) {
      this.noteEditing = s.name;
      this.noteDraft = s.note || "";
      this.$nextTick(() => {
        const el = document.querySelector('.note-edit input');
        if (el) el.focus();
      });
    },

    noteCancel() {
      this.noteEditing = null;
      this.noteDraft = "";
    },

    async noteSave(s) {
      this.voiceBusy = true;
      try {
        const r = await fetch(`/api/speakers/${encodeURIComponent(s.name)}/note`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ note: (this.noteDraft || "").trim() }),
        });
        if (!r.ok) throw new Error((await r.json()).detail || `HTTP ${r.status}`);
        this.noteEditing = null;
        this.noteDraft = "";
        await this.loadSpeakers();
      } catch (e) {
        this.voiceMsg = e.message;
        setTimeout(() => this.voiceMsg = "", 3000);
      } finally { this.voiceBusy = false; }
    },

    async speakerRemove(name) {
      this.voiceBusy = true;
      try {
        await fetch(`/api/speakers/${encodeURIComponent(name)}`, { method: "DELETE" });
        await this.loadSpeakers();
      } finally { this.voiceBusy = false; }
    },

    async clipEnroll(clipId) {
      const name = (this.clipNames[clipId] || "").trim();
      if (!name) return;
      this.voiceBusy = true;
      try {
        const r = await fetch(`/api/speakers/clips/${encodeURIComponent(clipId)}/enroll`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name }),
        });
        if (!r.ok) throw new Error((await r.json()).detail || `HTTP ${r.status}`);
        delete this.clipNames[clipId];
        await this.loadSpeakers();
      } catch (e) {
        this.voiceMsg = e.message;
        setTimeout(() => this.voiceMsg = "", 3000);
      } finally { this.voiceBusy = false; }
    },

    async clipDelete(clipId) {
      this.voiceBusy = true;
      try {
        await fetch(`/api/speakers/clips/${encodeURIComponent(clipId)}`, { method: "DELETE" });
        await this.loadSpeakers();
      } finally { this.voiceBusy = false; }
    },

    // ----- secrets -----
    async loadSecrets() {
      try {
        const r = await fetch("/api/secrets");
        const data = await r.json();
        this.secrets = data.secrets || [];
      } catch (e) { console.warn("loadSecrets:", e); }
    },

    secretsByKind(kind) {
      return (this.secrets || []).filter(s => s.kind === kind);
    },

    envToProvider(env) {
      // OPENAI_API_KEY → openai, ELEVENLABS_API_KEY → elevenlabs, etc.
      const m = {
        OPENAI_API_KEY: "openai",
        GROQ_API_KEY: "groq",
        OPENROUTER_API_KEY: "openrouter",
        ANTHROPIC_API_KEY: "anthropic",
        GEMINI_API_KEY: "gemini",
        FISH_API_KEY: "fish",
        ELEVENLABS_API_KEY: "elevenlabs",
      };
      return m[env] || "";
    },

    startEditSecret(env) {
      this.secretEdits[env] = "";
      this.secretShow[env] = false;
    },

    cancelEditSecret(env) {
      // Wipe the draft from memory immediately — no traces in component state.
      delete this.secretEdits[env];
      delete this.secretShow[env];
      delete this.secretMsg[env];
    },

    async saveSecret(env) {
      const value = this.secretEdits[env] ?? "";
      try {
        const r = await fetch("/api/secrets", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ env, value }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        // Wipe draft NOW — server confirmed the write, we don't need it anymore.
        delete this.secretEdits[env];
        delete this.secretShow[env];
        this.secretMsg[env] = value ? "saved" : "cleared";
        await this.loadSecrets();
        setTimeout(() => { delete this.secretMsg[env]; }, 1800);
      } catch (e) {
        this.secretMsg[env] = `error: ${e}`;
      }
    },

    async testProviderKey(provider, env) {
      this.secretBusy[env] = true;
      this.secretMsg[env] = "testing…";
      try {
        // Dynamic provider blocks get their own live probe.
        if (env && env.startsWith("PROVIDER_")) {
          const pid = env.removeprefix("PROVIDER_").removesuffix("_API_KEY").toLowerCase();
          const r = await fetch(`/api/providers/${encodeURIComponent(pid)}/test`, { method: "POST" });
          const data = await r.json();
          this.secretMsg[env] = data.ok ? `✓ ${data.preview || data.note || "ok"}` : `✗ ${data.error || "failed"}`;
          setTimeout(() => { delete this.secretMsg[env]; }, 4000);
          return;
        }
        const r = await fetch("/api/secrets/test", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ provider }),
        });
        const data = await r.json();
        this.secretMsg[env] = data.ok ? `✓ ${data.preview || "ok"}` : `✗ ${data.error || "failed"}`;
      } catch (e) {
        this.secretMsg[env] = `✗ ${e}`;
      } finally {
        this.secretBusy[env] = false;
        setTimeout(() => { delete this.secretMsg[env]; }, 4000);
      }
    },

    // ----- avatar -----
    async refreshAvatarStatus() {
      try {
        const r = await fetch("/api/avatar/status");
        this.avatarStatus = await r.json();
      } catch { this.avatarStatus = { enabled: false, connected: false }; }
    },

    async fetchHotkeys() {
      this.avatarHotkeysFetched = true;
      try {
        const r = await fetch("/api/avatar/hotkeys");
        const data = await r.json();
        this.avatarHotkeys = data.hotkeys || [];
        if (this.avatarHotkeys.length === 0) {
          this.avatarTestMsg = "no hotkeys returned (model has none defined?)";
          setTimeout(() => (this.avatarTestMsg = ""), 3000);
        }
      } catch (e) {
        this.avatarTestMsg = "fetch failed: " + e;
        setTimeout(() => (this.avatarTestMsg = ""), 3000);
      }
    },

    async testEmotion(slot) {
      try {
        const r = await fetch("/api/test/expression", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ expression: slot }),
        });
        const data = await r.json();
        this.avatarTestMsg = r.ok ? `→ ${slot}` : data.detail || "failed";
      } catch (e) {
        this.avatarTestMsg = "error: " + e;
      }
      setTimeout(() => (this.avatarTestMsg = ""), 1800);
    },

    async testRawHotkey(name) {
      try {
        await fetch("/api/test/expression", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ expression: name }),
        });
        this.avatarTestMsg = `fired: ${name}`;
      } catch (e) {
        this.avatarTestMsg = "error: " + e;
      }
      setTimeout(() => (this.avatarTestMsg = ""), 1800);
    },

    async testLook(x, y) {
      try {
        await fetch("/api/test/avatar_look", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ x, y, hold_sec: 0.8 }),
        });
      } catch {}
    },

    // Model suggestions for the vision-test override: the known vision-capable
    // models of the chosen provider ([] for endpoints we can't enumerate — the
    // field stays free-form so any model id can be typed in).
    visionTestModelOptions() {
      const provider = this.visionTestProvider || this.cfg.llm.provider;
      return (MODEL_OPTIONS[provider] || []).filter(m => m.vision);
    },

    async testVision() {
      const overrideBody = {
        provider: this.visionTestProvider,
        model: this.visionTestModel,
        base_url: this.visionTestUrl,
      };
      const usingOverride = !!(this.visionTestProvider || this.visionTestModel || this.visionTestUrl);
      // Only persist pending edits when testing the SAVED config — an override
      // probe must never write the candidate model into the profile.
      if (!usingOverride) await this.save();
      this.testing = true;
      this.visionTestResult = usingOverride
        ? `capturing screen + testing ${this.visionTestProvider || this.cfg.llm.provider}`
          + (this.visionTestModel ? `:${this.visionTestModel}` : "") + "..."
        : "capturing screen + sending to model...";
      this.visionTestMeta = "";
      try {
        const r = await fetch("/api/test/vision", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(usingOverride ? overrideBody : {}),
        });
        const data = await r.json();
        if (r.ok) {
          this.visionTestResult = data.text || "(empty response)";
          this.visionTestMeta =
            `${data.provider}:${data.model}${data.override ? " · override (not saved)" : ""}`
            + ` · frame ${data.frame_size?.join("×")} · ${(data.frame_bytes/1024).toFixed(1)} KB`;
        } else {
          this.visionTestResult = "ERROR: " + (data.detail || JSON.stringify(data));
        }
      } catch (e) {
        this.visionTestResult = "ERROR: " + e;
      } finally {
        this.testing = false;
      }
    },

    // ----- profiles -----
    async loadProfiles() {
      const r = await fetch("/api/profiles");
      const data = await r.json();
      this.profiles = data.profiles;
      this.activeProfile = data.active;
      await this.loadVoicePartners();
    },

    async switchProfile(name) {
      // The server keeps a partnered profile's saved voices in step with its
      // partner on activate, so this response carries what that did (or why it
      // refused) — reported below instead of syncing silently.
      const r = await fetch(`/api/profiles/${encodeURIComponent(name)}/activate`, { method: "PUT" });
      const data = await r.json().catch(() => ({}));
      await this.loadConfig();
      await this.loadProfiles();
      await this.loadVoiceLibrary();   // saved voices are per profile
      await this.loadVoicePartners();  // ...and so is the voice partner
      this.notePartnerSync(data.partner_sync);
    },

    async promptNewProfile() {
      const name = prompt("New profile name:");
      if (!name) return;
      await fetch("/api/profiles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      await this.loadProfiles();
      await this.loadConfig();
    },

    async promptCloneProfile() {
      const name = prompt(`Clone "${this.activeProfile}" as:`);
      if (!name) return;
      await fetch("/api/profiles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, clone_from: this.activeProfile }),
      });
      await this.loadProfiles();
      await this.loadConfig();
    },

    async confirmDeleteProfile() {
      if (!confirm(`Delete profile "${this.activeProfile}"? This cannot be undone.`)) return;
      await fetch(`/api/profiles/${encodeURIComponent(this.activeProfile)}`, { method: "DELETE" });
      await this.loadProfiles();
      await this.loadConfig();
    },

    // ----- config I/O -----
    async loadConfig() {
      this.loadFailed = false;
      this.loadErrorMsg = "";
      let fetched = null;
      try {
        const r = await fetch("/api/config");
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        fetched = await r.json();
      } catch (e) {
        // Never merge an error body (or nothing) over the empty cfg: the UI
        // would show factory defaults as if they were the user's saved
        // settings, and a save in that state would WIPE the profile.
        this.loadFailed = true;
        this.loadErrorMsg = String(e.message || e);
        return;
      }
      // Self-heal notice: the server blanked refs that pointed at deleted /
      // wrong-category blocks (typically a hand-edited YAML profile).
      if (Array.isArray(fetched.healed_refs) && fetched.healed_refs.length) {
        this.healedMsg = `Fixed stale provider references: ${fetched.healed_refs.join(", ")}`;
      }
      delete fetched.healed_refs;
      // Merge into empty to ensure newly added fields exist.
      const base = emptyCfg();
      this.cfg = deepMerge(base, fetched);
      // Theme rides on the profile: each persona keeps its own accent.
      this.applyTheme(this.cfg.dashboard_theme);
      // Textarea <-> list bridge for the thought seed pool.
      this.cfg.random_thoughts.seed_topics_text =
        (this.cfg.random_thoughts.seed_topics || []).join("\n");
    },

    // ----- dashboard accent theme (saved per profile) -----
    applyTheme(theme) {
      const t = THEME_OPTIONS.includes(theme) ? theme : "cyan";
      document.body.dataset.theme = t;
    },

    async setTheme(theme) {
      this.cfg.dashboard_theme = theme;
      this.applyTheme(theme);
      // Persist immediately onto the ACTIVE profile (the server merges this
      // single key into the saved config) — a theme pick shouldn't require
      // hitting Save, nor persist unrelated unsaved edits.
      try {
        const r = await fetch("/api/config", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dashboard_theme: theme }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
      } catch (e) {
        console.warn("setTheme:", e);
      }
    },

    dismissHealed() {
      this.healedMsg = "";
    },

    // ----- saved-device existence check (TTS output + hearing loopback) -----
    async checkSavedDevices() {
      try {
        const r = await fetch("/api/device-check", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            tts_output: this.cfg.tts.output_device || "",
            loopback: this.cfg.hearing.loopback_device || "",
          }),
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
        const names = (k) => (data[k] && data[k].exists === false) ? (this.cfg[k === "tts_output" ? "tts" : "hearing"][k === "tts_output" ? "output_device" : "loopback_device"] || "") : "";
        this.deviceWarn = { tts_output: names("tts_output"), loopback: names("loopback") };
      } catch (e) {
        // Check is best-effort: a failed probe never blocks saving.
        this.deviceWarn = { tts_output: "", loopback: "" };
      }
    },

    async save() {
      // Fold the seed-topics textarea back into the list before saving.
      if (this.cfg.random_thoughts) {
        this.cfg.random_thoughts.seed_topics = (this.cfg.random_thoughts.seed_topics_text || "")
          .split("\n").map(s => s.trim()).filter(Boolean);
      }
      try {
        const r = await fetch("/api/config", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(this.cfg),
        });
        if (!r.ok) {
          const detail = await r.json().catch(() => ({}));
          throw new Error(detail.detail || `HTTP ${r.status}`);
        }
        this.saveMsg = "saved";
        // Check AFTER persisting, against what is actually on disk now: the
        // saved TTS output / hearing loopback names may point at devices that
        // no longer exist (unplugged headset, Windows audio device change).
        this.checkSavedDevices();
      } catch (e) {
        // Keep the message on screen longer — a 1.4s "fail" flash is easy
        // to miss, and the user may believe the change was persisted.
        this.saveMsg = "✗ save failed: " + (e.message || e);
        setTimeout(() => (this.saveMsg = ""), 6000);
        return;
      }
      setTimeout(() => (this.saveMsg = ""), 1400);
    },

    // ----- durable memory (long-term store) -----
    toggleCallbackKind(kind) {
      const kinds = this.cfg.random_thoughts.memory_callback_kinds || [];
      const i = kinds.indexOf(kind);
      if (i === -1) kinds.push(kind);
      else if (kinds.length > 1) kinds.splice(i, 1);  // keep at least one tier
    },

    async loadLtm() {
      try {
        const r = await fetch("/api/longterm");
        if (r.ok) this.ltm = await r.json();
      } catch (e) { console.warn("loadLtm:", e); }
    },

    ltmVisible() {
      const q = (this.ltmQuery || "").toLowerCase();
      return [...(this.ltm.long_term || []), ...(this.ltm.short_term || [])]
        .filter(m => this.ltmFilter === "all" || m.kind === this.ltmFilter)
        .filter(m => !q
          || (m.text || "").toLowerCase().includes(q)
          || (m.tag || "").toLowerCase().includes(q));
    },

    async ltmAdd() {
      const text = this.ltmDraft.text.trim();
      if (!text) return;
      this.memoryBusy = true;
      try {
        const r = await fetch("/api/longterm", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, kind: this.ltmDraft.kind, tag: this.ltmDraft.tag, about: (this.ltmDraft.about || "").trim() }),
        });
        if (!r.ok) throw new Error((await r.json()).detail || `HTTP ${r.status}`);
        this.ltmDraft.text = "";
        this.ltmDraft.tag = "";
        await this.loadLtm();
      } catch (e) {
        this.memoryMsg = `error: ${e}`;
        setTimeout(() => (this.memoryMsg = ""), 2500);
      } finally { this.memoryBusy = false; }
    },

    ltmStartEdit(m) {
      this.ltmEditId = m.id;
      this.ltmEdit = { text: m.text, tag: m.tag || "", kind: m.kind, about: m.about || "" };
    },

    async ltmSave() {
      if (this.ltmEditId == null) return;
      this.memoryBusy = true;
      try {
        const r = await fetch(`/api/longterm/${this.ltmEditId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: this.ltmEdit.text,
            tag: this.ltmEdit.tag,
            about: this.ltmEdit.about || "",
            kind: this.ltmEdit.kind,
          }),
        });
        if (!r.ok) throw new Error((await r.json()).detail || `HTTP ${r.status}`);
        this.ltmEditId = null;
        await this.loadLtm();
      } catch (e) {
        this.memoryMsg = `error: ${e}`;
        setTimeout(() => (this.memoryMsg = ""), 2500);
      } finally { this.memoryBusy = false; }
    },

    async ltmRemove(id) {
      this.memoryBusy = true;
      try {
        await fetch(`/api/longterm/${id}`, { method: "DELETE" });
        await this.loadLtm();
      } finally { this.memoryBusy = false; }
    },

    async ltmClearAll() {
      if (!confirm("Delete ALL memories (long + short)? This cannot be undone.")) return;
      this.memoryBusy = true;
      try {
        await fetch("/api/longterm/clear", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
        });
        await this.loadLtm();
      } finally { this.memoryBusy = false; }
    },

    async captureMemoryNow() {
      this.memoryBusy = true;
      this.memoryMsg = "extracting…";
      try {
        const r = await fetch("/api/memory/capture-now", { method: "POST" });
        const d = await r.json();
        if (r.ok) {
          const facts = d.captured || [];
          this.memoryMsg = facts.length
            ? `captured ${facts.length}: ${facts.map(f => f.text).join(" | ")}`
            : "nothing new worth remembering";
          await this.loadLtm();
        } else {
          this.memoryMsg = d.detail || `HTTP ${r.status}`;
        }
      } catch (e) {
        this.memoryMsg = `error: ${e}`;
      } finally {
        this.memoryBusy = false;
        setTimeout(() => (this.memoryMsg = ""), 5000);
      }
    },

    async consolidateMemoryNow() {
      this.memoryBusy = true;
      this.memoryMsg = "consolidating…";
      try {
        const r = await fetch("/api/memory/consolidate-now", { method: "POST" });
        const d = await r.json();
        if (r.ok) {
          const { removed, added, skipped } = d;
          this.memoryMsg = skipped
            ? "not enough mergeable entries yet (need ≥ 2)"
            : `consolidated: ${removed} old memories → ${added} summaries`;
          await this.loadLtm();
        } else {
          this.memoryMsg = d.detail || `HTTP ${r.status}`;
        }
      } catch (e) {
        this.memoryMsg = `error: ${e}`;
      } finally {
        this.memoryBusy = false;
        setTimeout(() => (this.memoryMsg = ""), 5000);
      }
    },

    // ----- orchestrator -----
    async refreshStatus() {
      try {
        const r = await fetch("/api/status");
        this.status = await r.json();
        this.running = !!this.status.running;
      } catch { this.running = false; }
    },

    // ----- engagement-gate inspector (what was skipped + force a reply) -----
    gate: { open: false, loading: false, items: [], busyId: "", msg: "" },

    async openGateInspector() {
      this.gate.open = true;
      await this.gateRefresh();
    },
    async gateRefresh() {
      if (this.gate.loading) return;
      this.gate.loading = true;
      try {
        const r = await fetch("/api/gate/skipped");
        this.gate.items = r.ok ? await r.json() : [];
      } catch { this.gate.items = []; }
      finally { this.gate.loading = false; }
    },
    async gateForceReply(id) {
      if (this.gate.busyId) return;
      this.gate.busyId = id;
      this.gate.msg = "";
      try {
        const r = await fetch(`/api/gate/force-reply/${encodeURIComponent(id)}`, { method: "POST" });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
        this.gate.msg = "✓ queued as her very next turn";
        await this.gateRefresh();
      } catch (e) {
        this.gate.msg = "✗ " + (e.message || e);
      } finally {
        this.gate.busyId = "";
        setTimeout(() => (this.gate.msg = ""), 5000);
      }
    },

    async runPreflight() {
      this.preflightLoading = true;
      try {
        const r = await fetch("/api/preflight");
        this.preflight = r.ok ? await r.json() : [];
      } catch { this.preflight = []; }
      finally { this.preflightLoading = false; }
      return this.preflight;
    },
    preflightErrors() { return this.preflight.filter(p => p.level === "error"); },
    dismissPreflight() {
      this.preflightOpen = false;
      this.preflightDismissed = true;
      this.startError = "";
    },
    async start() {
      this.startBusy = true;
      this.startError = "";
      try {
        // Preflight: static config check. Errors open the checklist instead of
        // failing deep inside build_orchestrator with a cryptic exception.
        if (!this.preflightDismissed) {
          const issues = await this.runPreflight();
          if (issues.some(p => p.level === "error")) {
            this.preflightOpen = true;
            return;
          }
        }
        const r = await fetch("/api/start", { method: "POST" });
        if (!r.ok) {
          // Surface WHY the session refused to start (bad TTS model, missing
          // key, unreachable endpoint…) instead of a silently dead button.
          let detail = `HTTP ${r.status}`;
          try {
            const data = await r.json();
            detail = data.detail || detail;
          } catch {
            const txt = await r.text().catch(() => "");
            const m = txt.match(/(TTSError|LLMError|RuntimeError|ValueError)[:\s]+([^<\n]+)/);
            if (m) detail = m[2] || m[1];
          }
          this.startError = detail.slice(0, 240);
        }
      } catch (e) {
        this.startError = String(e);
      } finally {
        this.startBusy = false;
        await this.refreshStatus();
        if (this.running) this.startError = "";
      }
    },
    async stop()  { await fetch("/api/stop",  { method: "POST" }); await this.refreshStatus(); },

    // ----- donation test strip -----
    async testDonation(source) {
      this.testing = true;
      this.donationTest.msg = "queueing…";
      try {
        const r = await fetch("/api/test/donation", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            source,
            donor: this.donationTest.donor || "TestDonor",
            amount: this.donationTest.amount || 10,
            currency: "BRL",
            message: this.donationTest.message || "",
          }),
        });
        const d = await r.json();
        this.donationTest.msg = r.ok && d.ok
          ? `✓ queued (${source}) — Wallie will react if the queue path is live`
          : `✗ ${d.error || d.detail || "failed"}`;
      } catch (e) { this.donationTest.msg = `✗ ${e}`; }
      finally { this.testing = false; }
    },
    async testWebhookPath() {
      this.testing = true;
      this.donationTest.msg = "posting to webhook…";
      try {
        const r = await fetch("/api/donations/webhook-test", { method: "POST" });
        const d = await r.json();
        this.donationTest.msg = d.status === 200
          ? "✓ webhook validated and queued the event"
          : `✗ webhook answered ${d.status}: ${(d.body && d.body.error) || "error"}`;
      } catch (e) { this.donationTest.msg = `✗ ${e}`; }
      finally { this.testing = false; }
    },
    async testStreamlabsToken() {
      this.testing = true;
      this.donationTest.msg = "checking Streamlabs…";
      try {
        const r = await fetch("/api/test/streamlabs-connect", { method: "POST" });
        const d = await r.json();
        this.donationTest.msg = d.ok
          ? "✓ Streamlabs token valid — socket can connect"
          : `✗ ${d.error || "token invalid"}`;
      } catch (e) { this.donationTest.msg = `✗ ${e}`; }
      finally { this.testing = false; }
    },

    async installMinecraft() {
      this.playBusy = true; this.playLog = "Installing… downloading Fabric + mods, this can take a minute.";
      try {
        const r = await fetch("/api/play/install", { method: "POST" });
        const d = await r.json();
        this.playLog = (d.log || "done.").trim();
      } catch (e) { this.playLog = "Install failed: " + e; }
      this.playBusy = false;
    },
    async launchPlay() {
      await this.save();
      this.playBusy = true; this.playLog = "Launching Wallie Play… focus the Minecraft window. F8 stops it.";
      try {
        const r = await fetch("/api/play/launch", { method: "POST" });
        const d = await r.json();
        this.playLog = d.ok ? (d.msg || "launched.") : ("Could not launch: " + (d.detail || ""));
      } catch (e) { this.playLog = "Launch failed: " + e; }
      this.playBusy = false;
    },

    async loadKokoroStatus() {
      try {
        const r = await fetch("/api/tts/kokoro/status");
        if (!r.ok) return null;
        const d = await r.json();
        const job = d.install || {};
        this.kokoro = {
          installed: !!d.installed,
          kokoro_version: d.kokoro_version || "",
          missing: d.missing || [],
          installer_present: d.installer_present !== false,
          min_version: d.min_version || "",
          python_ok: d.python_ok !== false,
          python_version: d.python_version || "",
          python_range: d.python_range || "3.10–3.12",
          can_install: d.can_install !== false,
          running: !!job.running,
          returncode: job.returncode ?? null,
          log: job.log || "",
        };
        return this.kokoro;
      } catch (e) {
        return null;
      }
    },
    async installKokoro() {
      if (this.kokoroBusy) return;
      this.kokoroBusy = true;
      this.kokoro.log = "starting install…";
      try {
        const r = await fetch("/api/tts/kokoro/install", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            lang: this.cfg.tts.kokoro_lang_code || "a",
            voice: this.cfg.tts.kokoro_voice || "",
          }),
        });
        const d = await r.json().catch(() => ({}));
        if (!r.ok && r.status !== 409) {
          this.kokoro.log = "install failed: " + (d.detail || ("HTTP " + r.status));
          this.kokoroBusy = false;
          return;
        }
        await this.loadKokoroStatus();
        this._pollKokoroInstall();
      } catch (e) {
        this.kokoro.log = "install failed: " + e;
        this.kokoroBusy = false;
      }
    },
    _pollKokoroInstall() {
      if (this._kokoroPoll) clearInterval(this._kokoroPoll);
      this.kokoroBusy = true;
      this._kokoroPoll = setInterval(async () => {
        const s = await this.loadKokoroStatus();
        if (!s || !s.running) {
          clearInterval(this._kokoroPoll);
          this._kokoroPoll = null;
          this.kokoroBusy = false;
        }
      }, 2000);
    },

    // ----- Piper (optional local TTS) — install + per-voice downloads -----
    async loadPiperStatus() {
      try {
        const r = await fetch("/api/tts/piper/status");
        if (!r.ok) return null;
        const d = await r.json();
        this.piper = {
          installed: !!d.installed,
          piper_tts_version: d.piper_tts_version || "",
          voices: d.voices || [],
          voice_count: d.voice_count || 0,
          voices_dir: d.voices_dir || "",
          installer_present: d.installer_present !== false,
          can_install: d.can_install !== false,
          job: d.job || { running: false, kind: "", voice: "", returncode: null, log: "" },
        };
        return this.piper;
      } catch (e) {
        return null;
      }
    },
    async installPiper() {
      if (this.piperBusy) return;
      this.piperBusy = true;
      this.piper.job = { ...this.piper.job, log: "starting install…" };
      try {
        const r = await fetch("/api/tts/piper/install", { method: "POST" });
        const d = await r.json().catch(() => ({}));
        if (!r.ok && r.status !== 409) {
          this.piper.job = { ...this.piper.job, log: "install failed: " + (d.detail || ("HTTP " + r.status)) };
          this.piperBusy = false;
          return;
        }
        await this.loadPiperStatus();
        this._pollPiperJob();
      } catch (e) {
        this.piper.job = { ...this.piper.job, log: "install failed: " + e };
        this.piperBusy = false;
      }
    },
    async downloadPiperVoice() {
      const voice = (this.piperDownloadVoice || "").trim();
      if (!voice || this.piperBusy) return;
      this.piperBusy = true;
      this.piper.job = { ...this.piper.job, log: `downloading ${voice}…` };
      try {
        const r = await fetch("/api/tts/piper/download", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ voice }),
        });
        const d = await r.json().catch(() => ({}));
        if (!r.ok && r.status !== 409) {
          this.piper.job = { ...this.piper.job, log: "✗ " + (d.detail || ("HTTP " + r.status)) };
          this.piperBusy = false;
          return;
        }
        this.piperDownloadVoice = "";
        await this.loadPiperStatus();
        this._pollPiperJob();
      } catch (e) {
        this.piper.job = { ...this.piper.job, log: "✗ " + e };
        this.piperBusy = false;
      }
    },
    _pollPiperJob() {
      if (this._piperPoll) clearInterval(this._piperPoll);
      this.piperBusy = true;
      this._piperPoll = setInterval(async () => {
        const s = await this.loadPiperStatus();
        if (!s || !s.job || !s.job.running) {
          clearInterval(this._piperPoll);
          this._piperPoll = null;
          this.piperBusy = false;
          // A finished download adds a voice to voices/ — refresh the menu.
          if (s && s.job && s.job.kind === "download") this.loadPiperStatus();
        }
      }, 1000);
    },

    // Online catalogue of download-able voices (HuggingFace rhasspy/piper-voices).
    async loadPiperCatalog(refresh = false) {
      if (this.piperCatalog.loading) return;
      this.piperCatalog.loading = true;
      this.piperCatalog.msg = "loading the catalogue…";
      try {
        const r = await fetch("/api/tts/piper/catalog" + (refresh ? "?refresh=true" : ""));
        const d = await r.json().catch(() => ({}));
        if (!r.ok) {
          this.piperCatalog.msg = "✗ " + (d.detail || `HTTP ${r.status}`);
          return;
        }
        this.piperCatalog.voices = d.voices || [];
        this.piperCatalog.msg = `${this.piperCatalog.voices.length} voices available`;
      } catch (e) {
        this.piperCatalog.msg = "✗ " + e;
      } finally {
        this.piperCatalog.loading = false;
      }
    },
    piperCatalogLangs() {
      const seen = new Map();
      for (const v of this.piperCatalog.voices) {
        if (v.language && !seen.has(v.language)) seen.set(v.language, v.language_name || v.language);
      }
      return [...seen].map(([code, label]) => ({ code, label }))
        .sort((a, b) => a.label.localeCompare(b.label));
    },
    piperCatalogResults() {
      const q = (this.piperCatalog.query || "").trim().toLowerCase();
      const lang = this.piperCatalog.lang || "";
      const out = [];
      for (const v of this.piperCatalog.voices) {
        if (lang && v.language !== lang) continue;
        if (q) {
          const hay = `${v.name} ${v.voice || ""} ${v.language_name || ""}`.toLowerCase();
          if (!hay.includes(q)) continue;
        }
        out.push(v);
        if (out.length >= 80) break;   // cap the DOM; refine the search for more
      }
      return out;
    },
    downloadPiperCatalogVoice(v) {
      this.piperDownloadVoice = v.name;
      this.downloadPiperVoice();
    },
    fmtMB(bytes) {
      const n = Number(bytes) || 0;
      if (!n) return "—";
      return (n / (1024 * 1024)).toFixed(n >= 100 * 1024 * 1024 ? 0 : 1) + " MB";
    },

    // ----- Per-model saved voices (Piper / Kokoro) -----
    async loadLocalVoices() {
      try {
        const r = await fetch("/api/voices/local");
        if (!r.ok) return;
        const d = await r.json();
        const p = d.providers || {};
        this.localVoices = { piper: p.piper || [], kokoro: p.kokoro || [] };
      } catch {}
    },
    async saveLocalVoice(provider) {
      const name = (this.localVoiceName[provider] || "").trim();
      if (!name || this.localVoiceBusy) return;
      this.localVoiceBusy = provider;
      this.localVoiceMsg = "";
      try {
        const t = this.cfg.tts;
        const body = {
          name,
          voice_id: provider === "piper" ? (t.piper_model_path || "") : (t.kokoro_voice || ""),
          tts: provider === "piper"
            ? { piper_length_scale: t.piper_length_scale, piper_noise_scale: t.piper_noise_scale, piper_noise_w: t.piper_noise_w }
            : { kokoro_lang_code: t.kokoro_lang_code, kokoro_speed: t.kokoro_speed },
        };
        const r = await fetch(`/api/voices/local/${provider}`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) {
          this.localVoiceMsg = "✗ " + (d.detail || `HTTP ${r.status}`);
          return;
        }
        const p = d.providers || {};
        this.localVoices = { piper: p.piper || [], kokoro: p.kokoro || [] };
        this.localVoiceName[provider] = "";
        this.localVoiceMsg = `✓ saved “${name}” for ${provider}`;
      } catch (e) {
        this.localVoiceMsg = "✗ " + e;
      } finally {
        this.localVoiceBusy = "";
        setTimeout(() => (this.localVoiceMsg = ""), 6000);
      }
    },
    async applyLocalVoice(provider, voice) {
      Object.assign(this.cfg.tts, voice.tts || {});
      this.cfg.tts.provider = provider;
      if (provider === "piper" && voice.voice_id) this.cfg.tts.piper_model_path = voice.voice_id;
      if (provider === "kokoro" && voice.voice_id) this.cfg.tts.kokoro_voice = voice.voice_id;
      await this.save();
    },
    async deleteLocalVoice(provider, name) {
      if (!confirm(`Delete the saved ${provider} voice “${name}”?`)) return;
      try {
        const r = await fetch(`/api/voices/local/${provider}/${encodeURIComponent(name)}`, { method: "DELETE" });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) {
          this.localVoiceMsg = "✗ " + (d.detail || `HTTP ${r.status}`);
          return;
        }
        const p = d.providers || {};
        this.localVoices = { piper: p.piper || [], kokoro: p.kokoro || [] };
        this.localVoiceMsg = `✓ deleted “${name}”`;
      } catch (e) {
        this.localVoiceMsg = "✗ " + e;
      } finally {
        setTimeout(() => (this.localVoiceMsg = ""), 6000);
      }
    },

    async resetAudio() {
      try {
        await fetch("/api/audio/reset", { method: "POST" });
      } catch (e) { console.warn(e); }
    },

    // ----- captions panel -----
    async refreshCaptions() {
      try {
        const r = await fetch("/api/captions/status");
        this.captionsStatus = await r.json();
      } catch { this.captionsStatus = { enabled: false, clients: 0, url: null }; }
    },

    async testCaption() {
      this.testing = true;
      this.captionsTestMsg = "pushing…";
      try {
        const r = await fetch("/api/test/caption", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: this.captionsTestText || "this is how the caption overlay looks on stream",
            clear_after: true,
          }),
        });
        const d = await r.json();
        this.captionsTestMsg = r.ok && d.ok
          ? "✓ pushed — check the overlay page, it should clear itself"
          : (d.detail || "failed");
      } catch (e) { this.captionsTestMsg = "error: " + e; }
      finally {
        this.testing = false;
        setTimeout(() => (this.captionsTestMsg = ""), 3500);
      }
    },

    copyCaptionsUrl() {
      const url = this.captionsStatus?.url;
      if (!url) return;
      navigator.clipboard?.writeText(url).catch(() => {});
      this.captionsTestMsg = "URL copied — paste it in OBS → Browser Source";
      setTimeout(() => (this.captionsTestMsg = ""), 3000);
    },

    // ----- live who's-talking indicator (speaker ID) -----
    nowSpeakerLabel() {
      const sid = this.status.speaker_id;
      if (!sid || !sid.enabled) return "";
      const now = (sid.now || "").toLowerCase();
      if (!now) return "silent";
      if (now === "owner") return "Owner";
      if (now === "other") return "Someone else";
      if (now === "unknown") return "Unknown voice";
      return sid.now;   // enrolled name as-is
    },

    nowSpeakerClass() {
      const sid = this.status.speaker_id;
      if (!sid || !sid.enabled || !sid.now) return "";
      const now = (sid.now || "").toLowerCase();
      if (now === "owner") return "speaker-owner";
      if (now === "other" || now === "unknown") return "speaker-other";
      return "speaker-enrolled";
    },

    // ----- live log -----
    connectWs() {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      this._ws = new WebSocket(`${proto}://${location.host}/ws/events`);
      this._ws.onmessage = (ev) => {
        try {
          const entry = JSON.parse(ev.data);
          if (entry.type === "log") {
            entry.id = this._nextLogId++;
            this.logs.push(entry);
            if (this.logs.length > 400) this.logs.splice(0, this.logs.length - 400);
          } else if (entry.type === "memory") {
            this.onMemoryEvent(entry.data);
          }
        } catch {}
      };
      this._ws.onclose = () => setTimeout(() => this.connectWs(), 2000);
    },

    // ----- live memory capture feed -----
    onMemoryEvent(ev) {
      if (!ev || ev.id == null) return;
      ev.key = `${ev.id}-${this._nextMemorySeq++}`;   // same fact can re-fire (dedupe refresh)
      ev.ts = ev.ts || Date.now() / 1000;
      this.memoryFeed.unshift(ev);
      if (this.memoryFeed.length > 50) this.memoryFeed.splice(50);
    },

    async undoMemory(ev) {
      try {
        const r = await fetch(`/api/longterm/${ev.id}`, { method: "DELETE" });
        // 404 = already gone (deleted from the list) → still counts as undone.
        if (r.ok || r.status === 404) {
          this.memoryUndone[ev.id] = true;
          await this.loadLtm();
        }
      } catch (e) { console.warn("undoMemory:", e); }
    },

    // ----- test strip -----
    async testPersona(kind) {
      // Save first so the server reads the latest persona.
      await this.save();
      this.testing = true;
      this.testResult = "…";
      try {
        const r = await fetch("/api/test/persona", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ kind }),
        });
        const data = await r.json();
        this.testResult = data.text || "(no response)";
      } catch (e) {
        this.testResult = `error: ${e}`;
      } finally {
        this.testing = false;
      }
    },

    async speakTest() {
      if (!this.testResult) return;
      await this._voice(this.testResult);
    },

    async testVoice() {
      if (!this.voiceTestText) return;
      await this._voice(this.voiceTestText);
    },

    async testHearing(source) {
      if (this.hearingTestBusy) return;
      this.hearingTestBusy = true;
      this.hearingSource = source;
      this.hearingTestMsg = source === "mic"
        ? "recording 5s… speak now"
        : "recording 5s… play something on the machine";
      this.hearingTestResult = "";
      try {
        const r = await fetch("/api/test/hearing", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ seconds: 5, source }),
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          this.hearingTestMsg = "✗ " + (data.detail || `HTTP ${r.status}`);
          return;
        }
        this.hearingTestResult = data.text || "";
        this.hearingTestMsg = "✓ " + (data.note || "transcribed");
        if (data.hallucination) {
          this.hearingTestMsg += " — likely Whisper hallucination over silence/noise";
        } else if (!data.text) {
          this.hearingTestMsg += " — nothing recognizable was said (try again closer to the mic/speakers)";
        }
      } catch (e) {
        this.hearingTestMsg = "✗ " + e;
      } finally {
        this.hearingTestBusy = false;
        setTimeout(() => { this.hearingTestMsg = ""; }, 8000);
      }
    },

    async testExpression() {
      if (!this.avatarTestExpr) return;
      await this.save();
      this.testing = true;
      this.avatarTestMsg = "…";
      try {
        const r = await fetch("/api/test/expression", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ expression: this.avatarTestExpr }),
        });
        const data = await r.json();
        this.avatarTestMsg = data.ok ? "✓ triggered" : (data.detail || "failed");
      } catch (e) {
        this.avatarTestMsg = `error: ${e}`;
      } finally {
        this.testing = false;
        setTimeout(() => (this.avatarTestMsg = ""), 2500);
      }
    },

    async _voice(text) {
      await this.save();
      this.testing = true;
      this.voiceTestMsg = "";
      try {
        const r = await fetch("/api/test/voice", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          this.voiceTestMsg = "✗ " + (data.detail || `HTTP ${r.status}`);
          return;
        }
        this.voiceTestMsg = "✓ " + (data.note || "playing");
      } catch (e) {
        this.voiceTestMsg = "✗ " + e;
      } finally {
        this.testing = false;
        setTimeout(() => (this.voiceTestMsg = ""), 6000);
      }
    },

    // ----- Voice Lab — saved voices + cloning -----
    cloneProvider() {
      return this.voiceLab.providers.find(p => p.id === this.voiceLabClone.provider) || null;
    },

    async loadVoiceLibrary() {
      try {
        const r = await fetch("/api/voices/library");
        if (!r.ok) return;
        const data = await r.json();
        this.voiceLab.presets = data.presets || [];
        this.voiceLab.providers = data.providers || [];
        this._pruneVoicePick();
        // Preselect a provider the account can actually clone with.
        const current = this.cloneProvider();
        if (!current || !current.ready) {
          const usable = this.voiceLab.providers.find(p => p.ready);
          if (usable) this.voiceLabClone.provider = usable.id;
        }
      } catch {}
    },

    // Snapshot what the Voice page currently uses as a named, reusable voice.
    async saveCurrentVoice() {
      const name = (this.voiceLabSaveName || "").trim();
      if (!name || this.voiceLabBusy) return;
      const tts = { ...this.cfg.tts };
      delete tts.output_device;   // routes audio — not a property of a voice
      this.voiceLabBusy = "save";
      this.voiceLabMsg = "";
      try {
        const r = await fetch("/api/voices/library", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name,
            provider: this.cfg.tts.provider,
            voice_id: this.cfg.tts.voice_id || "",
            tts,
          }),
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          this.voiceLabMsg = "✗ " + (data.detail || `HTTP ${r.status}`);
          return;
        }
        this.voiceLab.presets = data.presets || this.voiceLab.presets;
        this._pruneVoicePick();
        this.voiceLabSaveName = "";
        this.voiceLabMsg = `✓ saved “${name}”`;
        this.loadVoicePartners();     // a new voice may have broken the parity
      } catch (e) {
        this.voiceLabMsg = "✗ " + e;
      } finally {
        this.voiceLabBusy = "";
        setTimeout(() => (this.voiceLabMsg = ""), 6000);
      }
    },

    // Writes the preset's settings into the config and persists them.
    async applyVoice(preset) {
      Object.assign(this.cfg.tts, preset.tts || {});
      if (preset.provider) this.cfg.tts.provider = preset.provider;
      if (preset.voice_id) this.cfg.tts.voice_id = preset.voice_id;
      await this.save();
    },

    async testVoicePreset(preset) {
      await this.applyVoice(preset);
      const text = (this.voiceTestText || "").trim()
        || "testing this voice — one, two, three.";
      this.voiceLabBusy = "test";
      this.voiceLabMsg = "… synthesizing with the saved voice";
      try {
        const r = await fetch("/api/test/voice", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        const data = await r.json().catch(() => ({}));
        this.voiceLabMsg = r.ok
          ? "✓ " + (data.note || "playing")
          : "✗ " + (data.detail || `HTTP ${r.status}`);
      } catch (e) {
        this.voiceLabMsg = "✗ " + e;
      } finally {
        this.voiceLabBusy = "";
        setTimeout(() => (this.voiceLabMsg = ""), 8000);
      }
    },

    // ----- Pull: bring voices FROM another profile (inverse of ⇪) -----
    async loadPullSource() {
      const src = (this.voicePull.source || "").trim();
      this.voicePull.presets = [];
      this.voicePull.names = [];
      this.voicePull.msg = "";
      if (!src || src === this.activeProfile) return;
      this.voicePull.loading = true;
      try {
        const r = await fetch(`/api/voices/library?profile=${encodeURIComponent(src)}`);
        const d = await r.json().catch(() => ({}));
        if (!r.ok) {
          this.voicePull.msg = "✗ " + (d.detail || `HTTP ${r.status}`);
          return;
        }
        this.voicePull.presets = d.presets || [];
        // Everything that profile has is ticked by default — untick to exclude.
        this.voicePull.names = this.voicePull.presets.map(p => p.name);
        this.voicePull.msg = this.voicePull.presets.length
          ? `“${src}” has ${this.voicePull.presets.length} saved voice(s) — untick any you don't want`
          : `“${src}” has no saved voices`;
      } catch (e) {
        this.voicePull.msg = "✗ " + e;
      } finally {
        this.voicePull.loading = false;
      }
    },

    pullPicked(name) {
      return this.voicePull.names.includes(name);
    },

    togglePullPick(name, on) {
      if (on && !this.pullPicked(name)) this.voicePull.names = [...this.voicePull.names, name];
      else if (!on) this.voicePull.names = this.voicePull.names.filter(n => n !== name);
    },

    async pullVoices() {
      const src = (this.voicePull.source || "").trim();
      if (this.voiceLabBusy || !src || !this.voicePull.names.length) return;
      const names = [...this.voicePull.names];
      this.voiceLabBusy = "pull";
      this.voicePull.msg = `pulling ${names.length} voice(s) from “${src}”…`;
      try {
        const { r, d, cancelled } = await this._mergeRequest("/api/voices/pull", {
          source_profile: src, names,
        });
        if (cancelled) {
          this.voicePull.msg = "kept this profile's voices — nothing was replaced";
          return;
        }
        if (!r.ok) {
          this.voicePull.msg = "✗ " + this._apiError(d, r.status);
          return;
        }
        this.voiceLab.presets = d.presets || this.voiceLab.presets;
        this._pruneVoicePick();
        const extra = d.replaced ? ` (${d.replaced} replaced)` : "";
        this.voicePull.msg = `✓ pulled ${d.copied} voice(s) from “${src}” — this profile now has ${d.total}${extra}`;
      } catch (e) {
        this.voicePull.msg = "✗ " + e;
      } finally {
        this.voiceLabBusy = "";
      }
    },

    // ----- Mirror: make two profiles hold the same voices, both ways -----
    async loadMirrorPlan() {
      const other = (this.voiceMirror.other || "").trim();
      this.voiceMirror.plan = null;
      this.voiceMirror.msg = "";
      if (!other || other === this.activeProfile) return;
      this.voiceMirror.loading = true;
      try {
        const r = await fetch(`/api/voices/mirror?other=${encodeURIComponent(other)}`);
        const d = await r.json().catch(() => ({}));
        if (!r.ok) {
          this.voiceMirror.msg = "✗ " + this._apiError(d, r.status);
          return;
        }
        this.voiceMirror.plan = d;
      } catch (e) {
        this.voiceMirror.msg = "✗ " + e;
      } finally {
        this.voiceMirror.loading = false;
      }
    },

    // One line per category of the diff, so the change is visible before it is applied.
    mirrorLines() {
      const p = this.voiceMirror.plan;
      if (!p) return [];
      const names = list => (list.length > 6
        ? ` (${list.slice(0, 6).join(", ")}…)` : ` (${list.join(", ")})`);
      const lines = [
        `→ ${p.other_profile}: ${p.to_other.length} to add${p.to_other.length ? names(p.to_other) : ""}`,
        `← ${p.this_profile}: ${p.to_this.length} to add${p.to_this.length ? names(p.to_this) : ""}`,
        `= ${p.identical.length} already identical`,
      ];
      if (p.conflicts.length) {
        lines.push(`! ${p.conflicts.length} differ: `
          + p.conflicts.slice(0, 4).map(c => `${c.name} (${c.this} vs ${c.other})`).join("; ")
          + (p.conflicts.length > 4 ? ` +${p.conflicts.length - 4} more` : ""));
      }
      return lines;
    },

    // Anything to do? With the "skip" policy a pile of conflicts alone changes nothing.
    mirrorPending() {
      const p = this.voiceMirror.plan;
      if (!p) return false;
      if (p.to_other.length || p.to_this.length) return true;
      return this.voiceMirror.policy !== "skip" && p.conflicts.length > 0;
    },

    async mirrorVoices() {
      const other = (this.voiceMirror.other || "").trim();
      if (this.voiceLabBusy || !this.mirrorPending()) return;
      this.voiceLabBusy = "mirror";
      this.voiceMirror.msg = `mirroring with “${other}”…`;
      try {
        const r = await fetch("/api/voices/mirror", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ other_profile: other, conflicts: this.voiceMirror.policy }),
        });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) {
          this.voiceMirror.msg = "✗ " + this._apiError(d, r.status);
          return;
        }
        this.voiceLab.presets = d.presets || this.voiceLab.presets;
        this._pruneVoicePick();
        this.voiceMirror.plan = d.plan || null;      // the diff AFTER applying
        const skipped = d.skipped ? `, ${d.skipped} differing left alone` : "";
        this.voiceMirror.msg = `✓ mirrored: ${d.to_other} → “${other}”, ${d.to_this} → this`
          + ` — now ${d.this_total} here and ${d.other_total} there${skipped}`;
        this.loadVoicePartners();     // the two libraries just changed
      } catch (e) {
        this.voiceMirror.msg = "✗ " + e;
      } finally {
        this.voiceLabBusy = "";
      }
    },

    // ----- Voice partner: the profile this one should stay in sync with -----
    // The badge in the profile picker warns while the two libraries differ. Its
    // state comes from the server (GET /api/voices/partners), computed from the
    // very same diff ⇄ mirror shows, so the badge can't disagree with the mirror.
    // What the switch route did to the voice libraries: said out loud, because
    // it is a write the user asked for only by naming a partner.
    notePartnerSync(sync) {
      clearTimeout(this._partnerSyncTimer);
      this.partnerSyncMsg = "";
      if (!sync || !sync.partner) return;
      if (sync.error) {
        this.partnerSyncMsg = `⚠ voices not synced with “${sync.partner}” on switch: ${sync.error}`;
      } else if (sync.synced) {
        const bits = [];
        if (sync.to_other) bits.push(`${sync.to_other} → “${sync.partner}”`);
        if (sync.to_this) bits.push(`${sync.to_this} → this profile`);
        const left = sync.skipped ? `, ${sync.skipped} differing left alone` : "";
        this.partnerSyncMsg = `⇄ auto-synced with “${sync.partner}” on switch: `
          + (bits.join(", ") || "nothing to copy") + left;
      } else {
        return;                      // already in step — the badge says so
      }
      this._partnerSyncTimer = setTimeout(() => (this.partnerSyncMsg = ""), 12000);
    },

    partnerState(name) {
      return this.voicePartners[name || this.activeProfile]
        || { partner: "", stale: false, out_of_sync: false, to_other: 0, to_this: 0, conflicts: 0 };
    },

    partnerIs(name) {
      return !!name && name !== this.activeProfile && this.partnerState().partner === name;
    },

    partnerBadge() {
      const s = this.partnerState();
      if (!s.partner) return { show: false, text: "", cls: "", title: "" };
      if (s.stale) {
        return { show: true, text: "⇄ partner missing", cls: "warn",
          title: `“${s.partner}” is no longer a profile — click to pick another voice partner in the Voice Lab.` };
      }
      if (!s.out_of_sync) {
        return { show: true, text: "⇄ in sync", cls: "ok",
          title: `Saved voices match “${s.partner}”. Click to open the diff in the Voice Lab.` };
      }
      const bits = [];
      if (s.to_this) bits.push(`${s.to_this} to pull`);
      if (s.to_other) bits.push(`${s.to_other} to copy`);
      if (s.conflicts) bits.push(`${s.conflicts} differing`);
      return { show: true, text: "⇄ out of sync", cls: "warn",
        title: `Saved voices differ from “${s.partner}”: ${bits.join(", ")}. Click to open the Voice Lab with the diff ready to mirror.` };
    },

    // The badge is a shortcut to the place that fixes the divergence: open the
    // Voice Lab with the partner already chosen as the mirror target, so the
    // diff (and the ⇄ mirror button's state) is there on arrival. A stale or
    // missing partner has nothing to select — the hint below the strip says so.
    openPartnerMirror() {
      this.section = "voicelab";
      const s = this.partnerState();
      if (s.partner && !s.stale && s.partner !== this.activeProfile) {
        this.voiceMirror.other = s.partner;
        this.voiceMirror.msg = "";
        this.loadMirrorPlan();
      }
      this.scrollToMirror();
    },

    // The section is only revealed a frame AFTER ``section`` flips (Alpine
    // defers the display change to requestAnimationFrame), so scrolling right
    // away would measure a still-hidden block and go nowhere. Wait for layout,
    // with a timer as the fallback for frames a backgrounded tab never paints.
    scrollToMirror() {
      const go = () => {
        const el = document.getElementById("voice-mirror");
        if (el && el.getBoundingClientRect().height > 0) el.scrollIntoView({ block: "center" });
      };
      this.$nextTick(() => requestAnimationFrame(() => requestAnimationFrame(go)));
      setTimeout(go, 300);
    },

    // Marks a profile in the dropdown itself, so divergence is visible before
    // you switch to it. Options can't hold markup — a glyph is all we get.
    profileOptionLabel(name) {
      const s = this.voicePartners[name];
      return s && s.partner && !s.stale && s.out_of_sync ? `${name} ⇄` : name;
    },

    partnerHint() {
      const other = (this.voiceMirror.other || "").trim();
      if (!other || other === this.activeProfile) return "";
      if (this.partnerIs(other)) {
        return `★ “${other}” is this profile's voice partner — the profile picker shows ⇄ while the two libraries differ.`;
      }
      const s = this.partnerState();
      if (s.partner) return `This profile's voice partner is “${s.partner}” — setting a new one replaces it.`;
      return `☆ Optional: make “${other}” this profile's voice partner and the picker warns whenever the two libraries drift apart.`;
    },

    async loadVoicePartners() {
      try {
        const r = await fetch("/api/voices/partners");
        if (!r.ok) return;
        const d = await r.json();
        this.voicePartners = d.profiles || {};
      } catch { /* the badge is advisory — never block a load on it */ }
    },

    async setVoicePartner() {
      const other = (this.voiceMirror.other || "").trim();
      if (!other || other === this.activeProfile) return;
      const partner = this.partnerIs(other) ? "" : other;   // the same button toggles off
      this.voiceLabBusy = "partner";
      try {
        const r = await fetch("/api/voices/partner", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ partner }),
        });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) {
          this.voiceMirror.msg = "✗ " + this._apiError(d, r.status);
          return;
        }
        await this.loadVoicePartners();
        this.voiceMirror.msg = partner
          ? `★ “${partner}” is now this profile's voice partner`
            + (d.out_of_sync ? " — the two libraries differ, ⇄ mirror to even them out" : " — the two libraries already match")
          : "☆ voice partner cleared";
      } catch (e) {
        this.voiceMirror.msg = "✗ " + e;
      } finally {
        this.voiceLabBusy = "";
      }
    },

    // Copies saved voices to the target profile. Pass ``names`` to copy just
    // those (the ⇪ button on a row, or the ticked selection); no names = all.
    async exportVoices(names) {
      const target = (this.voiceExport.target || "").trim();
      if (this.voiceLabBusy) return;
      if (!target) {
        this.voiceExport.msg = "✗ pick a destination profile in the strip below first";
        setTimeout(() => (this.voiceExport.msg = ""), 6000);
        return;
      }
      if (target === this.activeProfile) return;
      const picked = (Array.isArray(names) ? names : []).filter(Boolean);
      const one = picked.length === 1 ? picked : null;
      const many = picked.length > 1;
      this.voiceLabBusy = "export";
      this.voiceExport.msg = one
        ? `copying “${one[0]}”…`
        : many ? `copying ${picked.length} voices…` : "copying…";
      try {
        const body = { target_profile: target };
        if (picked.length) body.names = picked;
        const { r, d, cancelled } = await this._mergeRequest("/api/voices/export", body);
        if (cancelled) {
          this.voiceExport.msg = `kept “${target}” as it was — nothing was replaced`;
          return;
        }
        if (!r.ok) {
          this.voiceExport.msg = "✗ " + this._apiError(d, r.status);
          return;
        }
        const extra = d.replaced ? ` (${d.replaced} replaced)` : "";
        this.voiceExport.msg = one
          ? `✓ copied “${one[0]}” to “${target}” — it now has ${d.total} voice(s)${extra}`
          : many
            ? `✓ copied ${d.copied} of ${picked.length} selected voice(s) to “${target}” — it now has ${d.total}${extra}`
            : `✓ copied ${d.copied} voice(s) to “${target}” — it now has ${d.total}${extra}`;
      } catch (e) {
        this.voiceExport.msg = "✗ " + e;
      } finally {
        this.voiceLabBusy = "";
        setTimeout(() => (this.voiceExport.msg = ""), 8000);
      }
    },

    // ----- Selection of several saved voices -----
    voicePicked(name) {
      return this.voicePick.includes(name);
    },

    toggleVoicePick(name, on) {
      if (on && !this.voicePicked(name)) this.voicePick = [...this.voicePick, name];
      else if (!on) this.voicePick = this.voicePick.filter(n => n !== name);
    },

    allVoicesPicked() {
      return this.voiceLab.presets.length > 0
        && this.voicePick.length === this.voiceLab.presets.length;
    },

    pickAllVoices(on) {
      this.voicePick = on ? this.voiceLab.presets.map(p => p.name) : [];
    },

    // A ticked voice that no longer exists (deleted, or replaced by an import)
    // must not stay in the selection.
    _pruneVoicePick() {
      this.voicePick = this.voicePick.filter(
        n => this.voiceLab.presets.some(p => p.name === n));
    },

    // Error text for a failed response — ``detail`` is a string for most routes
    // and an object ({message, conflicts, plan}) for the voice-merge ones.
    _apiError(data, status) {
      const det = data && data.detail;
      if (typeof det === "string" && det) return det;
      if (det && typeof det === "object") return det.message || `HTTP ${status}`;
      return `HTTP ${status}`;
    },

    // Posts a voice merge (copy / pull / import). Same-named voices whose
    // settings differ come back as a 409 with the diff, so the user is asked
    // before anything is replaced; only then is the request re-sent with
    // ``overwrite``.
    async _mergeRequest(url, body) {
      const post = (payload) => fetch(url, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      let r = await post(body);
      let d = await r.json().catch(() => ({}));
      if (r.status !== 409 || !d.detail || typeof d.detail !== "object") return { r, d };

      const conflicts = d.detail.conflicts || [];
      const lines = conflicts.slice(0, 6)
        .map(c => `• ${c.name}: ${c.current} → ${c.incoming}`).join("\n");
      const more = conflicts.length > 6 ? `\n…and ${conflicts.length - 6} more` : "";
      if (!confirm(`${d.detail.message}\n\n${lines}${more}\n\nReplace them?`)) {
        return { cancelled: true };
      }
      r = await post({ ...body, overwrite: true });
      d = await r.json().catch(() => ({}));
      return { r, d };
    },

    // Filename-safe version of a voice or profile name for the downloaded backup.
    _jsonFileBase(label) {
      const s = String(label || "voices").trim()
        .replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[.\-]+|[.\-]+$/g, "");
      return s || "voices";
    },

    // Downloads saved voices as a portable .json file — the whole library, one
    // voice (the ⤓ button on a row) or the ticked ones. The server owns the format.
    async backupVoices(names) {
      if (this.voiceLabBusy) return;
      const picked = (Array.isArray(names) ? names : []).filter(Boolean);
      const one = picked.length === 1 ? picked : null;
      this.voiceLabBusy = "backup";
      this.voiceBackup.msg = one
        ? `packing “${one[0]}”…`
        : picked.length ? `packing ${picked.length} voices…` : "packing…";
      try {
        const q = picked.length
          ? "?names=" + encodeURIComponent(picked.join(","))
          : "";
        const r = await fetch("/api/voices/backup" + q);
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          this.voiceBackup.msg = "✗ " + (data.detail || `HTTP ${r.status}`);
          return;
        }
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = one
          ? `wallie-voice-${this._jsonFileBase(one[0])}.json`
          : picked.length
            ? "wallie-voices-selected.json"
            : `wallie-voices-${this._jsonFileBase(this.activeProfile)}.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 2000);
        const n = (data.presets || []).length;
        this.voiceBackup.msg = one
          ? `✓ downloaded “${one[0]}” as a .json file`
          : picked.length
            ? `✓ downloaded ${n} selected voice(s) as a .json file`
            : `✓ downloaded ${n} saved voice(s) as a .json file`;
      } catch (e) {
        this.voiceBackup.msg = "✗ " + e;
      } finally {
        this.voiceLabBusy = "";
        setTimeout(() => (this.voiceBackup.msg = ""), 8000);
      }
    },

    // Reads a backup .json in the browser and asks the server to merge it into
    // this profile's library (same-named voices are replaced).
    async importVoices(event) {
      const file = (event.target.files || [])[0];
      event.target.value = "";       // the same file can be re-picked
      if (!file || this.voiceLabBusy) return;
      this.voiceLabBusy = "import";
      this.voiceBackup.msg = `reading ${file.name}…`;
      try {
        const content = await file.text();
        const { r, d, cancelled } = await this._mergeRequest("/api/voices/import", { content });
        if (cancelled) {
          this.voiceBackup.msg = `kept this profile's voices — nothing from ${file.name} was applied`;
          return;
        }
        if (!r.ok) {
          this.voiceBackup.msg = "✗ " + this._apiError(d, r.status);
          return;
        }
        this.voiceLab.presets = d.presets || this.voiceLab.presets;
        this.voiceLab.providers = d.providers || this.voiceLab.providers;
        this._pruneVoicePick();
        const extra = d.replaced ? ` (${d.replaced} replaced)` : "";
        this.voiceBackup.msg = `✓ imported ${d.imported} voice(s) from ${file.name} — ${d.total} saved now${extra}`;
      } catch (e) {
        this.voiceBackup.msg = "✗ could not read " + file.name + ": " + e;
      } finally {
        this.voiceLabBusy = "";
        setTimeout(() => (this.voiceBackup.msg = ""), 10000);
      }
    },

    async deleteVoice(name) {
      if (!confirm(`Delete the saved voice “${name}”?\n\nThe voice itself stays on the provider.`)) return;
      try {
        const r = await fetch(`/api/voices/library/${encodeURIComponent(name)}`, { method: "DELETE" });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          this.voiceLabMsg = "✗ " + (data.detail || `HTTP ${r.status}`);
          return;
        }
        this.voiceLab.presets = data.presets || [];
        this._pruneVoicePick();
        this.voiceLabMsg = `✓ deleted “${name}”`;
        this.loadVoicePartners();     // deleting can also break the parity
      } catch (e) {
        this.voiceLabMsg = "✗ " + e;
      } finally {
        setTimeout(() => (this.voiceLabMsg = ""), 6000);
      }
    },

    // Reference clips are read in the browser and posted as base64 — the
    // dashboard has no multipart parser, and the server builds the provider's
    // multipart upload itself.
    _readBase64(file) {
      return new Promise((resolve, reject) => {
        const rd = new FileReader();
        rd.onload = () => resolve(String(rd.result).split(",", 2)[1] || "");
        rd.onerror = () => reject(new Error("could not read " + file.name));
        rd.readAsDataURL(file);
      });
    },

    async onCloneFiles(event) {
      const files = Array.from(event.target.files || []);
      event.target.value = "";   // same file can be re-picked
      this.voiceLabCloneMsg = "";
      for (const f of files) {
        if (this.voiceLabClone.samples.length >= 5) {
          this.voiceLabCloneMsg = "✗ at most 5 samples";
          break;
        }
        if (f.size > 12 * 1024 * 1024) {
          this.voiceLabCloneMsg = `✗ ${f.name} is larger than 12 MB`;
          continue;
        }
        try {
          const data = await this._readBase64(f);
          this.voiceLabClone.samples.push({ filename: f.name, data });
        } catch (e) {
          this.voiceLabCloneMsg = "✗ " + e;
        }
      }
    },

    async recordCloneSample() {
      if (this.voiceLabRecording) return;
      if (this.voiceLabClone.samples.length >= 5) {
        this.voiceLabCloneMsg = "✗ at most 5 samples — remove one first";
        return;
      }
      const seconds = Math.max(2, Math.min(30, Number(this.voiceLabRecordSeconds) || 8));
      this.voiceLabRecording = true;
      this.voiceLabCloneMsg = `🎙 recording ${seconds}s — speak clearly into the default mic…`;
      try {
        const r = await fetch("/api/voices/record", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ seconds }),
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          this.voiceLabCloneMsg = "✗ " + (data.detail || `HTTP ${r.status}`);
          return;
        }
        this.voiceLabClone.samples.push({ filename: `mic-${data.seconds}s.wav`, data: data.wav_b64 });
        this.voiceLabCloneMsg = `✓ recorded ${data.seconds}s — audition it below`;
      } catch (e) {
        this.voiceLabCloneMsg = "✗ " + e;
      } finally {
        this.voiceLabRecording = false;
      }
    },

    async createClonedVoice() {
      const { provider, name, description, samples } = this.voiceLabClone;
      if (!name.trim() || !samples.length || this.voiceLabBusy) return;
      this.voiceLabBusy = "clone";
      this.voiceLabCloneMsg = `🧬 creating “${name.trim()}” at ${provider} — this can take a moment…`;
      try {
        const r = await fetch("/api/voices/clone", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ provider, name: name.trim(), description, samples }),
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          this.voiceLabCloneMsg = "✗ " + (data.detail || `HTTP ${r.status}`);
          return;
        }
        this.voiceLab.presets = data.presets || this.voiceLab.presets;
        this.voiceLabCloneMsg =
          `✓ voice created · ${data.provider} : ${data.voice_id} — saved to the library above`;
        this.voiceLabClone.samples = [];
        this.voiceLabClone.name = "";
        this.voiceLabClone.description = "";
      } catch (e) {
        this.voiceLabCloneMsg = "✗ " + e;
      } finally {
        this.voiceLabBusy = "";
      }
    },

    // ----- Voice Lab A/B — same line, two voices, clips side by side -----
    abLabel(side) {
      return (side === "a" ? this.voiceAb.a : this.voiceAb.b) || "current Voice page settings";
    },

    // Nothing is saved and nothing plays through the output device: the route
    // hands the audio back so two clips can be compared (and replayed).
    async _previewVoice(name) {
      const text = (this.voiceAb.text || "").trim() || AB_DEFAULT_TEXT;
      let body = { text };
      if (name) {
        body.name = name;
      } else {
        // Compare against what the Voice page currently shows — including
        // edits that haven't been saved yet.
        const tts = { ...this.cfg.tts };
        delete tts.output_device;
        body = {
          ...body,
          provider: this.cfg.tts.provider,
          voice_id: this.cfg.tts.voice_id || "",
          tts,
        };
      }
      const r = await fetch("/api/voices/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        throw new Error(`${name || "current settings"}: ` + (data.detail || `HTTP ${r.status}`));
      }
      return {
        url: "data:audio/wav;base64," + data.wav_b64,
        source: name || "current Voice page settings",
        provider: data.provider,
        voice_id: data.voice_id,
        bytes: data.bytes,
        seconds: data.seconds,
        decoded: data.decoded,
      };
    },

    async compareVoices() {
      if (this.voiceAb.busy) return;
      this.voiceAb.busy = true;
      this.voiceAb.msg = "synthesizing both voices…";
      this.abClips = { a: null, b: null };
      this.abErrors = { a: "", b: "" };
      try {
        // allSettled, not all: a voice the provider refuses (missing key, plan
        // limits) must not throw away the clip that DID synthesize — it was
        // already billed, and half a comparison still tells you something.
        const res = await Promise.allSettled([
          this._previewVoice(this.voiceAb.a),
          this._previewVoice(this.voiceAb.b),
        ]);
        const sides = ["a", "b"];
        res.forEach((r, i) => {
          const side = sides[i];
          if (r.status === "fulfilled") this.abClips[side] = r.value;
          else this.abErrors[side] = r.reason?.message || String(r.reason);
        });
        const ok = sides.filter(s => this.abClips[s]).length;
        const failed = sides.filter(s => this.abErrors[s]);
        this.voiceAb.msg = ok === 2
          ? "✓ both clips ready — play A → B, or listen side by side"
          : failed.length
            ? `✗ ${failed.join(" and ").toUpperCase()} failed — see the card below`
            : "✗ no clip could be synthesized";
      } finally {
        this.voiceAb.busy = false;
      }
    },

    async playAbClip(side) {
      const el = this.$refs[side === "a" ? "abAudioA" : "abAudioB"];
      if (!el) return;
      el.currentTime = 0;
      try { await el.play(); } catch {}
    },

    // Back-to-back playback is what makes the choice obvious — one click, A then B.
    async playAbSequence() {
      const a = this.$refs.abAudioA;
      const b = this.$refs.abAudioB;
      if (!a) return;
      if (b) {
        a.onended = () => {
          a.onended = null;
          b.currentTime = 0;
          b.play().catch(() => {});
        };
      }
      a.currentTime = 0;
      try { await a.play(); } catch {}
    },

    // ----- helpers -----
    toggleInList(list, value) {
      const i = list.indexOf(value);
      if (i >= 0) list.splice(i, 1);
      else list.push(value);
    },

    kokoroLangs: KOKORO_LANGS,

    // Voice ids valid for the currently selected Kokoro language.
    kokoroVoiceOptions() {
      return KOKORO_VOICES[this.cfg.tts.kokoro_lang_code] || [];
    },

    // A voice id from another language silently mispronounces (or fails at
    // synth time), so switching language moves the pick onto that language's
    // default voice.
    onKokoroLangChange() {
      const voices = this.kokoroVoiceOptions();
      if (voices.length && !voices.includes(this.cfg.tts.kokoro_voice)) {
        this.cfg.tts.kokoro_voice = voices[0];
      }
    },

    modelOptions() {
      const provider = this.cfg.llm.provider;
      if (provider === "ollama") return [];
      const models = (MODEL_OPTIONS[provider] || []).slice();
      if (this.cfg.llm.model && !models.some(m => m.id === this.cfg.llm.model)) {
        models.unshift({ id: this.cfg.llm.model, label: this.cfg.llm.model, vision: this.cfg.llm.vision_capable, custom: true });
      }
      return models;
    },

    onModelSelect() {
      const models = MODEL_OPTIONS[this.cfg.llm.provider] || [];
      const selected = models.find(m => m.id === this.cfg.llm.model);
      if (selected) this.cfg.llm.vision_capable = selected.vision;
    },

    onProviderChange() {
      const models = MODEL_OPTIONS[this.cfg.llm.provider] || [];
      if (models.length > 0 && !models.some(m => m.id === this.cfg.llm.model)) {
        this.cfg.llm.model = models[0].id;
        this.cfg.llm.vision_capable = models[0].vision;
      }
    },

    // ---------- first-run setup wizard ----------
    wizardMaybeOpen() {
      try { if (localStorage.getItem("wallie_setup_done")) return; } catch {}
      const hasLLMKey = (this.secrets || []).some(s => s.kind === "llm" && s.is_set);
      if (!hasLLMKey) { this.wizard.step = 1; this.wizard.open = true; }
    },
    openWizard() { this.wizard.step = 1; this.wizard.open = true; },
    closeWizard() { this.wizard.open = false; },
    finishWizard() {
      try { localStorage.setItem("wallie_setup_done", "1"); } catch {}
      this.wizard.open = false;
    },
    wizardBack() { if (this.wizard.step > 1) this.wizard.step--; },

    async wizardPickProfile(name) {
      this.wizard.busy = true;
      try { await this.switchProfile(name); } finally { this.wizard.busy = false; }
      this.wizard.step = 2;
    },
    wizardStartFresh() { this.wizard.step = 2; },

    async wizardChoosePath(path) {
      const P = WIZARD_PATHS[path];
      if (!P) return;
      this.wizard.path = path;
      this.cfg.llm.provider = P.llm;
      this.cfg.llm.model = P.model;
      this.cfg.llm.vision_capable = true;
      let tts = P.tts;
      // Free path default is Piper, but if Kokoro is already installed use the
      // better local voice instead of leaving it on the manual-download one.
      if (path === "free") {
        const s = this.kokoro.installed ? this.kokoro : await this.loadKokoroStatus();
        if (s && s.installed) tts = "kokoro";
      }
      this.wizard.pickedKokoro = tts === "kokoro";
      this.cfg.tts.provider = tts;
      this.wizard.busy = true;
      try { await this.save(); } finally { this.wizard.busy = false; }
      this.wizard.step = 3;
    },

    wizardKeys() {
      return (WIZARD_PATHS[this.wizard.path]?.keys || []).map(env => WIZARD_KEYMETA[env]).filter(Boolean);
    },
    wizardKeyIsSet(env) {
      return (this.secrets || []).some(s => s.env === env && s.is_set);
    },
    wizardPathIsFree() { return this.wizard.path === "free"; },
    async wizardSaveKey(env) {
      const value = this.wizard.keyDrafts[env] ?? "";
      if (!value) return;
      this.wizard.keyBusy[env] = true;
      this.wizard.keyMsg[env] = "saving…";
      try {
        const r = await fetch("/api/secrets", {
          method: "PUT", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ env, value }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        delete this.wizard.keyDrafts[env];
        await this.loadSecrets();
        const meta = WIZARD_KEYMETA[env];
        const tr = await fetch("/api/secrets/test", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ provider: meta.provider }),
        });
        const td = await tr.json();
        this.wizard.keyMsg[env] = td.ok ? "✓ working" : `✗ ${td.error || "saved — test failed"}`;
      } catch (e) {
        this.wizard.keyMsg[env] = `✗ ${e}`;
      } finally {
        this.wizard.keyBusy[env] = false;
      }
    },
    wizardKeysReady() {
      const keys = WIZARD_PATHS[this.wizard.path]?.keys || [];
      return keys.length > 0 && keys.every(env => this.wizardKeyIsSet(env));
    },
    async wizardFinishAndStart() {
      this.wizard.busy = true;
      try { await this.save(); await this.start(); } finally { this.wizard.busy = false; }
      this.finishWizard();
    },

    formatHMS,
  };
}

// Chip input reusable component.
function chipInput(getList) {
  return {
    draft: "",
    model() { return getList() || []; },
    add() {
      const v = (this.draft || "").trim();
      if (!v) return;
      const list = getList();
      if (!list.includes(v)) list.push(v);
      this.draft = "";
    },
    remove(i) { getList().splice(i, 1); },
  };
}

// Deep merge: keys in 'override' replace keys in 'base'; arrays are taken whole from override.
function deepMerge(base, override) {
  if (override === null || override === undefined) return base;
  if (typeof base !== "object" || typeof override !== "object" || Array.isArray(base) || Array.isArray(override)) {
    return override;
  }
  const out = { ...base };
  for (const k of Object.keys(override)) {
    if (k in base && typeof base[k] === "object" && !Array.isArray(base[k]) && base[k] !== null) {
      out[k] = deepMerge(base[k], override[k]);
    } else {
      out[k] = override[k];
    }
  }
  return out;
}
