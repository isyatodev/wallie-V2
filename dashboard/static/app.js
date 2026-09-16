// =====================================================================
// Wallie dashboard — Alpine component
// =====================================================================

const SECTIONS = [
  { id: "identity",    label: "Identity",     ico: "🪪" },
  { id: "personality", label: "Personality",  ico: "🎭" },
  { id: "voice",       label: "Voice",        ico: "🎙" },
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
    { id: "gemini-2.5-pro", label: "Gemini 2.5 Pro", vision: true },
    { id: "gemini-2.5-flash", label: "Gemini 2.5 Flash", vision: true },
    { id: "gemini-2.0-flash", label: "Gemini 2.0 Flash", vision: true },
  ],
  ollama: [],
};

// First-run wizard: budget path -> provider config + required keys.
const WIZARD_PATHS = {
  free:    { llm: "gemini",    model: "gemini-2.5-flash",                            tts: "piper",      keys: ["GEMINI_API_KEY"] },
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
    persona: {
      name: "", handle: "", language: "en", pronouns: "", age_range: "", origin: "", archetype: "",
      backstory: "",
      energy: "warm", humor_style: ["ironic", "observational"],
      profanity: "mild", formality: "casual", sentence_length: "short",
      catchphrases: [], running_gags: [], banned_words: [],
      extra_style_notes: "",
      strong_opinions: true, admit_uncertainty: true, break_fourth_wall: false,
      favorite_topics: [], taboo_topics: [],
      address_style: "by_name", reply_length: "snappy", react_to_highlights_hype: true,
      vision_first_person: true, vision_commentary_density: "balanced",
    },
    llm: { provider: "groq", model: "", temperature: 0.85, top_p: 0.95, max_tokens: 500, presence_penalty: 0.3, frequency_penalty: 0.4, vision_capable: false, ollama_base_url: "http://localhost:11434", ollama_keep_alive: "5m", vision_provider: "main", vision_model: "", vision_provider_ref: "", vision_openai_compatible_base_url: "", vision_openai_compatible_timeout: 30, vision_max_tokens: 200, provider_ref: "", openai_compatible_base_url: "", openai_compatible_timeout: 25 },
    tts: { provider: "fish", voice_id: "", sample_rate: 24000, el_model_id: "eleven_turbo_v2_5", el_stability: 0.45, el_similarity_boost: 0.75, el_style: 0.0, fish_latency_mode: "balanced", fish_chunk_length: 100, piper_model_path: "", piper_length_scale: 1.0, kokoro_voice: "af_heart", kokoro_lang_code: "a", kokoro_speed: 1.0, openai_compatible_base_url: "", openai_compatible_model: "", openai_compatible_timeout: 30, openai_compatible_voice: "alloy", openai_compatible_speed: 1.0, openai_compatible_pcm_sample_rate: 24000, provider_ref: "" },
    vision: { enabled: false, source: "monitor", monitor_index: 1, interval_sec: 3.0, min_change_threshold: 8, max_edge_px: 768, startup_delay_sec: 5 },
    play: { enabled: false, game: "minecraft", goal: "Build a thriving Minecraft empire LIVE for an audience — gather, craft full gear, build, fight and explore. Make the journey entertaining, not a speedrun.", talk_from_agent: true, hide_chat: true, avoid_water: true },
    hearing: { enabled: false, window_sec: 5.0, model_size: "small", language: "", silence_threshold: 0.006, sound_event_threshold: 0.06, max_context_age_sec: 12.0, engine: "", openai_compatible_base_url: "", openai_compatible_model: "whisper-1", openai_compatible_timeout: 20, openai_compatible_prompt: "", provider_ref: "", speaker_id: { enabled: false, threshold: 0.68, unknown_threshold: 0.45, collect_other_voices: false } },
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
    avatarTestExpr: "",
    avatarTestMsg: "",
    visionTestResult: "",
    visionTestMeta: "",
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
    providerBusy: false,
    providerMsg: "",
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

    // First-run setup wizard (additive — reuses config/secrets/start APIs, breaks nothing).
    wizard: { open: false, step: 1, path: "", busy: false, keyDrafts: {}, keyMsg: {}, keyBusy: {} },

    async init() {
      await this.loadProfiles();
      await this.loadConfig();
      await this.refreshStatus();
      await this.loadSecrets();
      await this.loadProviders();
      await this.loadSpeakers();
      await this.loadLtm();
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

    providerRemove(i) {
      this.providers.splice(i, 1);
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
        this.providerMsg = "saved";
        await this.loadSecrets();   // new blocks immediately get their key field
        await this.loadConfig();    // re-sync cfg (dropdowns read block ids)
      } catch (e) {
        this.providerMsg = e.message || "fail";
      } finally {
        this.providerBusy = false;
        setTimeout(() => (this.providerMsg = ""), 1800);
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

    providerKeyEnv(p) {
      const slug = (p.id || p.name || "provider").toLowerCase().replace(/[^a-z0-9_]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 40) || "provider";
      return "PROVIDER_" + slug.toUpperCase() + "_API_KEY";
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

    async testVision() {
      await this.save();
      this.testing = true;
      this.visionTestResult = "capturing screen + sending to model...";
      this.visionTestMeta = "";
      try {
        const r = await fetch("/api/test/vision", { method: "POST" });
        const data = await r.json();
        if (r.ok) {
          this.visionTestResult = data.text || "(empty response)";
          this.visionTestMeta =
            `${data.provider}:${data.model} · frame ${data.frame_size?.join("×")} · ${(data.frame_bytes/1024).toFixed(1)} KB`;
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
    },

    async switchProfile(name) {
      await fetch(`/api/profiles/${encodeURIComponent(name)}/activate`, { method: "PUT" });
      await this.loadConfig();
      await this.loadProfiles();
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
      const r = await fetch("/api/config");
      const fetched = await r.json();
      // Merge into empty to ensure newly added fields exist.
      const base = emptyCfg();
      this.cfg = deepMerge(base, fetched);
      // Textarea <-> list bridge for the thought seed pool.
      this.cfg.random_thoughts.seed_topics_text =
        (this.cfg.random_thoughts.seed_topics || []).join("\n");
    },

    async save() {
      // Fold the seed-topics textarea back into the list before saving.
      if (this.cfg.random_thoughts) {
        this.cfg.random_thoughts.seed_topics = (this.cfg.random_thoughts.seed_topics_text || "")
          .split("\n").map(s => s.trim()).filter(Boolean);
      }
      const r = await fetch("/api/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(this.cfg),
      });
      this.saveMsg = r.ok ? "saved" : "fail";
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

    // ----- helpers -----
    toggleInList(list, value) {
      const i = list.indexOf(value);
      if (i >= 0) list.splice(i, 1);
      else list.push(value);
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
      this.cfg.tts.provider = P.tts;
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
