// ─── Dictation config ────────────────────────────────────

/// Which Whisper model to use for STT. Larger = more accurate but slower.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WhisperModel {
    Tiny,
    TinyEn,
    Base,
    BaseEn,
    Small,
    SmallEn,
    Medium,
}

impl WhisperModel {
    /// Filename of the GGML model weights (downloaded by `theword init`).
    pub fn filename(&self) -> &'static str {
        match self {
            Self::Tiny    => "ggml-tiny.bin",
            Self::TinyEn  => "ggml-tiny.en.bin",
            Self::Base    => "ggml-base.bin",
            Self::BaseEn  => "ggml-base.en.bin",
            Self::Small   => "ggml-small.bin",
            Self::SmallEn => "ggml-small.en.bin",
            Self::Medium  => "ggml-medium.bin",
        }
    }

    pub fn download_url(&self) -> String {
        format!(
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
            self.filename()
        )
    }
}

impl Default for WhisperModel {
    fn default() -> Self {
        Self::BaseEn
    }
}

/// WebRTC VAD aggressiveness. Higher = more aggressive at filtering non-speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadMode {
    Quality,
    LowBitrate,
    Aggressive,
    VeryAggressive,
}

impl Default for VadMode {
    fn default() -> Self {
        Self::Aggressive
    }
}

/// Whether to transcribe text or interpret as an agent command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DictationMode {
    /// Always clean and type the transcribed text.
    Transcribe,
    /// Always treat input as an agent command.
    Command,
    /// Let the LLM decide: type text or execute a tool.
    Auto,
}

impl Default for DictationMode {
    fn default() -> Self {
        Self::Auto
    }
}

/// Where cleaned text is delivered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMethod {
    /// Simulate typing via enigo.
    Type,
    /// Write to system clipboard.
    Clipboard,
    /// Both: write clipboard then type.
    Both,
}

impl Default for OutputMethod {
    fn default() -> Self {
        Self::Type
    }
}

/// A keyboard hotkey definition.
#[derive(Debug, Clone)]
pub struct HotkeyConfig {
    /// Key code string as recognised by rdev (e.g. "AltGr", "RightAlt", "F9").
    pub key: String,
    /// Hold to talk (true) vs toggle (false).
    pub hold_to_talk: bool,
}

impl Default for HotkeyConfig {
    fn default() -> Self {
        Self {
            key: "AltGr".to_string(),
            hold_to_talk: true,
        }
    }
}

/// All dictation-specific configuration for theword.
#[derive(Debug, Clone)]
pub struct DictationConfig {
    // ── STT ───────────────────────────────────────────────
    /// Whisper model to use.
    pub whisper_model: WhisperModel,
    /// Path to the GGML model file. If None, defaults to `~/.theword/models/<filename>`.
    pub whisper_model_path: Option<String>,
    /// BCP-47 language hint for Whisper (e.g. "en"). None = auto-detect.
    pub language: Option<String>,

    // ── VAD ───────────────────────────────────────────────
    /// WebRTC VAD aggressiveness mode.
    pub vad_mode: VadMode,
    /// Stop recording after this many milliseconds of consecutive silence.
    pub silence_threshold_ms: u64,
    /// Discard clips shorter than this many milliseconds of speech.
    pub min_speech_ms: u64,

    // ── Hotkey ────────────────────────────────────────────
    pub hotkey: HotkeyConfig,

    // ── LLM post-processing ───────────────────────────────
    /// Run the transcription through a local LLM for cleanup.
    pub rewrite_enabled: bool,
    /// Ollama model name for the rewrite pass.
    pub rewrite_model: String,

    // ── Behaviour ─────────────────────────────────────────
    pub mode: DictationMode,
    pub output_method: OutputMethod,
    /// Persist vocabulary corrections and writing patterns to the graph.
    pub learn_corrections: bool,
    /// Number of graph nodes fetched for the context briefing passed to the rewrite LLM.
    pub briefing_max_nodes: usize,
}

impl Default for DictationConfig {
    fn default() -> Self {
        Self {
            whisper_model: WhisperModel::default(),
            whisper_model_path: None,
            language: Some("en".to_string()),
            vad_mode: VadMode::default(),
            silence_threshold_ms: 700,
            min_speech_ms: 300,
            hotkey: HotkeyConfig::default(),
            rewrite_enabled: true,
            rewrite_model: "qwen2.5:1.5b".to_string(),
            mode: DictationMode::default(),
            output_method: OutputMethod::default(),
            learn_corrections: true,
            briefing_max_nodes: 12,
        }
    }
}

// ─── Core agent config ───────────────────────────────────

/// Runtime configuration for a CortexEmbedded instance.
#[derive(Debug, Clone)]
pub struct Config {
    /// Message count threshold before context compaction triggers.
    pub compaction_threshold: usize,
    /// Maximum agent loop iterations before forced stop.
    pub max_iterations: usize,
    /// Seconds between background decay sweeps.
    pub decay_interval_secs: u64,
    /// Minimum cosine similarity to create a `RelatesTo` auto-link edge.
    pub auto_link_cosine_threshold: f64,
    /// Minimum cosine similarity (with negation pattern) to flag as contradiction.
    pub contradiction_cosine_threshold: f64,
    /// When the HNSW linear buffer exceeds this count, trigger a full rebuild.
    pub hnsw_rebuild_threshold: usize,
    /// Maximum entries in the embedding LRU cache.
    pub embedding_cache_size: usize,
    /// Default number of nearest-neighbours returned by recall().
    pub default_recall_top_k: usize,
    /// Default BFS depth for graph walk during recall().
    pub default_graph_depth: usize,
    /// Lambda for recency weighting: exp(-lambda * hours_since_access).
    pub decay_lambda: f64,
    /// Number of HNSW neighbours to fetch for auto-link analysis.
    pub auto_link_candidates: usize,
    /// Number of most-recent session nodes (UserInput + Fact) always included
    /// in a chat turn's briefing, regardless of semantic similarity.
    pub session_recency_window: usize,
    /// Enable the bash/shell execution tool.
    pub bash_enabled: bool,
    /// Maximum seconds a bash command can run before being killed.
    pub bash_timeout_secs: u64,
    /// Maximum bytes of command output returned to the LLM.
    pub bash_max_output_bytes: usize,
    /// Shell command prefixes that are always blocked (case-insensitive substring match).
    pub bash_blocked_patterns: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            compaction_threshold: 20,
            max_iterations: 10,
            decay_interval_secs: 60,
            auto_link_cosine_threshold: 0.75,
            contradiction_cosine_threshold: 0.85,
            hnsw_rebuild_threshold: 200,
            embedding_cache_size: 10_000,
            default_recall_top_k: 10,
            default_graph_depth: 2,
            decay_lambda: 0.01,
            auto_link_candidates: 20,
            session_recency_window: 7,
            bash_enabled: true,
            bash_timeout_secs: 30,
            bash_max_output_bytes: 10_000,
            bash_blocked_patterns: vec![
                "rm -rf /".into(),
                "mkfs".into(),
                "dd if=".into(),
                ":(){:|:&};:".into(),
                "shutdown".into(),
                "reboot".into(),
                "halt".into(),
                "init 0".into(),
                "init 6".into(),
            ],
        }
    }
}
