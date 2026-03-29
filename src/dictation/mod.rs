/// Dictation orchestration for theword.
///
/// Ties together: audio capture → VAD → Whisper STT → graph memory recall
/// → LLM rewrite/command dispatch → keyboard output.
///
/// The main entry point is [`DictationEngine`], which is constructed once at
/// startup and either called directly (`dictate_once`) or runs as a persistent
/// hotkey-listening daemon (`run_listen_loop`).
use std::sync::Arc;
use std::time::Duration;

use enigo::{Direction, Enigo, Keyboard, Settings};
use rdev::{listen, Event, EventType, Key as RdevKey};
use tokio::sync::mpsc;

use crate::audio::{AudioCapture, to_vad_mode};
use crate::config::{DictationConfig, DictationMode, OutputMethod};
use crate::db::queries;
use crate::error::{CortexError, Result};
use crate::llm::LlmClient;
use crate::stt::WhisperHandle;
use crate::types::*;
use crate::CortexEmbedded;

// ─── LLM rewrite prompt ──────────────────────────────────

const REWRITE_SYSTEM: &str = "\
You are a dictation assistant. The user has just spoken and the raw transcript \
is provided below. Your job is to clean it up and return JSON.\n\n\
Rules:\n\
- Fix punctuation and capitalisation.\n\
- Remove filler words: um, uh, er, like, you know, sort of, kind of.\n\
- Do NOT add content, change meaning, or be creative.\n\
- If the user is clearly issuing a command (e.g. 'open terminal', \
  'press escape', 'make that formal', 'select all'), output:\n\
  {\"type\":\"command\",\"intent\":\"<short intent>\",\"original\":\"<raw text>\"}\n\
- Otherwise output:\n\
  {\"type\":\"text\",\"content\":\"<cleaned text>\"}\n\n\
Return only valid JSON. No markdown fences, no explanation.\n\n\
Context about this user (use to correct domain vocabulary, names, jargon):\n";

// ─── DictationResult ────────────────────────────────────

#[derive(Debug)]
pub enum DictationResult {
    /// Text was transcribed and should be typed.
    Text(String),
    /// A command was detected; intent describes what to do.
    Command { intent: String, original: String },
    /// Utterance was too short or silent — nothing to do.
    Silent,
}

// ─── DictationEngine ────────────────────────────────────

/// The core dictation engine. Holds all resources needed for the full pipeline.
pub struct DictationEngine {
    cortex: Arc<CortexEmbedded>,
    whisper: Arc<WhisperHandle>,
    llm: Option<Arc<dyn LlmClient>>,
    config: DictationConfig,
}

impl DictationEngine {
    pub fn new(
        cortex: Arc<CortexEmbedded>,
        whisper: Arc<WhisperHandle>,
        llm: Option<Arc<dyn LlmClient>>,
        config: DictationConfig,
    ) -> Self {
        Self { cortex, whisper, llm, config }
    }

    /// Record one utterance and run the full pipeline.
    ///
    /// Blocks the calling thread while recording (up to `timeout`).
    pub async fn dictate_once(&self, timeout: Duration) -> Result<DictationResult> {
        // 1. Capture audio
        let vad_mode = to_vad_mode(&self.config.vad_mode);
        let silence_ms = self.config.silence_threshold_ms;
        let min_speech_ms = self.config.min_speech_ms;
        let capture = AudioCapture::new(vad_mode, silence_ms, min_speech_ms);

        let utterance = tokio::task::spawn_blocking(move || {
            capture.record_utterance(timeout)
        })
        .await
        .map_err(|e| CortexError::Audio(format!("spawn_blocking: {e}")))??;

        let utterance = match utterance {
            Some(u) => u,
            None => return Ok(DictationResult::Silent),
        };

        // 2. Transcribe
        let whisper = self.whisper.clone();
        let samples = utterance.samples.clone();
        let raw_text = tokio::task::spawn_blocking(move || whisper.transcribe(&samples))
            .await
            .map_err(|e| CortexError::Stt(format!("spawn_blocking: {e}")))??;

        if raw_text.trim().is_empty() {
            return Ok(DictationResult::Silent);
        }

        // 3. LLM rewrite (if enabled)
        let result = if self.config.rewrite_enabled {
            if let Some(ref llm) = self.llm {
                self.rewrite(&raw_text, llm.as_ref()).await?
            } else {
                // No LLM configured — just use raw transcript
                DictationResult::Text(raw_text.clone())
            }
        } else {
            DictationResult::Text(raw_text.clone())
        };

        // 4. Persist DictationTurn to graph
        if self.config.learn_corrections {
            let cleaned = match &result {
                DictationResult::Text(t) => t.clone(),
                DictationResult::Command { original, .. } => original.clone(),
                DictationResult::Silent => String::new(),
            };
            self.persist_turn(&raw_text, &cleaned).await?;
        }

        // 5. Dispatch output
        self.dispatch(&result).await?;

        Ok(result)
    }

    /// Run the hotkey listen loop. Blocks forever (call from a dedicated thread).
    ///
    /// Spawns a background thread for rdev's blocking `listen()` call and
    /// communicates via a channel. When the configured hotkey is pressed
    /// (hold-to-talk) or toggled, triggers `dictate_once`.
    pub async fn run_listen_loop(self: Arc<Self>) -> Result<()> {
        let (tx, mut rx) = mpsc::channel::<HotkeyEvent>(8);
        let hotkey = self.config.hotkey.clone();

        // rdev::listen is blocking — run it on a dedicated OS thread
        std::thread::spawn(move || {
            let key_name = hotkey.key.clone();
            let hold_to_talk = hotkey.hold_to_talk;
            let mut recording = false;

            let tx_clone = tx.clone();
            if let Err(e) = listen(move |event: Event| {
                let matched = matches_hotkey(&event, &key_name);
                if matched {
                    match event.event_type {
                        EventType::KeyPress(_) if hold_to_talk && !recording => {
                            recording = true;
                            let _ = tx_clone.blocking_send(HotkeyEvent::StartRecording);
                        }
                        EventType::KeyRelease(_) if hold_to_talk && recording => {
                            recording = false;
                            let _ = tx_clone.blocking_send(HotkeyEvent::StopRecording);
                        }
                        EventType::KeyPress(_) if !hold_to_talk => {
                            recording = !recording;
                            if recording {
                                let _ = tx_clone.blocking_send(HotkeyEvent::StartRecording);
                            } else {
                                let _ = tx_clone.blocking_send(HotkeyEvent::StopRecording);
                            }
                        }
                        _ => {}
                    }
                }
            }) {
                eprintln!("[hotkey] rdev listen error: {e:?}");
            }
        });

        println!("theword listening. Hold {} to dictate.", self.config.hotkey.key);

        while let Some(event) = rx.recv().await {
            match event {
                HotkeyEvent::StartRecording => {
                    print!("[recording] ");
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                }
                HotkeyEvent::StopRecording => {
                    println!("processing...");
                    let engine = self.clone();
                    tokio::spawn(async move {
                        match engine.dictate_once(Duration::from_secs(30)).await {
                            Ok(DictationResult::Text(t)) => println!("[typed] {t}"),
                            Ok(DictationResult::Command { intent, .. }) => {
                                println!("[command] {intent}")
                            }
                            Ok(DictationResult::Silent) => println!("[silent]"),
                            Err(e) => eprintln!("[error] {e}"),
                        }
                    });
                }
            }
        }

        Ok(())
    }

    // ─── Private helpers ─────────────────────────────────

    async fn rewrite(&self, raw: &str, llm: &dyn LlmClient) -> Result<DictationResult> {
        // Build context briefing from graph memory
        let briefing = self
            .cortex
            .briefing(raw, self.config.briefing_max_nodes)
            .await
            .map(|b| b.context_doc)
            .unwrap_or_default();

        let system = format!("{REWRITE_SYSTEM}{briefing}");
        let messages = vec![
            Message::system(system),
            Message::user(format!("Raw transcript: {raw}")),
        ];

        let resp = llm.complete(&messages).await?;
        parse_rewrite_response(&resp.text, raw)
    }

    async fn dispatch(&self, result: &DictationResult) -> Result<()> {
        match result {
            DictationResult::Text(text) => {
                let text = text.clone();
                let method = self.config.output_method;
                tokio::task::spawn_blocking(move || output_text(&text, method))
                    .await
                    .map_err(|e| CortexError::Tool(format!("spawn_blocking: {e}")))??;
            }
            DictationResult::Command { intent, original } => {
                // For commands, the agent loop in the CLI handles dispatch.
                // Here we just log — the CLI's `listen` command reads the result.
                println!("[command detected] intent={intent} original={original}");
            }
            DictationResult::Silent => {}
        }
        Ok(())
    }

    async fn persist_turn(&self, raw: &str, cleaned: &str) -> Result<()> {
        let title = if raw.chars().count() > 60 {
            format!("{}…", raw.chars().take(60).collect::<String>())
        } else {
            raw.to_string()
        };

        let body = serde_json::json!({
            "raw": raw,
            "cleaned": cleaned,
        })
        .to_string();

        let node = Node::new(NodeKind::DictationTurn, title).with_body(body);
        self.cortex.remember(node).await?;

        // If the raw and cleaned differ meaningfully, store correction as VocabEntry
        if raw != cleaned && levenshtein_ratio(raw, cleaned) < 0.85 {
            let vocab = Node::new(
                NodeKind::VocabEntry,
                format!("Correction: {} → {}", truncate(raw, 30), truncate(cleaned, 30)),
            )
            .with_body(
                serde_json::json!({"from": raw, "to": cleaned}).to_string(),
            );
            self.cortex.remember(vocab).await?;
        }

        Ok(())
    }
}

// ─── Output helpers ──────────────────────────────────────

fn output_text(text: &str, method: OutputMethod) -> Result<()> {
    match method {
        OutputMethod::Type => type_text(text),
        OutputMethod::Clipboard => set_clipboard(text),
        OutputMethod::Both => {
            set_clipboard(text)?;
            type_text(text)
        }
    }
}

fn type_text(text: &str) -> Result<()> {
    let mut enigo = Enigo::new(&Settings::default())
        .map_err(|e| CortexError::Tool(format!("enigo init: {e}")))?;
    enigo
        .text(text)
        .map_err(|e| CortexError::Tool(format!("type_text: {e}")))
}

fn set_clipboard(text: &str) -> Result<()> {
    arboard::Clipboard::new()
        .and_then(|mut cb| cb.set_text(text))
        .map_err(|e| CortexError::Tool(format!("set_clipboard: {e}")))
}

// ─── Hotkey matching ─────────────────────────────────────

#[derive(Debug)]
enum HotkeyEvent {
    StartRecording,
    StopRecording,
}

fn matches_hotkey(event: &Event, key_name: &str) -> bool {
    let key = match &event.event_type {
        EventType::KeyPress(k) | EventType::KeyRelease(k) => k,
        _ => return false,
    };
    let name = key_name.to_lowercase();
    matches!(
        (key, name.as_str()),
        (RdevKey::AltGr, "altgr")
            | (RdevKey::Alt, "alt")
            | (RdevKey::F9, "f9")
            | (RdevKey::F10, "f10")
            | (RdevKey::F11, "f11")
            | (RdevKey::F12, "f12")
    )
}

// ─── LLM response parsing ────────────────────────────────

fn parse_rewrite_response(json_str: &str, raw_fallback: &str) -> Result<DictationResult> {
    let v: serde_json::Value = serde_json::from_str(json_str.trim()).unwrap_or_default();

    match v["type"].as_str() {
        Some("text") => {
            let content = v["content"]
                .as_str()
                .unwrap_or(raw_fallback)
                .to_string();
            Ok(DictationResult::Text(content))
        }
        Some("command") => {
            let intent = v["intent"].as_str().unwrap_or("unknown").to_string();
            let original = v["original"].as_str().unwrap_or(raw_fallback).to_string();
            Ok(DictationResult::Command { intent, original })
        }
        _ => {
            // LLM returned something unexpected — fall back to raw text
            Ok(DictationResult::Text(raw_fallback.to_string()))
        }
    }
}

// ─── Small utilities ─────────────────────────────────────

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max_chars).collect::<String>())
    }
}

/// Approximate similarity ratio between two strings (0.0 = totally different,
/// 1.0 = identical). Used to decide whether a correction is significant enough
/// to store as a VocabEntry.
fn levenshtein_ratio(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let n = a.len();
    let m = b.len();
    if n == 0 && m == 0 {
        return 1.0;
    }
    let max_len = n.max(m);
    let dist = levenshtein(&a, &b);
    1.0 - (dist as f64 / max_len as f64)
}

fn levenshtein(a: &[char], b: &[char]) -> usize {
    let n = a.len();
    let m = b.len();
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..=n { dp[i][0] = i; }
    for j in 0..=m { dp[0][j] = j; }
    for i in 1..=n {
        for j in 1..=m {
            dp[i][j] = if a[i - 1] == b[j - 1] {
                dp[i - 1][j - 1]
            } else {
                1 + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1])
            };
        }
    }
    dp[n][m]
}
