/// Speech-to-text module for theword.
///
/// Wraps whisper-rs (which in turn wraps whisper.cpp) for fully local,
/// offline transcription. The `WhisperHandle` is constructed once at startup
/// and kept alive for the process lifetime to avoid the ~500 ms model-load
/// cost on every utterance.
use std::path::{Path, PathBuf};

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::config::WhisperModel;
use crate::error::{CortexError, Result};

/// A loaded Whisper model ready for transcription.
pub struct WhisperHandle {
    ctx: WhisperContext,
    language: Option<String>,
}

impl WhisperHandle {
    /// Load the model from disk. `path` must point to a GGML `.bin` weights file.
    ///
    /// Set `language` to a BCP-47 code (e.g. `"en"`) to skip language detection
    /// and improve accuracy. Pass `None` to auto-detect.
    pub fn load(path: impl AsRef<Path>, language: Option<String>) -> Result<Self> {
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| CortexError::Stt("model path is not valid UTF-8".into()))?;

        let params = WhisperContextParameters::default();
        let ctx = WhisperContext::new_with_params(path_str, params)
            .map_err(|e| CortexError::Stt(format!("failed to load whisper model: {e}")))?;

        Ok(Self { ctx, language })
    }

    /// Transcribe raw 16 kHz mono PCM samples.
    ///
    /// Returns the concatenated text of all detected segments.
    pub fn transcribe(&self, samples: &[i16]) -> Result<String> {
        // whisper-rs expects f32 samples in [-1.0, 1.0]
        let samples_f32: Vec<f32> = samples
            .iter()
            .map(|&s| s as f32 / i16::MAX as f32)
            .collect();

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // Suppress non-speech tokens and task tokens for cleaner output
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);

        if let Some(ref lang) = self.language {
            params.set_language(Some(lang.as_str()));
        }

        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| CortexError::Stt(format!("failed to create whisper state: {e}")))?;

        state
            .full(params, &samples_f32)
            .map_err(|e| CortexError::Stt(format!("whisper inference failed: {e}")))?;

        let n_segments = state
            .full_n_segments()
            .map_err(|e| CortexError::Stt(format!("failed to get segment count: {e}")))?;

        let mut text = String::new();
        for i in 0..n_segments {
            let segment = state
                .full_get_segment_text(i)
                .map_err(|e| CortexError::Stt(format!("failed to get segment {i}: {e}")))?;
            text.push_str(segment.trim());
            text.push(' ');
        }

        Ok(text.trim().to_string())
    }
}

/// Resolve the path to the model weights file.
///
/// If `explicit_path` is provided, use it directly. Otherwise look in
/// `~/.theword/models/<filename>`.
pub fn resolve_model_path(
    model: &WhisperModel,
    explicit_path: Option<&str>,
) -> Result<PathBuf> {
    if let Some(p) = explicit_path {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
        return Err(CortexError::Stt(format!(
            "whisper model not found at: {}",
            p
        )));
    }

    let home = dirs_next::home_dir()
        .ok_or_else(|| CortexError::Stt("cannot determine home directory".into()))?;

    let path = home
        .join(".theword")
        .join("models")
        .join(model.filename());

    if !path.exists() {
        return Err(CortexError::Stt(format!(
            "whisper model not found at {}. Run `theword init` to download it.",
            path.display()
        )));
    }

    Ok(path)
}

/// Download a Whisper model weights file to `~/.theword/models/`.
///
/// Uses a blocking HTTP GET — call this from a dedicated thread or spawn_blocking.
pub fn download_model(model: &WhisperModel) -> Result<PathBuf> {
    let home = dirs_next::home_dir()
        .ok_or_else(|| CortexError::Stt("cannot determine home directory".into()))?;

    let models_dir = home.join(".theword").join("models");
    std::fs::create_dir_all(&models_dir)
        .map_err(|e| CortexError::Stt(format!("failed to create models dir: {e}")))?;

    let dest = models_dir.join(model.filename());
    if dest.exists() {
        println!("Model already downloaded: {}", dest.display());
        return Ok(dest);
    }

    let url = model.download_url();
    println!("Downloading {} ...", url);

    let response = ureq::get(&url)
        .call()
        .map_err(|e| CortexError::Stt(format!("download failed: {e}")))?;

    let mut file = std::fs::File::create(&dest)
        .map_err(|e| CortexError::Stt(format!("failed to create file: {e}")))?;

    std::io::copy(&mut response.into_reader(), &mut file)
        .map_err(|e| CortexError::Stt(format!("failed to write model: {e}")))?;

    println!("Saved to {}", dest.display());
    Ok(dest)
}
