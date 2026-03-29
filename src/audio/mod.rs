/// Audio capture module for theword.
///
/// Streams PCM audio from the default input device, runs WebRTC VAD to detect
/// speech boundaries, and returns a complete utterance as a `Vec<i16>` ready
/// for Whisper transcription.
///
/// Sample rate is fixed at 16 kHz mono — the format Whisper and WebRTC VAD
/// both expect, so we avoid any resampling.
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, SampleRate, StreamConfig};
use ringbuf::{HeapRb, traits::{Consumer, Producer, Split}};
use webrtc_vad::{Vad, SampleRate as VadRate, VadMode};

use crate::error::{CortexError, Result};

/// Fixed sample rate for all audio in theword. Whisper and WebRTC VAD both
/// require 16 kHz input; capturing at this rate avoids resampling.
pub const SAMPLE_RATE: u32 = 16_000;

/// VAD frame size in samples. WebRTC VAD requires exactly 10 ms, 20 ms, or
/// 30 ms frames. We use 30 ms = 480 samples at 16 kHz.
pub const VAD_FRAME_SAMPLES: usize = 480; // 30 ms @ 16 kHz

/// One completed spoken utterance captured from the microphone.
pub struct Utterance {
    /// Raw 16 kHz mono PCM samples.
    pub samples: Vec<i16>,
    /// Approximate duration of the utterance.
    pub duration: Duration,
}

/// Blocking audio capture with VAD-based segmentation.
///
/// Call [`AudioCapture::record_utterance`] to block until the user speaks and
/// then falls silent, returning the utterance PCM.
pub struct AudioCapture {
    vad_mode: VadMode,
    silence_threshold_ms: u64,
    min_speech_ms: u64,
}

impl AudioCapture {
    pub fn new(vad_mode: VadMode, silence_threshold_ms: u64, min_speech_ms: u64) -> Self {
        Self { vad_mode, silence_threshold_ms, min_speech_ms }
    }

    /// Block until a complete utterance is captured.
    ///
    /// The function starts recording when any speech is detected, and stops
    /// `silence_threshold_ms` after the last speech frame. Returns `None` if
    /// no speech was detected before `timeout` expires, or if the captured
    /// audio is shorter than `min_speech_ms`.
    pub fn record_utterance(&self, timeout: Duration) -> Result<Option<Utterance>> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| CortexError::Audio("no input device found".into()))?;

        let config = StreamConfig {
            channels: 1,
            sample_rate: SampleRate(SAMPLE_RATE),
            buffer_size: cpal::BufferSize::Default,
        };

        // Shared ring buffer: stream callback → main thread
        let rb = HeapRb::<i16>::new(SAMPLE_RATE as usize * 30); // 30 s headroom
        let (mut prod, mut cons) = rb.split();
        let prod = Arc::new(Mutex::new(prod));

        let prod_clone = prod.clone();
        let stream = device
            .build_input_stream(
                &config,
                move |data: &[i16], _| {
                    if let Ok(mut p) = prod_clone.lock() {
                        // Push samples; drop oldest if full (shouldn't happen with 30 s buffer)
                        let _ = p.push_slice(data);
                    }
                },
                |err| eprintln!("[audio] stream error: {err}"),
                None,
            )
            .map_err(|e| CortexError::Audio(e.to_string()))?;

        stream.play().map_err(|e| CortexError::Audio(e.to_string()))?;

        // Clone mode since VadMode doesn't implement Copy
        let vad_mode = match self.vad_mode {
            VadMode::Quality        => VadMode::Quality,
            VadMode::LowBitrate     => VadMode::LowBitrate,
            VadMode::Aggressive     => VadMode::Aggressive,
            VadMode::VeryAggressive => VadMode::VeryAggressive,
        };
        let mut vad = Vad::new_with_rate_and_mode(VadRate::Rate16kHz, vad_mode);

        let silence_frames_needed =
            (self.silence_threshold_ms / 30).max(1) as usize; // 30 ms per frame

        let min_speech_frames =
            (self.min_speech_ms / 30).max(1) as usize;

        let mut all_samples: Vec<i16> = Vec::new();
        let mut frame_buf: Vec<i16> = Vec::with_capacity(VAD_FRAME_SAMPLES);
        let mut tmp = vec![0i16; VAD_FRAME_SAMPLES];

        let mut speech_started = false;
        let mut silence_count = 0usize;
        let mut speech_frames = 0usize;

        let deadline = std::time::Instant::now() + timeout;

        loop {
            if std::time::Instant::now() > deadline {
                break;
            }

            // Drain available samples into our frame buffer
            let n = cons.pop_slice(&mut tmp);
            frame_buf.extend_from_slice(&tmp[..n]);

            // Process complete 30 ms frames
            while frame_buf.len() >= VAD_FRAME_SAMPLES {
                let frame: Vec<i16> = frame_buf.drain(..VAD_FRAME_SAMPLES).collect();

                let is_speech = vad.is_voice_segment(&frame).unwrap_or(false);

                if is_speech {
                    speech_started = true;
                    silence_count = 0;
                    speech_frames += 1;
                    all_samples.extend_from_slice(&frame);
                } else if speech_started {
                    silence_count += 1;
                    all_samples.extend_from_slice(&frame); // keep trailing silence for context
                    if silence_count >= silence_frames_needed {
                        // Utterance complete
                        drop(stream);
                        if speech_frames < min_speech_frames {
                            return Ok(None);
                        }
                        let duration = Duration::from_millis(
                            all_samples.len() as u64 * 1000 / SAMPLE_RATE as u64,
                        );
                        return Ok(Some(Utterance { samples: all_samples, duration }));
                    }
                }
            }

            // Brief sleep to avoid busy-looping
            std::thread::sleep(Duration::from_millis(5));
        }

        drop(stream);

        if speech_frames < min_speech_frames {
            return Ok(None);
        }

        let duration = Duration::from_millis(
            all_samples.len() as u64 * 1000 / SAMPLE_RATE as u64,
        );
        Ok(Some(Utterance { samples: all_samples, duration }))
    }
}

/// Convert the cede `VadMode` config enum to the webrtc-vad `VadMode`.
pub fn to_vad_mode(mode: &crate::config::VadMode) -> VadMode {
    match mode {
        crate::config::VadMode::Quality        => VadMode::Quality,
        crate::config::VadMode::LowBitrate     => VadMode::LowBitrate,
        crate::config::VadMode::Aggressive     => VadMode::Aggressive,
        crate::config::VadMode::VeryAggressive => VadMode::VeryAggressive,
    }
}
