use std::sync::{Arc, Mutex};
use lru::LruCache;
use std::num::NonZeroUsize;

use crate::error::{CortexError, Result};

/// Handle to the embedding model. Thread-safe, with an LRU cache to avoid
/// redundant inference for identical texts.
#[derive(Clone)]
pub struct EmbedHandle {
    inner: Arc<Mutex<EmbedInner>>,
}

struct EmbedInner {
    model: fastembed::TextEmbedding,
    cache: LruCache<String, Vec<f32>>,
}

impl EmbedHandle {
    /// Initialise fastembed with BAAI/bge-small-en-v1.5 (384-dim).
    /// Downloads model on first run (~33 MB).
    pub fn new(cache_size: usize) -> Result<Self> {
        let mut init_opts = fastembed::InitOptions::default();
        init_opts.model_name = fastembed::EmbeddingModel::BGESmallENV15;
        init_opts.show_download_progress = true;
        let model = fastembed::TextEmbedding::try_new(init_opts)
            .map_err(|e| CortexError::Embedding(format!("failed to load model: {e}")))?;

        let cache = LruCache::new(
            NonZeroUsize::new(cache_size.max(1)).unwrap(),
        );

        Ok(Self {
            inner: Arc::new(Mutex::new(EmbedInner { model, cache })),
        })
    }

    /// Embed a single text, using the cache when possible.
    /// Runs the model on `spawn_blocking` to avoid blocking tokio.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let text = text.to_string();
        let inner = self.inner.clone();

        tokio::task::spawn_blocking(move || {
            let mut guard = inner
                .lock()
                .map_err(|e| CortexError::Embedding(format!("lock: {e}")))?;

            // Cache hit?
            if let Some(cached) = guard.cache.get(&text) {
                return Ok(cached.clone());
            }

            // Run model inference
            let results = guard
                .model
                .embed(vec![text.clone()], None)
                .map_err(|e| CortexError::Embedding(format!("embed: {e}")))?;

            let vec = results
                .into_iter()
                .next()
                .ok_or_else(|| CortexError::Embedding("empty embedding result".into()))?;

            guard.cache.put(text, vec.clone());
            Ok(vec)
        })
        .await
        .map_err(|e| CortexError::Embedding(format!("join: {e}")))?
    }

    /// Embed a batch of texts. No caching for batch (simple implementation).
    pub async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let guard = inner
                .lock()
                .map_err(|e| CortexError::Embedding(format!("lock: {e}")))?;
            guard
                .model
                .embed(texts, None)
                .map_err(|e| CortexError::Embedding(format!("embed batch: {e}")))
        })
        .await
        .map_err(|e| CortexError::Embedding(format!("join: {e}")))?
    }

    /// Embedding dimensionality (384 for bge-small-en-v1.5).
    pub fn dim(&self) -> usize {
        384
    }
}
