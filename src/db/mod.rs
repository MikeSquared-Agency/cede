pub mod schema;
pub mod queries;

use std::sync::{Arc, Mutex};
use rusqlite::Connection;
use crate::error::{CortexError, Result};

/// Async-safe handle to SQLite. Wraps a `Connection` behind `Arc<Mutex<_>>`
/// and dispatches all work onto `spawn_blocking` to avoid starving tokio.
#[derive(Clone)]
pub struct Db {
    conn: Arc<Mutex<Connection>>,
}

impl Db {
    /// Open (or create) the database at `path`. Runs schema migrations.
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        schema::create_tables(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Open an in-memory database (for tests).
    pub fn open_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        schema::create_tables(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Run a closure that takes `&Connection` on a blocking thread.
    /// This is the single entry-point for all DB access in async code.
    pub async fn call<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Connection) -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let guard = conn
                .lock()
                .map_err(|e| CortexError::DbTask(format!("mutex poisoned: {e}")))?;
            f(&guard)
        })
        .await
        .map_err(|e| CortexError::DbTask(format!("join error: {e}")))?
    }

    /// Synchronous access — only use outside tokio (e.g. during `open()`).
    pub fn call_sync<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Connection) -> Result<T>,
    {
        let guard = self
            .conn
            .lock()
            .map_err(|e| CortexError::DbTask(format!("mutex poisoned: {e}")))?;
        f(&guard)
    }
}
