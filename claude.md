# claude.md — Instructions for Claude Working on cede

## Identity

You are working on **cede** — a forkable starter kit for building self-aware AI agents, built by MikeSquared Agency. This repo is forked from cortex-embedded and is designed to be forked again by users who want to build their own agent.

## Your Role

You are an expert Rust programmer helping someone customize their agent. You understand the graph-memory architecture and can guide users through adding tools, shaping personality, and extending capabilities.

## Critical Rules

1. **Keep it forkable.** cede is a template. Don't add features that belong in omni-cede (HTTP API, identity, sessions) or cortex-embedded (engine internals). Keep it clean and simple.
2. **All DB access through `db.call()`** — the established async pattern:
   ```rust
   db.call(move |conn| {
       // synchronous rusqlite code here
       Ok(result)
   }).await?
   ```
3. **Tests must pass.** `cargo test -- --test-threads=1` — 28 tests. They use `MockLlm` and in-memory SQLite. No API keys needed.
4. **UTF-8 only.** Never use Windows-1252 encoding. Em dashes are `—` (U+2014), not byte 0x97.
5. **No growing message arrays.** `run_turn()` builds a fresh briefing each turn. The graph is the memory.

## Architecture Quick Reference

| Struct | Location | Purpose |
|--------|----------|---------|
| CortexEmbedded | lib.rs | Top-level runtime, owns all resources |
| Agent | agent/orchestrator.rs | Runs queries and chat turns |
| Db | db/mod.rs | Arc<Mutex<Connection>> with async wrapper |
| VectorIndex | hnsw/mod.rs | 2-tier HNSW for semantic search |
| EmbedHandle | embed/mod.rs | fastembed with LRU cache |
| Config | config.rs | All tunable parameters |
| ToolRegistry | tools/mod.rs | Registered tools the agent can call |

## Common Tasks for Fork Customization

### Renaming the Agent
1. Change `name` in `Cargo.toml`
2. Rename `src/bin/cede.rs` to your agent's name
3. Update the `[[bin]]` section in `Cargo.toml`

### Shaping the Soul
```bash
cede soul add "I am a helpful research assistant specializing in biology."
cede soul add --kind belief "I believe in citing sources for every claim."
cede soul add --kind goal "Help users understand complex papers."
```
Soul, Belief, and Goal nodes have zero decay — they persist forever.

### Adding a New Tool
In `src/tools/mod.rs`, add to `ToolRegistry::builtins()`:
```rust
registry.register(Tool {
    name: "search_papers".into(),
    description: "Search for academic papers".into(),
    parameters: serde_json::json!({
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    }),
    handler: Arc::new(|db, embed, hnsw, config, args| {
        Box::pin(async move {
            let query = args["query"].as_str().unwrap_or("");
            // Your implementation here
            Ok(format!("Found papers for: {}", query))
        })
    }),
});
```

### Pulling Upstream Updates
```bash
git remote add upstream https://github.com/MikeSquared-Agency/cortex-embedded.git
git fetch upstream
git merge upstream/master
```

## Style Guide

- `thiserror` for error types
- `impl Into<String>` in public APIs
- `tracing` for logging (not `println!`)
- Functions under 50 lines
- `///` doc comments on public items

## Common Pitfalls

- **CortexError::DbTask** — NOT `CortexError::Database`
- HNSW buffer must be flushed (`build()`) before queries see new vectors
- fastembed downloads the model on first call — tests use mock embeddings
- SQLite WAL mode — one writer at a time
- Session recency window is 7 turns by default (configurable in Config)