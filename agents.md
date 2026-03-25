# agents.md — Guide for AI Agents Working on cede

You are working on **cede**, a forkable starter kit for building self-aware AI agents. This file tells you how to navigate the codebase and contribute effectively.

## What This Repo Is

cede is forked from **cortex-embedded** (the upstream engine). It is a Rust crate providing a graph-memory runtime where every piece of knowledge, every tool call, every LLM interaction is a node in a single SQLite-backed graph.

**This repo is meant to be forked.** Users clone cede, rename the binary, shape the soul, add tools, and ship their own self-aware agent.

### Ecosystem Position
- **cortex-embedded** (upstream) — the engine, not meant for direct forking
- **cede** (this repo) — forkable starter kit
- **omni-cede** — omnichannel variant with HTTP API, identity, sessions

## Repository Layout

```
src/
  lib.rs              # CortexEmbedded struct, background tasks, decay, consolidation
  types.rs            # All types: Node, Edge, NodeKind, EdgeKind, Message, LlmResponse, etc.
  error.rs            # CortexError enum, Result type alias
  config.rs           # Config struct with all tunable parameters
  agent/
    mod.rs            # Re-exports Agent
    orchestrator.rs   # Agent struct, run() and run_turn() methods, tool-call loop
    subagent.rs       # Sub-agent spawning and delegation
  db/
    mod.rs            # Db struct (Arc<Mutex<Connection>>), async call() wrapper
    schema.rs         # CREATE TABLE statements, migrations
    queries.rs        # All SQL queries as functions
  embed/
    mod.rs            # EmbedHandle — fastembed wrapper with LRU cache
  hnsw/
    mod.rs            # VectorIndex — 2-tier HNSW (built index + linear buffer)
  graph/
    mod.rs            # BFS traversal, graph walk scoring
  memory/
    mod.rs            # recall(), briefing(), briefing_with_kinds(), recency window
  tools/
    mod.rs            # ToolRegistry, builtin tools (remember, recall, forget, etc.)
  llm/
    mod.rs            # LlmClient trait, AnthropicClient, OllamaClient, MockLlm
  cli/
    mod.rs            # CLI commands: Chat, Ask, Memory, Soul, Sessions, Graph, etc.
    graph_tui.rs      # Interactive TUI graph explorer with chat panel
    graph_viz.rs      # ASCII graph visualization
  bin/
    cede.rs           # Binary entry point — calls cli::run()
tests/
  integration.rs      # 22 integration tests covering all phases
```

## Key Architecture

### Graph-Native Memory
Every node has: id (UUID), kind, content, importance, decay_rate, embedding, created_at. Nodes are connected by edges (RelatesTo, Contradicts, Supports, DerivesFrom, PartOf, Supersedes).

### Node Kinds (18)
Fact, Entity, Concept, Decision, Soul, Belief, Goal, UserInput, Session, Turn, LlmCall, ToolCall, LoopIteration, SubAgent, Delegation, Synthesis, Pattern, Limitation, Capability

### Chat Sessions
`run_turn(session_id, input)`: stores input as a node, builds a fresh semantic briefing, merges a recency window (last 7 turns), sends to LLM. No growing message array.

### Auto-Linking
Cosine similarity >= 0.75 adds RelatesTo edges. >= 0.85 with negation keywords triggers contradiction detection (3-tier: keyword → LLM → fallback).

### Db Pattern
All DB access: `db.call(move |conn| { ... }).await` — spawns blocking task, returns result.

## How to Customize (as a human/agent forking this)

1. **Rename:** Change crate name in Cargo.toml, rename `bin/cede.rs`
2. **Shape soul:** Use `cede soul add "..."` to define personality, beliefs, goals
3. **Add tools:** Add to `ToolRegistry::builtins()` in `src/tools/mod.rs`
4. **Tune config:** See `src/config.rs` for all tunable parameters
5. **Pull upstream:** `git remote add upstream <cortex-embedded-url>; git fetch upstream; git merge upstream/master`

## Build and Test

```bash
cargo build
cargo test -- --test-threads=1    # 28 tests (6 unit + 22 integration)
```

Tests use `MockLlm` and in-memory SQLite — no API keys needed.

## Conventions

- Async DB: `db.call(move |conn| { ... }).await`
- Embeddings: 384-dim f32 (BAAI/bge-small-en-v1.5)
- Node IDs: UUID v4 strings
- Timestamps: Unix seconds (i64)
- Error handling: `CortexError` enum, `Result<T>` alias

## Branch Policy

- `master` is protected: no direct push, PRs required
- Work on `dev` branch, merge via PR