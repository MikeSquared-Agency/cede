# cede

**Fork this. Build your own self-aware agent.**

cede is a forkable starter kit built on top of [cortex-embedded](https://github.com/MikeSquared-Agency/cortex-embedded) — the graph-memory cognitive engine. Fork this repo, shape the soul, add your own tools, and ship an agent that remembers everything.

> **Want omnichannel?** See [**omni-cede**](https://github.com/MikeSquared-Agency/omni-cede) — a fork of cede that adds an HTTP API, identity resolution, and per-channel session management.

## Ecosystem

```
cortex-embedded          <-- the engine (upstream)
  |-- cede               <-- you are here (fork this to build your own agent)
       |-- omni-cede     <-- omnichannel variant (HTTP API, identity, sessions)
```

## What You Get

Everything from cortex-embedded, packaged as a ready-to-run agent:

- **Graph memory** — 18 node kinds, 6 edge kinds, full provenance tracking
- **Graph-native chat sessions** — fresh HNSW-based briefing per turn, no growing message history
- **Hybrid recall** — semantic search + graph traversal + recency window (last 7 messages always included)
- **Local embeddings** — BAAI/bge-small-en-v1.5 via fastembed (384-dim, no API calls)
- **Auto-linking + contradiction detection** — three-tier: cosine -> negation keywords -> LLM adjudication
- **Importance decay + trust propagation** — the agent's beliefs strengthen or weaken over time
- **Context compaction** — LLM extracts key facts from long conversations
- **LLM backends** — Anthropic Claude or Ollama (local)
- **Tool registry** — add custom tools that write results into the graph with provenance
- **Sub-agents** — delegate tasks to scoped sub-agents in the shared graph
- **TUI graph explorer** — interactive terminal UI with chat, node inspection, visualization
- **CLI** — chat, ask, memory search, soul/identity management, graph visualization, diagnostics

## Quick Start

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/cede.git
cd cede

# Build
cargo build --release

# Initialize database and download embedding model
cede init

# Start chatting (pick one LLM backend)
ANTHROPIC_API_KEY=sk-ant-... cede chat
cede --ollama llama3 chat

# Single query
cede ask "What do you know?"

# Interactive graph explorer
ANTHROPIC_API_KEY=sk-ant-... cede graph explore

# Graph overview
cede graph overview

# Memory operations
cede memory stats
cede memory search "topic"
cede memory show <node_id>

# Identity
cede soul show

# Diagnostics
cede doctor
cede consolidate
```

## How to Make It Yours

### 1. Shape the Soul

Edit the seed identity in `src/lib.rs` — the `seed_identity()` function creates the initial `Soul` node. Change the body text to define who your agent is.

### 2. Add Custom Tools

Register tools in `src/tools/mod.rs`. Each tool is a function that takes parameters and returns a `ToolResult`. Tool calls are automatically tracked as `ToolCall` nodes in the graph.

### 3. Tune the Config

Adjust `src/config.rs` defaults:

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `session_recency_window` | 7 | Recent messages always included in briefing |
| `auto_link_cosine_threshold` | 0.75 | Minimum similarity for auto-linking |
| `contradiction_cosine_threshold` | 0.85 | Trigger contradiction detection |
| `compaction_threshold` | 20 | Messages before context compaction |
| `max_iterations` | 10 | Agent loop iteration limit |
| `decay_interval_secs` | 60 | Background decay sweep frequency |

### 4. Staying Updated

cede tracks cortex-embedded as the `upstream` remote. To pull engine improvements:

```bash
git fetch upstream
git merge upstream/master
```

## Architecture

```
+-----------------------------------------+
|                  cede                    |
+---------+----------+---------+----------+
|  recall | briefing |  tools  |  agent   |
| (HNSW + | (scored  | (custom | (loop +  |
|  graph) |  context)|  + std) | subagent)|
+---------+----------+---------+----------+
|            graph + memory               |
|       (BFS, scoring, decay)             |
+---------+-------------------------------+
|  HNSW   |         SQLite                |
| (2-tier)|  (WAL, bundled rusqlite)      |
+---------+-------------------------------+
|            fastembed                     |
|      (BAAI/bge-small-en-v1.5)           |
+-----------------------------------------+
```

## Tests

```bash
# Run all 28 tests
cargo test -- --test-threads=1
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| `rusqlite` (bundled) | SQLite with WAL mode |
| `instant-distance` | HNSW approximate nearest neighbor |
| `fastembed` | Local text embeddings (ONNX) |
| `tokio` | Async runtime |
| `reqwest` | HTTP client for Anthropic |
| `clap` | CLI argument parsing |
| `ratatui` + `crossterm` | TUI graph explorer |
| `async-channel` | Background task channels |

## License

MIT