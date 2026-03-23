# cortex-embedded

**One crate. One SQLite file. A complete AI agent with graph memory, sub-agents, and a CLI.**

Everything — identity, knowledge, tool calls, LLM calls, sub-agent work, loop iterations, self-model — is a node in the graph. The agent queries its own history the same way it queries any other knowledge.

## Features

- **Graph memory** — 18 node kinds, 6 edge kinds, full provenance tracking
- **Hybrid recall** — HNSW ANN search + BFS graph traversal + trust scoring + recency decay
- **Embeddings** — BAAI/bge-small-en-v1.5 via fastembed (384-dim, runs locally)
- **Auto-link** — background task creates `RelatesTo` and `Contradicts` edges automatically
- **Decay** — importance fades over time; Soul/Belief/Goal nodes are immune
- **Trust propagation** — `Supports` edges boost trust, `Contradicts` edges reduce it
- **Context compaction** — LLM extracts key facts from long conversations into the graph
- **LLM backends** — Anthropic Claude, Ollama (local), Mock (testing)
- **Tool registry** — tools write provenance-tracked results into the graph
- **Sub-agents** — spawn into the shared graph with scoped identity
- **CLI** — chat, ask, memory search, identity management, consolidation, diagnostics

## Quick Start

```bash
# Build
cargo build --release

# Initialize database and download embedding model
cortex init

# Check graph health
cortex doctor

# View identity
cortex soul show

# Memory stats
cortex memory stats

# Interactive chat (requires LLM)
ANTHROPIC_API_KEY=sk-ant-... cortex chat
# or with Ollama
cortex --ollama llama3 chat

# Single query
cortex ask "What do you know about JWT tokens?"

# Semantic search
cortex memory search "authentication"

# Run trust consolidation
cortex consolidate
```

## Architecture

```
┌─────────────────────────────────────────────┐
│                cortex-embedded               │
├──────────┬──────────┬──────────┬────────────┤
│  recall  │ briefing │  tools   │   agent    │
│ (hybrid  │ (context │ (registry│  (loop +   │
│  search) │  doc)    │  + trust)│ sub-agents)│
├──────────┴──────────┴──────────┴────────────┤
│              graph + memory                  │
│         (BFS walk, scoring, decay)           │
├──────────┬──────────────────────────────────┤
│   HNSW   │           SQLite                  │
│ (2-tier) │  (WAL mode, bundled rusqlite)     │
├──────────┴──────────────────────────────────┤
│              fastembed                        │
│        (BAAI/bge-small-en-v1.5)              │
└─────────────────────────────────────────────┘
```

### Node Kinds

| Category | Kinds |
|----------|-------|
| Knowledge | `Fact`, `Entity`, `Concept`, `Decision` |
| Identity | `Soul`, `Belief`, `Goal` |
| Operational | `Session`, `Turn`, `LlmCall`, `ToolCall`, `LoopIteration` |
| Sub-agents | `SubAgent`, `Delegation`, `Synthesis` |
| Meta | `Pattern`, `Capability`, `Limitation`, `Contradiction` |

### Edge Kinds

`RelatesTo` · `Contradicts` · `Supports` · `DerivesFrom` · `PartOf` · `Supersedes`

## How It Works

Every interaction creates a provenance chain:

```
Fact → ToolCall → LoopIteration → Session
```

The agent knows not just *what* it knows, but *how it came to know it*, *when*, *via which tool*, and *how much to trust it*.

**Recall pipeline:**
1. Embed query → HNSW k-NN search
2. BFS graph walk from candidates
3. Score: `importance × trust × recency × proximity_bonus`
4. Return ranked nodes with contradiction warnings

**Background tasks:**
- **Auto-link** — new nodes are compared against the graph; similar nodes get `RelatesTo` edges, contradicting nodes get `Contradicts` edges
- **Decay** — every 60s, nodes not accessed in 24h lose importance (floor: 0.01)

## Using as a Library

```rust
use cortex_embedded::{CortexEmbedded, types::*};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cx = CortexEmbedded::open("my_agent.db").await?;

    // Store knowledge
    let node = Node::new(NodeKind::Fact, "Rust is fast")
        .with_body("Rust provides zero-cost abstractions and memory safety.");
    cx.remember(node).await?;

    // Recall
    let results = cx.recall("performance", RecallOptions::default()).await?;
    for r in &results {
        println!("[{}] {} — score: {:.3}", r.node.kind, r.node.title, r.score);
    }

    // Build briefing for LLM
    let briefing = cx.briefing("system design", 12).await?;
    println!("{}", briefing.context_doc);

    Ok(())
}
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| `rusqlite` (bundled) | SQLite with WAL mode |
| `instant-distance` | HNSW approximate nearest neighbor search |
| `fastembed` | Local text embeddings (ONNX runtime) |
| `tokio` | Async runtime |
| `reqwest` | HTTP client for Anthropic API |
| `clap` | CLI argument parsing |
| `async-channel` | Background task communication |

## Tests

```bash
# Run all tests (22 total)
cargo test -- --test-threads=1

# Just HNSW unit tests
cargo test --lib hnsw

# Just integration tests
cargo test --test integration -- --test-threads=1
```

## License

MIT
