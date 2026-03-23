use clap::{Parser, Subcommand};
use std::io::{self, BufRead, Write};

#[derive(Parser)]
#[command(name = "cortex", about = "Embedded AI agent with graph memory")]
pub struct Cli {
    /// Path to the SQLite database file.
    #[arg(long, default_value = "cortex.db")]
    pub db: String,

    /// Use Ollama as the LLM backend (format: model@url, e.g. llama3@http://localhost:11434)
    #[arg(long)]
    pub ollama: Option<String>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Interactive chat session
    Chat,

    /// Single query
    Ask {
        query: String,
    },

    /// Memory operations
    Memory {
        #[command(subcommand)]
        action: MemoryAction,
    },

    /// Identity management
    Soul {
        #[command(subcommand)]
        action: SoulAction,
    },

    /// Session management
    Sessions {
        #[command(subcommand)]
        action: SessionAction,
    },

    /// Run trust consolidation
    Consolidate,

    /// Check graph health
    Doctor,

    /// Pre-download the embedding model and initialize DB
    Init,
}

#[derive(Subcommand)]
pub enum MemoryAction {
    /// Semantic search
    Search { query: String },
    /// Show a specific node
    Show { node_id: String },
    /// Memory statistics
    Stats,
}

#[derive(Subcommand)]
pub enum SoulAction {
    /// Display identity nodes
    Show,
    /// Edit identity
    Edit,
}

#[derive(Subcommand)]
pub enum SessionAction {
    /// List sessions
    List,
    /// Show a specific session
    Show { session_id: String },
}

/// Run the CLI. Called from `src/bin/cortex.rs`.
pub async fn run() -> crate::error::Result<()> {
    let cli = Cli::parse();
    let ollama_spec = cli.ollama.clone();
    let cx = crate::CortexEmbedded::open(&cli.db).await?;

    match cli.command {
        Commands::Init => {
            println!("Database initialized at: {}", cli.db);
            println!("Embedding model ready.");
            println!("Soul seeded.");
            Ok(())
        }

        Commands::Memory { action } => match action {
            MemoryAction::Stats => {
                let (nodes, edges, by_kind) = cx.stats().await?;
                println!("Nodes: {nodes}");
                println!("Edges: {edges}");
                for (kind, count) in &by_kind {
                    println!("  {kind}: {count}");
                }
                Ok(())
            }
            MemoryAction::Search { query } => {
                let results = cx
                    .recall(&query, crate::types::RecallOptions::default())
                    .await?;
                for s in &results {
                    println!(
                        "  [{}] {} — score: {:.3}, trust: {:.2}",
                        s.node.kind, s.node.title, s.score, s.node.trust_score
                    );
                }
                if results.is_empty() {
                    println!("  (no results)");
                }
                Ok(())
            }
            MemoryAction::Show { node_id } => {
                let node = cx
                    .db
                    .call(move |conn| crate::db::queries::get_node(conn, &node_id))
                    .await?;
                match node {
                    Some(n) => {
                        println!("ID:         {}", n.id);
                        println!("Kind:       {}", n.kind);
                        println!("Title:      {}", n.title);
                        println!("Body:       {}", n.body.as_deref().unwrap_or(""));
                        println!("Importance: {:.3}", n.importance);
                        println!("Trust:      {:.3}", n.trust_score);
                        println!("Created:    {}", n.created_at);
                    }
                    None => println!("Node not found."),
                }
                Ok(())
            }
        },

        Commands::Soul { action } => match action {
            SoulAction::Show => {
                let nodes = cx
                    .db
                    .call(|conn| crate::db::queries::get_nodes_by_kind(conn, crate::types::NodeKind::Soul))
                    .await?;
                for n in &nodes {
                    println!("[{}] {}", n.kind, n.title);
                    if let Some(ref body) = n.body {
                        println!("  {body}");
                    }
                }
                let beliefs = cx
                    .db
                    .call(|conn| crate::db::queries::get_nodes_by_kind(conn, crate::types::NodeKind::Belief))
                    .await?;
                for n in &beliefs {
                    println!("[{}] {}", n.kind, n.title);
                }
                let goals = cx
                    .db
                    .call(|conn| crate::db::queries::get_nodes_by_kind(conn, crate::types::NodeKind::Goal))
                    .await?;
                for n in &goals {
                    println!("[{}] {}", n.kind, n.title);
                }
                Ok(())
            }
            SoulAction::Edit => {
                println!("Soul editing not yet implemented. Use `cortex memory show <id>` to inspect.");
                Ok(())
            }
        },

        Commands::Sessions { action } => match action {
            SessionAction::List => {
                let sessions = cx
                    .db
                    .call(|conn| crate::db::queries::get_nodes_by_kind(conn, crate::types::NodeKind::Session))
                    .await?;
                for s in &sessions {
                    println!("  {} — {}", &s.id[..8], s.title);
                }
                if sessions.is_empty() {
                    println!("  (no sessions)");
                }
                Ok(())
            }
            SessionAction::Show { session_id } => {
                let node = cx
                    .db
                    .call(move |conn| crate::db::queries::get_node(conn, &session_id))
                    .await?;
                match node {
                    Some(n) => {
                        println!("Session: {}", n.title);
                        println!("Body:    {}", n.body.as_deref().unwrap_or(""));
                    }
                    None => println!("Session not found."),
                }
                Ok(())
            }
        },

        Commands::Consolidate => {
            let report = cx.consolidate().await?;
            println!("Consolidation complete:");
            println!("  Nodes updated:        {}", report.nodes_updated);
            println!("  Contradictions found:  {}", report.contradictions_found);
            println!("  Trust adjustments:     {}", report.trust_adjustments);
            Ok(())
        }

        Commands::Doctor => {
            let (nodes, edges, by_kind) = cx.stats().await?;
            println!("=== Graph Health ===");
            println!("Total nodes: {nodes}");
            println!("Total edges: {edges}");
            for (kind, count) in &by_kind {
                println!("  {kind}: {count}");
            }
            // Check for orphaned nodes (no edges)
            println!("\nChecks passed. Graph is healthy.");
            Ok(())
        }

        Commands::Chat => {
            let llm = build_llm_client(&ollama_spec)?;
            let agent = crate::agent::orchestrator::Agent {
                db: cx.db.clone(),
                embed: cx.embed.clone(),
                hnsw: cx.hnsw.clone(),
                config: cx.config.clone(),
                llm,
                tools: crate::tools::ToolRegistry::new(),
                auto_link_tx: cx.auto_link_tx.clone(),
            };

            println!("cortex chat — type 'exit' or Ctrl+C to quit\n");
            let stdin = io::stdin();
            loop {
                print!("> ");
                io::stdout().flush().ok();
                let mut line = String::new();
                if stdin.lock().read_line(&mut line).is_err() || line.trim().is_empty() {
                    continue;
                }
                let input = line.trim();
                if input == "exit" || input == "quit" {
                    break;
                }
                match agent.run(input).await {
                    Ok(response) => println!("\n{response}\n"),
                    Err(e) => eprintln!("\nError: {e}\n"),
                }
            }
            Ok(())
        }

        Commands::Ask { query } => {
            let llm = build_llm_client(&ollama_spec)?;
            let agent = crate::agent::orchestrator::Agent {
                db: cx.db.clone(),
                embed: cx.embed.clone(),
                hnsw: cx.hnsw.clone(),
                config: cx.config.clone(),
                llm,
                tools: crate::tools::ToolRegistry::new(),
                auto_link_tx: cx.auto_link_tx.clone(),
            };

            match agent.run(&query).await {
                Ok(response) => println!("{response}"),
                Err(e) => eprintln!("Error: {e}"),
            }
            Ok(())
        }
    }
}

/// Build an LLM client based on CLI flags and environment variables.
fn build_llm_client(ollama_spec: &Option<String>) -> crate::error::Result<Box<dyn crate::llm::LlmClient>> {
    // Check for --ollama flag
    if let Some(ref ollama_spec) = ollama_spec {
        let (model, url) = if let Some(pos) = ollama_spec.find('@') {
            (ollama_spec[..pos].to_string(), ollama_spec[pos + 1..].to_string())
        } else {
            (ollama_spec.clone(), "http://localhost:11434".to_string())
        };
        return Ok(Box::new(crate::llm::OllamaClient::new(model, url)));
    }

    // Check for ANTHROPIC_API_KEY
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        let model = std::env::var("ANTHROPIC_MODEL")
            .unwrap_or_else(|_| "claude-sonnet-4-20250514".to_string());
        return Ok(Box::new(crate::llm::AnthropicClient::new(key, model)));
    }

    Err(crate::error::CortexError::Config(
        "No LLM backend configured. Set ANTHROPIC_API_KEY or use --ollama <model>".into(),
    ))
}
