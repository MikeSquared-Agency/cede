use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

use crate::config::Config;
use crate::db::Db;
use crate::db::queries;
use crate::embed::EmbedHandle;
use crate::error::Result;
use crate::hnsw::VectorIndex;
use crate::llm::LlmClient;
use crate::memory;
use crate::tools::ToolRegistry;
use crate::types::*;

/// The agent. Owns the LLM client, tool registry, and a handle to the shared
/// `CortexEmbedded` infrastructure (db, embed, hnsw).
pub struct Agent {
    pub db: Db,
    pub embed: EmbedHandle,
    pub hnsw: Arc<RwLock<VectorIndex>>,
    pub config: Config,
    pub llm: Box<dyn LlmClient>,
    pub tools: ToolRegistry,
    pub auto_link_tx: async_channel::Sender<NodeId>,
}

impl Agent {
    /// Run the agent loop for a single user input.
    pub async fn run(&self, input: &str) -> Result<String> {
        // Create session node
        let session = Node::session(input);
        let session_id = session.id.clone();
        self.db
            .call({
                let s = session.clone();
                move |conn| queries::insert_node(conn, &s)
            })
            .await?;

        // Build briefing for system prompt
        let brief = memory::briefing_with_kinds(
            &self.db,
            &self.embed,
            &self.hnsw,
            &self.config,
            input,
            &[
                NodeKind::Soul,
                NodeKind::Belief,
                NodeKind::Goal,
                NodeKind::Fact,
                NodeKind::Decision,
                NodeKind::Pattern,
                NodeKind::Capability,
                NodeKind::Limitation,
            ],
            12,
        )
        .await?;

        let mut messages = vec![
            Message::system(brief.context_doc),
            Message::user(input),
        ];

        let mut iter: usize = 0;

        loop {
            iter += 1;

            // Write LoopIteration node
            let iter_node = Node::loop_iteration(iter, &session_id);
            let iter_id = iter_node.id.clone();
            self.db
                .call({
                    let n = iter_node.clone();
                    move |conn| queries::insert_node(conn, &n)
                })
                .await?;

            // Link iteration to session
            let edge = Edge::new(iter_id.clone(), session_id.clone(), EdgeKind::PartOf);
            self.db
                .call(move |conn| queries::insert_edge(conn, &edge))
                .await?;

            // LLM call
            let start = Instant::now();
            let response = self.llm.complete(&messages).await?;
            let latency_ms = start.elapsed().as_millis() as u64;

            // Record LlmCall node
            let llm_node = Node {
                kind: NodeKind::LlmCall,
                title: format!("LLM call iter {iter}"),
                body: Some(
                    serde_json::json!({
                        "model": self.llm.model_name(),
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "latency_ms": latency_ms,
                    })
                    .to_string(),
                ),
                ..Node::new(NodeKind::LlmCall, format!("LLM call iter {iter}"))
            };
            let llm_id = llm_node.id.clone();
            self.db
                .call({
                    let n = llm_node;
                    move |conn| queries::insert_node(conn, &n)
                })
                .await?;
            let llm_edge = Edge::new(llm_id, iter_id.clone(), EdgeKind::PartOf);
            self.db
                .call(move |conn| queries::insert_edge(conn, &llm_edge))
                .await?;

            match response.stop_reason {
                StopReason::ToolUse => {
                    let tool_name = response.tool_name.unwrap_or_default();
                    let tool_input = response.tool_input.unwrap_or(serde_json::Value::Null);
                    let result = self
                        .tools
                        .execute(
                            &tool_name,
                            tool_input,
                            iter_id,
                            &self.db,
                            &self.auto_link_tx,
                        )
                        .await?;
                    messages.push(Message::assistant(&response.text));
                    messages.push(Message::tool_result(result.output));
                }
                StopReason::EndTurn | StopReason::MaxTokens => {
                    // Store fact from response
                    let fact = Node::fact_from_response(&response.text, &session_id);
                    let fact_id = fact.id.clone();
                    self.db
                        .call({
                            let f = fact;
                            move |conn| queries::insert_node(conn, &f)
                        })
                        .await?;
                    let derives = Edge::new(
                        fact_id.clone(),
                        session_id.clone(),
                        EdgeKind::DerivesFrom,
                    );
                    self.db
                        .call(move |conn| queries::insert_edge(conn, &derives))
                        .await?;
                    let _ = self.auto_link_tx.try_send(fact_id);

                    // Context compaction: extract facts from long conversations
                    if messages.len() > self.config.compaction_threshold {
                        let _ = crate::compact_session(
                            &self.db,
                            &self.embed,
                            &self.hnsw,
                            &self.config,
                            &self.auto_link_tx,
                            &session_id,
                            &messages,
                            self.llm.as_ref(),
                        )
                        .await;
                    }

                    return Ok(response.text);
                }
            }

            // Guard: max iterations
            if iter >= self.config.max_iterations {
                let limit_node = Node::new(NodeKind::Limitation, "Hit max iterations")
                    .with_body(format!(
                        "Task: {}. Stopped at {} iterations.",
                        input, iter
                    ))
                    .with_importance(0.7)
                    .with_decay_rate(0.02);
                self.db
                    .call(move |conn| queries::insert_node(conn, &limit_node))
                    .await?;
                break;
            }
        }

        Ok("Reached iteration limit without final answer.".into())
    }
}
