use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::db::Db;
use crate::db::queries;
use crate::error::{CortexError, Result};
use crate::types::*;

/// A tool the agent can call. The handler is an async function that
/// takes JSON input and returns a `ToolResult`.
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub trust: f32,
    pub handler: Arc<
        dyn Fn(serde_json::Value) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send>>
            + Send
            + Sync,
    >,
}

/// Registry of available tools.
pub struct ToolRegistry {
    tools: HashMap<String, Tool>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Tool) {
        self.tools.insert(tool.name.clone(), tool);
    }

    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.tools.get(name)
    }

    pub fn list(&self) -> Vec<&Tool> {
        self.tools.values().collect()
    }

    /// Execute a tool and write ToolCall + Fact nodes to the graph.
    pub async fn execute(
        &self,
        name: &str,
        input: serde_json::Value,
        iter_node: NodeId,
        db: &Db,
        auto_link_tx: &async_channel::Sender<NodeId>,
    ) -> Result<ToolResult> {
        let tool = self
            .get(name)
            .ok_or_else(|| CortexError::Tool(format!("unknown tool: {name}")))?;

        let trust = tool.trust;
        let result = (tool.handler)(input.clone()).await?;

        // Write ToolCall node
        let tool_call_node = Node {
            kind: NodeKind::ToolCall,
            title: format!("ToolCall: {name}"),
            body: Some(serde_json::json!({
                "tool": name,
                "input": input,
                "output": &result.output,
                "success": result.success,
            }).to_string()),
            trust_score: trust as f64,
            ..Node::new(NodeKind::ToolCall, format!("ToolCall: {name}"))
        };
        let tc_id = tool_call_node.id.clone();
        db.call({
            let node = tool_call_node.clone();
            move |conn| queries::insert_node(conn, &node)
        })
        .await?;

        // Link ToolCall → LoopIteration via PartOf
        let edge = Edge::new(tc_id.clone(), iter_node, EdgeKind::PartOf);
        db.call(move |conn| queries::insert_edge(conn, &edge)).await?;

        // If success, write Fact derived from tool result
        if result.success {
            let fact = Node::new(NodeKind::Fact, format!("Result: {name}"))
                .with_body(&result.output)
                .with_trust(trust as f64);
            let fact_id = fact.id.clone();
            db.call({
                let fact = fact.clone();
                move |conn| queries::insert_node(conn, &fact)
            })
            .await?;

            let derives = Edge::new(fact_id.clone(), tc_id, EdgeKind::DerivesFrom);
            db.call(move |conn| queries::insert_edge(conn, &derives)).await?;

            // Enqueue for auto-linking
            let _ = auto_link_tx.try_send(fact_id);
        }

        Ok(result)
    }

    /// Build a JSON schema description of all tools (for LLM system prompt).
    pub fn schema_json(&self) -> serde_json::Value {
        let tools: Vec<serde_json::Value> = self
            .tools
            .values()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                })
            })
            .collect();
        serde_json::Value::Array(tools)
    }
}
