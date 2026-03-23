use crate::error::Result;
use crate::types::*;

// ─── LLM Client trait ───────────────────────────────────

/// Abstraction over LLM backends. Implement this trait for Anthropic, Ollama,
/// OpenAI, or a mock client for testing.
#[async_trait::async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete(&self, messages: &[Message]) -> Result<LlmResponse>;

    /// Return the model name for recording in LlmCall nodes.
    fn model_name(&self) -> &str;
}

// ─── Mock client for testing ────────────────────────────

/// A mock LLM client that returns pre-scripted responses in FIFO order.
pub struct MockLlmClient {
    pub responses: std::sync::Mutex<std::collections::VecDeque<LlmResponse>>,
    pub name: String,
}

impl MockLlmClient {
    pub fn new(responses: Vec<LlmResponse>) -> Self {
        Self {
            responses: std::sync::Mutex::new(responses.into()),
            name: "mock".to_string(),
        }
    }
}

#[async_trait::async_trait]
impl LlmClient for MockLlmClient {
    async fn complete(&self, _messages: &[Message]) -> Result<LlmResponse> {
        let mut queue = self.responses.lock().unwrap();
        queue
            .pop_front()
            .ok_or_else(|| crate::error::CortexError::Llm("no more mock responses".into()))
    }

    fn model_name(&self) -> &str {
        &self.name
    }
}

// ─── Anthropic client (Phase 5) ─────────────────────────

pub struct AnthropicClient {
    pub client: reqwest::Client,
    pub api_key: String,
    pub model: String,
}

impl AnthropicClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
        }
    }
}

#[async_trait::async_trait]
impl LlmClient for AnthropicClient {
    async fn complete(&self, messages: &[Message]) -> Result<LlmResponse> {
        // Split system message from conversation
        let system_msg = messages
            .iter()
            .find(|m| m.role == Role::System)
            .map(|m| m.content.clone())
            .unwrap_or_default();

        let chat_messages: Vec<serde_json::Value> = messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|m| {
                serde_json::json!({
                    "role": match m.role {
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        Role::Tool => "user", // tool results sent as user
                        Role::System => unreachable!(),
                    },
                    "content": m.content,
                })
            })
            .collect();

        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": 4096,
            "system": system_msg,
            "messages": chat_messages,
        });

        let resp = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::error::CortexError::Llm(format!("request: {e}")))?;

        let status = resp.status();
        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| crate::error::CortexError::Llm(format!("parse: {e}")))?;

        if !status.is_success() {
            return Err(crate::error::CortexError::Llm(format!(
                "API error {status}: {}",
                json
            )));
        }

        // Parse response
        let stop = json["stop_reason"].as_str().unwrap_or("end_turn");
        let stop_reason = match stop {
            "tool_use" => StopReason::ToolUse,
            "max_tokens" => StopReason::MaxTokens,
            _ => StopReason::EndTurn,
        };

        let mut text = String::new();
        let mut tool_name = None;
        let mut tool_input = None;

        if let Some(content) = json["content"].as_array() {
            for block in content {
                match block["type"].as_str() {
                    Some("text") => {
                        if let Some(t) = block["text"].as_str() {
                            text.push_str(t);
                        }
                    }
                    Some("tool_use") => {
                        tool_name = block["name"].as_str().map(String::from);
                        tool_input = Some(block["input"].clone());
                    }
                    _ => {}
                }
            }
        }

        let input_tokens = json["usage"]["input_tokens"].as_u64().unwrap_or(0) as usize;
        let output_tokens = json["usage"]["output_tokens"].as_u64().unwrap_or(0) as usize;

        Ok(LlmResponse {
            text,
            stop_reason,
            tool_name,
            tool_input,
            input_tokens,
            output_tokens,
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// ─── Ollama client (Phase 5) ────────────────────────────

pub struct OllamaClient {
    pub client: reqwest::Client,
    pub url: String,
    pub model: String,
}

impl OllamaClient {
    pub fn new(model: String, url: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            url,
            model,
        }
    }
}

#[async_trait::async_trait]
impl LlmClient for OllamaClient {
    async fn complete(&self, messages: &[Message]) -> Result<LlmResponse> {
        let msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": match m.role {
                        Role::System => "system",
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        Role::Tool => "user",
                    },
                    "content": m.content,
                })
            })
            .collect();

        let body = serde_json::json!({
            "model": self.model,
            "messages": msgs,
            "stream": false,
        });

        let resp = self
            .client
            .post(format!("{}/api/chat", self.url))
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::error::CortexError::Llm(format!("ollama: {e}")))?;

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| crate::error::CortexError::Llm(format!("ollama parse: {e}")))?;

        let text = json["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(LlmResponse {
            text,
            stop_reason: StopReason::EndTurn,
            tool_name: None,
            tool_input: None,
            input_tokens: 0,
            output_tokens: 0,
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
