//! Core types for the Open Agent SDK

use crate::hooks::Hooks;
use crate::tools::Tool;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Options for configuring an AI agent
#[derive(Clone)]
pub struct AgentOptions {
    /// System prompt to set the agent's behavior
    pub system_prompt: String,

    /// Model name (e.g., "qwen2.5-32b-instruct")
    pub model: String,

    /// OpenAI-compatible endpoint URL
    pub base_url: String,

    /// API key (most local servers don't need this)
    pub api_key: String,

    /// Maximum conversation turns
    pub max_turns: u32,

    /// Maximum tokens to generate (None uses provider default)
    pub max_tokens: Option<u32>,

    /// Sampling temperature (0.0 to 2.0)
    pub temperature: f32,

    /// Request timeout in seconds
    pub timeout: u64,

    /// Tools available to the agent
    pub tools: Vec<Arc<Tool>>,

    /// Enable automatic tool execution
    pub auto_execute_tools: bool,

    /// Maximum tool iterations in auto mode
    pub max_tool_iterations: u32,

    /// Lifecycle hooks for monitoring and control
    pub hooks: Hooks,
}

impl std::fmt::Debug for AgentOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentOptions")
            .field("system_prompt", &self.system_prompt)
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("api_key", &"***")
            .field("max_turns", &self.max_turns)
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .field("timeout", &self.timeout)
            .field("tools", &format!("{} tools", self.tools.len()))
            .field("auto_execute_tools", &self.auto_execute_tools)
            .field("max_tool_iterations", &self.max_tool_iterations)
            .field("hooks", &self.hooks)
            .finish()
    }
}

impl Default for AgentOptions {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            model: String::new(),
            base_url: String::new(),
            api_key: "not-needed".to_string(),
            max_turns: 1,
            max_tokens: Some(4096),
            temperature: 0.7,
            timeout: 60,
            tools: Vec::new(),
            auto_execute_tools: false,
            max_tool_iterations: 5,
            hooks: Hooks::new(),
        }
    }
}

impl AgentOptions {
    /// Create a new builder for AgentOptions
    pub fn builder() -> AgentOptionsBuilder {
        AgentOptionsBuilder::default()
    }
}

/// Builder for AgentOptions
#[derive(Default)]
pub struct AgentOptionsBuilder {
    system_prompt: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
    api_key: Option<String>,
    max_turns: Option<u32>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    timeout: Option<u64>,
    tools: Vec<Arc<Tool>>,
    auto_execute_tools: Option<bool>,
    max_tool_iterations: Option<u32>,
    hooks: Hooks,
}

impl std::fmt::Debug for AgentOptionsBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentOptionsBuilder")
            .field("system_prompt", &self.system_prompt)
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("tools", &format!("{} tools", self.tools.len()))
            .finish()
    }
}

impl AgentOptionsBuilder {
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn max_turns(mut self, turns: u32) -> Self {
        self.max_turns = Some(turns);
        self
    }

    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn timeout(mut self, timeout: u64) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn auto_execute_tools(mut self, auto: bool) -> Self {
        self.auto_execute_tools = Some(auto);
        self
    }

    pub fn max_tool_iterations(mut self, iterations: u32) -> Self {
        self.max_tool_iterations = Some(iterations);
        self
    }

    pub fn tool(mut self, tool: Tool) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools.extend(tools.into_iter().map(Arc::new));
        self
    }

    pub fn hooks(mut self, hooks: Hooks) -> Self {
        self.hooks = hooks;
        self
    }

    pub fn build(self) -> crate::Result<AgentOptions> {
        let model = self
            .model
            .ok_or_else(|| crate::Error::config("model is required"))?;

        let base_url = self
            .base_url
            .ok_or_else(|| crate::Error::config("base_url is required"))?;

        Ok(AgentOptions {
            system_prompt: self.system_prompt.unwrap_or_default(),
            model,
            base_url,
            api_key: self.api_key.unwrap_or_else(|| "not-needed".to_string()),
            max_turns: self.max_turns.unwrap_or(1),
            max_tokens: self.max_tokens.or(Some(4096)),
            temperature: self.temperature.unwrap_or(0.7),
            timeout: self.timeout.unwrap_or(60),
            tools: self.tools,
            auto_execute_tools: self.auto_execute_tools.unwrap_or(false),
            max_tool_iterations: self.max_tool_iterations.unwrap_or(5),
            hooks: self.hooks,
        })
    }
}

/// Message role in the conversation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

/// Content block types that can appear in messages
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text(TextBlock),
    ToolUse(ToolUseBlock),
    ToolResult(ToolResultBlock),
}

/// Text content block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBlock {
    pub text: String,
}

impl TextBlock {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// Tool use content block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUseBlock {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

impl ToolUseBlock {
    pub fn new(id: impl Into<String>, name: impl Into<String>, input: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            input,
        }
    }
}

/// Tool result block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultBlock {
    pub tool_use_id: String,
    pub content: serde_json::Value,
}

impl ToolResultBlock {
    pub fn new(tool_use_id: impl Into<String>, content: serde_json::Value) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content,
        }
    }
}

/// A message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: Vec<ContentBlock>,
}

impl Message {
    pub fn new(role: MessageRole, content: Vec<ContentBlock>) -> Self {
        Self { role, content }
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: vec![ContentBlock::Text(TextBlock::new(text))],
        }
    }

    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content,
        }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: vec![ContentBlock::Text(TextBlock::new(text))],
        }
    }

    /// Create a user message with custom content blocks
    pub fn user_with_blocks(content: Vec<ContentBlock>) -> Self {
        Self {
            role: MessageRole::User,
            content,
        }
    }
}

/// OpenAI API message format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// OpenAI tool call format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: OpenAIFunction,
}

/// OpenAI function format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunction {
    pub name: String,
    pub arguments: String,
}

/// OpenAI API request
#[derive(Debug, Clone, Serialize)]
pub struct OpenAIRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
}

/// OpenAI API streaming chunk
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<OpenAIChoice>,
}

/// OpenAI choice in streaming response
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIChoice {
    pub index: u32,
    pub delta: OpenAIDelta,
    pub finish_reason: Option<String>,
}

/// OpenAI delta in streaming response
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCallDelta>>,
}

/// OpenAI tool call delta
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIToolCallDelta {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "type")]
    pub call_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<OpenAIFunctionDelta>,
}

/// OpenAI function delta
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIFunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_options_builder() {
        let options = AgentOptions::builder()
            .system_prompt("Test prompt")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .api_key("test-key")
            .max_turns(5)
            .max_tokens(1000)
            .temperature(0.5)
            .timeout(30)
            .auto_execute_tools(true)
            .max_tool_iterations(10)
            .build()
            .unwrap();

        assert_eq!(options.system_prompt, "Test prompt");
        assert_eq!(options.model, "test-model");
        assert_eq!(options.base_url, "http://localhost:1234/v1");
        assert_eq!(options.api_key, "test-key");
        assert_eq!(options.max_turns, 5);
        assert_eq!(options.max_tokens, Some(1000));
        assert_eq!(options.temperature, 0.5);
        assert_eq!(options.timeout, 30);
        assert!(options.auto_execute_tools);
        assert_eq!(options.max_tool_iterations, 10);
    }

    #[test]
    fn test_agent_options_builder_defaults() {
        let options = AgentOptions::builder()
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        assert_eq!(options.system_prompt, "");
        assert_eq!(options.api_key, "not-needed");
        assert_eq!(options.max_turns, 1);
        assert_eq!(options.max_tokens, Some(4096));
        assert_eq!(options.temperature, 0.7);
        assert_eq!(options.timeout, 60);
        assert!(!options.auto_execute_tools);
        assert_eq!(options.max_tool_iterations, 5);
    }

    #[test]
    fn test_agent_options_builder_missing_required() {
        // Missing model
        let result = AgentOptions::builder()
            .base_url("http://localhost:1234/v1")
            .build();
        assert!(result.is_err());

        // Missing base_url
        let result = AgentOptions::builder().model("test-model").build();
        assert!(result.is_err());
    }

    #[test]
    fn test_message_user() {
        let msg = Message::user("Hello");
        assert!(matches!(msg.role, MessageRole::User));
        assert_eq!(msg.content.len(), 1);
        match &msg.content[0] {
            ContentBlock::Text(text) => assert_eq!(text.text, "Hello"),
            _ => panic!("Expected TextBlock"),
        }
    }

    #[test]
    fn test_message_system() {
        let msg = Message::system("System prompt");
        assert!(matches!(msg.role, MessageRole::System));
        assert_eq!(msg.content.len(), 1);
        match &msg.content[0] {
            ContentBlock::Text(text) => assert_eq!(text.text, "System prompt"),
            _ => panic!("Expected TextBlock"),
        }
    }

    #[test]
    fn test_message_assistant() {
        let content = vec![ContentBlock::Text(TextBlock::new("Response"))];
        let msg = Message::assistant(content);
        assert!(matches!(msg.role, MessageRole::Assistant));
        assert_eq!(msg.content.len(), 1);
    }

    #[test]
    fn test_text_block() {
        let block = TextBlock::new("Hello");
        assert_eq!(block.text, "Hello");
    }

    #[test]
    fn test_tool_use_block() {
        let input = serde_json::json!({"arg": "value"});
        let block = ToolUseBlock::new("call_123", "tool_name", input.clone());
        assert_eq!(block.id, "call_123");
        assert_eq!(block.name, "tool_name");
        assert_eq!(block.input, input);
    }

    #[test]
    fn test_tool_result_block() {
        let content = serde_json::json!({"result": "success"});
        let block = ToolResultBlock::new("call_123", content.clone());
        assert_eq!(block.tool_use_id, "call_123");
        assert_eq!(block.content, content);
    }

    #[test]
    fn test_message_role_serialization() {
        assert_eq!(
            serde_json::to_string(&MessageRole::User).unwrap(),
            "\"user\""
        );
        assert_eq!(
            serde_json::to_string(&MessageRole::System).unwrap(),
            "\"system\""
        );
        assert_eq!(
            serde_json::to_string(&MessageRole::Assistant).unwrap(),
            "\"assistant\""
        );
        assert_eq!(
            serde_json::to_string(&MessageRole::Tool).unwrap(),
            "\"tool\""
        );
    }

    #[test]
    fn test_openai_request_serialization() {
        let request = OpenAIRequest {
            model: "gpt-3.5".to_string(),
            messages: vec![OpenAIMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: true,
            max_tokens: Some(100),
            temperature: Some(0.7),
            tools: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("gpt-3.5"));
        assert!(json.contains("Hello"));
        assert!(json.contains("\"stream\":true"));
    }

    #[test]
    fn test_openai_chunk_deserialization() {
        let json = r#"{
            "id": "chunk_1",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-3.5",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": "Hello"
                },
                "finish_reason": null
            }]
        }"#;

        let chunk: OpenAIChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.id, "chunk_1");
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
    }

    #[test]
    fn test_content_block_serialization() {
        let text_block = ContentBlock::Text(TextBlock::new("Hello"));
        let json = serde_json::to_string(&text_block).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("Hello"));
    }

    #[test]
    fn test_agent_options_clone() {
        let options1 = AgentOptions::builder()
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let options2 = options1.clone();
        assert_eq!(options1.model, options2.model);
        assert_eq!(options1.base_url, options2.base_url);
    }
}
