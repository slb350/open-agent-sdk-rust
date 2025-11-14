//! Core type definitions for the Open Agent SDK.
//!
//! This module contains the fundamental data structures used throughout the SDK for
//! configuring and interacting with AI agents. The type system is organized into three
//! main categories:
//!
//! # Agent Configuration
//!
//! - [`AgentOptions`]: Main configuration struct for agent behavior, model settings,
//!   and tool management
//! - [`AgentOptionsBuilder`]: Builder pattern implementation for constructing
//!   [`AgentOptions`] with validation
//!
//! # Message System
//!
//! The SDK uses a flexible message system that supports multi-modal content:
//!
//! - [`Message`]: Container for conversation messages with role and content
//! - [`MessageRole`]: Enum defining who sent the message (System, User, Assistant, Tool)
//! - [`ContentBlock`]: Enum for different content types (text, tool use, tool results)
//! - [`TextBlock`]: Simple text content
//! - [`ToolUseBlock`]: Represents an AI request to execute a tool
//! - [`ToolResultBlock`]: Contains the result of a tool execution
//!
//! # OpenAI API Compatibility
//!
//! The SDK communicates with LLM providers using the OpenAI-compatible API format.
//! These types handle serialization/deserialization for streaming responses:
//!
//! - [`OpenAIRequest`]: Request payload sent to the API
//! - [`OpenAIMessage`]: Message format for OpenAI API
//! - [`OpenAIChunk`]: Streaming response chunk from the API
//! - [`OpenAIToolCall`], [`OpenAIFunction`]: Tool calling format
//! - [`OpenAIDelta`], [`OpenAIToolCallDelta`]: Incremental updates in streaming
//!
//! # Architecture Overview
//!
//! The type system is designed to:
//!
//! 1. **Separate concerns**: Internal SDK types (Message, ContentBlock) are distinct
//!    from API wire format (OpenAI types), allowing flexibility in provider support
//! 2. **Enable streaming**: OpenAI types support incremental delta parsing for
//!    real-time responses
//! 3. **Support tool use**: First-class support for function calling with proper
//!    request/response tracking
//! 4. **Provide ergonomics**: Builder pattern and convenience constructors make
//!    common operations simple
//!
//! # Example
//!
//! ```no_run
//! use open_agent::{AgentOptions, Message};
//!
//! // Build agent configuration
//! let options = AgentOptions::builder()
//!     .model("qwen2.5-32b-instruct")
//!     .base_url("http://localhost:1234/v1")
//!     .system_prompt("You are a helpful assistant")
//!     .max_turns(10)
//!     .auto_execute_tools(true)
//!     .build()
//!     .expect("Valid configuration");
//!
//! // Create a user message
//! let msg = Message::user("Hello, how are you?");
//! ```

use crate::Error;
use crate::hooks::Hooks;
use crate::tools::Tool;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ============================================================================
// NEWTYPE WRAPPERS FOR COMPILE-TIME TYPE SAFETY
// ============================================================================

/// Validated model name with compile-time type safety.
///
/// This newtype wrapper ensures that model names are validated at construction time
/// rather than at runtime, catching invalid configurations earlier in development.
///
/// # Validation Rules
///
/// - Must not be empty
/// - Must not be only whitespace
///
/// # Example
///
/// ```
/// use open_agent::ModelName;
///
/// // Valid model name
/// let model = ModelName::new("qwen2.5-32b-instruct").unwrap();
/// assert_eq!(model.as_str(), "qwen2.5-32b-instruct");
///
/// // Invalid: empty string
/// assert!(ModelName::new("").is_err());
///
/// // Invalid: whitespace only
/// assert!(ModelName::new("   ").is_err());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelName(String);

impl ModelName {
    /// Creates a new `ModelName` after validation.
    ///
    /// # Errors
    ///
    /// Returns an error if the model name is empty or contains only whitespace.
    pub fn new(name: impl Into<String>) -> crate::Result<Self> {
        let name = name.into();
        let trimmed = name.trim();

        if trimmed.is_empty() {
            return Err(Error::invalid_input(
                "Model name cannot be empty or whitespace",
            ));
        }

        Ok(ModelName(name))
    }

    /// Returns the model name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the `ModelName` and returns the inner `String`.
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl std::fmt::Display for ModelName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Validated base URL with compile-time type safety.
///
/// This newtype wrapper ensures that base URLs are validated at construction time
/// rather than at runtime, catching invalid configurations earlier in development.
///
/// # Validation Rules
///
/// - Must not be empty
/// - Must start with `http://` or `https://`
///
/// # Example
///
/// ```
/// use open_agent::BaseUrl;
///
/// // Valid base URLs
/// let url = BaseUrl::new("http://localhost:1234/v1").unwrap();
/// assert_eq!(url.as_str(), "http://localhost:1234/v1");
///
/// let url = BaseUrl::new("https://api.openai.com/v1").unwrap();
/// assert_eq!(url.as_str(), "https://api.openai.com/v1");
///
/// // Invalid: no http/https prefix
/// assert!(BaseUrl::new("localhost:1234").is_err());
///
/// // Invalid: empty string
/// assert!(BaseUrl::new("").is_err());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BaseUrl(String);

impl BaseUrl {
    /// Creates a new `BaseUrl` after validation.
    ///
    /// # Errors
    ///
    /// Returns an error if the URL is empty or doesn't start with http:// or https://.
    pub fn new(url: impl Into<String>) -> crate::Result<Self> {
        let url = url.into();
        let trimmed = url.trim();

        if trimmed.is_empty() {
            return Err(Error::invalid_input("base_url cannot be empty"));
        }

        if !trimmed.starts_with("http://") && !trimmed.starts_with("https://") {
            return Err(Error::invalid_input(
                "base_url must start with http:// or https://",
            ));
        }

        Ok(BaseUrl(url))
    }

    /// Returns the base URL as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the `BaseUrl` and returns the inner `String`.
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl std::fmt::Display for BaseUrl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Validated temperature value with compile-time type safety.
///
/// This newtype wrapper ensures that temperature values are validated at construction time
/// rather than at runtime, catching invalid configurations earlier in development.
///
/// # Validation Rules
///
/// - Must be between 0.0 and 2.0 (inclusive)
///
/// # Example
///
/// ```
/// use open_agent::Temperature;
///
/// // Valid temperatures
/// let temp = Temperature::new(0.7).unwrap();
/// assert_eq!(temp.value(), 0.7);
///
/// let temp = Temperature::new(0.0).unwrap();
/// assert_eq!(temp.value(), 0.0);
///
/// let temp = Temperature::new(2.0).unwrap();
/// assert_eq!(temp.value(), 2.0);
///
/// // Invalid: below range
/// assert!(Temperature::new(-0.1).is_err());
///
/// // Invalid: above range
/// assert!(Temperature::new(2.1).is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Temperature(f32);

impl Temperature {
    /// Creates a new `Temperature` after validation.
    ///
    /// # Errors
    ///
    /// Returns an error if the temperature is not between 0.0 and 2.0 (inclusive).
    pub fn new(temp: f32) -> crate::Result<Self> {
        if !(0.0..=2.0).contains(&temp) {
            return Err(Error::invalid_input(
                "temperature must be between 0.0 and 2.0",
            ));
        }

        Ok(Temperature(temp))
    }

    /// Returns the temperature value.
    pub fn value(&self) -> f32 {
        self.0
    }
}

impl std::fmt::Display for Temperature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ============================================================================
// AGENT CONFIGURATION
// ============================================================================

/// Configuration options for an AI agent instance.
///
/// `AgentOptions` controls all aspects of agent behavior including model selection,
/// conversation management, tool usage, and lifecycle hooks. This struct should be
/// constructed using [`AgentOptions::builder()`] rather than direct instantiation
/// to ensure required fields are validated.
///
/// # Architecture
///
/// The options are organized into several functional areas:
///
/// - **Model Configuration**: `model`, `base_url`, `api_key`, `temperature`, `max_tokens`
/// - **Conversation Control**: `system_prompt`, `max_turns`, `timeout`
/// - **Tool Management**: `tools`, `auto_execute_tools`, `max_tool_iterations`
/// - **Lifecycle Hooks**: `hooks` for monitoring and interception
///
/// # Thread Safety
///
/// Tools are wrapped in `Arc<Tool>` to allow efficient cloning and sharing across
/// threads, as agents may need to be cloned for parallel processing.
///
/// # Examples
///
/// ```no_run
/// use open_agent::AgentOptions;
///
/// let options = AgentOptions::builder()
///     .model("qwen2.5-32b-instruct")
///     .base_url("http://localhost:1234/v1")
///     .system_prompt("You are a helpful coding assistant")
///     .max_turns(5)
///     .temperature(0.7)
///     .build()
///     .expect("Valid configuration");
/// ```
#[derive(Clone)]
pub struct AgentOptions {
    /// System prompt that defines the agent's behavior and personality.
    ///
    /// This is sent as the first message in the conversation to establish
    /// context and instructions. Can be empty if no system-level guidance
    /// is needed.
    system_prompt: String,

    /// Model identifier for the LLM to use (e.g., "qwen2.5-32b-instruct", "gpt-4").
    ///
    /// This must match a model available at the configured `base_url`.
    /// Different models have varying capabilities for tool use, context
    /// length, and response quality.
    model: String,

    /// OpenAI-compatible API endpoint URL (e.g., "http://localhost:1234/v1").
    ///
    /// The SDK communicates using the OpenAI chat completions API format,
    /// which is widely supported by local inference servers (LM Studio,
    /// llama.cpp, vLLM) and cloud providers.
    base_url: String,

    /// API authentication key for the provider.
    ///
    /// Many local servers don't require authentication, so the default
    /// "not-needed" is often sufficient. For cloud providers like OpenAI,
    /// set this to your actual API key.
    api_key: String,

    /// Maximum number of conversation turns (user message + assistant response = 1 turn).
    ///
    /// This limits how long a conversation can continue. In auto-execution mode
    /// with tools, this prevents infinite loops. Set to 1 for single-shot
    /// interactions or higher for multi-turn conversations.
    max_turns: u32,

    /// Maximum tokens the model should generate in a single response.
    ///
    /// `None` uses the provider's default. Lower values constrain response
    /// length, which can be useful for cost control or ensuring concise answers.
    /// Note this is separate from the model's context window size.
    max_tokens: Option<u32>,

    /// Sampling temperature for response generation (typically 0.0 to 2.0).
    ///
    /// - 0.0: Deterministic, always picks most likely tokens
    /// - 0.7: Balanced creativity and consistency (default)
    /// - 1.0+: More random and creative responses
    ///
    /// Lower temperatures are better for factual tasks, higher for creative ones.
    temperature: f32,

    /// HTTP request timeout in seconds.
    ///
    /// Maximum time to wait for the API to respond. Applies per API call,
    /// not to the entire conversation. Increase for slower models or when
    /// expecting long responses.
    timeout: u64,

    /// Tools available for the agent to use during conversations.
    ///
    /// Tools are wrapped in `Arc` for efficient cloning. When the agent
    /// receives a tool use request, it looks up the tool by name in this
    /// vector. Empty by default.
    tools: Vec<Arc<Tool>>,

    /// Whether to automatically execute tools and continue the conversation.
    ///
    /// - `true`: SDK automatically executes tool calls and sends results back
    ///   to the model, continuing until no more tools are requested
    /// - `false`: Tool calls are returned to the caller, who must manually
    ///   execute them and provide results
    ///
    /// Auto-execution is convenient but gives less control. Manual execution
    /// allows for approval workflows and selective tool access.
    auto_execute_tools: bool,

    /// Maximum iterations of tool execution in automatic mode.
    ///
    /// Prevents infinite loops where the agent continuously requests tools.
    /// Each tool execution attempt counts as one iteration. Only relevant
    /// when `auto_execute_tools` is true.
    max_tool_iterations: u32,

    /// Lifecycle hooks for observing and intercepting agent operations.
    ///
    /// Hooks allow you to inject custom logic at various points:
    /// - Before/after API requests
    /// - Tool execution interception
    /// - Response streaming callbacks
    ///
    /// Useful for logging, metrics, debugging, and implementing custom
    /// authorization logic.
    hooks: Hooks,
}

/// Custom Debug implementation to prevent sensitive data leakage.
///
/// We override the default Debug implementation because:
/// 1. The `api_key` field may contain sensitive credentials that shouldn't
///    appear in logs or error messages
/// 2. The `tools` vector contains Arc-wrapped closures that don't debug nicely,
///    so we show a count instead
///
/// This ensures that debug output is safe for logging while remaining useful
/// for troubleshooting.
impl std::fmt::Debug for AgentOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentOptions")
            .field("system_prompt", &self.system_prompt)
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            // Mask API key to prevent credential leakage in logs
            .field("api_key", &"***")
            .field("max_turns", &self.max_turns)
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .field("timeout", &self.timeout)
            // Show tool count instead of trying to debug Arc<Tool> contents
            .field("tools", &format!("{} tools", self.tools.len()))
            .field("auto_execute_tools", &self.auto_execute_tools)
            .field("max_tool_iterations", &self.max_tool_iterations)
            .field("hooks", &self.hooks)
            .finish()
    }
}

/// Default values optimized for common single-turn use cases.
///
/// These defaults are chosen to:
/// - Require explicit configuration of critical fields (model, base_url)
/// - Provide safe, sensible defaults for optional fields
/// - Work with local inference servers that don't need authentication
impl Default for AgentOptions {
    fn default() -> Self {
        Self {
            // Empty string forces users to explicitly set context
            system_prompt: String::new(),
            // Empty string forces users to explicitly choose a model
            model: String::new(),
            // Empty string forces users to explicitly configure the endpoint
            base_url: String::new(),
            // Most local servers (LM Studio, llama.cpp) don't require auth
            api_key: "not-needed".to_string(),
            // Default to single-shot interaction; users opt into conversations
            max_turns: 1,
            // 4096 is a reasonable default that works with most models
            // while preventing runaway generation costs
            max_tokens: Some(4096),
            // 0.7 balances creativity with consistency for general use
            temperature: 0.7,
            // 60 seconds handles most requests without timing out prematurely
            timeout: 60,
            // No tools by default; users explicitly add capabilities
            tools: Vec::new(),
            // Manual tool execution by default for safety and control
            auto_execute_tools: false,
            // 5 iterations prevent infinite loops while allowing multi-step workflows
            max_tool_iterations: 5,
            // Empty hooks for no-op behavior
            hooks: Hooks::new(),
        }
    }
}

impl AgentOptions {
    /// Creates a new builder for constructing [`AgentOptions`].
    ///
    /// The builder pattern is used because:
    /// 1. Some fields are required (model, base_url) and need validation
    /// 2. Many fields have sensible defaults that can be overridden
    /// 3. The API is more discoverable and readable than struct initialization
    ///
    /// # Example
    ///
    /// ```no_run
    /// use open_agent::AgentOptions;
    ///
    /// let options = AgentOptions::builder()
    ///     .model("qwen2.5-32b-instruct")
    ///     .base_url("http://localhost:1234/v1")
    ///     .build()
    ///     .expect("Valid configuration");
    /// ```
    pub fn builder() -> AgentOptionsBuilder {
        AgentOptionsBuilder::default()
    }

    /// Returns the system prompt.
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// Returns the model identifier.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns the base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Returns the API key.
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Returns the maximum number of conversation turns.
    pub fn max_turns(&self) -> u32 {
        self.max_turns
    }

    /// Returns the maximum tokens setting.
    pub fn max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    /// Returns the sampling temperature.
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Returns the HTTP timeout in seconds.
    pub fn timeout(&self) -> u64 {
        self.timeout
    }

    /// Returns a reference to the tools vector.
    pub fn tools(&self) -> &[Arc<Tool>] {
        &self.tools
    }

    /// Returns whether automatic tool execution is enabled.
    pub fn auto_execute_tools(&self) -> bool {
        self.auto_execute_tools
    }

    /// Returns the maximum tool execution iterations.
    pub fn max_tool_iterations(&self) -> u32 {
        self.max_tool_iterations
    }

    /// Returns a reference to the hooks configuration.
    pub fn hooks(&self) -> &Hooks {
        &self.hooks
    }
}

/// Builder for constructing [`AgentOptions`] with validation.
///
/// This builder implements the typestate pattern using `Option<T>` to track
/// which required fields have been set. The [`build()`](AgentOptionsBuilder::build)
/// method validates that all required fields are present before creating
/// the final [`AgentOptions`].
///
/// # Required Fields
///
/// - `model`: The LLM model identifier
/// - `base_url`: The API endpoint URL
///
/// All other fields have sensible defaults.
///
/// # Usage Pattern
///
/// 1. Call [`AgentOptions::builder()`]
/// 2. Chain method calls to set configuration
/// 3. Call [`build()`](AgentOptionsBuilder::build) to validate and create the final options
///
/// Methods return `self` for chaining, following the fluent interface pattern.
///
/// # Examples
///
/// ```no_run
/// use open_agent::AgentOptions;
/// use open_agent::Tool;
///
/// let calculator = Tool::new(
///     "calculate",
///     "Perform arithmetic",
///     serde_json::json!({
///         "type": "object",
///         "properties": {
///             "expression": {"type": "string"}
///         }
///     }),
///     |input| Box::pin(async move {
///         Ok(serde_json::json!({"result": 42}))
///     }),
/// );
///
/// let options = AgentOptions::builder()
///     .model("qwen2.5-32b-instruct")
///     .base_url("http://localhost:1234/v1")
///     .system_prompt("You are a helpful assistant")
///     .max_turns(10)
///     .temperature(0.8)
///     .tool(calculator)
///     .auto_execute_tools(true)
///     .build()
///     .expect("Valid configuration");
/// ```
#[derive(Default)]
pub struct AgentOptionsBuilder {
    /// Optional system prompt; defaults to empty if not set
    system_prompt: Option<String>,
    /// Required: model identifier
    model: Option<String>,
    /// Required: API endpoint URL
    base_url: Option<String>,
    /// Optional API key; defaults to "not-needed"
    api_key: Option<String>,
    /// Optional max turns; defaults to 1
    max_turns: Option<u32>,
    /// Optional max tokens; defaults to Some(4096)
    max_tokens: Option<u32>,
    /// Optional temperature; defaults to 0.7
    temperature: Option<f32>,
    /// Optional timeout; defaults to 60 seconds
    timeout: Option<u64>,
    /// Tools to provide; starts empty
    tools: Vec<Arc<Tool>>,
    /// Optional auto-execute flag; defaults to false
    auto_execute_tools: Option<bool>,
    /// Optional max iterations; defaults to 5
    max_tool_iterations: Option<u32>,
    /// Lifecycle hooks; defaults to empty
    hooks: Hooks,
}

/// Custom Debug implementation for builder to show minimal useful information.
///
/// Similar to [`AgentOptions`], we provide a simplified debug output that:
/// - Omits sensitive fields like API keys (not shown at all in builder)
/// - Shows tool count rather than tool details
/// - Focuses on the most important configuration fields
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

/// Builder methods for configuring agent options.
///
/// All methods follow the builder pattern: they consume `self`, update a field,
/// and return `self` for method chaining. The generic `impl Into<String>` parameters
/// allow passing `&str`, `String`, or any other type that converts to `String`.
impl AgentOptionsBuilder {
    /// Sets the system prompt that defines agent behavior.
    ///
    /// The system prompt is sent at the beginning of every conversation to
    /// establish context, personality, and instructions for the agent.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use open_agent::AgentOptions;
    /// let options = AgentOptions::builder()
    ///     .model("qwen2.5-32b-instruct")
    ///     .base_url("http://localhost:1234/v1")
    ///     .system_prompt("You are a helpful coding assistant. Be concise.")
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Sets the model identifier (required).
    ///
    /// This must match a model available at your configured endpoint.
    /// Common examples: "qwen2.5-32b-instruct", "gpt-4", "claude-3-sonnet".
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use open_agent::AgentOptions;
    /// let options = AgentOptions::builder()
    ///     .model("gpt-4")
    ///     .base_url("https://api.openai.com/v1")
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the API endpoint URL (required).
    ///
    /// Must be an OpenAI-compatible endpoint. Common values:
    /// - Local: "http://localhost:1234/v1" (LM Studio default)
    /// - OpenAI: <https://api.openai.com/v1>
    /// - Custom: Your inference server URL
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use open_agent::AgentOptions;
    /// let options = AgentOptions::builder()
    ///     .model("qwen2.5-32b-instruct")
    ///     .base_url("http://localhost:1234/v1")
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Sets the API key for authentication.
    ///
    /// Required for cloud providers like OpenAI. Most local servers don't
    /// need this - the default "not-needed" works fine.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use open_agent::AgentOptions;
    /// let options = AgentOptions::builder()
    ///     .model("gpt-4")
    ///     .base_url("https://api.openai.com/v1")
    ///     .api_key("sk-...")
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the maximum number of conversation turns.
    ///
    /// One turn = user message + assistant response. Higher values enable
    /// longer conversations but may increase costs and latency.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use open_agent::AgentOptions;
    /// let options = AgentOptions::builder()
    ///     .model("qwen2.5-32b-instruct")
    ///     .base_url("http://localhost:1234/v1")
    ///     .max_turns(10)  // Allow multi-turn conversation
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn max_turns(mut self, turns: u32) -> Self {
        self.max_turns = Some(turns);
        self
    }

    /// Sets the maximum tokens to generate per response.
    ///
    /// Constrains response length. Lower values reduce costs but may truncate
    /// responses. Higher values allow longer, more complete answers.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use open_agent::AgentOptions;
    /// let options = AgentOptions::builder()
    ///     .model("qwen2.5-32b-instruct")
    ///     .base_url("http://localhost:1234/v1")
    ///     .max_tokens(1000)  // Limit to shorter responses
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Sets the sampling temperature for response generation.
    ///
    /// Controls randomness:
    /// - 0.0: Deterministic, always picks most likely tokens
    /// - 0.7: Balanced (default)
    /// - 1.0+: More creative/random
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use open_agent::AgentOptions;
    /// let options = AgentOptions::builder()
    ///     .model("qwen2.5-32b-instruct")
    ///     .base_url("http://localhost:1234/v1")
    ///     .temperature(0.0)  // Deterministic for coding tasks
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Sets the HTTP request timeout in seconds.
    ///
    /// How long to wait for the API to respond. Increase for slower models
    /// or when expecting long responses.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use open_agent::AgentOptions;
    /// let options = AgentOptions::builder()
    ///     .model("qwen2.5-32b-instruct")
    ///     .base_url("http://localhost:1234/v1")
    ///     .timeout(120)  // 2 minutes for complex tasks
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn timeout(mut self, timeout: u64) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Enables or disables automatic tool execution.
    ///
    /// When true, the SDK automatically executes tool calls and continues
    /// the conversation. When false, tool calls are returned for manual
    /// handling, allowing approval workflows.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use open_agent::AgentOptions;
    /// let options = AgentOptions::builder()
    ///     .model("qwen2.5-32b-instruct")
    ///     .base_url("http://localhost:1234/v1")
    ///     .auto_execute_tools(true)  // Automatic execution
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn auto_execute_tools(mut self, auto: bool) -> Self {
        self.auto_execute_tools = Some(auto);
        self
    }

    /// Sets the maximum tool execution iterations in automatic mode.
    ///
    /// Prevents infinite loops where the agent continuously calls tools.
    /// Only relevant when `auto_execute_tools` is true.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use open_agent::AgentOptions;
    /// let options = AgentOptions::builder()
    ///     .model("qwen2.5-32b-instruct")
    ///     .base_url("http://localhost:1234/v1")
    ///     .auto_execute_tools(true)
    ///     .max_tool_iterations(10)  // Allow up to 10 tool calls
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn max_tool_iterations(mut self, iterations: u32) -> Self {
        self.max_tool_iterations = Some(iterations);
        self
    }

    /// Adds a single tool to the agent's available tools.
    ///
    /// The tool is wrapped in `Arc` for efficient sharing. Can be called
    /// multiple times to add multiple tools.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use open_agent::AgentOptions;
    /// # use open_agent::Tool;
    /// let calculator = Tool::new(
    ///     "calculate",
    ///     "Evaluate a math expression",
    ///     serde_json::json!({"type": "object"}),
    ///     |input| Box::pin(async move { Ok(serde_json::json!({"result": 42})) }),
    /// );
    ///
    /// let options = AgentOptions::builder()
    ///     .model("qwen2.5-32b-instruct")
    ///     .base_url("http://localhost:1234/v1")
    ///     .tool(calculator)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn tool(mut self, tool: Tool) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    /// Adds multiple tools at once to the agent's available tools.
    ///
    /// Convenience method for bulk tool addition. All tools are wrapped
    /// in `Arc` automatically.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use open_agent::AgentOptions;
    /// # use open_agent::Tool;
    /// let tools = vec![
    ///     Tool::new("add", "Add numbers", serde_json::json!({}),
    ///         |input| Box::pin(async move { Ok(serde_json::json!({})) })),
    ///     Tool::new("multiply", "Multiply numbers", serde_json::json!({}),
    ///         |input| Box::pin(async move { Ok(serde_json::json!({})) })),
    /// ];
    ///
    /// let options = AgentOptions::builder()
    ///     .model("qwen2.5-32b-instruct")
    ///     .base_url("http://localhost:1234/v1")
    ///     .tools(tools)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools.extend(tools.into_iter().map(Arc::new));
        self
    }

    /// Sets lifecycle hooks for monitoring and intercepting agent operations.
    ///
    /// Hooks allow custom logic at various points: before/after API calls,
    /// tool execution, response streaming, etc. Useful for logging, metrics,
    /// debugging, and custom authorization.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use open_agent::{AgentOptions, Hooks, HookDecision};
    /// let hooks = Hooks::new()
    ///     .add_user_prompt_submit(|event| async move {
    ///         println!("User prompt: {}", event.prompt);
    ///         Some(HookDecision::continue_())
    ///     });
    ///
    /// let options = AgentOptions::builder()
    ///     .model("qwen2.5-32b-instruct")
    ///     .base_url("http://localhost:1234/v1")
    ///     .hooks(hooks)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn hooks(mut self, hooks: Hooks) -> Self {
        self.hooks = hooks;
        self
    }

    /// Validates configuration and builds the final [`AgentOptions`].
    ///
    /// This method performs validation to ensure required fields are set and
    /// applies default values for optional fields. Returns an error if
    /// validation fails.
    ///
    /// # Required Fields
    ///
    /// - `model`: Must be set or build() returns an error
    /// - `base_url`: Must be set or build() returns an error
    ///
    /// # Errors
    ///
    /// Returns a configuration error if any required field is missing.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use open_agent::AgentOptions;
    /// // Success - all required fields set
    /// let options = AgentOptions::builder()
    ///     .model("qwen2.5-32b-instruct")
    ///     .base_url("http://localhost:1234/v1")
    ///     .build()
    ///     .expect("Valid configuration");
    ///
    /// // Error - missing model
    /// let result = AgentOptions::builder()
    ///     .base_url("http://localhost:1234/v1")
    ///     .build();
    /// assert!(result.is_err());
    /// ```
    pub fn build(self) -> crate::Result<AgentOptions> {
        // Validate required fields - these must be explicitly set by the user
        // because they're fundamental to connecting to an LLM provider
        let model = self
            .model
            .ok_or_else(|| crate::Error::config("model is required"))?;

        let base_url = self
            .base_url
            .ok_or_else(|| crate::Error::config("base_url is required"))?;

        // Validate model is not empty or whitespace
        if model.trim().is_empty() {
            return Err(crate::Error::invalid_input(
                "model cannot be empty or whitespace",
            ));
        }

        // Validate base_url is not empty and has valid URL format
        if base_url.trim().is_empty() {
            return Err(crate::Error::invalid_input("base_url cannot be empty"));
        }
        // Check if URL has a valid scheme (http:// or https://)
        if !base_url.starts_with("http://") && !base_url.starts_with("https://") {
            return Err(crate::Error::invalid_input(
                "base_url must start with http:// or https://",
            ));
        }

        // Validate temperature is in valid range (0.0 to 2.0)
        let temperature = self.temperature.unwrap_or(0.7);
        if !(0.0..=2.0).contains(&temperature) {
            return Err(crate::Error::invalid_input(
                "temperature must be between 0.0 and 2.0",
            ));
        }

        // Validate max_tokens if set
        let max_tokens = self.max_tokens.or(Some(4096));
        if let Some(tokens) = max_tokens {
            if tokens == 0 {
                return Err(crate::Error::invalid_input(
                    "max_tokens must be greater than 0",
                ));
            }
        }

        // Construct the final options, applying defaults where values weren't set
        Ok(AgentOptions {
            // Empty system prompt is valid - not all use cases need one
            system_prompt: self.system_prompt.unwrap_or_default(),
            model,
            base_url,
            // Default API key works for most local servers
            api_key: self.api_key.unwrap_or_else(|| "not-needed".to_string()),
            // Default to single-turn for simplicity
            max_turns: self.max_turns.unwrap_or(1),
            max_tokens,
            temperature,
            // Conservative timeout that works for most requests
            timeout: self.timeout.unwrap_or(60),
            // Tools vector was built up during configuration, use as-is
            tools: self.tools,
            // Manual execution by default for safety and control
            auto_execute_tools: self.auto_execute_tools.unwrap_or(false),
            // Reasonable limit to prevent runaway tool loops
            max_tool_iterations: self.max_tool_iterations.unwrap_or(5),
            // Hooks were built up during configuration, use as-is
            hooks: self.hooks,
        })
    }
}

/// Identifies the sender/role of a message in the conversation.
///
/// This enum follows the standard chat completion role system used by most
/// LLM APIs. The role determines how the message is interpreted and processed.
///
/// # Serialization
///
/// Serializes to lowercase strings via serde (`"system"`, `"user"`, etc.)
/// to match OpenAI API format.
///
/// # Role Semantics
///
/// - [`System`](MessageRole::System): Establishes context, instructions, and behavior
/// - [`User`](MessageRole::User): Input from the human or calling application
/// - [`Assistant`](MessageRole::Assistant): Response from the AI model
/// - [`Tool`](MessageRole::Tool): Results from tool/function execution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// System message that establishes agent behavior and context.
    ///
    /// Typically the first message in a conversation. Used for instructions,
    /// personality definition, and constraints that apply throughout the
    /// conversation.
    System,

    /// User message representing human or application input.
    ///
    /// The prompt or query that the agent should respond to. In multi-turn
    /// conversations, user messages alternate with assistant messages.
    User,

    /// Assistant message containing the AI model's response.
    ///
    /// Can include text, tool use requests, or both. When the model wants to
    /// call a tool, it includes ToolUseBlock content.
    Assistant,

    /// Tool result message containing function execution results.
    ///
    /// Sent back to the model after executing a requested tool. Contains the
    /// tool's output that the model can use in its next response.
    Tool,
}

/// Multi-modal content blocks that can appear in messages.
///
/// Messages are composed of one or more content blocks, allowing rich,
/// structured communication between the user, assistant, and tools.
///
/// # Serialization
///
/// Uses serde's "externally tagged" enum format with a `"type"` field:
/// ```json
/// {"type": "text", "text": "Hello"}
/// {"type": "tool_use", "id": "call_123", "name": "search", "input": {...}}
/// {"type": "tool_result", "tool_use_id": "call_123", "content": {...}}
/// ```
///
/// # Block Types
///
/// - [`Text`](ContentBlock::Text): Simple text content
/// - [`Image`](ContentBlock::Image): Image content (URL or base64)
/// - [`ToolUse`](ContentBlock::ToolUse): Request from model to execute a tool
/// - [`ToolResult`](ContentBlock::ToolResult): Result of tool execution
///
/// # Usage
///
/// Messages can contain multiple blocks. For example, a user message might
/// include text and an image, or an assistant message might include text
/// followed by a tool use request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Text content block containing a string message.
    Text(TextBlock),

    /// Image content block for vision-capable models.
    Image(ImageBlock),

    /// Tool use request from the model to execute a function.
    ToolUse(ToolUseBlock),

    /// Tool execution result sent back to the model.
    ToolResult(ToolResultBlock),
}

/// Simple text content in a message.
///
/// The most common content type, representing plain text communication.
/// Both users and assistants primarily use text blocks for their messages.
///
/// # Example
///
/// ```
/// use open_agent::{TextBlock, ContentBlock};
///
/// let block = TextBlock::new("Hello, world!");
/// let content = ContentBlock::Text(block);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBlock {
    /// The text content.
    pub text: String,
}

impl TextBlock {
    /// Creates a new text block from any string-like type.
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::TextBlock;
    ///
    /// let block = TextBlock::new("Hello");
    /// assert_eq!(block.text, "Hello");
    /// ```
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// Tool use request from the AI model.
///
/// When the model determines it needs to call a tool/function, it returns
/// a ToolUseBlock specifying which tool to call and with what parameters.
/// The application must then execute the tool and return results via
/// [`ToolResultBlock`].
///
/// # Fields
///
/// - `id`: Unique identifier for this tool call, used to correlate results
/// - `name`: Name of the tool to execute (must match a registered tool)
/// - `input`: JSON parameters to pass to the tool
///
/// # Example
///
/// ```
/// use open_agent::{ToolUseBlock, ContentBlock};
/// use serde_json::json;
///
/// let block = ToolUseBlock::new(
///     "call_123",
///     "calculate",
///     json!({"expression": "2 + 2"})
/// );
/// assert_eq!(block.id, "call_123");
/// assert_eq!(block.name, "calculate");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUseBlock {
    /// Unique identifier for this tool call.
    ///
    /// Generated by the model. Used to correlate the tool result back to
    /// this specific request, especially when multiple tools are called.
    pub id: String,

    /// Name of the tool to execute.
    ///
    /// Must match the name of a tool that was provided in the agent's
    /// configuration, otherwise execution will fail.
    pub name: String,

    /// JSON parameters to pass to the tool.
    ///
    /// The structure should match the tool's input schema. The tool's
    /// execution function receives this value as input.
    pub input: serde_json::Value,
}

impl ToolUseBlock {
    /// Creates a new tool use block.
    ///
    /// # Parameters
    ///
    /// - `id`: Unique identifier for this tool call
    /// - `name`: Name of the tool to execute
    /// - `input`: JSON parameters for the tool
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::ToolUseBlock;
    /// use serde_json::json;
    ///
    /// let block = ToolUseBlock::new(
    ///     "call_abc",
    ///     "search",
    ///     json!({"query": "Rust async programming"})
    /// );
    /// ```
    pub fn new(id: impl Into<String>, name: impl Into<String>, input: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            input,
        }
    }
}

/// Tool execution result sent back to the model.
///
/// After executing a tool requested via [`ToolUseBlock`], the application
/// creates a ToolResultBlock containing the tool's output and sends it back
/// to the model. The model then uses this information in its next response.
///
/// # Fields
///
/// - `tool_use_id`: Must match the `id` from the corresponding ToolUseBlock
/// - `content`: JSON result from the tool execution
///
/// # Example
///
/// ```
/// use open_agent::{ToolResultBlock, ContentBlock};
/// use serde_json::json;
///
/// let result = ToolResultBlock::new(
///     "call_123",
///     json!({"result": 4})
/// );
/// assert_eq!(result.tool_use_id, "call_123");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultBlock {
    /// ID of the tool use request this result corresponds to.
    ///
    /// Must match the `id` field from the ToolUseBlock that requested
    /// this tool execution. This correlation is essential for the model
    /// to understand which tool call produced which result.
    pub tool_use_id: String,

    /// JSON result from executing the tool.
    ///
    /// Contains the tool's output data. Can be any valid JSON structure -
    /// the model will interpret it based on the tool's description and
    /// output schema.
    pub content: serde_json::Value,
}

impl ToolResultBlock {
    /// Creates a new tool result block.
    ///
    /// # Parameters
    ///
    /// - `tool_use_id`: ID from the corresponding ToolUseBlock
    /// - `content`: JSON result from tool execution
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::ToolResultBlock;
    /// use serde_json::json;
    ///
    /// let result = ToolResultBlock::new(
    ///     "call_xyz",
    ///     json!({
    ///         "status": "success",
    ///         "data": {"temperature": 72}
    ///     })
    /// );
    /// ```
    pub fn new(tool_use_id: impl Into<String>, content: serde_json::Value) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content,
        }
    }
}

/// Image detail level for vision API calls.
///
/// Controls the resolution and token cost of image processing.
///
/// # Token Costs (gpt-4o)
///
/// - `Low`: Fixed 85 tokens, 512x512 resolution
/// - `High`: Variable tokens based on dimensions (170 tokens per 512px tile + 85 base)
/// - `Auto`: Model decides based on image characteristics (default)
///
/// # Examples
///
/// ```
/// use open_agent::ImageDetail;
///
/// let detail = ImageDetail::High;
/// assert_eq!(detail.to_string(), "high");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum ImageDetail {
    /// Low resolution (512x512), fixed 85 tokens
    Low,
    /// High resolution, variable tokens based on dimensions
    High,
    /// Automatic selection (default)
    #[default]
    Auto,
}

impl std::fmt::Display for ImageDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImageDetail::Low => write!(f, "low"),
            ImageDetail::High => write!(f, "high"),
            ImageDetail::Auto => write!(f, "auto"),
        }
    }
}

/// Image content block for vision-capable models.
///
/// Supports both URL-based images and base64-encoded images.
///
/// # Examples
///
/// ```
/// use open_agent::{ImageBlock, ImageDetail};
///
/// // From URL
/// let image = ImageBlock::from_url("https://example.com/image.jpg");
///
/// // From base64
/// let image = ImageBlock::from_base64("iVBORw0KGgo...", "image/png");
///
/// // With detail level
/// let image = ImageBlock::from_url("https://example.com/image.jpg")
///     .with_detail(ImageDetail::High);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageBlock {
    url: String,
    #[serde(default)]
    detail: ImageDetail,
}

impl ImageBlock {
    /// Creates a new image block from a URL.
    ///
    /// # Arguments
    ///
    /// * `url` - The image URL (must be HTTP, HTTPS, or data URI)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if:
    /// - URL is empty
    /// - URL scheme is not `http://`, `https://`, or `data:`
    /// - Data URI is malformed (missing MIME type or base64 encoding)
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::ImageBlock;
    ///
    /// let image = ImageBlock::from_url("https://example.com/cat.jpg")?;
    /// assert_eq!(image.url(), "https://example.com/cat.jpg");
    /// # Ok::<(), open_agent::Error>(())
    /// ```
    pub fn from_url(url: impl Into<String>) -> crate::Result<Self> {
        let url = url.into();

        // Validate URL is not empty
        if url.is_empty() {
            return Err(crate::Error::invalid_input("Image URL cannot be empty"));
        }

        // Validate URL scheme
        if url.starts_with("http://") || url.starts_with("https://") {
            // Valid HTTP/HTTPS URL
            Ok(Self {
                url,
                detail: ImageDetail::default(),
            })
        } else if let Some(mime_part) = url.strip_prefix("data:") {
            // Validate data URI format: data:MIME;base64,DATA
            if !url.contains(";base64,") {
                return Err(crate::Error::invalid_input(
                    "Data URI must be in format: data:image/TYPE;base64,DATA",
                ));
            }

            // Extract MIME type from data:MIME;base64,DATA
            if let Some(semicolon_pos) = mime_part.find(';') {
                let mime_type = &mime_part[..semicolon_pos];
                if mime_type.is_empty() || !mime_type.starts_with("image/") {
                    return Err(crate::Error::invalid_input(
                        "Data URI MIME type must start with 'image/'",
                    ));
                }
            } else {
                return Err(crate::Error::invalid_input(
                    "Malformed data URI: missing MIME type",
                ));
            }

            Ok(Self {
                url,
                detail: ImageDetail::default(),
            })
        } else {
            Err(crate::Error::invalid_input(
                "Image URL must start with http://, https://, or data:",
            ))
        }
    }

    /// Creates a new image block from base64-encoded data.
    ///
    /// # Arguments
    ///
    /// * `base64_data` - The base64-encoded image data
    /// * `mime_type` - The MIME type (e.g., "image/jpeg", "image/png")
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if:
    /// - Base64 data is empty
    /// - MIME type is empty
    /// - MIME type does not start with "image/"
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::ImageBlock;
    ///
    /// let base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
    /// let image = ImageBlock::from_base64(base64, "image/png")?;
    /// assert!(image.url().starts_with("data:image/png;base64,"));
    /// # Ok::<(), open_agent::Error>(())
    /// ```
    pub fn from_base64(
        base64_data: impl AsRef<str>,
        mime_type: impl AsRef<str>,
    ) -> crate::Result<Self> {
        let data = base64_data.as_ref();
        let mime = mime_type.as_ref();

        // Validate base64 data is not empty
        if data.is_empty() {
            return Err(crate::Error::invalid_input(
                "Base64 image data cannot be empty",
            ));
        }

        // Validate MIME type is not empty
        if mime.is_empty() {
            return Err(crate::Error::invalid_input("MIME type cannot be empty"));
        }

        // Validate MIME type starts with "image/"
        if !mime.starts_with("image/") {
            return Err(crate::Error::invalid_input(
                "MIME type must start with 'image/' (e.g., 'image/png', 'image/jpeg')",
            ));
        }

        let url = format!("data:{};base64,{}", mime, data);
        Ok(Self {
            url,
            detail: ImageDetail::default(),
        })
    }

    /// Sets the image detail level.
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::{ImageBlock, ImageDetail};
    ///
    /// let image = ImageBlock::from_url("https://example.com/image.jpg")
    ///     .with_detail(ImageDetail::High);
    /// ```
    pub fn with_detail(mut self, detail: ImageDetail) -> Self {
        self.detail = detail;
        self
    }

    /// Returns the image URL (or data URI for base64 images).
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Returns the image detail level.
    pub fn detail(&self) -> ImageDetail {
        self.detail
    }
}

/// A complete message in a conversation.
///
/// Messages are the primary unit of communication in the agent system. Each
/// message has a role (who sent it) and content (what it contains). Content
/// is structured as a vector of blocks to support multi-modal communication.
///
/// # Structure
///
/// - `role`: Who sent the message ([`MessageRole`])
/// - `content`: What the message contains (one or more [`ContentBlock`]s)
///
/// # Message Patterns
///
/// ## Simple Text Message
/// ```
/// use open_agent::Message;
///
/// let msg = Message::user("What's the weather?");
/// ```
///
/// ## Assistant Response with Tool Call
/// ```
/// use open_agent::{Message, ContentBlock, TextBlock, ToolUseBlock};
/// use serde_json::json;
///
/// let msg = Message::assistant(vec![
///     ContentBlock::Text(TextBlock::new("Let me check that for you.")),
///     ContentBlock::ToolUse(ToolUseBlock::new(
///         "call_123",
///         "get_weather",
///         json!({"location": "San Francisco"})
///     ))
/// ]);
/// ```
///
/// ## Tool Result
/// ```
/// use open_agent::{Message, ContentBlock, ToolResultBlock};
/// use serde_json::json;
///
/// let msg = Message::user_with_blocks(vec![
///     ContentBlock::ToolResult(ToolResultBlock::new(
///         "call_123",
///         json!({"temp": 72, "conditions": "sunny"})
///     ))
/// ]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// The role/sender of this message.
    pub role: MessageRole,

    /// The content blocks that make up this message.
    ///
    /// A message can contain multiple blocks of different types. For example,
    /// an assistant message might have both text and tool use blocks.
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// Creates a new message with the specified role and content.
    ///
    /// This is the most general constructor. For convenience, use the
    /// role-specific constructors like [`user()`](Message::user),
    /// [`assistant()`](Message::assistant), etc.
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::{Message, MessageRole, ContentBlock, TextBlock};
    ///
    /// let msg = Message::new(
    ///     MessageRole::User,
    ///     vec![ContentBlock::Text(TextBlock::new("Hello"))]
    /// );
    /// ```
    pub fn new(role: MessageRole, content: Vec<ContentBlock>) -> Self {
        Self { role, content }
    }

    /// Creates a user message with simple text content.
    ///
    /// This is the most common way to create user messages. For more complex
    /// content with multiple blocks, use [`user_with_blocks()`](Message::user_with_blocks).
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::Message;
    ///
    /// let msg = Message::user("What is 2+2?");
    /// ```
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: vec![ContentBlock::Text(TextBlock::new(text))],
        }
    }

    /// Creates an assistant message with the specified content blocks.
    ///
    /// Assistant messages often contain multiple content blocks (text + tool use).
    /// This method takes a vector of blocks for maximum flexibility.
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::{Message, ContentBlock, TextBlock};
    ///
    /// let msg = Message::assistant(vec![
    ///     ContentBlock::Text(TextBlock::new("The answer is 4"))
    /// ]);
    /// ```
    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content,
        }
    }

    /// Creates a system message with simple text content.
    ///
    /// System messages establish the agent's behavior and context. They're
    /// typically sent at the start of a conversation.
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::Message;
    ///
    /// let msg = Message::system("You are a helpful assistant. Be concise.");
    /// ```
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: vec![ContentBlock::Text(TextBlock::new(text))],
        }
    }

    /// Creates a user message with custom content blocks.
    ///
    /// Use this when you need to send structured content beyond simple text,
    /// such as tool results. For simple text messages, prefer
    /// [`user()`](Message::user).
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::{Message, ContentBlock, ToolResultBlock};
    /// use serde_json::json;
    ///
    /// let msg = Message::user_with_blocks(vec![
    ///     ContentBlock::ToolResult(ToolResultBlock::new(
    ///         "call_123",
    ///         json!({"result": "success"})
    ///     ))
    /// ]);
    /// ```
    pub fn user_with_blocks(content: Vec<ContentBlock>) -> Self {
        Self {
            role: MessageRole::User,
            content,
        }
    }

    /// Creates a user message with text and an image from a URL.
    ///
    /// This is a convenience method for the common pattern of sending text with
    /// an image. The image uses `ImageDetail::Auto` by default. For more control
    /// over detail level, use [`user_with_image_detail()`](Message::user_with_image_detail).
    ///
    /// # Arguments
    ///
    /// * `text` - The text prompt
    /// * `image_url` - URL of the image (http/https or data URI)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if the image URL is invalid (empty, wrong scheme, etc.)
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::Message;
    ///
    /// let msg = Message::user_with_image(
    ///     "What's in this image?",
    ///     "https://example.com/photo.jpg"
    /// )?;
    /// # Ok::<(), open_agent::Error>(())
    /// ```
    pub fn user_with_image(
        text: impl Into<String>,
        image_url: impl Into<String>,
    ) -> crate::Result<Self> {
        Ok(Self {
            role: MessageRole::User,
            content: vec![
                ContentBlock::Text(TextBlock::new(text)),
                ContentBlock::Image(ImageBlock::from_url(image_url)?),
            ],
        })
    }

    /// Creates a user message with text and an image with specified detail level.
    ///
    /// Use this when you need control over the image detail level for token cost
    /// management. `ImageDetail::Low` uses ~85 tokens, `ImageDetail::High` uses
    /// more tokens based on image dimensions, and `ImageDetail::Auto` lets the
    /// model decide.
    ///
    /// # Arguments
    ///
    /// * `text` - The text prompt
    /// * `image_url` - URL of the image (http/https or data URI)
    /// * `detail` - Detail level (Low, High, or Auto)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if the image URL is invalid (empty, wrong scheme, etc.)
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::{Message, ImageDetail};
    ///
    /// let msg = Message::user_with_image_detail(
    ///     "Analyze this diagram in detail",
    ///     "https://example.com/diagram.png",
    ///     ImageDetail::High
    /// )?;
    /// # Ok::<(), open_agent::Error>(())
    /// ```
    pub fn user_with_image_detail(
        text: impl Into<String>,
        image_url: impl Into<String>,
        detail: ImageDetail,
    ) -> crate::Result<Self> {
        Ok(Self {
            role: MessageRole::User,
            content: vec![
                ContentBlock::Text(TextBlock::new(text)),
                ContentBlock::Image(ImageBlock::from_url(image_url)?.with_detail(detail)),
            ],
        })
    }

    /// Creates a user message with text and a base64-encoded image.
    ///
    /// This is useful when you have image data in memory and want to send it
    /// without uploading to a URL first. The image will be encoded as a data URI.
    ///
    /// # Arguments
    ///
    /// * `text` - The text prompt
    /// * `base64_data` - Base64-encoded image data
    /// * `mime_type` - MIME type (e.g., "image/png", "image/jpeg")
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if the base64 data or MIME type is invalid
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::Message;
    ///
    /// let base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ...";
    /// let msg = Message::user_with_base64_image(
    ///     "What's this image?",
    ///     base64_data,
    ///     "image/png"
    /// )?;
    /// # Ok::<(), open_agent::Error>(())
    /// ```
    pub fn user_with_base64_image(
        text: impl Into<String>,
        base64_data: impl AsRef<str>,
        mime_type: impl AsRef<str>,
    ) -> crate::Result<Self> {
        Ok(Self {
            role: MessageRole::User,
            content: vec![
                ContentBlock::Text(TextBlock::new(text)),
                ContentBlock::Image(ImageBlock::from_base64(base64_data, mime_type)?),
            ],
        })
    }
}

/// OpenAI API message format for serialization.
///
/// This struct represents the wire format for messages when communicating
/// with OpenAI-compatible APIs. It differs from the internal [`Message`]
/// type to accommodate the specific serialization requirements of the
/// OpenAI API.
///
/// # Key Differences from Internal Message Type
///
/// - Content is a flat string rather than structured blocks
/// - Tool calls are represented in OpenAI's specific format
/// - Supports both sending tool calls (via `tool_calls`) and tool results
///   (via `tool_call_id`)
///
/// # Serialization
///
/// Optional fields are skipped when `None` to keep payloads minimal.
///
/// # Usage
///
/// This type is typically created by the SDK internally when converting
/// from [`Message`] to API format. Users rarely need to construct these
/// directly.
///
/// # OpenAI Content Format
///
/// OpenAI content format supporting both string and array.
///
/// For backward compatibility, text-only messages use string format.
/// Messages with images use array format with multiple content parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAIContent {
    /// Simple text string (backward compatible)
    Text(String),
    /// Array of content parts (text and/or images)
    Parts(Vec<OpenAIContentPart>),
}

/// A single content part in an OpenAI message.
///
/// Can be either text or an image URL. This is a tagged enum that prevents
/// invalid states (e.g., having both text and image_url, or neither).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIContentPart {
    /// Text content part
    Text {
        /// The text content
        text: String,
    },
    /// Image URL content part
    #[serde(rename = "image_url")]
    ImageUrl {
        /// The image URL details
        image_url: OpenAIImageUrl,
    },
}

impl OpenAIContentPart {
    /// Creates a text content part.
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::types::OpenAIContentPart;
    ///
    /// let part = OpenAIContentPart::text("Hello world");
    /// ```
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Creates an image URL content part.
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::{types::OpenAIContentPart, ImageDetail};
    ///
    /// let part = OpenAIContentPart::image_url("https://example.com/img.jpg", ImageDetail::High);
    /// ```
    pub fn image_url(url: impl Into<String>, detail: ImageDetail) -> Self {
        Self::ImageUrl {
            image_url: OpenAIImageUrl {
                url: url.into(),
                detail: Some(detail.to_string()),
            },
        }
    }
}

/// OpenAI image URL structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIImageUrl {
    /// Image URL or data URI
    pub url: String,
    /// Detail level: "low", "high", or "auto"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIMessage {
    /// Message role as a string ("system", "user", "assistant", "tool").
    pub role: String,

    /// Message content (string for text-only, array for text+images).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<OpenAIContent>,

    /// Tool calls requested by the assistant (assistant messages only).
    ///
    /// When the model wants to call tools, this field contains the list
    /// of tool invocations with their parameters. Only present in assistant
    /// messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,

    /// ID of the tool call this message is responding to (tool messages only).
    ///
    /// When sending tool results back to the model, this field links the
    /// result to the original tool call request. Only present in tool
    /// messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// OpenAI tool call representation in API messages.
///
/// Represents a request from the model to execute a specific function/tool.
/// This is the wire format used in the OpenAI API, distinct from the internal
/// [`ToolUseBlock`] representation.
///
/// # Structure
///
/// Each tool call has:
/// - A unique ID for correlation with results
/// - A type (always "function" in current OpenAI API)
/// - Function details (name and arguments)
///
/// # Example JSON
///
/// ```json
/// {
///   "id": "call_abc123",
///   "type": "function",
///   "function": {
///     "name": "get_weather",
///     "arguments": "{\"location\":\"San Francisco\"}"
///   }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolCall {
    /// Unique identifier for this tool call.
    ///
    /// Generated by the model. Used to correlate tool results back to
    /// this specific call.
    pub id: String,

    /// Type of the call (always "function" in current API).
    ///
    /// The `rename` attribute ensures this serializes as `"type"` in JSON
    /// since `type` is a Rust keyword.
    #[serde(rename = "type")]
    pub call_type: String,

    /// Function/tool details (name and arguments).
    pub function: OpenAIFunction,
}

/// OpenAI function call details.
///
/// Contains the function name and its arguments in the OpenAI API format.
/// Note that arguments are serialized as a JSON string, not a JSON object,
/// which is an OpenAI API quirk.
///
/// # Arguments Format
///
/// The `arguments` field is a **JSON string**, not a parsed JSON object.
/// For example: `"{\"x\": 1, \"y\": 2}"` not `{"x": 1, "y": 2}`.
/// This must be parsed before use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunction {
    /// Name of the function/tool to call.
    pub name: String,

    /// Function arguments as a **JSON string** (OpenAI API quirk).
    ///
    /// Must be parsed as JSON before use. For example, this might contain
    /// the string `"{\"location\":\"NYC\",\"units\":\"fahrenheit\"}"` which
    /// needs to be parsed into an actual JSON value.
    pub arguments: String,
}

/// Complete request payload for OpenAI chat completions API.
///
/// This struct is serialized and sent as the request body when making
/// API calls to OpenAI-compatible endpoints. It includes the model,
/// conversation history, and configuration parameters.
///
/// # Streaming
///
/// The SDK always uses streaming mode (`stream: true`) to enable real-time
/// response processing and better user experience.
///
/// # Optional Fields
///
/// Fields marked with `skip_serializing_if` are omitted from the JSON payload
/// when `None`, allowing the API provider to use its defaults.
///
/// # Example
///
/// ```ignore
/// use open_agent_sdk::types::{OpenAIRequest, OpenAIMessage};
///
/// let request = OpenAIRequest {
///     model: "gpt-4".to_string(),
///     messages: vec![
///         OpenAIMessage {
///             role: "user".to_string(),
///             content: "Hello!".to_string(),
///             tool_calls: None,
///             tool_call_id: None,
///         }
///     ],
///     stream: true,
///     max_tokens: Some(1000),
///     temperature: Some(0.7),
///     tools: None,
/// };
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct OpenAIRequest {
    /// Model identifier (e.g., "gpt-4", "qwen2.5-32b-instruct").
    pub model: String,

    /// Conversation history as a sequence of messages.
    ///
    /// Includes system prompt, user messages, assistant responses, and
    /// tool results. Order matters - messages are processed sequentially.
    pub messages: Vec<OpenAIMessage>,

    /// Whether to stream the response.
    ///
    /// The SDK always sets this to `true` for better user experience.
    /// Streaming allows incremental processing of responses rather than
    /// waiting for the entire completion.
    pub stream: bool,

    /// Maximum tokens to generate (optional).
    ///
    /// `None` uses the provider's default. Some providers require this
    /// to be set explicitly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Sampling temperature (optional).
    ///
    /// `None` uses the provider's default. Controls randomness in
    /// generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Tools/functions available to the model (optional).
    ///
    /// When present, enables function calling. Each tool is described
    /// with a JSON schema defining its parameters. `None` means no
    /// tools are available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
}

/// A single chunk from OpenAI's streaming response.
///
/// When the SDK requests streaming responses (`stream: true`), the API
/// returns the response incrementally as a series of chunks. Each chunk
/// represents a small piece of the complete response, allowing the SDK
/// to process and display content as it's generated.
///
/// # Streaming Architecture
///
/// Instead of waiting for the entire response, streaming sends many small
/// chunks in rapid succession. Each chunk contains:
/// - Metadata (id, model, timestamp)
/// - One or more choices (usually just one for single completions)
/// - Incremental deltas with new content
///
/// # Server-Sent Events Format
///
/// Chunks are transmitted as Server-Sent Events (SSE) over HTTP:
/// ```text
/// data: {"id":"chunk_1","object":"chat.completion.chunk",...}
/// data: {"id":"chunk_2","object":"chat.completion.chunk",...}
/// data: [DONE]
/// ```
///
/// # Example Chunk JSON
///
/// ```json
/// {
///   "id": "chatcmpl-123",
///   "object": "chat.completion.chunk",
///   "created": 1677652288,
///   "model": "gpt-4",
///   "choices": [{
///     "index": 0,
///     "delta": {"content": "Hello"},
///     "finish_reason": null
///   }]
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIChunk {
    /// Unique identifier for this completion.
    ///
    /// All chunks in a single streaming response share the same ID.
    /// Not actively used by the SDK but preserved for debugging.
    #[allow(dead_code)]
    pub id: String,

    /// Object type (always "chat.completion.chunk" for streaming).
    ///
    /// Not actively used by the SDK but preserved for debugging.
    #[allow(dead_code)]
    pub object: String,

    /// Unix timestamp of when this chunk was created.
    ///
    /// Not actively used by the SDK but preserved for debugging.
    #[allow(dead_code)]
    pub created: i64,

    /// Model that generated this chunk.
    ///
    /// Not actively used by the SDK but preserved for debugging.
    #[allow(dead_code)]
    pub model: String,

    /// Array of completion choices (usually contains one element).
    ///
    /// Each choice represents a possible completion. In normal usage,
    /// there's only one choice per chunk. This is the critical field
    /// that the SDK processes to extract content and tool calls.
    pub choices: Vec<OpenAIChoice>,
}

/// A single choice/completion option in a streaming chunk.
///
/// In streaming responses, each chunk can theoretically contain multiple
/// choices (parallel completions), but in practice there's usually just one.
/// Each choice contains a delta with incremental updates and optionally a
/// finish reason when the generation is complete.
///
/// # Delta vs Complete Content
///
/// Unlike non-streaming responses that send complete messages, streaming
/// sends deltas - just the new content added in this chunk. The SDK
/// accumulates these deltas to build the complete response.
///
/// # Finish Reason
///
/// - `None`: More content is coming
/// - `Some("stop")`: Normal completion
/// - `Some("length")`: Hit max token limit
/// - `Some("tool_calls")`: Model wants to call tools
/// - `Some("content_filter")`: Blocked by content policy
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIChoice {
    /// Index of this choice in the choices array.
    ///
    /// Usually 0 since most requests generate a single completion.
    /// Not actively used by the SDK but preserved for debugging.
    #[allow(dead_code)]
    pub index: u32,

    /// Incremental update/delta for this chunk.
    ///
    /// Contains the new content, tool calls, or other updates added in
    /// this specific chunk. The SDK processes this to update its internal
    /// state and accumulate the full response.
    pub delta: OpenAIDelta,

    /// Reason why generation finished (None if still generating).
    ///
    /// Only present in the final chunk of a stream:
    /// - `None`: Generation is still in progress
    /// - `Some("stop")`: Completed normally
    /// - `Some("length")`: Hit token limit
    /// - `Some("tool_calls")`: Model requested tools
    /// - `Some("content_filter")`: Content was filtered
    ///
    /// The SDK uses this to detect completion and determine next actions.
    pub finish_reason: Option<String>,
}

/// Incremental update in a streaming chunk.
///
/// Represents the new content/changes added in this specific chunk.
/// Unlike complete messages, deltas only contain what's new, not the
/// entire accumulated content. The SDK accumulates these deltas to
/// build the complete response.
///
/// # Incremental Nature
///
/// If the complete response is "Hello, world!", the deltas might be:
/// 1. `content: Some("Hello")`
/// 2. `content: Some(", ")`
/// 3. `content: Some("world")`
/// 4. `content: Some("!")`
///
/// The SDK concatenates these to build the full text.
///
/// # Tool Call Deltas
///
/// Tool calls are also streamed incrementally. The first delta might
/// include the tool ID and name, while subsequent deltas stream the
/// arguments JSON string piece by piece.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIDelta {
    /// Role of the message (only in first chunk).
    ///
    /// Typically "assistant". Only appears in the first delta of a response
    /// to establish who's speaking. Subsequent deltas omit this field.
    /// Not actively used by the SDK but preserved for completeness.
    #[allow(dead_code)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,

    /// Incremental text content added in this chunk.
    ///
    /// Contains the new text tokens generated. `None` if this chunk doesn't
    /// add text (e.g., it might only have tool call updates). The SDK
    /// concatenates these across chunks to build the complete response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Incremental tool call updates added in this chunk.
    ///
    /// When the model wants to call tools, tool call information is streamed
    /// incrementally. Each delta might add to different parts of the tool
    /// call (ID, name, arguments). The SDK accumulates these to reconstruct
    /// complete tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCallDelta>>,
}

/// Incremental update for a tool call in streaming.
///
/// Tool calls are streamed piece-by-piece, with different chunks potentially
/// updating different parts. The SDK must accumulate these deltas to
/// reconstruct complete tool calls.
///
/// # Streaming Pattern
///
/// A complete tool call is typically streamed as:
/// 1. First chunk: `index: 0, id: Some("call_123"), type: Some("function")`
/// 2. Second chunk: `index: 0, function: Some(FunctionDelta { name: Some("search"), ... })`
/// 3. Multiple chunks: `index: 0, function: Some(FunctionDelta { arguments: Some("part") })`
///
/// The SDK uses the `index` to know which tool call to update, as multiple
/// tool calls can be streamed simultaneously.
///
/// # Index-Based Accumulation
///
/// The `index` field is crucial for tracking which tool call is being updated.
/// When the model calls multiple tools, each has a different index, and deltas
/// specify which one they're updating.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIToolCallDelta {
    /// Index identifying which tool call this delta updates.
    ///
    /// When multiple tools are called, each has an index (0, 1, 2, ...).
    /// The SDK uses this to route delta updates to the correct tool call
    /// in its accumulation buffer.
    pub index: u32,

    /// Tool call ID (only in first delta for this tool call).
    ///
    /// Generated by the model. Present in the first chunk for each tool
    /// call, then omitted in subsequent chunks. The SDK stores this to
    /// correlate results later.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Type of call (always "function" when present).
    ///
    /// Only appears in the first delta for each tool call. Subsequent
    /// deltas omit this field. Not actively used by the SDK but preserved
    /// for completeness.
    #[allow(dead_code)]
    #[serde(skip_serializing_if = "Option::is_none", rename = "type")]
    pub call_type: Option<String>,

    /// Incremental function details (name and/or arguments).
    ///
    /// Contains partial updates to the function name and arguments.
    /// The SDK accumulates these across chunks to build the complete
    /// function call specification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<OpenAIFunctionDelta>,
}

/// Incremental update for function details in streaming tool calls.
///
/// As the model streams a tool call, the function name and arguments are
/// sent incrementally. The name usually comes first in one chunk, then
/// arguments are streamed piece-by-piece as a JSON string.
///
/// # Arguments Streaming
///
/// The arguments field is particularly important to understand. It contains
/// **fragments of a JSON string** that must be accumulated and then parsed:
///
/// 1. Chunk 1: `arguments: Some("{")`
/// 2. Chunk 2: `arguments: Some("\"query\":")`
/// 3. Chunk 3: `arguments: Some("\"hello\"")`
/// 4. Chunk 4: `arguments: Some("}")`
///
/// The SDK concatenates these into `"{\"query\":\"hello\"}"` and then
/// parses it as JSON.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIFunctionDelta {
    /// Function/tool name (only in first delta for this function).
    ///
    /// Present when the model first starts calling this function, then
    /// omitted in subsequent chunks. The SDK stores this to know which
    /// tool to execute.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Incremental fragment of the arguments JSON string.
    ///
    /// Contains a piece of the complete JSON arguments string. The SDK
    /// must concatenate all argument fragments across chunks, then parse
    /// the complete string as JSON to get the actual parameters.
    ///
    /// For example, if the complete arguments should be:
    /// `{"x": 1, "y": 2}`
    ///
    /// This might be streamed as:
    /// - `Some("{\"x\": ")`
    /// - `Some("1, \"y\": ")`
    /// - `Some("2}")`
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
    fn test_message_user_with_image() {
        let msg =
            Message::user_with_image("What's in this image?", "https://example.com/image.jpg")
                .unwrap();
        assert!(matches!(msg.role, MessageRole::User));
        assert_eq!(msg.content.len(), 2);

        // Should have text first, then image
        match &msg.content[0] {
            ContentBlock::Text(text) => assert_eq!(text.text, "What's in this image?"),
            _ => panic!("Expected TextBlock at position 0"),
        }
        match &msg.content[1] {
            ContentBlock::Image(image) => {
                assert_eq!(image.url(), "https://example.com/image.jpg");
                assert_eq!(image.detail(), ImageDetail::Auto);
            }
            _ => panic!("Expected ImageBlock at position 1"),
        }
    }

    #[test]
    fn test_message_user_with_image_and_detail() {
        let msg = Message::user_with_image_detail(
            "Analyze this in detail",
            "https://example.com/diagram.png",
            ImageDetail::High,
        )
        .unwrap();
        assert!(matches!(msg.role, MessageRole::User));
        assert_eq!(msg.content.len(), 2);

        match &msg.content[1] {
            ContentBlock::Image(image) => {
                assert_eq!(image.detail(), ImageDetail::High);
            }
            _ => panic!("Expected ImageBlock"),
        }
    }

    #[test]
    fn test_message_user_with_base64_image() {
        let base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ";
        let msg =
            Message::user_with_base64_image("What's this?", base64_data, "image/png").unwrap();
        assert!(matches!(msg.role, MessageRole::User));
        assert_eq!(msg.content.len(), 2);

        match &msg.content[1] {
            ContentBlock::Image(image) => {
                assert!(image.url().starts_with("data:image/png;base64,"));
                assert!(image.url().contains(base64_data));
            }
            _ => panic!("Expected ImageBlock"),
        }
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
                content: Some(OpenAIContent::Text("Hello".to_string())),
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

    #[test]
    fn test_temperature_validation() {
        // Temperature too low (< 0.0)
        let result = AgentOptions::builder()
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .temperature(-0.1)
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("temperature"));

        // Temperature too high (> 2.0)
        let result = AgentOptions::builder()
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .temperature(2.1)
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("temperature"));

        // Valid temperatures should work
        let result = AgentOptions::builder()
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .temperature(0.0)
            .build();
        assert!(result.is_ok());

        let result = AgentOptions::builder()
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .temperature(2.0)
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_url_validation() {
        // Empty URL should fail
        let result = AgentOptions::builder()
            .model("test-model")
            .base_url("")
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("base_url"));

        // Invalid URL format should fail
        let result = AgentOptions::builder()
            .model("test-model")
            .base_url("not-a-url")
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("base_url"));

        // Valid URLs should work
        let result = AgentOptions::builder()
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build();
        assert!(result.is_ok());

        let result = AgentOptions::builder()
            .model("test-model")
            .base_url("https://api.openai.com/v1")
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_model_validation() {
        // Empty model should fail
        let result = AgentOptions::builder()
            .model("")
            .base_url("http://localhost:1234/v1")
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("model"));

        // Whitespace-only model should fail
        let result = AgentOptions::builder()
            .model("   ")
            .base_url("http://localhost:1234/v1")
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("model"));
    }

    #[test]
    fn test_max_tokens_validation() {
        // max_tokens = 0 should fail
        let result = AgentOptions::builder()
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .max_tokens(0)
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("max_tokens"));

        // Valid max_tokens should work
        let result = AgentOptions::builder()
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .max_tokens(1)
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_agent_options_getters() {
        // Test that AgentOptions provides getter methods for field access
        let options = AgentOptions::builder()
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .system_prompt("Test prompt")
            .api_key("test-key")
            .max_turns(5)
            .max_tokens(1000)
            .temperature(0.5)
            .timeout(30)
            .auto_execute_tools(true)
            .max_tool_iterations(10)
            .build()
            .unwrap();

        // All fields should be accessible via getter methods, not direct field access
        assert_eq!(options.system_prompt(), "Test prompt");
        assert_eq!(options.model(), "test-model");
        assert_eq!(options.base_url(), "http://localhost:1234/v1");
        assert_eq!(options.api_key(), "test-key");
        assert_eq!(options.max_turns(), 5);
        assert_eq!(options.max_tokens(), Some(1000));
        assert_eq!(options.temperature(), 0.5);
        assert_eq!(options.timeout(), 30);
        assert!(options.auto_execute_tools());
        assert_eq!(options.max_tool_iterations(), 10);
        assert_eq!(options.tools().len(), 0);
    }

    // ========================================================================
    // Image Support Tests (Phase 1 - TDD RED)
    // ========================================================================

    #[test]
    fn test_image_block_from_url() {
        // Should create ImageBlock from URL
        let block = ImageBlock::from_url("https://example.com/image.jpg").unwrap();
        assert_eq!(block.url(), "https://example.com/image.jpg");
        assert!(matches!(block.detail(), ImageDetail::Auto));
    }

    #[test]
    fn test_image_block_from_base64() {
        // Should create ImageBlock from base64
        let block = ImageBlock::from_base64("abc123", "image/jpeg").unwrap();
        assert!(block.url().starts_with("data:image/jpeg;base64,"));
        assert!(matches!(block.detail(), ImageDetail::Auto));
    }

    #[test]
    fn test_image_block_with_detail() {
        // Should set detail level
        let block = ImageBlock::from_url("https://example.com/image.jpg")
            .unwrap()
            .with_detail(ImageDetail::High);
        assert!(matches!(block.detail(), ImageDetail::High));
    }

    #[test]
    fn test_image_detail_serialization() {
        // Should serialize ImageDetail to correct strings
        let json = serde_json::to_string(&ImageDetail::Low).unwrap();
        assert_eq!(json, "\"low\"");

        let json = serde_json::to_string(&ImageDetail::High).unwrap();
        assert_eq!(json, "\"high\"");

        let json = serde_json::to_string(&ImageDetail::Auto).unwrap();
        assert_eq!(json, "\"auto\"");
    }

    #[test]
    fn test_content_block_image_variant() {
        // Should add Image variant to ContentBlock
        let image = ImageBlock::from_url("https://example.com/image.jpg").unwrap();
        let block = ContentBlock::Image(image);

        match block {
            ContentBlock::Image(img) => {
                assert_eq!(img.url(), "https://example.com/image.jpg");
            }
            _ => panic!("Expected Image variant"),
        }
    }

    #[test]
    fn test_openai_content_text_format() {
        // Should serialize text-only as string (backward compat)
        let content = OpenAIContent::Text("Hello".to_string());
        let json = serde_json::to_value(&content).unwrap();
        assert_eq!(json, serde_json::json!("Hello"));
    }

    #[test]
    fn test_openai_content_parts_format() {
        // Should serialize mixed content as array
        let parts = vec![
            OpenAIContentPart::text("What's in this image?"),
            OpenAIContentPart::image_url("https://example.com/img.jpg", ImageDetail::High),
        ];
        let content = OpenAIContent::Parts(parts);
        let json = serde_json::to_value(&content).unwrap();

        assert!(json.is_array());
        assert_eq!(json[0]["type"], "text");
        assert_eq!(json[0]["text"], "What's in this image?");
        assert_eq!(json[1]["type"], "image_url");
        assert_eq!(json[1]["image_url"]["url"], "https://example.com/img.jpg");
        assert_eq!(json[1]["image_url"]["detail"], "high");
    }

    // ========================================================================
    // OpenAIContentPart Enum Tests (Phase 4 - PR #3 Fixes)
    // ========================================================================

    #[test]
    fn test_openai_content_part_text_serialization() {
        // RED: Test that text variant serializes correctly with enum
        let part = OpenAIContentPart::text("Hello world");
        let json = serde_json::to_value(&part).unwrap();

        // Should have type field with value "text"
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "Hello world");
        // Should not have image_url field
        assert!(json.get("image_url").is_none());
    }

    #[test]
    fn test_openai_content_part_image_serialization() {
        // RED: Test that image_url variant serializes correctly with enum
        let part = OpenAIContentPart::image_url("https://example.com/img.jpg", ImageDetail::Low);
        let json = serde_json::to_value(&part).unwrap();

        // Should have type field with value "image_url"
        assert_eq!(json["type"], "image_url");
        assert_eq!(json["image_url"]["url"], "https://example.com/img.jpg");
        assert_eq!(json["image_url"]["detail"], "low");
        // Should not have text field
        assert!(json.get("text").is_none());
    }

    #[test]
    fn test_openai_content_part_enum_exhaustiveness() {
        // RED: Test that enum prevents invalid states
        // With tagged enum, it should be impossible to create a part with both text and image_url
        // or a part with neither. This test documents expected enum behavior.

        let text_part = OpenAIContentPart::text("test");
        let image_part = OpenAIContentPart::image_url("url", ImageDetail::Auto);

        // Pattern matching should be exhaustive
        match text_part {
            OpenAIContentPart::Text { .. } => {
                // Expected for text part
            }
            OpenAIContentPart::ImageUrl { .. } => {
                panic!("Text part should not match ImageUrl variant");
            }
        }

        match image_part {
            OpenAIContentPart::Text { .. } => {
                panic!("Image part should not match Text variant");
            }
            OpenAIContentPart::ImageUrl { .. } => {
                // Expected for image part
            }
        }
    }

    #[test]
    fn test_image_detail_display() {
        // Should convert ImageDetail to string
        assert_eq!(ImageDetail::Low.to_string(), "low");
        assert_eq!(ImageDetail::High.to_string(), "high");
        assert_eq!(ImageDetail::Auto.to_string(), "auto");
    }

    // ========================================================================
    // ImageBlock Validation Tests (Phase 1 - PR #3 Fixes)
    // ========================================================================

    #[test]
    fn test_image_block_from_url_rejects_empty() {
        // Should reject empty URL strings
        let result = ImageBlock::from_url("");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn test_image_block_from_url_rejects_invalid_scheme() {
        // Should reject non-HTTP/HTTPS/data schemes
        let result = ImageBlock::from_url("ftp://example.com/image.jpg");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("scheme") || err.to_string().contains("http"));
    }

    #[test]
    fn test_image_block_from_url_rejects_relative_path() {
        // Should reject relative paths
        let result = ImageBlock::from_url("/images/photo.jpg");
        assert!(result.is_err());
        // Error message should mention URL requirements
        assert!(matches!(result.unwrap_err(), crate::Error::InvalidInput(_)));
    }

    #[test]
    fn test_image_block_from_url_accepts_http() {
        // Should accept HTTP URLs
        let result = ImageBlock::from_url("http://example.com/image.jpg");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().url(), "http://example.com/image.jpg");
    }

    #[test]
    fn test_image_block_from_url_accepts_https() {
        // Should accept HTTPS URLs
        let result = ImageBlock::from_url("https://example.com/image.jpg");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().url(), "https://example.com/image.jpg");
    }

    #[test]
    fn test_image_block_from_url_accepts_data_uri() {
        // Should accept data URIs
        let data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
        let result = ImageBlock::from_url(data_uri);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().url(), data_uri);
    }

    #[test]
    fn test_image_block_from_url_rejects_malformed_data_uri() {
        // Should reject malformed data URIs
        let result = ImageBlock::from_url("data:notanimage");
        assert!(result.is_err());
        // Should return InvalidInput error for malformed data URI
        assert!(matches!(result.unwrap_err(), crate::Error::InvalidInput(_)));
    }

    #[test]
    fn test_image_block_from_base64_rejects_empty() {
        // Should reject empty base64 data
        let result = ImageBlock::from_base64("", "image/png");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn test_image_block_from_base64_rejects_invalid_mime() {
        // Should reject non-image MIME types
        let result = ImageBlock::from_base64("somedata", "text/plain");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("MIME") || err.to_string().contains("image"));
    }

    #[test]
    fn test_image_block_from_base64_accepts_valid_input() {
        // Should accept valid base64 data with image MIME type
        let base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
        let result = ImageBlock::from_base64(base64, "image/png");
        assert!(result.is_ok());
        let block = result.unwrap();
        assert!(block.url().starts_with("data:image/png;base64,"));
    }

    #[test]
    fn test_image_block_from_base64_accepts_all_image_types() {
        // Should accept all common image MIME types
        let base64 = "validdata";
        let mime_types = ["image/jpeg", "image/png", "image/gif", "image/webp"];

        for mime in &mime_types {
            let result = ImageBlock::from_base64(base64, *mime);
            assert!(result.is_ok(), "Should accept {}", mime);
            let block = result.unwrap();
            assert!(block.url().starts_with(&format!("data:{};base64,", mime)));
        }
    }

    #[test]
    fn test_image_block_from_base64_rejects_empty_mime() {
        // Should reject empty MIME type
        let result = ImageBlock::from_base64("somedata", "");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("MIME") || err.to_string().contains("empty"));
    }
}
