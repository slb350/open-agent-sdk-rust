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
