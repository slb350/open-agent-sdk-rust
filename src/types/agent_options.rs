/// Configuration for model requests, tool execution, and hooks.
///
/// Construct validated options with [`AgentOptions::builder`]. `Default` leaves the
/// model and base URL empty without validation. Clones share tool handlers.
/// `max_tokens` and `temperature` remain unset unless explicitly supplied.
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

    /// API endpoint URL (e.g., "http://localhost:1234/v1").
    ///
    /// The wire format is chosen by `protocol`: `OpenAiChat` posts to
    /// `{base_url}/chat/completions`, which local inference servers (LM Studio,
    /// llama.cpp, vLLM) and most cloud providers support; `Anthropic` posts to
    /// `{base_url}/messages`.
    base_url: String,

    /// API authentication key for the provider.
    ///
    /// Many local servers don't require authentication, so the default
    /// "not-needed" is often sufficient. For cloud providers like OpenAI,
    /// set this to your actual API key.
    api_key: String,

    /// Caller-supplied HTTP headers applied to every model request.
    ///
    /// These may override the protocol defaults, including authentication and content type.
    /// Header names are matched case-insensitively when the builder replaces an earlier value.
    headers: BTreeMap<String, String>,

    /// Retained compatibility setting, stored but not enforced by the SDK.
    ///
    /// Use `max_tool_iterations` to limit automatic tool rounds. Applications must
    /// limit conversation turns themselves.
    max_turns: u32,

    /// Maximum tokens the model should generate in a single response.
    ///
    /// `None` uses the provider's default. Lower values constrain response
    /// length, which can be useful for cost control or ensuring concise answers.
    /// Note this is separate from the model's context window size.
    max_tokens: Option<u32>,

    /// Optional sampling temperature. `None` omits the field and lets the server choose.
    ///
    /// The builder accepts 0.0 through 2.0; a provider may accept a narrower range or
    /// reject the parameter entirely.
    temperature: Option<f32>,

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

    /// Whether the SDK executes tool calls and continues until a text response or
    /// the configured iteration limit. Defaults to manual execution.
    auto_execute_tools: bool,

    /// Maximum automatic tool rounds. One round may contain several tool calls.
    max_tool_iterations: u32,

    /// Whether reasoning is emitted separately from content; defaults to false.
    ///
    /// Reasoning never enters assistant text or conversation history.
    include_reasoning: bool,

    /// The wire protocol this endpoint speaks.
    ///
    /// Selects the request path, the auth header, the body shape and the streaming
    /// vocabulary. Defaults to [`ApiProtocol::OpenAiChat`], which is what every endpoint the
    /// SDK supported before 0.9.0 speaks, so an existing configuration keeps its behaviour.
    protocol: ApiProtocol,

    /// Hooks for prompt submission and automatic pre/post-tool execution.
    hooks: Hooks,
}

/// Omits credentials and shows tool counts instead of handlers.
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
            .field("include_reasoning", &self.include_reasoning)
            .field("protocol", &self.protocol)
            .field("hooks", &self.hooks)
            .finish()
    }
}

/// Defaults with an empty model and base URL. Use the builder for runtime validation.
impl Default for AgentOptions {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            model: String::new(),
            base_url: String::new(),
            api_key: "not-needed".to_string(),
            headers: BTreeMap::new(),
            max_turns: 1,
            max_tokens: None,
            temperature: None,
            timeout: 60,
            tools: Vec::new(),
            auto_execute_tools: false,
            max_tool_iterations: 5,
            hooks: Hooks::new(),
            include_reasoning: false,
            protocol: ApiProtocol::OpenAiChat,
        }
    }
}

impl AgentOptions {
    /// Creates a builder that requires a model and base URL.
    ///
    /// See [`AgentOptionsBuilder`] for an example.
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

    /// Returns the caller-supplied HTTP headers.
    pub fn headers(&self) -> &BTreeMap<String, String> {
        &self.headers
    }

    /// Returns the retained `max_turns` compatibility value.
    ///
    /// The SDK does not enforce it. Use [`max_tool_iterations`](Self::max_tool_iterations)
    /// for automatic tool rounds and enforce conversation limits in your application.
    pub fn max_turns(&self) -> u32 {
        self.max_turns
    }

    /// Returns the maximum tokens setting.
    pub fn max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    /// Returns the sampling temperature, or `None` when the server should choose.
    pub fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    /// Returns the wire protocol this endpoint speaks.
    pub fn protocol(&self) -> ApiProtocol {
        self.protocol
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

    /// Returns whether the reasoning channel is surfaced to the caller.
    pub fn include_reasoning(&self) -> bool {
        self.include_reasoning
    }
}
