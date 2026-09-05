/// Builds [`AgentOptions`] with runtime validation.
///
/// `model` and `base_url` are required. Unset `temperature` and `max_tokens` are
/// omitted from requests; other settings use the defaults documented below.
///
/// ```rust
/// use open_agent::{AgentOptions, ApiProtocol};
///
/// let options = AgentOptions::builder()
///     .model("my-model")
///     .base_url("http://localhost:1234/v1")
///     .protocol(ApiProtocol::OpenAiChat)
///     .system_prompt("Answer concisely.")
///     .timeout(120)
///     .build()?;
/// # Ok::<(), open_agent::Error>(())
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
    /// Caller-supplied HTTP headers; starts empty
    headers: BTreeMap<String, String>,
    /// Optional max turns; defaults to 1
    max_turns: Option<u32>,
    /// Optional max tokens; unset means no client-imposed cap
    max_tokens: Option<u32>,
    /// Optional temperature; unset omits the parameter and lets the server choose
    temperature: Option<f32>,
    /// Optional timeout; defaults to 60 seconds
    timeout: Option<u64>,
    /// Tools to provide; starts empty
    tools: Vec<Arc<Tool>>,
    /// Optional auto-execute flag; defaults to false
    auto_execute_tools: Option<bool>,
    /// Optional max iterations; defaults to 5
    max_tool_iterations: Option<u32>,
    /// Optional reasoning capture; defaults to false
    include_reasoning: Option<bool>,
    /// Optional wire protocol; defaults to OpenAI chat completions
    protocol: Option<ApiProtocol>,
    /// Lifecycle hooks; defaults to empty
    hooks: Hooks,
}

/// Shows configuration identifiers and tool count, omitting credentials.
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
    /// Sets the system prompt. Defaults to empty; an empty prompt is omitted.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Sets the required model identifier. Empty or whitespace-only names are rejected.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the required base URL, beginning with `http://` or `https://`.
    ///
    /// The selected [`protocol`](Self::protocol) adds `/chat/completions` or `/messages`.
    /// Unlike [`BaseUrl::new`], the builder does not trim before checking the scheme.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Sets the authentication key; defaults to `"not-needed"`.
    ///
    /// An empty key suppresses the SDK authentication header. Caller headers may
    /// override protocol defaults.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Stores a compatibility value that the SDK does not enforce. Defaults to 1.
    ///
    /// This setting does not limit conversation length or enable multi-turn support.
    /// Use [`max_tool_iterations`](Self::max_tool_iterations) for automatic tool rounds;
    /// limit conversation turns in your application.
    pub fn max_turns(mut self, turns: u32) -> Self {
        self.max_turns = Some(turns);
        self
    }

    /// Sets a positive generation-token limit.
    ///
    /// Unset by default: omission lets the server choose. An explicit cap may truncate
    /// a response, including reasoning generated before visible text.
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Emits reasoning through [`StreamEvent::Reasoning`](crate::StreamEvent::Reasoning)
    /// and [`Client::reasoning`](crate::Client::reasoning). Defaults to false.
    ///
    /// Reasoning is kept separate from content and history regardless of this flag.
    pub fn include_reasoning(mut self, include: bool) -> Self {
        self.include_reasoning = Some(include);
        self
    }

    /// Sets a sampling temperature in the inclusive range 0.0–2.0.
    ///
    /// Unset by default, which omits the field. Providers may impose a narrower range
    /// or reject any explicit temperature.
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Selects the request path, credentials, body, and stream vocabulary.
    ///
    /// Defaults to [`ApiProtocol::OpenAiChat`]. Use [`ApiProtocol::Anthropic`] for a
    /// messages endpoint.
    pub fn protocol(mut self, protocol: ApiProtocol) -> Self {
        self.protocol = Some(protocol);
        self
    }

    /// Sets the timeout for each HTTP request, including its stream. Defaults to 60 seconds.
    ///
    /// This does not bound an entire conversation or automatic tool loop.
    pub fn timeout(mut self, timeout: u64) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Enables automatic tool calls and continuation. Defaults to false.
    ///
    /// Manual mode returns tool calls to the caller. Automatic mode buffers the response
    /// and returns final text after the tool loop.
    pub fn auto_execute_tools(mut self, auto: bool) -> Self {
        self.auto_execute_tools = Some(auto);
        self
    }

    /// Limits automatic tool rounds. Defaults to 5; each round may contain multiple calls.
    ///
    /// At the limit, [`Client::finish_reason`](crate::Client::finish_reason) reports
    /// [`FinishReason::MaxToolIterations`](crate::FinishReason::MaxToolIterations).
    pub fn max_tool_iterations(mut self, iterations: u32) -> Self {
        self.max_tool_iterations = Some(iterations);
        self
    }

    /// Appends one tool. Handlers are shared through [`Arc`].
    pub fn tool(mut self, tool: Tool) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    /// Appends tools without replacing those already registered.
    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools.extend(tools.into_iter().map(Arc::new));
        self
    }

    /// Replaces the hook registry. See [`Hooks`] for event ordering and examples.
    pub fn hooks(mut self, hooks: Hooks) -> Self {
        self.hooks = hooks;
        self
    }

    /// Validates configuration and applies defaults.
    ///
    /// # Errors
    ///
    /// Returns an error for missing model/base URL, empty names or endpoints, unsupported
    /// URL schemes, invalid header names/values, out-of-range temperature, or zero
    /// `max_tokens`. Provider-specific parameter restrictions are enforced by the server.
    pub fn build(self) -> crate::Result<AgentOptions> {
        let model = self
            .model
            .ok_or_else(|| crate::Error::config("model is required"))?;

        let base_url = self
            .base_url
            .ok_or_else(|| crate::Error::config("base_url is required"))?;

        if model.trim().is_empty() {
            return Err(crate::Error::invalid_input(
                "model cannot be empty or whitespace",
            ));
        }

        if base_url.trim().is_empty() {
            return Err(crate::Error::invalid_input("base_url cannot be empty"));
        }
        if !base_url.starts_with("http://") && !base_url.starts_with("https://") {
            return Err(crate::Error::invalid_input(
                "base_url must start with http:// or https://",
            ));
        }

        http_headers::validate(&self.headers)?;

        // Reuse the exported validator; an unset value remains omitted on the wire.
        self.temperature.map(Temperature::new).transpose()?;

        if let Some(tokens) = self.max_tokens {
            if tokens == 0 {
                return Err(crate::Error::invalid_input(
                    "max_tokens must be greater than 0",
                ));
            }
        }

        Ok(AgentOptions {
            system_prompt: self.system_prompt.unwrap_or_default(),
            model,
            base_url,
            api_key: self.api_key.unwrap_or_else(|| "not-needed".to_string()),
            headers: self.headers,
            max_turns: self.max_turns.unwrap_or(1),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            timeout: self.timeout.unwrap_or(60),
            tools: self.tools,
            auto_execute_tools: self.auto_execute_tools.unwrap_or(false),
            max_tool_iterations: self.max_tool_iterations.unwrap_or(5),
            hooks: self.hooks,
            include_reasoning: self.include_reasoning.unwrap_or(false),
            protocol: self.protocol.unwrap_or_default(),
        })
    }
}
