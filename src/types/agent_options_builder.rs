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
    /// Optional max tokens; unset means no client-imposed cap
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
    /// Leaving this unset omits `max_tokens` from the request entirely, so the server applies
    /// its own limit. That is the right default for long-context and reasoning models, which
    /// a client-side cap would cut off mid-response.
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

        // Validate max_tokens if set. An unset value is passed through as None so the field is
        // omitted from the wire request entirely, letting the server apply its own limit —
        // there is otherwise no way to express "no cap", and a client-imposed default truncates
        // long-context and reasoning models mid-response.
        if let Some(tokens) = self.max_tokens {
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
            max_tokens: self.max_tokens,
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
