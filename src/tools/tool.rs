/// A named asynchronous function with a JSON input schema.
///
/// Use [`tool()`] to build a tool incrementally or [`Tool::new`] to supply a schema
/// directly. Clones share the handler through [`Arc`]. The handler is responsible
/// for validating arguments; the schema is sent to the model as its description.
#[derive(Clone)]
pub struct Tool {
    /// Name used to match model tool calls to this handler.
    name: String,

    /// Description sent to the model.
    description: String,

    /// Normalized JSON input schema sent to the model.
    input_schema: Value,

    /// Shared asynchronous handler; clones of a tool share the same closure.
    handler: ToolHandler,
}

impl Tool {
    /// Creates a tool and normalizes its input schema.
    ///
    /// Accepted schema forms:
    ///
    /// - A full object containing `type` and `properties` is passed through unchanged.
    /// - A property map such as `{"query": "string"}` makes each property required.
    /// - Extended properties may include `optional`, `required`, and `default`.
    ///   Explicit `required: true` wins; otherwise `optional: true`, `required: false`,
    ///   or a default makes the property optional. The two flags are removed afterward.
    ///
    /// Type aliases include `str`, `int`, `i32`/`i64`/`u32`/`u64`, `float`/`f32`/`f64`,
    /// `bool`, `list`/`vec`, and `dict`/`map`. Unknown names become `string`. Non-object schema
    /// inputs produce an empty object schema. The handler still validates arguments.
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::Tool;
    /// use serde_json::json;
    ///
    /// let search = Tool::new(
    ///     "search",
    ///     "Search documents",
    ///     json!({
    ///         "query": "string",
    ///         "limit": {"type": "integer", "default": 10}
    ///     }),
    ///     |args| async move {
    ///         let limit = args["limit"].as_u64().unwrap_or(10);
    ///         Ok(json!({"query": args["query"], "limit": limit}))
    ///     },
    /// );
    /// ```
    pub fn new<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
        handler: F,
    ) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value>> + Send + 'static,
    {
        let name = name.into();
        let description = description.into();

        let input_schema = convert_schema_to_openai(input_schema);

        Self {
            name,
            description,
            input_schema,
            handler: Arc::new(move |args| Box::pin(handler(args))),
        }
    }

    /// Runs the handler with the supplied JSON arguments.
    ///
    /// The handler validates arguments; this method does not enforce the input schema.
    ///
    /// # Errors
    ///
    /// Propagates the handler's error unchanged. Automatic execution converts that
    /// error to a tool-result JSON object so the model can respond to it.
    pub async fn execute(&self, arguments: Value) -> Result<Value> {
        (self.handler)(arguments).await
    }

    /// Returns the tool name, description, and schema in OpenAI function-calling format.
    ///
    /// The returned JSON contains no handler. Anthropic translation extracts the same
    /// metadata when building its request.
    pub fn to_openai_format(&self) -> Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema
            }
        })
    }

    /// Returns the tool's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the tool's description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Returns a reference to the tool's input schema.
    pub fn input_schema(&self) -> &Value {
        &self.input_schema
    }
}

/// Shows tool metadata and omits the handler.
impl std::fmt::Debug for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("input_schema", &self.input_schema)
            // Handler is intentionally omitted - it's not debuggable
            .finish()
    }
}
