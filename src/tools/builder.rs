/// Builder for creating tools with a fluent API.
///
/// The `ToolBuilder` provides a convenient, readable way to construct tools
/// using method chaining. It's especially useful when building tools incrementally
/// or when the schema structure is determined dynamically.
///
/// ## Builder Pattern Benefits
///
/// - **Readability**: Method chains read like natural language
/// - **Flexibility**: Add parameters conditionally
/// - **Type safety**: Catches errors at compile time
/// - **Discoverability**: IDE autocomplete shows available options
///
/// ## Workflow
///
/// 1. Create builder with [`tool()`] or [`ToolBuilder::new()`]
/// 2. Add parameters with [`.param()`](ToolBuilder::param)
/// 3. Optionally set schema with [`.schema()`](ToolBuilder::schema)
/// 4. Finalize with [`.build()`](ToolBuilder::build) and provide handler
///
/// ## Examples
///
/// See the [`tool()`] function for detailed examples.
///
/// ## Note on Schema Mutation
///
/// If you call `.schema()` after `.param()`, the parameters will be replaced
/// by the new schema. Similarly, calling `.param()` after `.schema()` will
/// reset a non-object schema to an empty object before adding the parameter.
/// Generally, use either `.schema()` or `.param()`, not both.
pub struct ToolBuilder {
    /// The tool's unique identifier
    name: String,

    /// Human-readable description of the tool's purpose
    description: String,

    /// The input schema, built up through .param() calls or set via .schema()
    schema: Value,
}

impl ToolBuilder {
    /// Start building a new tool with a name and description.
    ///
    /// This creates a builder with an empty schema. You can then add parameters
    /// using [`.param()`](ToolBuilder::param) or set a complete schema with
    /// [`.schema()`](ToolBuilder::schema).
    ///
    /// ## Parameters
    ///
    /// - `name`: Tool identifier (converted to String via Into trait)
    /// - `description`: Human-readable explanation of what the tool does
    ///
    /// ## Examples
    ///
    /// ```rust
    /// # use open_agent::ToolBuilder;
    /// let builder = ToolBuilder::new("search", "Search for information");
    /// // builder.param(...).build(...)
    /// ```
    ///
    /// Typically you'll use the [`tool()`] convenience function instead of calling
    /// this directly.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            // Start with an empty object schema
            schema: serde_json::json!({}),
        }
    }

    /// Set the complete input schema.
    ///
    /// This replaces any schema or parameters set previously. Use this when you
    /// have a pre-built schema object (especially useful for complex schemas
    /// with nested structures).
    ///
    /// ## Schema Format
    ///
    /// Accepts any of the formats supported by [`Tool::new`]:
    /// - Simple type notation: `{"param": "string"}`
    /// - Extended schema: `{"param": {"type": "string", "description": "..."}}`
    /// - Full JSON Schema: `{"type": "object", "properties": {...}, "required": [...]}`
    ///
    /// ## Warning
    ///
    /// This overwrites any parameters added via `.param()`. Generally, choose
    /// one approach: either use `.param()` for simple cases or `.schema()` for
    /// complex cases, but not both.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// # use open_agent::tool;
    /// # use serde_json::json;
    /// let my_tool = tool("api_call", "Make an API call")
    ///     .schema(json!({
    ///         "endpoint": {
    ///             "type": "string",
    ///             "description": "API endpoint URL",
    ///             "pattern": "^https://"
    ///         },
    ///         "method": {
    ///             "type": "string",
    ///             "enum": ["GET", "POST", "PUT", "DELETE"]
    ///         }
    ///     }))
    ///     .build(|_| async { Ok(json!({})) });
    /// ```
    pub fn schema(mut self, schema: Value) -> Self {
        // Replace the current schema entirely
        self.schema = schema;
        self
    }

    /// Add a single parameter to the schema.
    ///
    /// This is a convenience method for building schemas incrementally. Each call
    /// adds one parameter with a simple type string.
    ///
    /// ## Parameters
    ///
    /// - `name`: Parameter name (will be required in tool calls)
    /// - `type_str`: Type string like "string", "number", "boolean", etc.
    ///   Supported types: "string", "number", "integer", "boolean", "array", "object".
    ///
    /// ## Behavior
    ///
    /// - If the current schema is not an object (e.g., you called `.schema()` with
    ///   a non-object value), it will be reset to an empty object first.
    /// - All parameters added via `.param()` are marked as required.
    /// - For optional parameters, use `.schema()` with extended property format.
    ///
    /// ## Method Chaining
    ///
    /// This method consumes `self` and returns it, enabling method chaining:
    /// ```rust
    /// # use open_agent::tool;
    /// # use serde_json::json;
    /// let my_tool = tool("calculate", "Perform calculation")
    ///     .param("operation", "string")
    ///     .param("x", "number")
    ///     .param("y", "number")
    ///     .build(|_| async { Ok(json!({})) });
    /// ```
    ///
    /// ## Examples
    ///
    /// ```rust
    /// # use open_agent::tool;
    /// # use serde_json::json;
    /// // Add multiple parameters
    /// let weather_tool = tool("get_weather", "Get weather for a location")
    ///     .param("location", "string")
    ///     .param("units", "string")
    ///     .build(|args| async move {
    ///         // Implementation
    ///         Ok(json!({"temp": 72}))
    ///     });
    /// ```
    pub fn param(mut self, name: &str, type_str: &str) -> Self {
        // Ensure schema is an object, reset if not
        // This handles the edge case where .schema() was called with a non-object
        if !self.schema.is_object() {
            self.schema = serde_json::json!({});
        }

        // Get mutable reference to the object. This should always succeed because we just
        // ensured it's an object above, but we use expect() for defensive programming.
        let obj = self
            .schema
            .as_object_mut()
            .expect("BUG: schema should be an object after initialization");

        // Insert the parameter as a simple type string
        // This will be converted to proper JSON Schema by convert_schema_to_openai
        obj.insert(name.to_string(), Value::String(type_str.to_string()));

        self
    }

    /// Build the final Tool with a handler function.
    ///
    /// This consumes the builder and produces a [`Tool`] ready for use. The handler
    /// function defines what happens when the tool is called.
    ///
    /// ## Handler Requirements
    ///
    /// The handler must be:
    /// - An async function or closure
    /// - Accept a single `Value` argument (the tool's input parameters)
    /// - Return a `Future<Output = Result<Value>>`
    /// - Implement `Send + Sync + 'static` for thread safety
    ///
    /// ## Generic Parameters
    ///
    /// - `F`: The handler function type (inferred from the closure/function you provide)
    /// - `Fut`: The future type returned by the handler (inferred automatically)
    ///
    /// ## Examples
    ///
    /// ### Simple Handler
    /// ```rust
    /// # use open_agent::tool;
    /// # use serde_json::json;
    /// let my_tool = tool("echo", "Echo back the input")
    ///     .param("message", "string")
    ///     .build(|args| async move {
    ///         Ok(args) // Echo arguments back
    ///     });
    /// ```
    ///
    /// ### Handler with External State
    /// ```rust
    /// # use open_agent::tool;
    /// # use serde_json::json;
    /// # use std::sync::Arc;
    /// let counter = Arc::new(std::sync::atomic::AtomicU32::new(0));
    ///
    /// let my_tool = tool("increment", "Increment a counter")
    ///     .build(move |_args| {
    ///         let counter = counter.clone();
    ///         async move {
    ///             let val = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    ///             Ok(json!({"count": val + 1}))
    ///         }
    ///     });
    /// ```
    ///
    /// ### Handler with Error Handling
    /// ```rust
    /// # use open_agent::{tool, Error};
    /// # use serde_json::json;
    /// let my_tool = tool("divide", "Divide two numbers")
    ///     .param("a", "number")
    ///     .param("b", "number")
    ///     .build(|args| async move {
    ///         let a = args["a"].as_f64().ok_or_else(|| Error::tool("Invalid 'a' parameter"))?;
    ///         let b = args["b"].as_f64().ok_or_else(|| Error::tool("Invalid 'b' parameter"))?;
    ///
    ///         if b == 0.0 {
    ///             return Err(Error::tool("Division by zero"));
    ///         }
    ///
    ///         Ok(json!({"result": a / b}))
    ///     });
    /// ```
    pub fn build<F, Fut>(self, handler: F) -> Tool
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value>> + Send + 'static,
    {
        // Delegate to Tool::new which handles schema conversion and handler wrapping
        Tool::new(self.name, self.description, self.schema, handler)
    }
}
