/// Tool definition for OpenAI-compatible function calling.
///
/// A `Tool` encapsulates everything needed for an LLM to understand and execute
/// a function: its identity, purpose, expected inputs, and implementation.
///
/// ## Design Philosophy
///
/// Tools are **immutable by design**. Once created, their metadata and handler
/// cannot be changed. This ensures:
/// - Thread safety through simple cloning (all fields are cheaply cloned)
/// - Predictable behavior - a tool's signature never changes mid-execution
/// - Safe concurrent access without locks
///
/// ## Cloning Behavior
///
/// The `Clone` implementation is efficient:
/// - `name` and `description`: String clones (heap allocation)
/// - `input_schema`: JSON Value clone (reference counted internally in some cases)
/// - `handler`: Arc clone (only increments atomic counter, shares same handler)
///
/// This means cloning a tool is relatively cheap and won't duplicate the actual
/// handler implementation.
///
/// ## Thread Safety
///
/// Tools are fully thread-safe:
/// - All fields are `Send + Sync`
/// - Handler is wrapped in `Arc` for shared ownership
/// - Can be stored in agent registries accessed by multiple threads
/// - Can be cloned and sent across thread boundaries
///
/// ## Examples
///
/// ```rust,no_run
/// use open_agent::Tool;
/// use serde_json::json;
///
/// // Create a tool using the constructor
/// let calculator = Tool::new(
///     "multiply",
///     "Multiply two numbers together",
///     json!({
///         "a": "number",
///         "b": "number"
///     }),
///     |args| Box::pin(async move {
///         let a = args["a"].as_f64().unwrap_or(1.0);
///         let b = args["b"].as_f64().unwrap_or(1.0);
///         Ok(json!({"result": a * b}))
///     })
/// );
///
/// // Access tool metadata
/// println!("Tool: {}", calculator.name());
/// println!("Description: {}", calculator.description());
/// println!("Schema: {}", calculator.input_schema());
/// ```
#[derive(Clone)]
pub struct Tool {
    /// Unique identifier for the tool.
    ///
    /// The name should be descriptive and follow these conventions:
    /// - Use lowercase with underscores (snake_case): `get_weather`, `search_database`
    /// - Be concise but clear: prefer `search` over `s`, but avoid overly long names
    /// - Avoid special characters that might cause issues in different contexts
    ///
    /// The LLM uses this name when deciding to invoke the tool, and it appears in
    /// function call responses. Choose names that clearly indicate the tool's purpose.
    ///
    /// # Examples
    /// - `get_weather` - Fetches weather data
    /// - `calculate` - Performs calculations
    /// - `search_documents` - Searches through document store
    name: String,

    /// Human-readable description of what the tool does.
    ///
    /// This description is sent to the LLM and significantly influences when the tool
    /// is invoked. A good description should:
    ///
    /// - Clearly state the tool's purpose and capabilities
    /// - Mention key parameters and what they control
    /// - Include any important limitations or requirements
    /// - Be concise but complete (typically 1-3 sentences)
    ///
    /// The LLM relies heavily on this description to determine if the tool is
    /// appropriate for a given user request.
    ///
    /// # Examples
    ///
    /// Good: "Get current weather conditions for a specific location. Requires a
    /// location name and optional temperature units (celsius/fahrenheit)."
    ///
    /// Poor: "Weather tool" (too vague, doesn't explain parameters or behavior)
    description: String,

    /// JSON Schema defining the tool's input parameters.
    ///
    /// This schema describes what arguments the tool expects and is automatically
    /// converted to OpenAI's function calling format. The schema serves two purposes:
    ///
    /// 1. **LLM Guidance**: Tells the LLM what arguments to provide when calling the tool
    /// 2. **Validation**: Can be used to validate arguments before handler execution
    ///
    /// The schema is stored in OpenAI's expected format after conversion:
    /// ```json
    /// {
    ///   "type": "object",
    ///   "properties": {
    ///     "param_name": {
    ///       "type": "string",
    ///       "description": "Parameter description"
    ///     }
    ///   },
    ///   "required": ["param_name"]
    /// }
    /// ```
    ///
    /// See [`Tool::new`] for details on how simple schemas are converted to this format.
    input_schema: Value,

    /// Async handler function that executes the tool's logic.
    ///
    /// The handler receives arguments as a JSON [`Value`] and returns a `Result<Value>`.
    /// It's wrapped in an [`Arc`] for efficient sharing and cloning.
    ///
    /// ## Argument Structure
    ///
    /// Arguments are passed as a JSON object matching the `input_schema`:
    /// ```json
    /// {
    ///   "param1": "value1",
    ///   "param2": 42,
    ///   "param3": [1, 2, 3]
    /// }
    /// ```
    ///
    /// ## Return Value
    ///
    /// Handlers should return a JSON value that will be sent back to the LLM.
    /// The structure is flexible but should be informative:
    ///
    /// ```json
    /// // Success response
    /// {
    ///   "status": "success",
    ///   "data": { /* results */ }
    /// }
    ///
    /// // Or just the data directly
    /// {
    ///   "temperature": 22,
    ///   "conditions": "sunny"
    /// }
    /// ```
    ///
    /// ## Error Handling
    ///
    /// If the handler returns `Err()`, the error will be propagated to the agent
    /// which can decide how to handle it (retry, report to LLM, etc.).
    ///
    /// ## Example Handler
    ///
    /// ```ignore
    /// use serde_json::{json, Value};
    /// use open_agent::{Result, Error};
    ///
    /// let handler = |args: Value| Box::pin(async move {
    ///     // Extract and validate arguments
    ///     let query = args["query"].as_str()
    ///         .ok_or_else(|| Error::tool("Missing query parameter"))?;
    ///
    ///     // Perform async operation
    ///     let results = perform_search(query).await?;
    ///
    ///     // Return structured response
    ///     Ok(json!({
    ///         "results": results,
    ///         "count": results.len()
    ///     }))
    /// });
    /// # async fn perform_search(query: &str) -> Result<Vec<String>> { Ok(vec![]) }
    /// ```
    handler: ToolHandler,
}

impl Tool {
    /// Create a new tool with flexible schema definition.
    ///
    /// This constructor handles schema conversion automatically, accepting multiple formats:
    ///
    /// ## Schema Formats
    ///
    /// ### 1. Simple Type Notation
    /// ```json
    /// {
    ///   "location": "string",
    ///   "temperature": "number"
    /// }
    /// ```
    /// All parameters are marked as required by default.
    ///
    /// ### 2. Extended Property Schema
    /// ```json
    /// {
    ///   "query": {
    ///     "type": "string",
    ///     "description": "Search query"
    ///   },
    ///   "limit": {
    ///     "type": "integer",
    ///     "optional": true
    ///   }
    /// }
    /// ```
    /// Use `"optional": true` or `"required": false` to mark parameters as optional.
    ///
    /// ### 3. Full JSON Schema
    /// ```json
    /// {
    ///   "type": "object",
    ///   "properties": {
    ///     "name": {"type": "string"}
    ///   },
    ///   "required": ["name"]
    /// }
    /// ```
    /// Already valid JSON Schema - passed through as-is.
    ///
    /// ## Handler Requirements
    ///
    /// The handler must satisfy several trait bounds:
    ///
    /// - `Fn(Value) -> Fut`: Takes JSON arguments, returns a future
    /// - `Send + Sync`: Can be shared across threads safely
    /// - `'static`: No non-static references (must own all data)
    /// - `Fut: Future<Output = Result<Value>> + Send`: Future is sendable and produces Result
    ///
    /// The constructor automatically wraps the handler in `Arc<...>` and boxes the futures,
    /// so you don't need to do this manually.
    ///
    /// ## Generic Parameters
    ///
    /// - `F`: The handler function type
    /// - `Fut`: The future type returned by the handler
    ///
    /// These are inferred automatically from the handler you provide.
    ///
    /// # Examples
    ///
    /// ## Simple Calculator Tool
    ///
    /// ```rust,no_run
    /// use open_agent::Tool;
    /// use serde_json::json;
    ///
    /// let add_tool = Tool::new(
    ///     "add",
    ///     "Add two numbers together",
    ///     json!({
    ///         "a": "number",
    ///         "b": "number"
    ///     }),
    ///     |args| {
    ///         Box::pin(async move {
    ///             let a = args.get("a")
    ///                 .and_then(|v| v.as_f64())
    ///                 .ok_or_else(|| open_agent::Error::invalid_input("Parameter 'a' must be a number"))?;
    ///             let b = args.get("b")
    ///                 .and_then(|v| v.as_f64())
    ///                 .ok_or_else(|| open_agent::Error::invalid_input("Parameter 'b' must be a number"))?;
    ///             Ok(json!({"result": a + b}))
    ///         })
    ///     }
    /// );
    /// ```
    ///
    /// ## Tool with Optional Parameters
    ///
    /// ```rust,no_run
    /// use open_agent::Tool;
    /// use serde_json::json;
    ///
    /// let search_tool = Tool::new(
    ///     "search",
    ///     "Search for information",
    ///     json!({
    ///         "query": {
    ///             "type": "string",
    ///             "description": "What to search for"
    ///         },
    ///         "max_results": {
    ///             "type": "integer",
    ///             "description": "Maximum results to return",
    ///             "optional": true,
    ///             "default": 10
    ///         }
    ///     }),
    ///     |args| Box::pin(async move {
    ///         let query = args["query"].as_str().unwrap_or("");
    ///         let max = args.get("max_results")
    ///             .and_then(|v| v.as_i64())
    ///             .unwrap_or(10);
    ///
    ///         // Perform search...
    ///         Ok(json!({"results": [], "query": query, "limit": max}))
    ///     })
    /// );
    /// ```
    ///
    /// ## Tool with External State
    ///
    /// ```rust,no_run
    /// use open_agent::Tool;
    /// use serde_json::json;
    /// use std::sync::Arc;
    ///
    /// // State that needs to be shared
    /// let api_key = Arc::new("secret-key".to_string());
    ///
    /// let tool = Tool::new(
    ///     "api_call",
    ///     "Make an API call",
    ///     json!({"endpoint": "string"}),
    ///     move |args| {
    ///         // Clone Arc to move into async block
    ///         let api_key = api_key.clone();
    ///         Box::pin(async move {
    ///             let endpoint = args["endpoint"].as_str().unwrap_or("");
    ///             // Use api_key in async operation
    ///             println!("Calling {} with key {}", endpoint, api_key);
    ///             Ok(json!({"status": "success"}))
    ///         })
    ///     }
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
        // Convert inputs to owned types
        let name = name.into();
        let description = description.into();

        // Convert the provided schema to OpenAI's expected JSON Schema format
        // This handles simple type notation, extended schemas, and full JSON Schema
        let input_schema = convert_schema_to_openai(input_schema);

        Self {
            name,
            description,
            input_schema,
            // Wrap the handler in Arc for cheap cloning and thread-safe sharing
            // Box::pin converts the future to a pinned, heap-allocated trait object
            handler: Arc::new(move |args| Box::pin(handler(args))),
        }
    }

    /// Execute the tool with the provided arguments.
    ///
    /// This method invokes the tool's handler asynchronously, passing the arguments
    /// and awaiting the result. It's the primary way to run a tool's logic.
    ///
    /// ## Execution Flow
    ///
    /// 1. Call the handler function (stored in `Arc`) with arguments
    /// 2. The handler returns a `Pin<Box<dyn Future>>`
    /// 3. Await the future to get the `Result<Value>`
    /// 4. Return the result (success value or error)
    ///
    /// ## Arguments
    ///
    /// Arguments should be a JSON object matching the tool's `input_schema`:
    /// ```json
    /// {
    ///   "param1": "value1",
    ///   "param2": 42
    /// }
    /// ```
    ///
    /// The handler is responsible for extracting and validating these arguments.
    ///
    /// ## Error Handling
    ///
    /// If the handler returns an error, it's propagated directly. The agent
    /// calling this method should handle errors appropriately (e.g., retry logic,
    /// error reporting to the LLM).
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use open_agent::Tool;
    /// # use serde_json::json;
    /// # async fn example() -> open_agent::Result<()> {
    /// let calculator = Tool::new(
    ///     "add",
    ///     "Add numbers",
    ///     json!({"a": "number", "b": "number"}),
    ///     |args| Box::pin(async move {
    ///         let sum = args["a"].as_f64().unwrap() + args["b"].as_f64().unwrap();
    ///         Ok(json!({"result": sum}))
    ///     })
    /// );
    ///
    /// // Execute the tool
    /// let result = calculator.execute(json!({"a": 5.0, "b": 3.0})).await?;
    /// assert_eq!(result["result"], 8.0);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn execute(&self, arguments: Value) -> Result<Value> {
        // Invoke the handler function with the arguments
        // The handler returns Pin<Box<dyn Future>>, which we immediately await
        (self.handler)(arguments).await
    }

    /// Convert the tool definition to OpenAI's function calling format.
    ///
    /// This method generates the JSON structure expected by OpenAI's Chat Completion
    /// API when using function calling. The format is also compatible with other
    /// LLM providers that follow OpenAI's conventions.
    ///
    /// ## Output Format
    ///
    /// Returns a JSON structure like:
    /// ```json
    /// {
    ///   "type": "function",
    ///   "function": {
    ///     "name": "tool_name",
    ///     "description": "Tool description",
    ///     "parameters": {
    ///       "type": "object",
    ///       "properties": { ... },
    ///       "required": [ ... ]
    ///     }
    ///   }
    /// }
    /// ```
    ///
    /// ## Usage in API Calls
    ///
    /// This format is typically used when constructing the `tools` array for
    /// API requests:
    /// ```json
    /// {
    ///   "model": "gpt-4",
    ///   "messages": [...],
    ///   "tools": [
    ///     // Output of to_openai_format() for each tool
    ///   ]
    /// }
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use open_agent::tool;
    /// # use serde_json::json;
    /// let my_tool = tool("search", "Search for information")
    ///     .param("query", "string")
    ///     .build(|_| async { Ok(json!({})) });
    ///
    /// let openai_format = my_tool.to_openai_format();
    ///
    /// // Verify the structure
    /// assert_eq!(openai_format["type"], "function");
    /// assert_eq!(openai_format["function"]["name"], "search");
    /// assert_eq!(openai_format["function"]["description"], "Search for information");
    /// assert!(openai_format["function"]["parameters"].is_object());
    /// ```
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

/// Custom Debug implementation for Tool.
///
/// The handler field is omitted from debug output because:
/// - Function pointers/closures don't have meaningful debug representations
/// - The `Arc<dyn Fn...>` type is complex and not useful to display
/// - Showing the handler would just print something like "Arc { ... }"
///
/// Only the metadata fields (name, description, input_schema) are shown,
/// which are the most useful for debugging tool definitions.
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
