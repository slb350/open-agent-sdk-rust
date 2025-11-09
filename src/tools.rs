//! # Tool System for Open Agent SDK
//!
//! This module provides a comprehensive tool definition system compatible with OpenAI's
//! function calling API and similar LLM tool-use patterns.
//!
//! ## Architecture Overview
//!
//! The tool system is built around three core concepts:
//!
//! 1. **Tool Definition** - The [`Tool`] struct contains metadata (name, description, schema)
//!    and an async handler function that executes the tool's logic.
//!
//! 2. **Schema Flexibility** - Supports both simple type notation and full JSON Schema,
//!    automatically converting to the OpenAI function calling format.
//!
//! 3. **Async Execution** - Tools run asynchronously with a pinned, boxed future pattern
//!    that enables dynamic dispatch and easy integration with async runtimes.
//!
//! ## Tool Lifecycle
//!
//! ```text
//! 1. Definition:   Create tool with name, description, schema, and handler
//! 2. Registration: Add tool to agent's tool registry
//! 3. Invocation:   LLM decides to call tool with specific arguments
//! 4. Execution:    Handler processes arguments and returns result
//! 5. Response:     Result is sent back to LLM for further processing
//! ```
//!
//! ## Schema Conversion
//!
//! The system intelligently handles multiple schema formats:
//!
//! - **Simple notation**: `{"location": "string", "units": "string"}`
//! - **Typed schema**: `{"param": {"type": "number", "description": "A value"}}`
//! - **Full JSON Schema**: Already valid JSON Schema with "type" and "properties"
//!
//! All formats are normalized to OpenAI's expected JSON Schema structure.
//!
//! ## Handler Pattern
//!
//! Tool handlers use `Pin<Box<dyn Future>>` for several critical reasons:
//!
//! - **Type Erasure**: Different async functions have different concrete types.
//!   Boxing allows storing handlers with varying types in a single collection.
//!
//! - **Pinning**: Futures in Rust must be pinned to a memory location before polling.
//!   Pin guarantees the future won't move, which is essential for self-referential types.
//!
//! - **Send + Sync**: These bounds ensure handlers can be safely shared across threads,
//!   crucial for concurrent agent operations.
//!
//! ## Examples
//!
//! ### Creating a Simple Tool
//!
//! ```rust,no_run
//! use open_agent::{tool, Result};
//! use serde_json::json;
//!
//! // Using the builder pattern
//! let weather_tool = tool("get_weather", "Get current weather for a location")
//!     .param("location", "string")
//!     .param("units", "string")
//!     .build(|args| async move {
//!         let location = args["location"].as_str().unwrap_or("Unknown");
//!         let units = args["units"].as_str().unwrap_or("celsius");
//!
//!         // Simulate API call
//!         Ok(json!({
//!             "location": location,
//!             "temperature": 22,
//!             "units": units
//!         }))
//!     });
//! ```
//!
//! ### Creating a Tool with Complex Schema
//!
//! ```rust,no_run
//! use open_agent::Tool;
//! use serde_json::json;
//!
//! let search_tool = Tool::new(
//!     "search",
//!     "Search the web for information",
//!     json!({
//!         "query": {
//!             "type": "string",
//!             "description": "Search query"
//!         },
//!         "max_results": {
//!             "type": "integer",
//!             "description": "Maximum number of results",
//!             "optional": true
//!         }
//!     }),
//!     |args| Box::pin(async move {
//!         // Implementation
//!         Ok(json!({"results": []}))
//!     })
//! );
//! ```

use crate::Result;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Type alias for tool handler functions.
///
/// ## Handler Anatomy
///
/// A tool handler is a complex type that enables dynamic async execution:
///
/// ```text
/// Arc<                                      // Thread-safe reference counting
///   dyn Fn(Value)                          // Function taking JSON arguments
///     -> Pin<Box<                           // Pinned heap allocation
///       dyn Future<Output = Result<Value>>  // Async computation
///         + Send>>                          // Can cross thread boundaries
///     + Send + Sync>                        // Handler itself is thread-safe
/// ```
///
/// ### Why Arc?
///
/// [`Arc`] (Atomic Reference Counted) allows multiple parts of the system to hold
/// references to the same handler without worrying about ownership. This is essential
/// because tools may be:
/// - Stored in an agent's tool registry
/// - Cloned when creating tool definitions for API calls
/// - Accessed concurrently by multiple agent threads
///
/// The atomic reference counting ensures thread-safe access without locks on the
/// handler reference itself (though the handler may still use internal synchronization).
///
/// ### Why Pin<Box<>>?
///
/// **Pinning** guarantees that the future won't be moved in memory after creation.
/// This is critical because async functions can create self-referential structures
/// (e.g., a future holding a reference to its own data). Moving such a structure
/// would invalidate internal pointers.
///
/// **Boxing** (heap allocation) enables:
/// - Storing futures of different concrete types (different handlers) in one container
/// - Having a predictable, small stack footprint (just a pointer, not the whole future)
/// - Dynamic dispatch - the actual future type is erased but still executable
///
/// ### Why Send + Sync?
///
/// - **Send**: The future can be sent across thread boundaries. Essential for
///   multi-threaded async runtimes (like Tokio) that may move tasks between threads.
///
/// - **Sync**: Multiple threads can safely hold references to the handler. This allows
///   tools to be called concurrently by different parts of the system.
///
/// ## Example Usage
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use std::pin::Pin;
/// use std::future::Future;
/// use serde_json::{json, Value};
/// use open_agent::Result;
///
/// // Define a handler that matches ToolHandler type
/// let handler: Arc<dyn Fn(Value) -> Pin<Box<dyn Future<Output = Result<Value>> + Send>> + Send + Sync> =
///     Arc::new(|args| {
///         Box::pin(async move {
///             // Handler implementation
///             Ok(json!({"status": "success"}))
///         })
///     });
///
/// // Can be cloned cheaply (only increments Arc counter)
/// let handler_clone = handler.clone();
/// ```
pub type ToolHandler =
    Arc<dyn Fn(Value) -> Pin<Box<dyn Future<Output = Result<Value>> + Send>> + Send + Sync>;

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

/// Convert various schema formats to OpenAI's JSON Schema format.
///
/// This function is the core of the schema conversion system. It accepts multiple
/// input formats and normalizes them to the standard JSON Schema structure expected
/// by OpenAI's function calling API.
///
/// ## Conversion Logic
///
/// ### 1. Full JSON Schema (Pass-through)
/// If the input already has `"type": "object"` and `"properties"`, it's assumed to
/// be a complete JSON Schema and returned as-is:
/// ```json
/// {
///   "type": "object",
///   "properties": { "name": {"type": "string"} },
///   "required": ["name"]
/// }
/// // → Returned unchanged
/// ```
///
/// ### 2. Simple Type Notation
/// A flat object with type strings is expanded to full JSON Schema:
/// ```json
/// {"location": "string", "temperature": "number"}
/// // → Converts to:
/// {
///   "type": "object",
///   "properties": {
///     "location": {"type": "string"},
///     "temperature": {"type": "number"}
///   },
///   "required": ["location", "temperature"]
/// }
/// ```
/// All parameters become required by default.
///
/// ### 3. Extended Property Schema
/// Object values with additional metadata (description, optional, etc.):
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
/// // → Converts to JSON Schema with "query" required, "limit" optional
/// ```
///
/// ## Required vs Optional Parameters
///
/// The function determines if a parameter is required using this logic:
/// 1. If `"required": true` is explicitly set → required
/// 2. If `"required": false` is explicitly set → optional
/// 3. If `"optional": true` is set → optional
/// 4. If parameter has a `"default"` value → optional
/// 5. Otherwise → required (default behavior)
///
/// The `"optional"` and `"required"` keys are removed from the final schema
/// as they're not part of standard JSON Schema (the `required` array is used instead).
///
/// ## Type Mapping
///
/// Simple type strings are converted via [`type_to_json_schema`]:
/// - `"string"`, `"str"` → `{"type": "string"}`
/// - `"number"`, `"float"`, `"f32"`, `"f64"` → `{"type": "number"}`
/// - `"integer"`, `"int"`, `"i32"`, `"i64"` → `{"type": "integer"}`
/// - `"boolean"`, `"bool"` → `{"type": "boolean"}`
/// - `"array"`, `"list"`, `"vec"` → `{"type": "array"}`
/// - `"object"`, `"dict"`, `"map"` → `{"type": "object"}`
///
/// ## Examples
///
/// See the test cases in this module for concrete examples of each conversion path.
fn convert_schema_to_openai(schema: Value) -> Value {
    // Check if the input is already a complete JSON Schema
    // A complete schema has both "type": "object" and a "properties" field
    if schema.is_object() {
        let obj = schema
            .as_object()
            .expect("BUG: is_object() returned true but as_object() returned None");
        if obj.contains_key("type") && obj.contains_key("properties") {
            // This is already a full JSON Schema - pass it through unchanged
            return schema;
        }

        // If we get here, we need to convert to full JSON Schema format
        // Initialize the properties map and required array
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        // Iterate through each parameter in the input schema
        for (param_name, param_type) in obj {
            if let Some(type_str) = param_type.as_str() {
                // Case 1: Simple type notation like "string", "number", etc.
                // Convert the type string to a proper JSON Schema type object
                properties.insert(param_name.clone(), type_to_json_schema(type_str));

                // Simple notation always means required (no way to specify optional)
                required.push(param_name.clone());
            } else if param_type.is_object() {
                // Case 2: Extended property schema with metadata
                // Clone the property schema so we can modify it
                let mut prop = param_type.clone();
                let prop_obj = prop
                    .as_object_mut()
                    .expect("BUG: is_object() returned true but as_object_mut() returned None");

                // Extract and remove the "optional" flag (not standard JSON Schema)
                let is_optional = prop_obj
                    .remove("optional")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                // Extract and remove the "required" flag (not standard JSON Schema)
                // This is different from the "required" array - it's per-property
                let is_required = prop_obj.remove("required").and_then(|v| v.as_bool());

                // Check if the property has a default value
                // Properties with defaults are typically optional
                let has_default = prop_obj.contains_key("default");

                // Add the cleaned property schema to the properties map
                properties.insert(param_name.clone(), prop);

                // Determine if this parameter should be in the required array
                // Priority order:
                // 1. Explicit required: true → add to required
                // 2. Explicit optional: true OR required: false → don't add
                // 3. Has default value → don't add (defaults make params optional)
                // 4. Otherwise → add to required (conservative default)
                if let Some(true) = is_required {
                    required.push(param_name.clone());
                } else if is_optional || is_required == Some(false) {
                    // Explicitly optional - don't add to required array
                } else if !has_default {
                    // No explicit optionality and no default → required
                    required.push(param_name.clone());
                }
                // Note: if has_default is true and no explicit required/optional,
                // we don't add to required (defaults imply optional)
            }
        }

        // Build and return the complete JSON Schema object
        return serde_json::json!({
            "type": "object",
            "properties": properties,
            "required": required
        });
    }

    // Fallback case: input is not an object (unexpected but handled gracefully)
    // Return an empty object schema that accepts any properties
    serde_json::json!({
        "type": "object",
        "properties": {},
        "required": []
    })
}

/// Convert a type string to a JSON Schema type object.
///
/// This function maps friendly, Rust-like type names to their JSON Schema equivalents.
/// It's designed to accept common variations developers might use, making tool
/// definition more intuitive.
///
/// ## Type Mappings
///
/// | Input Types | JSON Schema Type | Use Case |
/// |-------------|------------------|----------|
/// | `"string"`, `"str"` | `"string"` | Text data |
/// | `"number"`, `"float"`, `"f32"`, `"f64"` | `"number"` | Floating point numbers |
/// | `"integer"`, `"int"`, `"i32"`, `"i64"`, `"u32"`, `"u64"` | `"integer"` | Whole numbers |
/// | `"boolean"`, `"bool"` | `"boolean"` | True/false values |
/// | `"array"`, `"list"`, `"vec"` | `"array"` | Lists/arrays |
/// | `"object"`, `"dict"`, `"map"` | `"object"` | Nested objects/maps |
/// | anything else | `"string"` | Default fallback |
///
/// ## Design Rationale
///
/// The function accepts multiple aliases for each type to accommodate different
/// naming conventions:
/// - Standard JSON Schema names (`"string"`, `"integer"`, `"boolean"`)
/// - Common programming abbreviations (`"str"`, `"int"`, `"bool"`)
/// - Rust-specific types (`"i32"`, `"f64"`, `"vec"`)
/// - Python-style names (`"dict"`, `"list"`)
///
/// ## Default Behavior
///
/// Unknown type strings default to `"string"` rather than causing an error.
/// This prevents tool creation from failing due to typos, though it may lead
/// to unexpected schema behavior. Consider validating type strings at a higher
/// level if strict type checking is needed.
///
/// ## Output Format
///
/// Always returns a JSON object with a single `"type"` field:
/// ```json
/// {"type": "string"}
/// {"type": "number"}
/// {"type": "integer"}
/// // etc.
/// ```
///
/// ## Examples
///
/// ```rust
/// # use serde_json::json;
/// # fn type_to_json_schema(type_str: &str) -> serde_json::Value {
/// #     let json_type = match type_str {
/// #         "string" | "str" => "string",
/// #         "integer" | "int" | "i32" | "i64" | "u32" | "u64" => "integer",
/// #         "number" | "float" | "f32" | "f64" => "number",
/// #         "boolean" | "bool" => "boolean",
/// #         "array" | "list" | "vec" => "array",
/// #         "object" | "dict" | "map" => "object",
/// #         _ => "string",
/// #     };
/// #     json!({ "type": json_type })
/// # }
/// assert_eq!(type_to_json_schema("string"), json!({"type": "string"}));
/// assert_eq!(type_to_json_schema("i64"), json!({"type": "integer"}));
/// assert_eq!(type_to_json_schema("f32"), json!({"type": "number"}));
/// assert_eq!(type_to_json_schema("bool"), json!({"type": "boolean"}));
/// assert_eq!(type_to_json_schema("vec"), json!({"type": "array"}));
/// assert_eq!(type_to_json_schema("unknown"), json!({"type": "string"})); // fallback
/// ```
fn type_to_json_schema(type_str: &str) -> Value {
    // Match against known type strings (case-sensitive)
    // The match is designed to be comprehensive but not exhaustive
    let json_type = match type_str {
        // String types
        "string" | "str" => "string",

        // Integer types (various Rust integer types accepted)
        "integer" | "int" | "i32" | "i64" | "u32" | "u64" => "integer",

        // Floating point types
        "number" | "float" | "f32" | "f64" => "number",

        // Boolean types
        "boolean" | "bool" => "boolean",

        // Array/list types
        "array" | "list" | "vec" => "array",

        // Object/map types
        "object" | "dict" | "map" => "object",

        // Unknown type - default to string for safety
        // This prevents errors but may hide typos
        _ => "string",
    };

    // Return a JSON Schema type object
    serde_json::json!({ "type": json_type })
}

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

/// Create a tool using the builder pattern (convenience function).
///
/// This is the recommended way to create tools. It returns a [`ToolBuilder`] that
/// allows you to fluently configure the tool's schema and handler.
///
/// ## Typical Usage Pattern
///
/// ```text
/// tool(name, description)
///     .param(name, type)  // Add parameters (optional, can repeat)
///     .build(handler)     // Provide handler and create Tool
/// ```
///
/// ## Why Use This Instead of Tool::new?
///
/// - **More readable**: The builder pattern reads like natural language
/// - **Incremental schema building**: Add parameters one at a time
/// - **Flexible**: Can conditionally add parameters or use `.schema()` for complex cases
/// - **Type-safe**: Method chaining ensures you can't forget the handler
///
/// ## Parameters
///
/// - `name`: Unique identifier for the tool (snake_case recommended)
/// - `description`: Human-readable explanation of what the tool does
///
/// Both parameters accept any type that implements `Into<String>`, so you can
/// pass string literals, `String` values, or anything else convertible to String.
///
/// ## Examples
///
/// ### Basic Calculator Tool
///
/// ```rust,no_run
/// use open_agent::tool;
/// use serde_json::json;
///
/// let add_tool = tool("add", "Add two numbers")
///     .param("a", "number")
///     .param("b", "number")
///     .build(|args| async move {
///         let a = args.get("a")
///             .and_then(|v| v.as_f64())
///             .ok_or_else(|| open_agent::Error::invalid_input("Parameter 'a' must be a number"))?;
///         let b = args.get("b")
///             .and_then(|v| v.as_f64())
///             .ok_or_else(|| open_agent::Error::invalid_input("Parameter 'b' must be a number"))?;
///         Ok(json!({"result": a + b}))
///     });
/// ```
///
/// ### Tool with External HTTP Client
///
/// ```rust,no_run
/// use open_agent::{tool, Error};
/// use serde_json::json;
/// # use std::sync::Arc;
///
/// // Shared HTTP client (example - use your actual HTTP client)
/// # struct HttpClient;
/// # impl HttpClient {
/// #     fn new() -> Self { HttpClient }
/// #     async fn get(&self, url: &str) -> Result<String, Box<dyn std::error::Error>> {
/// #         Ok("response".to_string())
/// #     }
/// # }
/// let http_client = Arc::new(HttpClient::new());
///
/// let fetch_tool = tool("fetch_url", "Fetch content from a URL")
///     .param("url", "string")
///     .build(move |args| {
///         let client = http_client.clone();
///         async move {
///             let url = args["url"].as_str().unwrap_or("");
///             let content = client.get(url).await
///                 .map_err(|e| Error::tool(format!("Failed to fetch: {}", e)))?;
///             Ok(json!({"content": content}))
///         }
///     });
/// ```
///
/// ### Tool with Complex Schema
///
/// ```rust,no_run
/// use open_agent::tool;
/// use serde_json::json;
///
/// let search_tool = tool("search", "Search for information")
///     .schema(json!({
///         "query": {
///             "type": "string",
///             "description": "Search query"
///         },
///         "filters": {
///             "type": "object",
///             "description": "Optional filters",
///             "optional": true,
///             "properties": {
///                 "date_from": {"type": "string"},
///                 "date_to": {"type": "string"}
///             }
///         },
///         "max_results": {
///             "type": "integer",
///             "default": 10,
///             "optional": true
///         }
///     }))
///     .build(|args| async move {
///         // Implementation
///         Ok(json!({"results": []}))
///     });
/// ```
///
/// ### Conditional Parameter Addition
///
/// ```rust,no_run
/// use open_agent::tool;
/// use serde_json::json;
///
/// # let enable_advanced = true;
/// let mut builder = tool("process", "Process data")
///     .param("input", "string");
///
/// // Conditionally add parameters
/// if enable_advanced {
///     builder = builder.param("advanced_mode", "boolean");
/// }
///
/// let my_tool = builder.build(|args| async move {
///     Ok(json!({"status": "processed"}))
/// });
/// ```
///
/// ### Integration with Agent
///
/// ```rust,no_run
/// use open_agent::{Client, AgentOptions, tool};
/// use serde_json::json;
///
/// # async fn example() -> open_agent::Result<()> {
/// let weather_tool = tool("get_weather", "Get weather for a location")
///     .param("location", "string")
///     .build(|args| async move {
///         Ok(json!({"temp": 72, "conditions": "sunny"}))
///     });
///
/// let options = AgentOptions::builder()
///     .model("gpt-4")
///     .base_url("http://localhost:1234/v1")
///     .tool(weather_tool)
///     .build()?;
///
/// let client = Client::new(options)?;
/// // Client can now use the tool when responding to queries
/// # Ok(())
/// # }
/// ```
///
/// ## See Also
///
/// - [`Tool::new`] - Direct constructor if you prefer not using the builder
/// - [`ToolBuilder`] - The builder type returned by this function
/// - [`Tool`] - The final tool type produced by `.build()`
pub fn tool(name: impl Into<String>, description: impl Into<String>) -> ToolBuilder {
    ToolBuilder::new(name, description)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use serde_json::json;

    #[test]
    fn test_type_to_json_schema() {
        assert_eq!(type_to_json_schema("string"), json!({"type": "string"}));
        assert_eq!(type_to_json_schema("integer"), json!({"type": "integer"}));
        assert_eq!(type_to_json_schema("number"), json!({"type": "number"}));
        assert_eq!(type_to_json_schema("bool"), json!({"type": "boolean"}));
    }

    #[test]
    fn test_convert_simple_schema() {
        let schema = json!({
            "location": "string",
            "units": "string"
        });

        let result = convert_schema_to_openai(schema);

        assert_eq!(result["type"], "object");
        assert_eq!(result["properties"]["location"]["type"], "string");
        assert_eq!(result["properties"]["units"]["type"], "string");
        assert_eq!(result["required"], json!(["location", "units"]));
    }

    #[test]
    fn test_convert_full_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        });

        let result = convert_schema_to_openai(schema.clone());
        assert_eq!(result, schema);
    }

    #[tokio::test]
    async fn test_tool_creation() {
        let add_tool = tool("add", "Add two numbers")
            .param("a", "number")
            .param("b", "number")
            .build(|args| async move {
                let a = args
                    .get("a")
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| Error::invalid_input("Parameter 'a' must be a number"))?;
                let b = args
                    .get("b")
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| Error::invalid_input("Parameter 'b' must be a number"))?;
                Ok(json!({"result": a + b}))
            });

        assert_eq!(add_tool.name, "add");
        assert_eq!(add_tool.description, "Add two numbers");

        let result = add_tool.execute(json!({"a": 5.0, "b": 3.0})).await.unwrap();
        assert_eq!(result["result"], 8.0);
    }

    #[test]
    fn test_tool_to_openai_format() {
        let tool = tool("test", "Test tool")
            .param("param1", "string")
            .build(|_| async { Ok(json!({})) });

        let format = tool.to_openai_format();

        assert_eq!(format["type"], "function");
        assert_eq!(format["function"]["name"], "test");
        assert_eq!(format["function"]["description"], "Test tool");
        assert!(format["function"]["parameters"].is_object());
    }

    #[test]
    fn test_param_after_non_object_schema() {
        // Edge case: calling .param() after setting schema to non-object
        // Should reset schema and add param without panicking
        let tool = tool("test", "Test tool")
            .schema(json!("string")) // Set to non-object
            .param("key", "number") // Should reset schema to {} and add param
            .build(|_| async { Ok(json!({})) });

        let format = tool.to_openai_format();

        // Verify it worked - schema should be object with the param
        assert!(format["function"]["parameters"].is_object());
        assert!(format["function"]["parameters"]["properties"]["key"].is_object());
    }
}
