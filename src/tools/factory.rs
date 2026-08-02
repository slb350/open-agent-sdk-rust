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
