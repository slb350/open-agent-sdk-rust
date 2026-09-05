/// Starts a [`ToolBuilder`].
///
/// ```rust
/// use open_agent::{tool, Error};
/// use serde_json::json;
///
/// let add = tool("add", "Add two numbers")
///     .param("a", "number")
///     .param("b", "number")
///     .build(|args| async move {
///         let a = args["a"].as_f64()
///             .ok_or_else(|| Error::invalid_input("a must be a number"))?;
///         let b = args["b"].as_f64()
///             .ok_or_else(|| Error::invalid_input("b must be a number"))?;
///         Ok(json!({"result": a + b}))
///     });
/// ```
///
/// Use [`ToolBuilder::schema`] for optional parameters or a complete JSON Schema,
/// and [`ToolBuilder::build`] for an example with shared state.
pub fn tool(name: impl Into<String>, description: impl Into<String>) -> ToolBuilder {
    ToolBuilder::new(name, description)
}
