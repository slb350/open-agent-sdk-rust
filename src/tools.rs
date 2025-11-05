//! Tool system for Open Agent SDK
//!
//! Provides tool definition compatible with OpenAI's function calling API.

use crate::Result;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Type alias for tool handlers
///
/// Tool handlers are async functions that take arguments as a JSON Value
/// and return a Result containing a JSON Value
pub type ToolHandler =
    Arc<dyn Fn(Value) -> Pin<Box<dyn Future<Output = Result<Value>> + Send>> + Send + Sync>;

/// Tool definition for OpenAI-compatible function calling
#[derive(Clone)]
pub struct Tool {
    /// Unique tool identifier
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Parameter schema (JSON Schema format)
    pub input_schema: Value,
    /// Async function that executes the tool
    pub handler: ToolHandler,
}

impl Tool {
    /// Create a new tool with a simple schema
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use open_agent::Tool;
    /// use serde_json::json;
    ///
    /// let tool = Tool::new(
    ///     "add",
    ///     "Add two numbers",
    ///     json!({
    ///         "a": "number",
    ///         "b": "number"
    ///     }),
    ///     |args| {
    ///         Box::pin(async move {
    ///             let a = args["a"].as_f64().unwrap_or(0.0);
    ///             let b = args["b"].as_f64().unwrap_or(0.0);
    ///             Ok(json!({"result": a + b}))
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

    /// Execute the tool with given arguments
    pub async fn execute(&self, arguments: Value) -> Result<Value> {
        (self.handler)(arguments).await
    }

    /// Convert tool definition to OpenAI function calling format
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
}

impl std::fmt::Debug for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("input_schema", &self.input_schema)
            .finish()
    }
}

/// Convert simple type schema or JSON Schema to OpenAI format
///
/// Handles both:
/// - Simple schemas: `{"location": "string", "units": "string"}`
/// - Full JSON Schemas with "type" and "properties"
fn convert_schema_to_openai(schema: Value) -> Value {
    // If already a JSON Schema (has 'type' and 'properties'), return as-is
    if schema.is_object() {
        let obj = schema.as_object().unwrap();
        if obj.contains_key("type") && obj.contains_key("properties") {
            return schema;
        }

        // Convert simple schema to JSON Schema
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for (param_name, param_type) in obj {
            if let Some(type_str) = param_type.as_str() {
                // Simple type string
                properties.insert(param_name.clone(), type_to_json_schema(type_str));
                required.push(param_name.clone());
            } else if param_type.is_object() {
                // Already a property schema
                let mut prop = param_type.clone();
                let prop_obj = prop.as_object_mut().unwrap();

                let is_optional = prop_obj
                    .remove("optional")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                let is_required = prop_obj.remove("required").and_then(|v| v.as_bool());

                let has_default = prop_obj.contains_key("default");

                properties.insert(param_name.clone(), prop);

                if let Some(true) = is_required {
                    required.push(param_name.clone());
                } else if is_optional || is_required == Some(false) {
                    // Don't add to required
                } else if !has_default {
                    required.push(param_name.clone());
                }
            }
        }

        return serde_json::json!({
            "type": "object",
            "properties": properties,
            "required": required
        });
    }

    // Fallback: return empty object schema
    serde_json::json!({
        "type": "object",
        "properties": {},
        "required": []
    })
}

/// Convert type string to JSON Schema type
fn type_to_json_schema(type_str: &str) -> Value {
    let json_type = match type_str {
        "string" | "str" => "string",
        "integer" | "int" | "i32" | "i64" | "u32" | "u64" => "integer",
        "number" | "float" | "f32" | "f64" => "number",
        "boolean" | "bool" => "boolean",
        "array" | "list" | "vec" => "array",
        "object" | "dict" | "map" => "object",
        _ => "string", // Default to string
    };

    serde_json::json!({ "type": json_type })
}

/// Builder for creating tools with a fluent API
pub struct ToolBuilder {
    name: String,
    description: String,
    schema: Value,
}

impl ToolBuilder {
    /// Start building a new tool
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            schema: serde_json::json!({}),
        }
    }

    /// Set the input schema
    pub fn schema(mut self, schema: Value) -> Self {
        self.schema = schema;
        self
    }

    /// Add a parameter with a type string
    pub fn param(mut self, name: &str, type_str: &str) -> Self {
        let obj = self.schema.as_object_mut().unwrap();
        obj.insert(name.to_string(), Value::String(type_str.to_string()));
        self
    }

    /// Build the tool with a handler function
    pub fn build<F, Fut>(self, handler: F) -> Tool
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value>> + Send + 'static,
    {
        Tool::new(self.name, self.description, self.schema, handler)
    }
}

/// Create a tool using a builder pattern
///
/// # Examples
///
/// ```rust,no_run
/// use open_agent::tool;
/// use serde_json::json;
///
/// let add_tool = tool("add", "Add two numbers")
///     .param("a", "number")
///     .param("b", "number")
///     .build(|args| async move {
///         let a = args["a"].as_f64().unwrap_or(0.0);
///         let b = args["b"].as_f64().unwrap_or(0.0);
///         Ok(json!({"result": a + b}))
///     });
/// ```
pub fn tool(name: impl Into<String>, description: impl Into<String>) -> ToolBuilder {
    ToolBuilder::new(name, description)
}

#[cfg(test)]
mod tests {
    use super::*;
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
                let a = args["a"].as_f64().unwrap_or(0.0);
                let b = args["b"].as_f64().unwrap_or(0.0);
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
}
