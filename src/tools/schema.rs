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
