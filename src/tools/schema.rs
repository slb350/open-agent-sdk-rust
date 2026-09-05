/// Normalizes the schema forms documented by [`Tool::new`].
fn convert_schema_to_openai(schema: Value) -> Value {
    if schema.is_object() {
        let obj = schema
            .as_object()
            .expect("BUG: is_object() returned true but as_object() returned None");
        if obj.contains_key("type") && obj.contains_key("properties") {
            return schema;
        }

        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for (param_name, param_type) in obj {
            if let Some(type_str) = param_type.as_str() {
                properties.insert(param_name.clone(), type_to_json_schema(type_str));

                required.push(param_name.clone());
            } else if param_type.is_object() {
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

                properties.insert(param_name.clone(), prop);

                if let Some(true) = is_required {
                    required.push(param_name.clone());
                } else if is_optional || is_required == Some(false) {
                    // Explicitly optional properties stay out of the required list.
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

    serde_json::json!({
        "type": "object",
        "properties": {},
        "required": []
    })
}

/// Maps shorthand type names to JSON Schema types; unknown names become strings.
fn type_to_json_schema(type_str: &str) -> Value {
    let json_type = match type_str {
        "string" | "str" => "string",

        "integer" | "int" | "i32" | "i64" | "u32" | "u64" => "integer",

        "number" | "float" | "f32" | "f64" => "number",

        "boolean" | "bool" => "boolean",

        "array" | "list" | "vec" => "array",

        "object" | "dict" | "map" => "object",

        _ => "string",
    };

    serde_json::json!({ "type": json_type })
}
