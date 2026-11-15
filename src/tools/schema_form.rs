//! Tells a complete JSON Schema object apart from the shorthand property map.
//!
//! [`Tool::new`](super::Tool::new) accepts both, and they overlap: `{"type": "object"}` is a
//! complete schema for a tool without arguments, and it is also what a property map looks like
//! when it declares one parameter named `type`. Reading it as the property map turns an
//! argument-less tool into one that requires a `type` argument, so the distinction lives here,
//! where tests can pin it.

use serde_json::{Map, Value};

/// Object-schema keywords a caller may write beside `"type": "object"` instead of `properties`.
///
/// Every one is a JSON Schema vocabulary keyword that applies to an object. A shorthand
/// property map maps parameter names to a type name or a property object, so a key from this
/// list in a map that also says `"type": "object"` is read as the keyword, not as a parameter.
const OBJECT_SCHEMA_KEYWORDS: &[&str] = &[
    "$schema",
    "$id",
    "$defs",
    "definitions",
    "title",
    "description",
    "required",
    "additionalProperties",
    "patternProperties",
    "propertyNames",
    "minProperties",
    "maxProperties",
    "unevaluatedProperties",
    "dependentRequired",
    "dependentSchemas",
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "if",
    "then",
    "else",
    "examples",
];

/// Whether `schema` is already a complete object schema rather than a shorthand property map.
///
/// A map with `type` and `properties` always is. A map whose `type` is the string `"object"`
/// and whose every other key is an object-schema keyword is one too, with no properties of its
/// own. Anything else, including `{"type": "string", "query": "string"}` or `{"type": "object",
/// "query": "string"}`, is a property map that happens to name a parameter `type`.
pub(super) fn is_full_object_schema(schema: &Map<String, Value>) -> bool {
    if !schema.contains_key("type") {
        return false;
    }
    schema.contains_key("properties")
        || (schema.get("type").and_then(Value::as_str) == Some("object")
            && schema
                .keys()
                .all(|key| key == "type" || OBJECT_SCHEMA_KEYWORDS.contains(&key.as_str())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn is_full(schema: Value) -> bool {
        is_full_object_schema(schema.as_object().expect("test schema is an object"))
    }

    #[test]
    fn type_with_properties_is_full_whatever_the_type_says() {
        assert!(is_full(json!({"type": "object", "properties": {}})));
        assert!(is_full(json!({"type": "string", "properties": "x"})));
    }

    #[test]
    fn an_object_type_with_only_keywords_is_full() {
        assert!(is_full(json!({"type": "object"})));
        for keyword in OBJECT_SCHEMA_KEYWORDS {
            assert!(
                is_full(json!({"type": "object", *keyword: "x"})),
                "{keyword}"
            );
        }
    }

    #[test]
    fn a_property_map_is_not_full() {
        assert!(!is_full(json!({})));
        assert!(!is_full(json!({"query": "string"})));
        assert!(!is_full(json!({"type": "string"})));
        assert!(!is_full(json!({"type": "string", "title": "string"})));
        assert!(!is_full(json!({"type": "object", "query": "string"})));
        assert!(!is_full(
            json!({"type": {"type": "string"}, "title": "string"})
        ));
        assert!(!is_full(
            json!({"type": "object", "title": "string", "query": "string"})
        ));
    }

    #[test]
    fn a_map_without_type_is_never_full() {
        assert!(!is_full(json!({"required": ["a"], "title": "T"})));
        // Without `type`, `properties` is just a parameter that happens to carry that name.
        assert!(!is_full(json!({"properties": {"type": "string"}})));
    }
}
