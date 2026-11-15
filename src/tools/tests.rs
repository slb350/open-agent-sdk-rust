
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

    #[test]
    fn full_schema_without_properties_is_not_read_as_a_property_map() {
        // A tool that takes no arguments is commonly declared as a bare object schema. It must
        // not be mistaken for a property map with one required parameter named `type`.
        let result = convert_schema_to_openai(json!({"type": "object"}));
        assert_eq!(result["type"], "object");
        assert_eq!(result["properties"], json!({}));
        assert!(result.get("required").is_none_or(|r| r == &json!([])));

        let strict = json!({"type": "object", "additionalProperties": false});
        let result = convert_schema_to_openai(strict);
        assert_eq!(result["type"], "object");
        assert_eq!(result["properties"], json!({}));
        assert_eq!(result["additionalProperties"], json!(false));
        assert!(result.get("required").is_none_or(|r| r == &json!([])));
    }

    #[test]
    fn full_schema_keywords_survive_without_properties() {
        let schema = json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Lookup",
            "description": "Takes either field",
            "type": "object",
            "required": ["a"],
            "anyOf": [{"required": ["a"]}, {"required": ["b"]}]
        });
        let result = convert_schema_to_openai(schema.clone());
        let mut expected = schema;
        expected["properties"] = json!({});
        assert_eq!(result, expected);
    }

    #[test]
    fn a_property_named_type_in_a_property_map_stays_a_property() {
        let result = convert_schema_to_openai(json!({"type": "string", "query": "string"}));
        assert_eq!(result["properties"]["type"]["type"], "string");
        assert_eq!(result["properties"]["query"]["type"], "string");
        assert_eq!(result["required"], json!(["query", "type"]));

        // Mixed with a non-keyword sibling, `"type": "object"` is still a parameter.
        let result = convert_schema_to_openai(json!({"type": "object", "query": "string"}));
        assert_eq!(result["properties"]["type"]["type"], "object");
        assert_eq!(result["properties"]["query"]["type"], "string");
    }

    #[test]
    fn builder_schema_of_a_bare_object_exposes_no_parameters() {
        let no_args = tool("ping", "No arguments")
            .schema(json!({"type": "object"}))
            .build(|_| async { Ok(json!({})) });
        let parameters = no_args.to_openai_format()["function"]["parameters"].clone();
        assert_eq!(parameters, json!({"type": "object", "properties": {}}));
    }
