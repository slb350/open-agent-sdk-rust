
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
