//! Translation tests for [`AnthropicRequest::from_openai`].
//!
//! Every case here asserts on the serialized JSON rather than on the struct, because the
//! shape on the wire is the contract: a field that serializes under the wrong name, or one
//! that should have been omitted, is invisible at the struct level.

use super::*;
use crate::types::{OpenAIFunction, OpenAIImageUrl, OpenAIToolCall};

/// Builds a request carrying `messages` and nothing else optional.
fn request_with(messages: Vec<OpenAIMessage>) -> OpenAIRequest {
    OpenAIRequest {
        model: "m".to_string(),
        messages,
        stream: true,
        max_tokens: None,
        temperature: None,
        tools: None,
    }
}

/// A text message in the given role.
fn message(role: &str, text: &str) -> OpenAIMessage {
    OpenAIMessage {
        role: role.to_string(),
        content: Some(OpenAIContent::Text(text.to_string())),
        tool_calls: None,
        tool_call_id: None,
    }
}

/// Serializes a translated request so assertions read against the wire shape.
fn wire(request: &OpenAIRequest) -> Value {
    serde_json::to_value(AnthropicRequest::from_openai(request)).expect("serializes")
}

#[test]
fn the_system_prompt_leaves_the_messages_array() {
    let out = wire(&request_with(vec![
        message("system", "be terse"),
        message("user", "hi"),
    ]));

    assert_eq!(out["system"], json!("be terse"));
    assert_eq!(out["messages"].as_array().expect("array").len(), 1);
    assert_eq!(out["messages"][0]["role"], json!("user"));
}

#[test]
fn multiple_system_messages_are_joined_rather_than_dropped() {
    let out = wire(&request_with(vec![
        message("system", "first"),
        message("system", "second"),
        message("user", "hi"),
    ]));

    assert_eq!(out["system"], json!("first\n\nsecond"));
}

#[test]
fn a_request_with_no_system_message_omits_the_field() {
    let out = wire(&request_with(vec![message("user", "hi")]));

    assert!(
        out.get("system").is_none(),
        "system must be omitted, got {out}"
    );
}

#[test]
fn a_text_only_turn_serializes_as_a_bare_string() {
    let out = wire(&request_with(vec![message("user", "hi")]));

    assert_eq!(out["messages"][0]["content"], json!("hi"));
}

#[test]
fn a_turn_whose_only_part_is_text_also_serializes_as_a_bare_string() {
    // Vision callers build content as a part array even when the turn ends up carrying only
    // text. That is the same turn as the one above and goes on the wire the same way, rather
    // than as a one-element block array.
    let user = OpenAIMessage {
        role: "user".to_string(),
        content: Some(OpenAIContent::Parts(vec![OpenAIContentPart::Text {
            text: "hi".to_string(),
        }])),
        tool_calls: None,
        tool_call_id: None,
    };

    let out = wire(&request_with(vec![user]));

    assert_eq!(out["messages"][0]["content"], json!("hi"));
}

#[test]
fn a_turn_with_text_and_an_image_keeps_its_block_array() {
    // The bare-string shortcut applies only when text is the whole turn; a second part means
    // the array is the only representation that can carry it.
    let user = OpenAIMessage {
        role: "user".to_string(),
        content: Some(OpenAIContent::Parts(vec![
            OpenAIContentPart::Text {
                text: "what is this".to_string(),
            },
            OpenAIContentPart::ImageUrl {
                image_url: OpenAIImageUrl {
                    url: "https://example.com/a.png".to_string(),
                    detail: None,
                },
            },
        ])),
        tool_calls: None,
        tool_call_id: None,
    };

    let out = wire(&request_with(vec![user]));

    let blocks = out["messages"][0]["content"]
        .as_array()
        .expect("a multi-part turn keeps its array");
    assert_eq!(blocks.len(), 2, "{out}");
    assert_eq!(blocks[0]["type"], json!("text"));
    assert_eq!(blocks[1]["type"], json!("image"));
}

#[test]
fn system_text_survives_a_parts_message_and_images_do_not() {
    let system = OpenAIMessage {
        role: "system".to_string(),
        content: Some(OpenAIContent::Parts(vec![
            OpenAIContentPart::text("be "),
            OpenAIContentPart::ImageUrl {
                image_url: OpenAIImageUrl {
                    url: "https://example.com/a.png".to_string(),
                    detail: None,
                },
            },
            OpenAIContentPart::text("terse"),
        ])),
        tool_calls: None,
        tool_call_id: None,
    };

    let out = wire(&request_with(vec![system, message("user", "hi")]));

    assert_eq!(out["system"], json!("be terse"));
}

#[test]
fn an_assistant_tool_call_becomes_a_tool_use_block_with_parsed_input() {
    let assistant = OpenAIMessage {
        role: "assistant".to_string(),
        content: None,
        tool_calls: Some(vec![OpenAIToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: OpenAIFunction {
                name: "search".to_string(),
                arguments: r#"{"q":"rust"}"#.to_string(),
            },
        }]),
        tool_call_id: None,
    };

    let out = wire(&request_with(vec![message("user", "hi"), assistant]));
    let block = &out["messages"][1]["content"][0];

    assert_eq!(block["type"], json!("tool_use"));
    assert_eq!(block["id"], json!("call_1"));
    assert_eq!(block["name"], json!("search"));
    // Parsed into an object, not passed through as the JSON string OpenAI sends.
    assert_eq!(block["input"], json!({ "q": "rust" }));
}

#[test]
fn text_alongside_a_tool_call_precedes_it_in_the_block_array() {
    let assistant = OpenAIMessage {
        role: "assistant".to_string(),
        content: Some(OpenAIContent::Text("looking".to_string())),
        tool_calls: Some(vec![OpenAIToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: OpenAIFunction {
                name: "search".to_string(),
                arguments: "{}".to_string(),
            },
        }]),
        tool_call_id: None,
    };

    let out = wire(&request_with(vec![assistant]));
    let blocks = out["messages"][0]["content"]
        .as_array()
        .expect("block array");

    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0], json!({ "type": "text", "text": "looking" }));
    assert_eq!(blocks[1]["type"], json!("tool_use"));
}

#[test]
fn malformed_tool_arguments_become_an_empty_object() {
    let assistant = OpenAIMessage {
        role: "assistant".to_string(),
        content: None,
        tool_calls: Some(vec![OpenAIToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: OpenAIFunction {
                name: "search".to_string(),
                arguments: "{not json".to_string(),
            },
        }]),
        tool_call_id: None,
    };

    let out = wire(&request_with(vec![assistant]));

    assert_eq!(out["messages"][0]["content"][0]["input"], json!({}));
}

#[test]
fn a_tool_result_becomes_a_user_turn() {
    let result = OpenAIMessage {
        role: "tool".to_string(),
        content: Some(OpenAIContent::Text("42".to_string())),
        tool_calls: None,
        tool_call_id: Some("call_1".to_string()),
    };

    let out = wire(&request_with(vec![result]));
    let block = &out["messages"][0]["content"][0];

    assert_eq!(out["messages"][0]["role"], json!("user"));
    assert_eq!(block["type"], json!("tool_result"));
    assert_eq!(block["tool_use_id"], json!("call_1"));
    assert_eq!(block["content"], json!("42"));
}

#[test]
fn consecutive_tool_results_merge_into_one_user_turn() {
    let result = |id: &str, body: &str| OpenAIMessage {
        role: "tool".to_string(),
        content: Some(OpenAIContent::Text(body.to_string())),
        tool_calls: None,
        tool_call_id: Some(id.to_string()),
    };

    let out = wire(&request_with(vec![
        result("call_1", "a"),
        result("call_2", "b"),
    ]));

    let messages = out["messages"].as_array().expect("array");
    assert_eq!(messages.len(), 1, "parallel results share one turn: {out}");
    let blocks = messages[0]["content"].as_array().expect("block array");
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0]["tool_use_id"], json!("call_1"));
    assert_eq!(blocks[1]["tool_use_id"], json!("call_2"));
}

#[test]
fn a_tool_result_does_not_merge_into_a_real_user_turn() {
    let result = OpenAIMessage {
        role: "tool".to_string(),
        content: Some(OpenAIContent::Text("42".to_string())),
        tool_calls: None,
        tool_call_id: Some("call_1".to_string()),
    };

    let out = wire(&request_with(vec![message("user", "hi"), result]));

    let messages = out["messages"].as_array().expect("array");
    assert_eq!(messages.len(), 2, "the user turn is not a tool turn: {out}");
    assert_eq!(messages[0]["content"], json!("hi"));
    assert_eq!(messages[1]["content"][0]["type"], json!("tool_result"));
}

#[test]
fn a_tool_result_does_not_merge_into_a_user_turn_that_is_already_a_block_array() {
    // The turn a `content_blocks` conversion produces for an image is an array too, so
    // "content is an array" is not enough to identify a turn this module built. Absorbing a
    // tool result into it would attach an answer to the wrong turn.
    let with_image = OpenAIMessage {
        role: "user".to_string(),
        content: Some(OpenAIContent::Parts(vec![
            OpenAIContentPart::text("look"),
            OpenAIContentPart::ImageUrl {
                image_url: OpenAIImageUrl {
                    url: "https://example.com/a.png".to_string(),
                    detail: None,
                },
            },
        ])),
        tool_calls: None,
        tool_call_id: None,
    };
    let result = OpenAIMessage {
        role: "tool".to_string(),
        content: Some(OpenAIContent::Text("42".to_string())),
        tool_calls: None,
        tool_call_id: Some("call_1".to_string()),
    };

    let out = wire(&request_with(vec![with_image, result]));

    let messages = out["messages"].as_array().expect("array");
    assert_eq!(messages.len(), 2, "the image turn keeps to itself: {out}");
    assert_eq!(
        messages[0]["content"].as_array().expect("blocks").len(),
        2,
        "the image turn is unchanged: {out}"
    );
    assert_eq!(messages[1]["content"][0]["type"], json!("tool_result"));
}

#[test]
fn a_data_uri_image_becomes_a_base64_source() {
    let user = OpenAIMessage {
        role: "user".to_string(),
        content: Some(OpenAIContent::Parts(vec![OpenAIContentPart::ImageUrl {
            image_url: OpenAIImageUrl {
                url: "data:image/png;base64,AAAB".to_string(),
                detail: None,
            },
        }])),
        tool_calls: None,
        tool_call_id: None,
    };

    let out = wire(&request_with(vec![user]));

    assert_eq!(
        out["messages"][0]["content"][0],
        json!({
            "type": "image",
            "source": { "type": "base64", "media_type": "image/png", "data": "AAAB" },
        })
    );
}

#[test]
fn data_uri_parameters_are_not_part_of_the_media_type() {
    // `ImageBlock::from_url` reads the media type up to the first `;`, so it accepts this URI
    // and reports `image/png`. Splitting on `;base64` instead would send Anthropic
    // `image/png;charset=utf-8`, a media type it rejects, for an image the SDK had accepted.
    let user = OpenAIMessage {
        role: "user".to_string(),
        content: Some(OpenAIContent::Parts(vec![OpenAIContentPart::ImageUrl {
            image_url: OpenAIImageUrl {
                url: "data:image/png;charset=utf-8;base64,AAAB".to_string(),
                detail: None,
            },
        }])),
        tool_calls: None,
        tool_call_id: None,
    };

    let out = wire(&request_with(vec![user]));

    assert_eq!(
        out["messages"][0]["content"][0],
        json!({
            "type": "image",
            "source": { "type": "base64", "media_type": "image/png", "data": "AAAB" },
        })
    );
}

#[test]
fn a_remote_image_url_becomes_a_url_source() {
    let user = OpenAIMessage {
        role: "user".to_string(),
        content: Some(OpenAIContent::Parts(vec![OpenAIContentPart::ImageUrl {
            image_url: OpenAIImageUrl {
                url: "https://example.com/a.png".to_string(),
                detail: None,
            },
        }])),
        tool_calls: None,
        tool_call_id: None,
    };

    let out = wire(&request_with(vec![user]));

    assert_eq!(
        out["messages"][0]["content"][0]["source"],
        json!({ "type": "url", "url": "https://example.com/a.png" })
    );
}

#[test]
fn a_data_uri_that_is_not_base64_is_not_read_as_base64() {
    // `data:image/svg+xml,<svg/>` carries the payload verbatim. Declaring it base64 would
    // hand the API bytes that are not base64 at all.
    let user = OpenAIMessage {
        role: "user".to_string(),
        content: Some(OpenAIContent::Parts(vec![OpenAIContentPart::ImageUrl {
            image_url: OpenAIImageUrl {
                url: "data:image/svg+xml,<svg/>".to_string(),
                detail: None,
            },
        }])),
        tool_calls: None,
        tool_call_id: None,
    };

    let out = wire(&request_with(vec![user]));

    assert_eq!(
        out["messages"][0]["content"][0]["source"]["type"],
        json!("url")
    );
}

#[test]
fn a_tool_definition_is_flattened_out_of_its_function_envelope() {
    let mut request = request_with(vec![message("user", "hi")]);
    request.tools = Some(vec![json!({
        "type": "function",
        "function": {
            "name": "search",
            "description": "find things",
            "parameters": { "type": "object", "properties": { "q": { "type": "string" } } },
        },
    })]);

    let out = wire(&request);

    assert_eq!(
        out["tools"][0],
        json!({
            "name": "search",
            "description": "find things",
            "input_schema": { "type": "object", "properties": { "q": { "type": "string" } } },
        })
    );
}

#[test]
fn a_tool_definition_without_parameters_gets_an_empty_object_schema() {
    let mut request = request_with(vec![message("user", "hi")]);
    request.tools = Some(vec![json!({
        "type": "function",
        "function": { "name": "ping" },
    })]);

    let out = wire(&request);

    assert_eq!(
        out["tools"][0]["input_schema"],
        json!({ "type": "object", "properties": {} }),
        "input_schema is required by the API, so it is never omitted"
    );
}

#[test]
fn a_tool_already_in_anthropic_shape_passes_through_untouched() {
    let mut request = request_with(vec![message("user", "hi")]);
    let native = json!({
        "name": "search",
        "input_schema": { "type": "object", "properties": {} },
    });
    request.tools = Some(vec![native.clone()]);

    let out = wire(&request);

    assert_eq!(out["tools"][0], native);
}

#[test]
fn unset_ceilings_are_omitted_from_the_wire() {
    let out = wire(&request_with(vec![message("user", "hi")]));

    assert!(out.get("max_tokens").is_none(), "got {out}");
    assert!(out.get("temperature").is_none(), "got {out}");
    assert!(out.get("tools").is_none(), "got {out}");
}

#[test]
fn set_ceilings_reach_the_wire() {
    let mut request = request_with(vec![message("user", "hi")]);
    request.max_tokens = Some(1024);
    request.temperature = Some(0.2);

    let out = wire(&request);

    assert_eq!(out["max_tokens"], json!(1024));
    assert_eq!(out["model"], json!("m"));
    assert_eq!(out["stream"], json!(true));
    assert!(
        (out["temperature"].as_f64().expect("number") - 0.2).abs() < 1e-6,
        "got {out}"
    );
}

#[test]
fn streaming_is_carried_through_rather_than_assumed() {
    let mut request = request_with(vec![message("user", "hi")]);
    request.stream = false;

    assert_eq!(wire(&request)["stream"], json!(false));
}
