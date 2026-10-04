use super::*;
use serde_json::json;

/// A well-formed chunk whose single choice ends with `finish_reason`.
fn chunk(finish_reason: serde_json::Value) -> serde_json::Value {
    json!({
        "id": "1", "object": "chat.completion.chunk", "created": 0, "model": "m",
        "choices": [{ "index": 0, "delta": { "content": "hi" }, "finish_reason": finish_reason }],
    })
}

/// Decodes a payload that must be reported as an in-band error.
fn reported(payload: serde_json::Value) -> Error {
    decode_chunk(&payload.to_string()).expect_err("payload reports an error")
}

fn api_parts(error: Error) -> (Option<u16>, String) {
    match error {
        Error::Api { status, message } => (status, message),
        other => panic!("expected Error::Api, got {other:?}"),
    }
}

#[test]
fn a_well_formed_chunk_decodes_unchanged() {
    let decoded = decode_chunk(&chunk(json!(null)).to_string()).expect("chunk decodes");

    assert_eq!(decoded.choices[0].delta.content.as_deref(), Some("hi"));
}

#[test]
fn unparseable_payloads_keep_the_parse_failure_message() {
    for payload in ["not json", r#"{"unrelated": 1}"#, r#"{"error": "boom"}"#] {
        let error = decode_chunk(payload).expect_err("payload is not a chunk");

        assert!(
            matches!(&error, Error::Stream(message)
                if message.starts_with("Failed to parse SSE event data: ")),
            "{payload}: {error:?}"
        );
    }
}

#[test]
fn only_http_error_statuses_are_kept() {
    // 70_000 and 65_936 overflow u16; the latter truncates to 400 and must not pass for it.
    let cases = [
        (json!(399), None),
        (json!(400), Some(400)),
        (json!(500), Some(500)),
        (json!(599), Some(599)),
        (json!(600), None),
        (json!(70_000), None),
        (json!(65_936), None),
        (json!(-500), None),
        (json!(500.5), None),
        (json!("500"), None),
        (json!("context_length_exceeded"), None),
        (json!(null), None),
    ];
    for (code, expected) in cases {
        let (status, _) = api_parts(reported(
            json!({ "error": { "code": code, "message": "m" } }),
        ));

        assert_eq!(status, expected, "code {code}");
    }
}

#[test]
fn a_missing_code_has_no_status() {
    let (status, _) = api_parts(reported(json!({ "error": { "message": "m" } })));

    assert_eq!(status, None);
}

#[test]
fn the_message_names_the_error_type_when_there_is_one() {
    let cases = [
        (
            json!({ "type": "server_error", "message": "boom" }),
            "server_error: boom",
        ),
        (json!({ "message": "boom" }), "boom"),
        (json!({ "type": "server_error" }), "server_error"),
        (json!({ "type": "", "message": "boom" }), "boom"),
        (
            json!({ "type": "server_error", "message": "" }),
            "server_error",
        ),
        (json!({}), "the server reported an error without details"),
        (
            json!({ "type": "", "message": "" }),
            "the server reported an error without details",
        ),
    ];
    for (error, expected) in cases {
        let (_, message) = api_parts(reported(json!({ "error": error })));

        assert_eq!(message, expected, "error {error}");
    }
}

#[test]
fn a_null_error_is_not_an_error() {
    let mut payload = chunk(json!("error"));
    payload["error"] = json!(null);

    let decoded = decode_chunk(&payload.to_string()).expect("a null error reports nothing");

    assert_eq!(decoded.choices[0].finish_reason.as_deref(), Some("error"));
}

#[test]
fn a_terminating_error_chunk_reports_its_error_object() {
    let mut payload = chunk(json!("error"));
    payload["error"] = json!({ "code": 429, "message": "Rate limit exceeded" });

    let (status, message) = api_parts(reported(payload));

    assert_eq!(
        (status, message.as_str()),
        (Some(429), "Rate limit exceeded")
    );
}

#[test]
fn any_choice_may_be_the_terminating_one() {
    let mut payload = chunk(json!(null));
    payload["choices"]
        .as_array_mut()
        .expect("choices is an array")
        .push(json!({ "index": 1, "delta": {}, "finish_reason": "error" }));
    payload["error"] = json!({ "code": 502, "message": "upstream" });

    let (status, _) = api_parts(reported(payload));

    assert_eq!(status, Some(502));
}

#[test]
fn an_error_object_beside_live_content_does_not_discard_it() {
    // Only a chunk that terminates with `finish_reason: "error"` is the documented shape.
    // One that carries an `error` key beside content the server did not call failed is
    // delivered, because dropping its text would discard a response nobody said was lost.
    for finish_reason in [json!(null), json!("stop"), json!("length")] {
        let mut payload = chunk(finish_reason.clone());
        payload["error"] = json!({ "code": 500, "message": "boom" });

        let decoded = decode_chunk(&payload.to_string()).expect("content is delivered");

        assert_eq!(
            decoded.choices[0].delta.content.as_deref(),
            Some("hi"),
            "{finish_reason}"
        );
    }
}

#[test]
fn a_terminating_reason_without_an_error_object_decodes_as_a_chunk() {
    let decoded = decode_chunk(&chunk(json!("error")).to_string()).expect("chunk decodes");

    assert_eq!(decoded.choices[0].finish_reason.as_deref(), Some("error"));
}
