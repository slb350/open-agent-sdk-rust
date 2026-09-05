//! Assistant tool-call messages retain their content field in the actual HTTP request.

mod common;

use open_agent::{
    AgentOptions, ApiProtocol, Client, ContentBlock, Message, TextBlock, ToolUseBlock,
};
use serde_json::{Value, json};

#[tokio::test]
async fn assistant_tool_calls_keep_empty_or_present_text_on_the_wire() {
    let server = common::sse_server(common::DONE).await;
    for text in [None, Some("Calling the tool")] {
        let mut blocks = Vec::new();
        if let Some(text) = text {
            blocks.push(ContentBlock::Text(TextBlock::new(text)));
        }
        blocks.push(ContentBlock::ToolUse(ToolUseBlock::new(
            "call-1",
            "calculate",
            json!({"value": 42}),
        )));
        let mut client = Client::new(common::options_for(&server)).unwrap();
        client
            .send_message(Message::assistant(blocks))
            .await
            .unwrap();
        let requests = server.received_requests().await.unwrap();
        let body: Value = requests.last().unwrap().body_json().unwrap();
        assert_eq!(
            body["messages"],
            json!([{
                "role": "assistant", "content": text.unwrap_or_default(),
                "tool_calls": [{"id": "call-1", "type": "function", "function": {
                    "name": "calculate", "arguments": "{\"value\":42}"
                }}]
            }])
        );
    }
}

#[tokio::test]
async fn manual_tool_result_continuation_preserves_the_matching_call_id() {
    let anthropic_call = common::anthropic_frame(
        "content_block_start",
        json!({
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "tool_use", "id": "call-1", "name": "calculate", "input": {}}
        }),
    ) + &common::anthropic_frame(
        "content_block_delta",
        json!({
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "{\"value\":42}"}
        }),
    );
    for (protocol, response, expected_result) in [
        (
            ApiProtocol::OpenAiChat,
            common::tool_chunk("call-1", "calculate", r#"{"value":42}"#) + common::DONE,
            json!({"role": "tool", "tool_call_id": "call-1", "content": "{\"answer\":84}"}),
        ),
        (
            ApiProtocol::Anthropic,
            anthropic_call,
            json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "{\"answer\":84}"}]}),
        ),
    ] {
        let server = match protocol {
            ApiProtocol::OpenAiChat => common::sse_server(response).await,
            ApiProtocol::Anthropic => common::anthropic_sse_server(response).await,
            _ => unreachable!("only the two configured protocols are tested"),
        };
        let options = AgentOptions::builder()
            .model("m")
            .base_url(format!("{}/v1", server.uri()))
            .protocol(protocol)
            .build()
            .unwrap();
        let mut client = Client::new(options).unwrap();
        client.send("calculate").await.unwrap();
        let Some(ContentBlock::ToolUse(call)) = client.receive().await.unwrap() else {
            panic!("expected tool call from {protocol:?}");
        };
        client
            .add_tool_result(call.id(), json!({"answer": 84}))
            .unwrap();
        client.send("").await.unwrap();
        let requests = server.received_requests().await.unwrap();
        let body: Value = requests[1].body_json().unwrap();
        assert_eq!(body["messages"][2], expected_result, "{protocol:?}");
    }
}
