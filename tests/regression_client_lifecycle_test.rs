//! Request transitions must discard previously buffered output in either client mode.
mod common;

use common::{DONE, message_text, text_chunk, text_of};
use open_agent::{AgentOptions, Client, ContentBlock, Hooks, Message, MessageRole, tool};
use serde_json::json;
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn client(auto: bool) -> (MockServer, Client) {
    let server = MockServer::start().await;
    let round = AtomicUsize::new(0);
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_: &wiremock::Request| {
            let n = round.fetch_add(1, Ordering::SeqCst) + 1;
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(
                    text_chunk(&format!("turn {n} "), None)
                        + &text_chunk("done", Some("stop"))
                        + DONE,
                )
        })
        .mount(&server)
        .await;
    let options = AgentOptions::builder()
        .model("m")
        .base_url(format!("{}/v1", server.uri()))
        .auto_execute_tools(auto)
        .build()
        .unwrap();
    (server, Client::new(options).unwrap())
}

async fn drain(client: &mut Client) -> String {
    let mut blocks = Vec::new();
    while let Some(block) = client.receive().await.unwrap() {
        blocks.push(block);
    }
    text_of(&blocks)
}

#[tokio::test]
async fn new_auto_requests_replace_completed_or_abandoned_output() {
    for completed in [false, true] {
        for structured in [false, true] {
            let (_server, mut client) = client(true).await;
            client.send("first").await.unwrap();
            assert_eq!(
                text_of(&[client.receive().await.unwrap().unwrap()]),
                "turn 1 "
            );
            if completed {
                assert_eq!(drain(&mut client).await, "done");
                assert!(client.receive().await.unwrap().is_none());
            }
            if structured {
                client.send_message(Message::user("second")).await.unwrap();
            } else {
                client.send("second").await.unwrap();
            }
            assert_eq!(
                drain(&mut client).await,
                "turn 2 done",
                "completed={completed}, structured={structured}"
            );
        }
    }
}

#[tokio::test]
async fn interrupt_discards_auto_output_and_next_request_resumes() {
    let (_server, mut client) = client(true).await;
    client.send("first").await.unwrap();
    assert!(client.receive().await.unwrap().is_some());
    client.interrupt_handle().store(true, Ordering::SeqCst);
    assert!(client.receive().await.unwrap().is_none());
    client.send("second").await.unwrap();
    assert_eq!(drain(&mut client).await, "turn 2 done");
}

#[tokio::test]
async fn clear_history_abandons_pending_output_in_both_modes() {
    for auto in [false, true] {
        let (_server, mut client) = client(auto).await;
        client.send("first").await.unwrap();
        assert!(matches!(
            client.receive().await.unwrap(),
            Some(ContentBlock::Text(_))
        ));
        client.clear_history();
        assert!(client.receive().await.unwrap().is_none(), "auto={auto}");
        assert!(client.history().is_empty());
        client.send("second").await.unwrap();
        assert_eq!(drain(&mut client).await, "turn 2 done");
        assert_eq!(client.history().len(), 2);
    }
}

#[tokio::test]
async fn new_manual_send_discards_partial_assistant_history() {
    let (_server, mut client) = client(false).await;
    client.send("first").await.unwrap();
    assert!(client.receive().await.unwrap().is_some());
    client.send("second").await.unwrap();
    assert_eq!(drain(&mut client).await, "turn 2 done");
    let assistants: Vec<_> = client
        .history()
        .iter()
        .filter(|message| message.role == MessageRole::Assistant)
        .map(message_text)
        .collect();
    assert_eq!(assistants, ["turn 2 done"]);
}

#[tokio::test]
async fn cancellation_from_a_tool_or_hook_stops_remaining_work_and_continuation() {
    for (stage, index, expected_calls, expected_hooks) in [
        ("pre", 0, 0, [1, 0]),
        ("tool", 0, 1, [1, 0]),
        ("post", 0, 1, [1, 1]),
        ("tool", 1, 2, [2, 1]),
    ] {
        let server = common::sse_server(
            common::sse_frame(json!({
                "index": 0,
                "delta": {"tool_calls": (0..2).map(|index| json!({
                    "index": index, "id": format!("call-{index}"), "type": "function",
                    "function": {"name": "work", "arguments": json!({"index": index}).to_string()}
                })).collect::<Vec<_>>()},
                "finish_reason": "tool_calls"
            })) + DONE,
        )
        .await;
        let handle = Arc::new(OnceLock::<Arc<AtomicBool>>::new());
        let shared_handle = handle.clone();
        let cancel = move |at: &str, input: &serde_json::Value| {
            if at == stage && input["index"] == index {
                shared_handle.get().unwrap().store(true, Ordering::SeqCst);
            }
        };
        let pre_cancel = cancel.clone();
        let post_cancel = cancel.clone();
        let hooks_called = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
        let pre_called = hooks_called.clone();
        let post_called = hooks_called.clone();
        let calls = Arc::new(AtomicUsize::new(0));
        let tool_calls = calls.clone();
        let options = AgentOptions::builder()
            .model("m")
            .base_url(format!("{}/v1", server.uri()))
            .auto_execute_tools(true)
            .hooks(
                Hooks::new()
                    .add_pre_tool_use(move |event| {
                        pre_called[0].fetch_add(1, Ordering::SeqCst);
                        pre_cancel("pre", &event.tool_input);
                        async { None }
                    })
                    .add_post_tool_use(move |event| {
                        post_called[1].fetch_add(1, Ordering::SeqCst);
                        post_cancel("post", &event.tool_input);
                        async { None }
                    }),
            )
            .tool(tool("work", "Record execution").build(move |input| {
                tool_calls.fetch_add(1, Ordering::SeqCst);
                cancel("tool", &input);
                async { Ok(json!({"done": true})) }
            }))
            .build()
            .unwrap();
        let mut client = Client::new(options).unwrap();
        handle.set(client.interrupt_handle()).unwrap();
        client.send("go").await.unwrap();
        assert!(client.receive().await.unwrap().is_none(), "{stage}/{index}");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            expected_calls,
            "{stage}/{index}"
        );
        assert_eq!(
            hooks_called
                .each_ref()
                .map(|count| count.load(Ordering::SeqCst)),
            expected_hooks
        );
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            1,
            "{stage}/{index}"
        );
        assert!(client.interrupt_handle().load(Ordering::SeqCst));
        client.send("next request").await.unwrap();
        let requests = server.received_requests().await.unwrap();
        let body: serde_json::Value = requests[1].body_json().unwrap();
        let result_ids: Vec<_> = body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|message| message["tool_call_id"].as_str())
            .collect();
        assert_eq!(
            result_ids,
            ["call-0", "call-1"],
            "cancelled calls must not leave unmatched history"
        );
        let results = body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|message| message["role"] == "tool");
        for (index, message) in results.enumerate() {
            let result: serde_json::Value =
                serde_json::from_str(message["content"].as_str().unwrap()).unwrap();
            assert_eq!(
                result,
                if index < expected_calls {
                    json!({"done": true})
                } else {
                    json!({"error": "Tool execution cancelled"})
                }
            );
        }
    }
}
