//! Exercise hook decisions through a complete tool round and its continuation request.

mod common;

use common::{DONE, sse_frame, sse_server, text_chunk, tool_chunk};
use open_agent::{AgentOptions, Client, ContentBlock, Error, HookDecision, Hooks, tool};
use serde_json::{Value, json};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use wiremock::{
    Mock, MockServer, Request, ResponseTemplate,
    matchers::{method, path},
};

#[tokio::test]
async fn auto_hooks_block_modify_and_transform_results_before_the_continuation() {
    let server = MockServer::start().await;
    let calls = ["blocked", "calculate", "failing"];
    let body = sse_frame(json!({
        "index": 0,
        "delta": { "tool_calls": calls.iter().enumerate().map(|(index, name)| json!({
            "index": index, "id": name, "type": "function",
            "function": {"name": name, "arguments": "{\"value\":500}"}
        })).collect::<Vec<_>>() },
        "finish_reason": "tool_calls"
    })) + DONE;
    let requests = AtomicUsize::new(0);
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_: &Request| {
            let response = if requests.fetch_add(1, Ordering::SeqCst) == 0 {
                body.clone()
            } else {
                text_chunk("done", Some("stop")) + DONE
            };
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(response)
        })
        .expect(2)
        .mount(&server)
        .await;

    let executed = Arc::new(Mutex::new(Vec::new()));
    let captured_execution = Arc::clone(&executed);
    let observed = Arc::new(Mutex::new(Vec::new()));
    let captured_results = Arc::clone(&observed);
    let hooks = Hooks::new()
        .add_pre_tool_use(|event| async move {
            (event.tool_name == "blocked").then(|| HookDecision::block("denied"))
        })
        .add_pre_tool_use(|event| async move {
            Some(if event.tool_name == "calculate" {
                HookDecision::modify_input(json!({"value": 100}), "clamped")
            } else {
                HookDecision::continue_()
            })
        })
        .add_pre_tool_use(|_| async { panic!("first decision must stop the pre-hook chain") })
        .add_post_tool_use(move |event| {
            let captured_results = Arc::clone(&captured_results);
            async move {
                captured_results.lock().unwrap().push((
                    event.tool_name.clone(),
                    event.tool_input,
                    event.tool_result.clone(),
                ));
                let mut result = event.tool_result;
                result["handled"] = json!(true);
                Some(HookDecision::modify_input(result, "annotated"))
            }
        })
        .add_post_tool_use(|_| async { panic!("first decision must stop the post-hook chain") });
    let options = AgentOptions::builder()
        .model("m")
        .base_url(format!("{}/v1", server.uri()))
        .tool(
            tool("blocked", "Must be blocked").build(|_| async { panic!("blocked tool executed") }),
        )
        .tool(
            tool("calculate", "Double a value")
                .param("value", "number")
                .build(move |input| {
                    let captured_execution = Arc::clone(&captured_execution);
                    async move {
                        captured_execution.lock().unwrap().push(input.clone());
                        Ok(json!({"result": input["value"].as_i64().unwrap() * 2}))
                    }
                }),
        )
        .tool(
            tool("failing", "Report a failure")
                .build(|_| async { Err(Error::tool("expected failure")) }),
        )
        .hooks(hooks)
        .auto_execute_tools(true)
        .build()
        .unwrap();
    let mut client = Client::new(options).unwrap();
    client.send("run the tools").await.unwrap();
    let mut output = Vec::new();
    while let Some(block) = client.receive().await.unwrap() {
        output.push(block);
    }
    assert_eq!(common::text_of(&output), "done");
    assert!(
        output
            .iter()
            .all(|block| matches!(block, ContentBlock::Text(_)))
    );
    assert_eq!(*executed.lock().unwrap(), vec![json!({"value": 100})]);

    let expected_results = {
        let observed = observed.lock().unwrap();
        assert_eq!(
            observed
                .iter()
                .map(|(name, _, _)| name.as_str())
                .collect::<Vec<_>>(),
            calls
        );
        assert_eq!(observed[0].2["reason"], "denied");
        assert_eq!(observed[1].1, json!({"value": 100}));
        assert_eq!(observed[1].2, json!({"result": 200}));
        assert!(
            observed[2].2["error"]
                .as_str()
                .unwrap()
                .contains("expected failure")
        );
        observed
            .iter()
            .map(|(_, _, original)| {
                let mut expected = original.clone();
                expected["handled"] = json!(true);
                expected
            })
            .collect::<Vec<_>>()
    };

    let requests = server.received_requests().await.unwrap();
    let continuation: Value = requests[1].body_json().unwrap();
    assert_eq!(continuation["tools"].as_array().unwrap().len(), 3);
    assert_eq!(continuation["tools"][1]["function"]["name"], "calculate");
    let results: Vec<_> = continuation["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|message| message["role"] == "tool")
        .collect();
    assert_eq!(results.len(), 3);
    for ((message, name), expected) in results.iter().zip(calls).zip(expected_results) {
        assert_eq!(message["tool_call_id"], name);
        let result: Value = serde_json::from_str(message["content"].as_str().unwrap()).unwrap();
        assert_eq!(result, expected);
    }
}

#[tokio::test]
async fn manual_receive_leaves_tool_execution_to_the_caller_and_skips_tool_hooks() {
    let server = sse_server(tool_chunk("call-1", "double", r#"{"value":21}"#) + DONE).await;
    let hooks = Hooks::new()
        .add_pre_tool_use(|_| async { panic!("manual receive invoked a pre-tool hook") })
        .add_post_tool_use(|_| async { panic!("manual receive invoked a post-tool hook") });
    let options = AgentOptions::builder()
        .model("m")
        .base_url(format!("{}/v1", server.uri()))
        .hooks(hooks)
        .tool(tool("double", "Double a value").build(|input| async move {
            Ok(json!({"result": input["value"].as_i64().unwrap() * 2}))
        }))
        .build()
        .unwrap();
    let mut client = Client::new(options).unwrap();
    client.send("double 21").await.unwrap();
    let Some(ContentBlock::ToolUse(call)) = client.receive().await.unwrap() else {
        panic!("expected a manual tool call");
    };
    assert_eq!(call.id(), "call-1");
    assert!(client.receive().await.unwrap().is_none());
    let result = client
        .get_tool(call.name())
        .unwrap()
        .execute(call.input().clone())
        .await
        .unwrap();
    assert_eq!(result, json!({"result": 42}));
}
