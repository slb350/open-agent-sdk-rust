//! Regression coverage for structured conversation history in hook events.

use open_agent::{
    AgentOptions, Client, ContentBlock, HookDecision, Hooks, Message, TextBlock, ToolResultBlock,
    ToolUseBlock,
};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

#[tokio::test]
async fn user_prompt_hook_receives_complete_serialized_history() {
    let captured_history = Arc::new(Mutex::new(None::<Vec<Value>>));
    let hook_history = Arc::clone(&captured_history);
    let hooks = Hooks::new().add_user_prompt_submit(move |event| {
        let hook_history = Arc::clone(&hook_history);
        async move {
            *hook_history.lock().expect("history capture lock") = Some(event.history);
            Some(HookDecision::block("captured before request"))
        }
    });

    let options = AgentOptions::builder()
        .model("test-model")
        .base_url("http://127.0.0.1:9/v1")
        .hooks(hooks)
        .build()
        .expect("valid options");
    let mut client = Client::new(options).expect("valid client");

    let history = vec![
        Message::user("Use the calculator"),
        Message::assistant(vec![
            ContentBlock::Text(TextBlock::new("I'll calculate that.")),
            ContentBlock::ToolUse(ToolUseBlock::new(
                "call-1",
                "calculate",
                json!({"value": 42}),
            )),
        ]),
        Message::user_with_blocks(vec![ContentBlock::ToolResult(ToolResultBlock::new(
            "call-1",
            json!({"result": 84}),
        ))]),
    ];
    client.history_mut().extend(history.clone());

    let error = client
        .send("What happened?")
        .await
        .expect_err("hook should stop the request after capturing history");
    assert!(error.to_string().contains("captured before request"));

    let expected: Vec<Value> = history
        .iter()
        .map(|message| serde_json::to_value(message).expect("message serializes"))
        .collect();
    assert_eq!(
        captured_history
            .lock()
            .expect("history capture lock")
            .as_ref(),
        Some(&expected)
    );
}
