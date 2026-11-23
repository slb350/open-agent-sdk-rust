//! Optional provider smoke; deterministic behavior is covered by mock-server tests.

use open_agent::{AgentOptions, Client, ContentBlock};
use std::time::Duration;

#[tokio::test]
#[ignore = "requires Ollama at localhost:11434 with qwen3:8b installed"]
async fn ollama_auto_mode_returns_a_text_response() {
    let options = AgentOptions::builder()
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .auto_execute_tools(true)
        .max_tokens(512)
        .timeout(30)
        .build()
        .unwrap();
    let mut client = Client::new(options).unwrap();
    let response = tokio::time::timeout(Duration::from_secs(30), async {
        client.send("Say hello briefly.").await.unwrap();
        let mut response = String::new();
        while let Some(block) = client.receive().await.unwrap() {
            if let ContentBlock::Text(text) = block {
                response.push_str(&text.text);
            }
        }
        response
    })
    .await
    .expect("provider smoke completed within 30 seconds");
    assert!(!response.trim().is_empty());
}
