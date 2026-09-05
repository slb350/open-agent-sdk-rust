/// A stream of content, optional reasoning, and one final finish event.
pub type EventStream = Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>;

/// Starts a single-turn streaming query using the configured protocol.
///
/// Does not execute tools or lifecycle hooks. For conversation history and
/// automatic tool execution, use [`Client`].
pub async fn query(prompt: &str, options: &AgentOptions) -> Result<EventStream> {
    let client = model_http_client_builder(options)
        .build()
        .map_err(Error::Http)?;

    let mut messages = Vec::new();

    if !options.system_prompt().is_empty() {
        messages.push(OpenAIMessage {
            role: "system".to_string(),
            content: Some(OpenAIContent::Text(options.system_prompt().to_string())),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    messages.push(OpenAIMessage {
        role: "user".to_string(),
        content: Some(OpenAIContent::Text(prompt.to_string())),
        tool_calls: None,
        tool_call_id: None,
    });

    let request = request::build_request(options, messages);

    stream_request(&client, options, &request).await
}
