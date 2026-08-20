/// A pinned, boxed stream of events from the model.
///
/// This type alias represents an asynchronous stream that yields [`StreamEvent`] items.
/// Each item is wrapped in a `Result` to handle potential errors during streaming.
///
/// The stream is:
/// - **Pinned** (`Pin<Box<...>>`): Required for safe async operations and self-referential types
/// - **Boxed**: Allows dynamic dispatch and hides the concrete stream implementation
/// - **Send**: Can be safely transferred between threads
///
/// # Events
///
/// - [`StreamEvent::Block`]: a completed [`ContentBlock`] — assistant text, or a fully
///   assembled tool call
/// - [`StreamEvent::Reasoning`]: chain-of-thought text, only when
///   [`AgentOptions::include_reasoning`] is enabled
/// - [`StreamEvent::Finish`]: exactly once, as the final item, carrying the
///   [`FinishReason`]
///
/// # Migrating from `ContentStream` (0.7.x and earlier)
///
/// The stream used to yield bare `ContentBlock`s. Wrap the old match in
/// [`StreamEvent::into_block`] to get the previous behaviour, then handle
/// [`StreamEvent::Finish`] where the distinction between a clean stop and a truncated
/// response matters.
///
/// # Error Handling
///
/// Errors in the stream indicate issues like:
/// - Network failures or timeouts
/// - Malformed SSE events
/// - JSON parsing errors
/// - API errors from the model provider
///
/// When an error occurs, the stream typically terminates. It's the caller's responsibility
/// to handle errors appropriately.
///
/// # Examples
///
/// ```rust,no_run
/// use open_agent::{query, AgentOptions, ContentBlock, FinishReason, StreamEvent};
/// use futures::StreamExt;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let options = AgentOptions::builder()
///     .model("gpt-4")
///     .api_key("sk-...")
///     .build()?;
///
/// let mut stream = query("Hello!", &options).await?;
///
/// while let Some(result) = stream.next().await {
///     match result? {
///         StreamEvent::Block(ContentBlock::Text(text)) => print!("{}", text.text),
///         StreamEvent::Finish(reason) => println!("\nstopped: {reason}"),
///         _ => {}
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub type EventStream = Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>;

/// Simple query function for single-turn interactions without conversation history.
///
/// This is a stateless convenience function for simple queries that don't require
/// multi-turn conversations. It creates a temporary HTTP client, sends a single
/// prompt, and returns a stream of events.
///
/// For multi-turn conversations or more control over the interaction, use [`Client`] instead.
///
/// # Parameters
///
/// - `prompt`: The user's message to send to the model
/// - `options`: Configuration including model, API key, tools, etc.
///
/// # Returns
///
/// Returns an [`EventStream`] that yields events as they arrive from the model. The stream
/// must be polled to completion to receive all content and the terminating
/// [`StreamEvent::Finish`].
///
/// # Behavior
///
/// 1. Creates a temporary HTTP client with configured timeout
/// 2. Builds message array (system prompt + user prompt)
/// 3. Converts tools to the wire format if provided
/// 4. Makes an HTTP POST request to the path the configured
///    [`ApiProtocol`](crate::ApiProtocol) selects
/// 5. Parses Server-Sent Events (SSE) response stream
/// 6. Aggregates chunks into complete content blocks
/// 7. Returns stream that yields events as they complete, ending with `Finish`
///
/// # Error Handling
///
/// This function can return errors for:
/// - HTTP client creation failures
/// - Network errors during the request
/// - API errors (authentication, invalid model, rate limits, etc.)
/// - SSE parsing errors
/// - JSON deserialization errors
///
/// # Performance Notes
///
/// - Creates a new HTTP client for each call (consider using `Client` for repeated queries)
/// - Timeout is configurable via `AgentOptions::timeout` (default: 120 seconds)
/// - Streaming begins immediately; no buffering of the full response
///
/// # Examples
///
/// ## Basic Usage
///
/// ```rust,no_run
/// use open_agent::{query, AgentOptions, ContentBlock, FinishReason, StreamEvent};
/// use futures::StreamExt;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let options = AgentOptions::builder()
///         .system_prompt("You are a helpful assistant")
///         .model("gpt-4")
///         .api_key("sk-...")
///         .build()?;
///
///     let mut stream = query("What's the capital of France?", &options).await?;
///
///     while let Some(event) = stream.next().await {
///         match event? {
///             StreamEvent::Block(ContentBlock::Text(text)) => print!("{}", text.text),
///             StreamEvent::Finish(FinishReason::Length) => {
///                 eprintln!("response truncated at the token cap");
///             }
///             _ => {}
///         }
///     }
///
///     Ok(())
/// }
/// ```
///
/// ## With Tools
///
/// ```rust,no_run
/// use open_agent::{query, AgentOptions, Tool, ContentBlock, StreamEvent};
/// use futures::StreamExt;
/// use serde_json::json;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let calculator = Tool::new(
///     "calculator",
///     "Performs calculations",
///     json!({"type": "object"}),
///     |input| Box::pin(async move { Ok(json!({"result": 42})) })
/// );
///
/// let options = AgentOptions::builder()
///     .model("gpt-4")
///     .api_key("sk-...")
///     .tools(vec![calculator])
///     .build()?;
///
/// let mut stream = query("Calculate 2+2", &options).await?;
///
/// while let Some(event) = stream.next().await {
///     match event?.into_block() {
///         Some(ContentBlock::ToolUse(tool_use)) => {
///             println!("Model wants to use: {}", tool_use.name());
///             // Note: You'll need to manually execute tools and continue
///             // the conversation. For automatic execution, use Client.
///         }
///         Some(ContentBlock::Text(text)) => print!("{}", text.text),
///         _ => {}
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// ## Error Handling
///
/// ```rust,no_run
/// use open_agent::{query, AgentOptions};
/// use futures::StreamExt;
///
/// # async fn example() {
/// let options = AgentOptions::builder()
///     .model("gpt-4")
///     .api_key("invalid-key")
///     .build()
///     .unwrap();
///
/// match query("Hello", &options).await {
///     Ok(mut stream) => {
///         while let Some(result) = stream.next().await {
///             match result {
///                 Ok(event) => println!("Event: {:?}", event),
///                 Err(e) => {
///                     eprintln!("Stream error: {}", e);
///                     break;
///                 }
///             }
///         }
///     }
///     Err(e) => eprintln!("Query failed: {}", e),
/// }
/// # }
/// ```
pub async fn query(prompt: &str, options: &AgentOptions) -> Result<EventStream> {
    // Create HTTP client with configured timeout
    // The timeout applies to the entire request, not individual chunks
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(options.timeout()))
        .build()
        .map_err(Error::Http)?;

    // Build messages array for the API request
    // OpenAI format expects an array of message objects with role and content
    let mut messages = Vec::new();

    // Add system prompt if provided
    // System prompts set the assistant's behavior and context
    if !options.system_prompt().is_empty() {
        messages.push(OpenAIMessage {
            role: "system".to_string(),
            content: Some(OpenAIContent::Text(options.system_prompt().to_string())),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // Add user prompt
    // This is the actual query from the user
    messages.push(OpenAIMessage {
        role: "user".to_string(),
        content: Some(OpenAIContent::Text(prompt.to_string())),
        tool_calls: None,
        tool_call_id: None,
    });

    // Convert tools to OpenAI format if any are provided
    // Tools are described using JSON Schema for parameter validation
    let tools = if !options.tools().is_empty() {
        Some(
            options
                .tools()
                .iter()
                .map(|t| t.to_openai_format())
                .collect(),
        )
    } else {
        None
    };

    // Build the OpenAI-compatible request payload
    // stream=true enables Server-Sent Events for incremental responses
    let request = OpenAIRequest {
        model: options.model().to_string(),
        messages,
        stream: true, // Critical: enables SSE streaming
        max_tokens: options.max_tokens(),
        temperature: options.temperature(),
        tools,
    };

    stream_request(&client, options, &request).await
}
