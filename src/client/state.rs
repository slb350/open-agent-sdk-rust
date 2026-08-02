/// Stateful client for multi-turn conversations with automatic history management.
///
/// The `Client` is the primary interface for building conversational AI applications.
/// It maintains conversation history, manages streaming responses, and provides two
/// modes of operation: manual and automatic tool execution.
///
/// # State Management
///
/// The client maintains several pieces of state that persist across multiple turns:
///
/// - **Conversation History**: Complete record of all messages exchanged
/// - **Active Stream**: Currently active SSE stream being consumed
/// - **Interrupt Flag**: Thread-safe cancellation signal
/// - **Auto-Execution Buffer**: Cached blocks for auto-execution mode
///
/// # Operating Modes
///
/// ## Manual Mode (default)
///
/// In manual mode, the client streams blocks directly to the caller. When the model
/// requests a tool, you receive a `ToolUseBlock`, execute the tool yourself, add the
/// result with `add_tool_result()`, and continue the conversation.
///
/// **Advantages**:
/// - Full control over tool execution
/// - Custom error handling per tool
/// - Ability to modify tool inputs/outputs
/// - Interactive debugging capabilities
///
/// ## Automatic Mode (`auto_execute_tools = true`)
///
/// In automatic mode, the client executes tools transparently and only returns the
/// final text response after all tool iterations complete.
///
/// **Advantages**:
/// - Simpler API for common use cases
/// - Built-in retry logic via hooks
/// - Automatic conversation continuation
/// - Configurable iteration limits
///
/// # Thread Safety
///
/// The client is NOT thread-safe for concurrent use. However, the interrupt mechanism
/// uses `Arc<AtomicBool>` which can be safely shared across threads to signal cancellation.
///
/// # Memory Management
///
/// - History grows unbounded by default (consider clearing periodically)
/// - Streams are consumed lazily (low memory footprint during streaming)
/// - Auto-execution buffers entire response (higher memory in auto mode)
///
/// # Examples
///
/// ## Basic Multi-Turn Conversation
///
/// ```rust,no_run
/// use open_agent::{Client, AgentOptions, ContentBlock};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut client = Client::new(AgentOptions::builder()
///     .model("gpt-4")
///     .api_key("sk-...")
///     .build()?)?;
///
/// // First question
/// client.send("What's the capital of France?").await?;
/// while let Some(block) = client.receive().await? {
///     if let ContentBlock::Text(text) = block {
///         println!("{}", text.text); // "Paris is the capital of France."
///     }
/// }
///
/// // Follow-up question - history is automatically maintained
/// client.send("What's its population?").await?;
/// while let Some(block) = client.receive().await? {
///     if let ContentBlock::Text(text) = block {
///         println!("{}", text.text); // "Paris has approximately 2.2 million people."
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// ## Manual Tool Execution
///
/// ```rust,no_run
/// use open_agent::{Client, AgentOptions, ContentBlock, Tool};
/// use serde_json::json;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let calculator = Tool::new(
///     "calculator",
///     "Performs arithmetic",
///     json!({"type": "object"}),
///     |input| Box::pin(async move { Ok(json!({"result": 42})) })
/// );
///
/// let mut client = Client::new(AgentOptions::builder()
///     .model("gpt-4")
///     .api_key("sk-...")
///     .tools(vec![calculator])
///     .build()?)?;
///
/// client.send("What's 2+2?").await?;
///
/// while let Some(block) = client.receive().await? {
///     match block {
///         ContentBlock::ToolUse(tool_use) => {
///             // Execute tool manually
///             let result = json!({"result": 4});
///             client.add_tool_result(tool_use.id(), result)?;
///
///             // Continue conversation to get model's response
///             client.send("").await?;
///         }
///         ContentBlock::Text(text) => {
///             println!("{}", text.text); // "The result is 4."
///         }
///         ContentBlock::ToolResult(_) | ContentBlock::Image(_) => {}
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// ## Automatic Tool Execution
///
/// ```rust,no_run
/// use open_agent::{Client, AgentOptions, ContentBlock, Tool};
/// use serde_json::json;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let calculator = Tool::new(
///     "calculator",
///     "Performs arithmetic",
///     json!({"type": "object"}),
///     |input| Box::pin(async move { Ok(json!({"result": 42})) })
/// );
///
/// let mut client = Client::new(AgentOptions::builder()
///     .model("gpt-4")
///     .api_key("sk-...")
///     .tools(vec![calculator])
///     .auto_execute_tools(true)  // Enable auto-execution
///     .build()?)?;
///
/// client.send("What's 2+2?").await?;
///
/// // Tools execute automatically - you only receive final text
/// while let Some(block) = client.receive().await? {
///     if let ContentBlock::Text(text) = block {
///         println!("{}", text.text); // "The result is 4."
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// ## With Interruption
///
/// ```rust,no_run
/// use open_agent::{Client, AgentOptions};
/// use std::time::Duration;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut client = Client::new(AgentOptions::default())?;
///
/// // Start a long-running query
/// client.send("Write a very long story").await?;
///
/// // Spawn a task to interrupt after timeout
/// let interrupt_handle = client.interrupt_handle();
/// tokio::spawn(async move {
///     tokio::time::sleep(Duration::from_secs(5)).await;
///     interrupt_handle.store(true, std::sync::atomic::Ordering::SeqCst);
/// });
///
/// // This loop will stop when interrupted
/// while let Some(block) = client.receive().await? {
///     // Process blocks...
/// }
///
/// // Client is still usable after interruption
/// client.send("What's 2+2?").await?;
/// # Ok(())
/// # }
/// ```
pub struct Client {
    /// Configuration options including model, API key, tools, hooks, etc.
    ///
    /// This field contains all the settings that control how the client behaves.
    /// It's set once during construction and cannot be modified (though you can
    /// access it via `options()` for inspection).
    options: AgentOptions,

    /// Complete conversation history as a sequence of messages.
    ///
    /// Each message contains a role (System/User/Assistant/Tool) and content blocks.
    /// History grows unbounded by default - use `clear_history()` to reset.
    ///
    /// **Important**: The history includes ALL messages, not just user/assistant.
    /// This includes tool results and intermediate assistant messages from tool calls.
    history: Vec<Message>,

    /// Currently active SSE stream being consumed.
    ///
    /// This is `Some(stream)` while a response is being received, and `None` when
    /// no request is in flight or after a response completes.
    ///
    /// The stream is set by `send()` and consumed by `receive()`. When the stream
    /// is exhausted, `receive()` returns `Ok(None)` and sets this back to `None`.
    current_stream: Option<ContentStream>,

    /// Reusable HTTP client for making API requests.
    ///
    /// Configured once during construction with the timeout from `AgentOptions`.
    /// Reusing the same client across requests enables connection pooling and
    /// better performance for multi-turn conversations.
    http_client: reqwest::Client,

    /// Thread-safe interrupt flag for cancellation.
    ///
    /// This `Arc<AtomicBool>` can be cloned and shared across threads or async tasks
    /// to signal cancellation. When set to `true`, the next `receive()` call will
    /// return `Ok(None)` and clear the current stream.
    ///
    /// The flag is automatically reset to `false` at the start of each `send()` call.
    ///
    /// **Thread Safety**: Can be safely accessed from multiple threads using atomic
    /// operations. However, only one thread should call `send()`/`receive()`.
    interrupted: Arc<AtomicBool>,

    /// Buffer of content blocks for auto-execution mode.
    ///
    /// When `auto_execute_tools` is enabled, `receive()` internally calls the
    /// auto-execution loop which buffers all final text blocks here. Subsequent
    /// calls to `receive()` return blocks from this buffer one at a time.
    ///
    /// **Only used when `options.auto_execute_tools == true`**.
    ///
    /// The buffer is cleared when starting a new auto-execution loop.
    auto_exec_buffer: Vec<ContentBlock>,

    /// Current read position in the auto-execution buffer.
    ///
    /// Tracks which block to return next when `receive()` is called in auto mode.
    /// Reset to 0 when the buffer is refilled with a new response.
    ///
    /// **Only used when `options.auto_execute_tools == true`**.
    auto_exec_index: usize,

    /// Accumulator for assistant response blocks in manual mode.
    ///
    /// In manual mode, `receive()` streams blocks one at a time to the caller.
    /// This buffer collects those blocks so that when the stream ends, the
    /// complete assistant message can be added to conversation history.
    ///
    /// **Only used when `options.auto_execute_tools == false`**.
    manual_receive_buffer: Vec<ContentBlock>,
}
