impl Client {
    /// Sends a user message and initiates streaming of the model's response.
    ///
    /// This method performs several critical steps:
    ///
    /// 1. Executes UserPromptSubmit hooks (which can modify or block the prompt)
    /// 2. Adds the user message to conversation history
    /// 3. Builds and sends HTTP request to the OpenAI-compatible API
    /// 4. Parses the SSE stream and sets up aggregation
    /// 5. Stores the stream for consumption via `receive()`
    ///
    /// # Parameters
    ///
    /// - `prompt`: The user's message. Can be empty to continue conversation after
    ///   adding tool results (common pattern in manual tool execution mode).
    ///
    /// # Returns
    ///
    /// - `Ok(())`: Request sent successfully, call `receive()` to get blocks
    /// - `Err(e)`: Request failed (network error, API error, hook blocked, etc.)
    ///
    /// # Behavior Details
    ///
    /// ## Hook Execution
    ///
    /// Before sending, UserPromptSubmit hooks are executed. Hooks can:
    /// - Modify the prompt text
    /// - Block the request entirely
    /// - Access conversation history
    ///
    /// If a hook blocks the request, this method returns an error immediately.
    ///
    /// ## History Management
    ///
    /// The prompt is added to history BEFORE sending the request. This ensures
    /// that history is consistent even if the request fails.
    ///
    /// ## Stream Setup
    ///
    /// The response stream is set up but not consumed. You must call `receive()`
    /// repeatedly to get content blocks. The stream remains active until:
    /// - All blocks are consumed (stream naturally ends)
    /// - An error occurs
    /// - Interrupt is triggered
    ///
    /// ## Interrupt Handling
    ///
    /// The interrupt flag is reset to `false` at the start of this method,
    /// allowing a fresh request after a previous interruption.
    ///
    /// # State Changes
    ///
    /// - Resets `interrupted` flag to `false`
    /// - Appends user message to `history`
    /// - Sets `current_stream` to new SSE stream
    /// - Does NOT modify `auto_exec_buffer` or `auto_exec_index`
    ///
    /// # Errors
    ///
    /// Returns errors for:
    /// - Hook blocking the prompt
    /// - HTTP client errors (network failure, DNS, etc.)
    /// - API errors (auth failure, invalid model, rate limits)
    /// - Invalid response format
    ///
    /// After an error, the client remains usable for new requests.
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    ///
    /// ```rust,no_run
    /// # use open_agent::{Client, AgentOptions};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut client = Client::new(AgentOptions::default())?;
    /// client.send("Hello!").await?;
    ///
    /// while let Some(block) = client.receive().await? {
    ///     // Process blocks...
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Continuing After Tool Result
    ///
    /// ```rust,no_run
    /// # use open_agent::{Client, AgentOptions, ContentBlock};
    /// # use serde_json::json;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut client = Client::new(AgentOptions::default())?;
    /// client.send("Use the calculator").await?;
    ///
    /// while let Some(block) = client.receive().await? {
    ///     if let ContentBlock::ToolUse(tool_use) = block {
    ///         // Execute tool and add result
    ///         client.add_tool_result(tool_use.id(), json!({"result": 42}))?;
    ///
    ///         // Continue conversation with empty prompt
    ///         client.send("").await?;
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn send(&mut self, prompt: &str) -> Result<()> {
        use crate::hooks::UserPromptSubmitEvent;

        // Reset interrupt flag for new query
        // This allows the client to be reused after a previous interruption
        // Uses SeqCst ordering to ensure visibility across all threads
        self.interrupted.store(false, Ordering::SeqCst);

        // Discard any leftover manual-mode blocks from an abandoned stream.
        // If the prior stream completed normally, receive() already committed
        // the buffer to history on EOF. If the caller is calling send() before
        // the stream finished, the buffer is partial and must not be persisted.
        self.manual_receive_buffer.clear();
        self.current_stream = None;

        let mut final_prompt = prompt.to_string();
        if !self.options.hooks().user_prompt_submit.is_empty() {
            // Hooks run before adding to history, allowing modification or blocking.
            let history_snapshot = serialize_history_snapshot(&self.history)?;
            let event = UserPromptSubmitEvent::new(final_prompt.clone(), history_snapshot);

            if let Some(decision) = self.options.hooks().execute_user_prompt_submit(event).await {
                if !decision.continue_execution() {
                    return Err(Error::other(format!(
                        "Prompt blocked by hook: {}",
                        decision.reason().unwrap_or("")
                    )));
                }
                if let Some(modified) = decision.modified_prompt() {
                    final_prompt = modified.to_string();
                }
            }
        }

        // Add user message to history BEFORE sending request
        // This ensures history consistency even if request fails
        // Empty prompts are still added (needed for tool continuation)
        self.history.push(Message::user(final_prompt));

        self.start_request().await
    }

}
