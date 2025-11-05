//! Client for streaming queries and multi-turn conversations

use crate::types::{
    AgentOptions, ContentBlock, Message, MessageRole, OpenAIMessage, OpenAIRequest, TextBlock,
};
use crate::utils::{ToolCallAggregator, parse_sse_stream};
use crate::{Error, Result};
use futures::stream::{Stream, StreamExt};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Stream of content blocks from the model
pub type ContentStream = Pin<Box<dyn Stream<Item = Result<ContentBlock>> + Send>>;

/// Simple query function for single-turn interactions
///
/// # Examples
///
/// ```rust,no_run
/// use open_agent::{query, AgentOptions};
/// use futures::StreamExt;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let options = AgentOptions::builder()
///         .system_prompt("You are a helpful assistant")
///         .model("qwen2.5-32b-instruct")
///         .base_url("http://localhost:1234/v1")
///         .build()?;
///
///     let mut stream = query("What's the capital of France?", &options).await?;
///
///     while let Some(block) = stream.next().await {
///         match block? {
///             open_agent::ContentBlock::Text(text) => {
///                 print!("{}", text.text);
///             }
///             _ => {}
///         }
///     }
///
///     Ok(())
/// }
/// ```
pub async fn query(prompt: &str, options: &AgentOptions) -> Result<ContentStream> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(options.timeout))
        .build()
        .map_err(Error::Http)?;

    // Build messages
    let mut messages = Vec::new();

    if !options.system_prompt.is_empty() {
        messages.push(OpenAIMessage {
            role: "system".to_string(),
            content: options.system_prompt.clone(),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    messages.push(OpenAIMessage {
        role: "user".to_string(),
        content: prompt.to_string(),
        tool_calls: None,
        tool_call_id: None,
    });

    // Build tools array if tools are provided
    let tools = if !options.tools.is_empty() {
        Some(options.tools.iter().map(|t| t.to_openai_format()).collect())
    } else {
        None
    };

    // Build request
    let request = OpenAIRequest {
        model: options.model.clone(),
        messages,
        stream: true,
        max_tokens: options.max_tokens,
        temperature: Some(options.temperature),
        tools,
    };

    // Make request
    let url = format!("{}/chat/completions", options.base_url);
    let response = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", options.api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(Error::Http)?;

    // Check for errors
    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(Error::api(format!("API error {}: {}", status, body)));
    }

    // Parse SSE stream
    let sse_stream = parse_sse_stream(response);

    // Aggregate chunks into content blocks
    let stream = sse_stream.scan(ToolCallAggregator::new(), |aggregator, chunk_result| {
        let result = match chunk_result {
            Ok(chunk) => match aggregator.process_chunk(chunk) {
                Ok(blocks) => {
                    if blocks.is_empty() {
                        Some(None) // Continue streaming
                    } else {
                        Some(Some(Ok(blocks)))
                    }
                }
                Err(e) => Some(Some(Err(e))),
            },
            Err(e) => Some(Some(Err(e))),
        };
        futures::future::ready(result)
    });

    // Flatten the stream
    let flattened = stream
        .filter_map(|item| async move { item })
        .flat_map(|result| {
            futures::stream::iter(match result {
                Ok(blocks) => blocks.into_iter().map(Ok).collect(),
                Err(e) => vec![Err(e)],
            })
        });

    Ok(Box::pin(flattened))
}

/// Client for multi-turn conversations
///
/// # Examples
///
/// ```rust,no_run
/// use open_agent::{Client, AgentOptions, ContentBlock};
/// use futures::StreamExt;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let options = AgentOptions::builder()
///         .system_prompt("You are a helpful assistant")
///         .model("qwen2.5-32b-instruct")
///         .base_url("http://localhost:1234/v1")
///         .max_turns(10)
///         .build()?;
///
///     let mut client = Client::new(options);
///
///     // First query
///     client.send("What's the capital of France?").await?;
///
///     while let Some(block) = client.receive().await {
///         match block? {
///             ContentBlock::Text(text) => {
///                 println!("Assistant: {}", text.text);
///             }
///             _ => {}
///         }
///     }
///
///     // Follow-up query
///     client.send("What's its population?").await?;
///
///     while let Some(block) = client.receive().await {
///         match block? {
///             ContentBlock::Text(text) => {
///                 println!("Assistant: {}", text.text);
///             }
///             _ => {}
///         }
///     }
///
///     Ok(())
/// }
/// ```
pub struct Client {
    options: AgentOptions,
    history: Vec<Message>,
    current_stream: Option<ContentStream>,
    http_client: reqwest::Client,
    interrupted: Arc<AtomicBool>,

    // Auto-execution buffers (only used when auto_execute_tools=true)
    auto_exec_buffer: Vec<ContentBlock>,
    auto_exec_index: usize,
}

impl Client {
    /// Create a new client
    pub fn new(options: AgentOptions) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(options.timeout))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            options,
            history: Vec::new(),
            current_stream: None,
            http_client,
            interrupted: Arc::new(AtomicBool::new(false)),
            auto_exec_buffer: Vec::new(),
            auto_exec_index: 0,
        }
    }

    /// Send a query to the model
    pub async fn send(&mut self, prompt: &str) -> Result<()> {
        use crate::hooks::UserPromptSubmitEvent;

        // Reset interrupt flag for new query
        self.interrupted.store(false, Ordering::SeqCst);

        // Execute UserPromptSubmit hooks
        let mut final_prompt = prompt.to_string();
        let history_snapshot: Vec<serde_json::Value> = self
            .history
            .iter()
            .map(|_| serde_json::json!({})) // Simplified for now
            .collect();

        let event = UserPromptSubmitEvent::new(final_prompt.clone(), history_snapshot);
        if let Some(decision) = self.options.hooks.execute_user_prompt_submit(event).await {
            if !decision.continue_execution {
                return Err(Error::other(format!(
                    "Prompt blocked by hook: {}",
                    decision.reason.unwrap_or_default()
                )));
            }
            if let Some(modified) = decision.modified_prompt {
                final_prompt = modified;
            }
        }

        // Add user message to history
        self.history.push(Message::user(final_prompt));

        // Build messages for API
        let mut messages = Vec::new();

        if !self.options.system_prompt.is_empty() {
            messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: self.options.system_prompt.clone(),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // Convert history to OpenAI format
        for msg in &self.history {
            let content = msg
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text(text) => Some(text.text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");

            messages.push(OpenAIMessage {
                role: match msg.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    MessageRole::Tool => "tool",
                }
                .to_string(),
                content,
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // Build tools array if tools are provided
        let tools = if !self.options.tools.is_empty() {
            Some(
                self.options
                    .tools
                    .iter()
                    .map(|t| t.to_openai_format())
                    .collect(),
            )
        } else {
            None
        };

        // Build request
        let request = OpenAIRequest {
            model: self.options.model.clone(),
            messages,
            stream: true,
            max_tokens: self.options.max_tokens,
            temperature: Some(self.options.temperature),
            tools,
        };

        // Make request
        let url = format!("{}/chat/completions", self.options.base_url);
        let response = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.options.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(Error::Http)?;

        // Check for errors
        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(Error::api(format!("API error {}: {}", status, body)));
        }

        // Parse SSE stream
        let sse_stream = parse_sse_stream(response);

        // Aggregate chunks into content blocks
        let stream = sse_stream.scan(ToolCallAggregator::new(), |aggregator, chunk_result| {
            let result = match chunk_result {
                Ok(chunk) => match aggregator.process_chunk(chunk) {
                    Ok(blocks) => {
                        if blocks.is_empty() {
                            Some(None)
                        } else {
                            Some(Some(Ok(blocks)))
                        }
                    }
                    Err(e) => Some(Some(Err(e))),
                },
                Err(e) => Some(Some(Err(e))),
            };
            futures::future::ready(result)
        });

        // Flatten the stream
        let flattened = stream
            .filter_map(|item| async move { item })
            .flat_map(|result| {
                futures::stream::iter(match result {
                    Ok(blocks) => blocks.into_iter().map(Ok).collect(),
                    Err(e) => vec![Err(e)],
                })
            });

        self.current_stream = Some(Box::pin(flattened));

        Ok(())
    }

    /// Internal method that returns one block (original receive logic)
    ///
    /// This is the original receive() logic, now extracted for reuse.
    async fn receive_one(&mut self) -> Option<Result<ContentBlock>> {
        // Check if interrupted
        if self.interrupted.load(Ordering::SeqCst) {
            self.current_stream = None;
            return None;
        }

        if let Some(stream) = &mut self.current_stream {
            stream.next().await
        } else {
            None
        }
    }

    /// Collect all blocks from current stream
    ///
    /// Internal helper for auto-execution mode.
    /// Returns all blocks from the current response.
    async fn collect_all_blocks(&mut self) -> Result<Vec<ContentBlock>> {
        let mut blocks = Vec::new();

        while let Some(block_result) = self.receive_one().await {
            // Check interrupt during collection
            if self.interrupted.load(Ordering::SeqCst) {
                self.current_stream = None;
                return Err(Error::other(
                    "Operation interrupted during block collection",
                ));
            }

            blocks.push(block_result?);
        }

        Ok(blocks)
    }

    /// Execute a tool by name with given input
    ///
    /// Internal helper for auto-execution mode.
    /// Returns the tool result or error.
    async fn execute_tool_internal(
        &self,
        tool_name: &str,
        input: serde_json::Value,
    ) -> Result<serde_json::Value> {
        // Find tool in options.tools
        let tool = self
            .options
            .tools
            .iter()
            .find(|t| t.name == tool_name)
            .ok_or_else(|| Error::tool(format!("Tool '{}' not found", tool_name)))?;

        // Execute tool
        tool.execute(input).await
    }

    /// Auto-execution loop
    ///
    /// Executes tools automatically until text-only response or max iterations.
    /// Returns all content blocks (text only) from final response.
    async fn auto_execute_loop(&mut self) -> Result<Vec<ContentBlock>> {
        use crate::types::ToolResultBlock;

        let mut iteration = 0;
        let max_iterations = self.options.max_tool_iterations;

        loop {
            // 1. Collect all blocks from current stream
            let blocks = self.collect_all_blocks().await?;

            if blocks.is_empty() {
                return Ok(Vec::new());
            }

            // 2. Separate text and tool blocks
            let mut text_blocks = Vec::new();
            let mut tool_blocks = Vec::new();

            for block in blocks {
                match block {
                    ContentBlock::Text(_) => text_blocks.push(block),
                    ContentBlock::ToolUse(_) => tool_blocks.push(block),
                    _ => {} // Ignore other variants
                }
            }

            // 3. If no tool blocks, we're done - add to history and return
            if tool_blocks.is_empty() {
                // Add assistant message with text blocks to history
                if !text_blocks.is_empty() {
                    let assistant_msg = Message::assistant(text_blocks.clone());
                    self.history.push(assistant_msg);
                }
                return Ok(text_blocks);
            }

            // 4. Check iteration limit BEFORE executing tools
            iteration += 1;
            if iteration > max_iterations {
                // Max iterations reached - add what we have and return
                if !text_blocks.is_empty() {
                    let assistant_msg = Message::assistant(text_blocks.clone());
                    self.history.push(assistant_msg);
                }
                return Ok(text_blocks);
            }

            // 5. Add assistant message with ALL blocks (text + tool calls) to history
            let mut all_blocks = text_blocks.clone();
            all_blocks.extend(tool_blocks.clone());
            let assistant_msg = Message::assistant(all_blocks);
            self.history.push(assistant_msg);

            // 6. Execute all tools and collect results
            for block in tool_blocks {
                if let ContentBlock::ToolUse(tool_use) = block {
                    // Create history snapshot for hooks
                    let history_snapshot: Vec<serde_json::Value> =
                        self.history.iter().map(|_| serde_json::json!({})).collect();

                    // Execute PreToolUse hooks
                    use crate::hooks::PreToolUseEvent;
                    let pre_event = PreToolUseEvent::new(
                        tool_use.name.clone(),
                        tool_use.input.clone(),
                        tool_use.id.clone(),
                        history_snapshot.clone(),
                    );

                    let mut tool_input = tool_use.input.clone();
                    let mut should_execute = true;
                    let mut block_reason = None;

                    if let Some(decision) = self.options.hooks.execute_pre_tool_use(pre_event).await
                    {
                        if !decision.continue_execution {
                            should_execute = false;
                            block_reason = decision.reason;
                        } else if let Some(modified) = decision.modified_input {
                            tool_input = modified;
                        }
                    }

                    // Execute tool or create blocked result
                    let result = if should_execute {
                        match self
                            .execute_tool_internal(&tool_use.name, tool_input.clone())
                            .await
                        {
                            Ok(res) => res,
                            Err(e) => {
                                // Tool execution failed - add error result
                                serde_json::json!({
                                    "error": e.to_string(),
                                    "tool": tool_use.name,
                                    "id": tool_use.id
                                })
                            }
                        }
                    } else {
                        // Tool blocked by PreToolUse hook
                        serde_json::json!({
                            "error": "Tool execution blocked by hook",
                            "reason": block_reason.unwrap_or_else(|| "No reason provided".to_string()),
                            "tool": tool_use.name,
                            "id": tool_use.id
                        })
                    };

                    // Execute PostToolUse hooks
                    use crate::hooks::PostToolUseEvent;
                    let post_event = PostToolUseEvent::new(
                        tool_use.name.clone(),
                        tool_input,
                        tool_use.id.clone(),
                        result.clone(),
                        history_snapshot,
                    );

                    let mut final_result = result;
                    if let Some(decision) =
                        self.options.hooks.execute_post_tool_use(post_event).await
                    {
                        // PostToolUse can modify the result via modified_input field
                        if let Some(modified) = decision.modified_input {
                            final_result = modified;
                        }
                    }

                    // Add tool result to history
                    let tool_result = ToolResultBlock::new(&tool_use.id, final_result);
                    let tool_result_msg =
                        Message::user_with_blocks(vec![ContentBlock::ToolResult(tool_result)]);
                    self.history.push(tool_result_msg);
                }
            }

            // 7. Continue conversation (send empty string to get next response)
            self.send("").await?;

            // Loop continues to collect next response
        }
    }

    /// Receive the next content block from the current stream
    pub async fn receive(&mut self) -> Option<Result<ContentBlock>> {
        if self.options.auto_execute_tools {
            // Check if we have buffered blocks to yield
            if self.auto_exec_index < self.auto_exec_buffer.len() {
                let block = self.auto_exec_buffer[self.auto_exec_index].clone();
                self.auto_exec_index += 1;
                return Some(Ok(block));
            }

            // No buffered blocks - run auto-execution loop
            if self.auto_exec_buffer.is_empty() {
                match self.auto_execute_loop().await {
                    Ok(blocks) => {
                        self.auto_exec_buffer = blocks;
                        self.auto_exec_index = 0;

                        if self.auto_exec_buffer.is_empty() {
                            return None;
                        }

                        let block = self.auto_exec_buffer[0].clone();
                        self.auto_exec_index = 1;
                        return Some(Ok(block));
                    }
                    Err(e) => return Some(Err(e)),
                }
            }

            None
        } else {
            // Manual mode
            self.receive_one().await
        }
    }

    /// Interrupt the current operation
    ///
    /// This method cancels any in-progress streaming and leaves the client
    /// in a valid state for reuse. It's safe to call multiple times (idempotent)
    /// and safe to call when no operation is in progress (no-op).
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use open_agent::{Client, AgentOptions};
    /// use std::time::Duration;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut client = Client::new(AgentOptions::default());
    ///
    /// // Start a query
    /// client.send("Tell me a long story").await?;
    ///
    /// // Interrupt after a timeout
    /// tokio::time::sleep(Duration::from_secs(2)).await;
    /// client.interrupt();
    ///
    /// // Client is ready for new queries
    /// client.send("What's 2+2?").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn interrupt(&self) {
        self.interrupted.store(true, Ordering::SeqCst);
    }

    /// Get a reference to the conversation history
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Get a mutable reference to the conversation history
    pub fn history_mut(&mut self) -> &mut Vec<Message> {
        &mut self.history
    }

    /// Get a reference to the agent options
    pub fn options(&self) -> &AgentOptions {
        &self.options
    }

    /// Clear the conversation history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Add a tool result to the conversation history
    ///
    /// This method is used for manual tool execution. After receiving a ToolUseBlock,
    /// execute the tool and call this method with the result.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use open_agent::{Client, AgentOptions, ContentBlock};
    /// use futures::StreamExt;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut client = Client::new(AgentOptions::default());
    /// client.send("Use the calculator").await?;
    ///
    /// while let Some(block) = client.receive().await {
    ///     match block? {
    ///         ContentBlock::ToolUse(tool_use) => {
    ///             // Execute tool manually
    ///             let result = serde_json::json!({"result": 42});
    ///             client.add_tool_result(&tool_use.id, result);
    ///
    ///             // Continue conversation to get response
    ///             client.send("").await?;
    ///         }
    ///         _ => {}
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_tool_result(&mut self, tool_use_id: &str, content: serde_json::Value) {
        use crate::types::ToolResultBlock;

        // Add tool result to history
        let result_block = ToolResultBlock::new(tool_use_id, content);

        // Tool results are added as part of the conversation
        // In the OpenAI format, we'll need to convert these appropriately
        // For now, store in history (we'll handle conversion in send())
        self.history.push(Message::new(
            MessageRole::Tool,
            vec![ContentBlock::Text(TextBlock::new(
                serde_json::to_string(&result_block.content).unwrap_or_default(),
            ))],
        ));
    }

    /// Get the tool registry for executing tools
    pub fn get_tool(&self, name: &str) -> Option<&crate::tools::Tool> {
        self.options
            .tools
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let client = Client::new(options);
        assert_eq!(client.history().len(), 0);
    }

    #[test]
    fn test_interrupt_flag_initial_state() {
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let client = Client::new(options);
        // Initially not interrupted
        assert!(!client.interrupted.load(Ordering::SeqCst));
    }

    #[test]
    fn test_interrupt_sets_flag() {
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let client = Client::new(options);
        client.interrupt();
        assert!(client.interrupted.load(Ordering::SeqCst));
    }

    #[test]
    fn test_interrupt_idempotent() {
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let client = Client::new(options);
        client.interrupt();
        assert!(client.interrupted.load(Ordering::SeqCst));

        // Call again - should still be interrupted
        client.interrupt();
        assert!(client.interrupted.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_receive_returns_none_when_interrupted() {
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let mut client = Client::new(options);

        // Interrupt before receiving
        client.interrupt();

        // receive() should return None
        let result = client.receive().await;
        assert!(result.is_none());
    }
}
