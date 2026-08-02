impl Client {
    /// Receives the next content block from the current stream.
    ///
    /// This is the primary method for consuming responses from the model. It works
    /// differently depending on the operating mode:
    ///
    /// ## Manual Mode (default)
    ///
    /// Streams blocks directly from the API response as they arrive. You receive:
    /// - `TextBlock`: Incremental text from the model
    /// - `ToolUseBlock`: Requests to execute tools
    /// - Other block types as they're emitted
    ///
    /// When you receive a `ToolUseBlock`, you must:
    /// 1. Execute the tool yourself
    /// 2. Call `add_tool_result()` with the result
    /// 3. Call `send("")` to continue the conversation
    ///
    /// ## Automatic Mode (`auto_execute_tools = true`)
    ///
    /// Transparently executes tools and only returns final text blocks. The first
    /// call to `receive()` triggers the auto-execution loop which:
    /// 1. Collects all blocks from the stream
    /// 2. Executes any tool calls automatically
    /// 3. Continues the conversation until reaching a text-only response
    /// 4. Buffers the final text blocks
    /// 5. Returns them one at a time on subsequent `receive()` calls
    ///
    /// # Returns
    ///
    /// - `Ok(Some(block))`: Successfully received a content block
    /// - `Ok(None)`: Stream ended normally or was interrupted
    /// - `Err(e)`: An error occurred during streaming or tool execution
    ///
    /// # Behavior Details
    ///
    /// ## Interruption
    ///
    /// Checks the interrupt flag on every call. If interrupted, immediately returns
    /// `Ok(None)` and clears the stream. The client can be reused after interruption.
    ///
    /// ## Stream Lifecycle
    ///
    /// 1. After `send()`, stream is active
    /// 2. Each `receive()` call yields one block
    /// 3. When stream ends, returns `Ok(None)`
    /// 4. Subsequent calls continue returning `Ok(None)` until next `send()`
    ///
    /// ## Auto-Execution Buffer
    ///
    /// In auto mode, blocks are buffered in memory. The buffer persists until
    /// fully consumed (index reaches length), at which point it's cleared.
    ///
    /// # State Changes
    ///
    /// - Advances stream position
    /// - In auto mode: May trigger entire execution loop and modify history
    /// - In manual mode: Only reads from stream, no history changes
    /// - Increments `auto_exec_index` when returning buffered blocks
    ///
    /// # Examples
    ///
    /// ## Manual Mode - Basic
    ///
    /// ```rust,no_run
    /// # use open_agent::{Client, AgentOptions, ContentBlock};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut client = Client::new(AgentOptions::default())?;
    /// client.send("Hello!").await?;
    ///
    /// while let Some(block) = client.receive().await? {
    ///     match block {
    ///         ContentBlock::Text(text) => print!("{}", text.text),
    ///         ContentBlock::ToolUse(_) | ContentBlock::ToolResult(_) | ContentBlock::Image(_) => {}
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Manual Mode - With Tools
    ///
    /// ```rust,no_run
    /// # use open_agent::{Client, AgentOptions, ContentBlock};
    /// # use serde_json::json;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut client = Client::new(AgentOptions::default())?;
    /// client.send("Use the calculator").await?;
    ///
    /// while let Some(block) = client.receive().await? {
    ///     match block {
    ///         ContentBlock::Text(text) => {
    ///             println!("{}", text.text);
    ///         }
    ///         ContentBlock::ToolUse(tool_use) => {
    ///             println!("Executing: {}", tool_use.name());
    ///
    ///             // Execute tool manually
    ///             let result = json!({"result": 42});
    ///
    ///             // Add result and continue
    ///             client.add_tool_result(tool_use.id(), result)?;
    ///             client.send("").await?;
    ///         }
    ///         ContentBlock::ToolResult(_) | ContentBlock::Image(_) => {}
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Auto Mode
    ///
    /// ```rust,no_run
    /// # use open_agent::{Client, AgentOptions, ContentBlock, Tool};
    /// # use serde_json::json;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut client = Client::new(AgentOptions::builder()
    ///     .auto_execute_tools(true)
    ///     .build()?)?;
    ///
    /// client.send("Calculate 2+2").await?;
    ///
    /// // Tools execute automatically - you only get final text
    /// while let Some(block) = client.receive().await? {
    ///     if let ContentBlock::Text(text) = block {
    ///         println!("{}", text.text);
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## With Error Handling
    ///
    /// ```rust,no_run
    /// # use open_agent::{Client, AgentOptions};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut client = Client::new(AgentOptions::default())?;
    /// client.send("Hello").await?;
    ///
    /// loop {
    ///     match client.receive().await {
    ///         Ok(Some(block)) => {
    ///             // Process block
    ///         }
    ///         Ok(None) => {
    ///             // Stream ended
    ///             break;
    ///         }
    ///         Err(e) => {
    ///             eprintln!("Error: {}", e);
    ///             break;
    ///         }
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    pub async fn receive(&mut self) -> Result<Option<ContentBlock>> {
        // ========================================================================
        // AUTO-EXECUTION MODE
        // ========================================================================
        if self.options.auto_execute_tools() {
            // Check if we have buffered blocks to return
            // In auto mode, all final text blocks are buffered and returned one at a time
            if self.auto_exec_index < self.auto_exec_buffer.len() {
                // Return next buffered block
                let block = self.auto_exec_buffer[self.auto_exec_index].clone();
                self.auto_exec_index += 1;
                return Ok(Some(block));
            }

            // No buffered blocks - need to run auto-execution loop
            // This only happens on the first receive() call after send()
            if self.auto_exec_buffer.is_empty() {
                match self.auto_execute_loop().await {
                    Ok(blocks) => {
                        // Buffer all final text blocks
                        self.auto_exec_buffer = blocks;
                        self.auto_exec_index = 0;

                        // If no blocks, return None (empty response)
                        if self.auto_exec_buffer.is_empty() {
                            return Ok(None);
                        }

                        // Return first buffered block
                        let block = self.auto_exec_buffer[0].clone();
                        self.auto_exec_index = 1;
                        return Ok(Some(block));
                    }
                    Err(e) => return Err(e),
                }
            }

            // Buffer exhausted - return None
            Ok(None)
        } else {
            // ====================================================================
            // MANUAL MODE
            // ====================================================================
            // Stream blocks to caller while accumulating them so we can add
            // the complete assistant message to history when the stream ends.
            match self.receive_one().await {
                Err(e) => {
                    // Stream error — discard partial output so a retry
                    // doesn't flush truncated blocks into history.
                    self.manual_receive_buffer.clear();
                    Err(e)
                }
                Ok(Some(block)) => {
                    self.manual_receive_buffer.push(block.clone());
                    Ok(Some(block))
                }
                Ok(None) => {
                    if self.interrupted.load(Ordering::SeqCst) && self.current_stream.is_some() {
                        // Interrupted a live stream — discard partial output.
                        // current_stream is still Some because receive_one()
                        // only clears it on natural EOF, not on interrupt.
                        self.current_stream = None;
                        self.manual_receive_buffer.clear();
                    } else if !self.manual_receive_buffer.is_empty() {
                        // Either natural EOF or interrupt after stream already
                        // finished — commit the (complete) assistant message.
                        let blocks = std::mem::take(&mut self.manual_receive_buffer);
                        self.history.push(Message::assistant(blocks));
                    }
                    Ok(None)
                }
            }
        }
    }

    /// Interrupts the current operation by setting the interrupt flag.
    ///
    /// This method provides a thread-safe way to cancel any in-progress streaming
    /// operation. The interrupt flag is checked by `receive()` before each block,
    /// allowing responsive cancellation.
    ///
    /// # Behavior
    ///
    /// - Sets the atomic interrupt flag to `true`
    /// - Next `receive()` call will return `Ok(None)` and clear the stream
    /// - Flag is automatically reset to `false` on next `send()` call
    /// - Safe to call from any thread (uses atomic operations)
    /// - Idempotent: calling multiple times has same effect as calling once
    /// - No-op if no operation is in progress
    ///
    /// # Thread Safety
    ///
    /// This method uses `Arc<AtomicBool>` internally, which can be safely shared
    /// across threads. You can clone the interrupt handle and use it from different
    /// threads or async tasks:
    ///
    /// ```rust,no_run
    /// # use open_agent::{Client, AgentOptions};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut client = Client::new(AgentOptions::default())?;
    /// let interrupt_handle = client.interrupt_handle();
    ///
    /// // Use from another thread
    /// tokio::spawn(async move {
    ///     tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    ///     interrupt_handle.store(true, std::sync::atomic::Ordering::SeqCst);
    /// });
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # State Changes
    ///
    /// - Sets `interrupted` flag to `true`
    /// - Does NOT modify stream, history, or other state directly
    /// - Effect takes place on next `receive()` call
    ///
    /// # Use Cases
    ///
    /// - User cancellation (e.g., stop button in UI)
    /// - Timeout enforcement
    /// - Resource cleanup
    /// - Emergency shutdown
    ///
    /// # Examples
    ///
    /// ## Basic Interruption
    ///
    /// ```rust,no_run
    /// use open_agent::{Client, AgentOptions};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut client = Client::new(AgentOptions::default())?;
    ///
    /// client.send("Tell me a long story").await?;
    ///
    /// // Interrupt after receiving some blocks
    /// let mut count = 0;
    /// while let Some(block) = client.receive().await? {
    ///     count += 1;
    ///     if count >= 5 {
    ///         client.interrupt();
    ///     }
    /// }
    ///
    /// // Client is ready for new queries
    /// client.send("What's 2+2?").await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## With Timeout
    ///
    /// ```rust,no_run
    /// use open_agent::{Client, AgentOptions};
    /// use std::time::Duration;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut client = Client::new(AgentOptions::default())?;
    ///
    /// client.send("Long request").await?;
    ///
    /// // Spawn timeout task
    /// let interrupt_handle = client.interrupt_handle();
    /// tokio::spawn(async move {
    ///     tokio::time::sleep(Duration::from_secs(10)).await;
    ///     interrupt_handle.store(true, std::sync::atomic::Ordering::SeqCst);
    /// });
    ///
    /// while let Some(_block) = client.receive().await? {
    ///     // Process until timeout
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn interrupt(&self) {
        // Set interrupt flag using SeqCst for immediate visibility across all threads
        self.interrupted.store(true, Ordering::SeqCst);
    }

    /// Returns a clone of the interrupt handle for thread-safe cancellation.
    ///
    /// This method provides access to the shared `Arc<AtomicBool>` interrupt flag,
    /// allowing it to be used from other threads or async tasks to signal cancellation.
    ///
    /// # Returns
    ///
    /// A cloned `Arc<AtomicBool>` that can be used to interrupt operations from any thread.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use open_agent::{Client, AgentOptions};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut client = Client::new(AgentOptions::default())?;
    /// let interrupt_handle = client.interrupt_handle();
    ///
    /// // Use from another thread
    /// tokio::spawn(async move {
    ///     tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    ///     interrupt_handle.store(true, std::sync::atomic::Ordering::SeqCst);
    /// });
    /// # Ok(())
    /// # }
    /// ```
    pub fn interrupt_handle(&self) -> Arc<AtomicBool> {
        self.interrupted.clone()
    }

    /// Returns a reference to the conversation history.
    ///
    /// The history contains all messages exchanged in the conversation, including:
    /// - User messages
    /// - Assistant messages (with text and tool use blocks)
    /// - Tool result messages
    ///
    /// # Returns
    ///
    /// A slice of `Message` objects in chronological order.
    ///
    /// # Use Cases
    ///
    /// - Inspecting conversation context
    /// - Debugging tool execution flow
    /// - Saving conversation state
    /// - Implementing custom history management
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use open_agent::{Client, AgentOptions};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Client::new(AgentOptions::default())?;
    ///
    /// // Initially empty
    /// assert_eq!(client.history().len(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Returns a mutable reference to the conversation history.
    ///
    /// This allows you to modify the history directly for advanced use cases like:
    /// - Removing old messages to manage context length
    /// - Editing messages for retry scenarios
    /// - Injecting synthetic messages for testing
    ///
    /// # Warning
    ///
    /// Modifying history directly can lead to inconsistent conversation state if not
    /// done carefully. The SDK expects history to follow the proper message flow
    /// (user → assistant → tool results → assistant, etc.).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use open_agent::{Client, AgentOptions};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut client = Client::new(AgentOptions::default())?;
    ///
    /// // Remove oldest messages to stay within context limit
    /// if client.history().len() > 50 {
    ///     client.history_mut().drain(0..10);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn history_mut(&mut self) -> &mut Vec<Message> {
        &mut self.history
    }

    /// Returns a reference to the agent configuration options.
    ///
    /// Provides read-only access to the `AgentOptions` used to configure this client.
    ///
    /// # Use Cases
    ///
    /// - Inspecting current configuration
    /// - Debugging issues
    /// - Conditional logic based on settings
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use open_agent::{Client, AgentOptions};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Client::new(AgentOptions::builder()
    ///     .model("gpt-4")
    ///     .base_url("http://localhost:1234/v1")
    ///     .build()?)?;
    ///
    /// println!("Using model: {}", client.options().model());
    /// # Ok(())
    /// # }
    /// ```
    pub fn options(&self) -> &AgentOptions {
        &self.options
    }

}
