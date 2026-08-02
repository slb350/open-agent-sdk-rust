impl Client {
    /// Clears all conversation history.
    ///
    /// This resets the conversation to a blank slate while preserving the client
    /// configuration (tools, hooks, model, etc.). The next message will start a
    /// fresh conversation with no prior context.
    ///
    /// # State Changes
    ///
    /// - Clears `history` vector
    /// - Does NOT modify current stream, options, or other state
    ///
    /// # Use Cases
    ///
    /// - Starting a new conversation
    /// - Preventing context length issues
    /// - Clearing sensitive data
    /// - Implementing conversation sessions
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use open_agent::{Client, AgentOptions, ContentBlock};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut client = Client::new(AgentOptions::default())?;
    ///
    /// // First conversation
    /// client.send("Hello").await?;
    /// while let Some(_) = client.receive().await? {}
    ///
    /// // Clear and start fresh
    /// client.clear_history();
    ///
    /// // New conversation with no memory of previous
    /// client.send("Hello again").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.manual_receive_buffer.clear();
    }

    /// Adds a tool result to the conversation history for manual tool execution.
    ///
    /// This method is used exclusively in **manual mode** after receiving a `ToolUseBlock`.
    /// The workflow is:
    ///
    /// 1. `receive()` returns a `ToolUseBlock`
    /// 2. You execute the tool yourself
    /// 3. Call `add_tool_result()` with the tool's output
    /// 4. Call `send("")` to continue the conversation
    /// 5. The model receives the tool result and generates a response
    ///
    /// # Parameters
    ///
    /// - `tool_use_id`: The unique ID from the `ToolUseBlock` (must match exactly)
    /// - `content`: The tool's output as a JSON value
    ///
    /// # Behavior
    ///
    /// Creates a `ToolResultBlock` and adds it to conversation history as a tool message.
    /// This preserves the tool call/result pairing that the model needs to understand
    /// the conversation flow.
    ///
    /// # State Changes
    ///
    /// - Appends a tool message to `history`
    /// - Does NOT modify stream or trigger any requests
    ///
    /// # Important Notes
    ///
    /// - **Not used in auto mode**: Auto-execution handles tool results automatically
    /// - **ID must match**: The `tool_use_id` must match the ID from the `ToolUseBlock`
    /// - **No validation**: This method doesn't validate the result format
    /// - **Must call send()**: After adding result(s), call `send("")` to continue
    ///
    /// # Examples
    ///
    /// ## Basic Manual Tool Execution
    ///
    /// ```rust,no_run
    /// use open_agent::{Client, AgentOptions, ContentBlock};
    /// use serde_json::json;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut client = Client::new(AgentOptions::default())?;
    /// client.send("Use the calculator").await?;
    ///
    /// while let Some(block) = client.receive().await? {
    ///     match block {
    ///         ContentBlock::ToolUse(tool_use) => {
    ///             // Execute tool manually
    ///             let result = json!({"result": 42});
    ///
    ///             // Add result to history
    ///             client.add_tool_result(tool_use.id(), result)?;
    ///
    ///             // Continue conversation to get model's response
    ///             client.send("").await?;
    ///         }
    ///         ContentBlock::Text(text) => {
    ///             println!("{}", text.text);
    ///         }
    ///         ContentBlock::ToolResult(_) | ContentBlock::Image(_) => {}
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Handling Tool Errors
    ///
    /// ```rust,no_run
    /// use open_agent::{Client, AgentOptions, ContentBlock};
    /// use serde_json::json;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut client = Client::new(AgentOptions::default())?;
    /// # client.send("test").await?;
    /// while let Some(block) = client.receive().await? {
    ///     if let ContentBlock::ToolUse(tool_use) = block {
    ///         // Try to execute tool
    ///         let result = match execute_tool(tool_use.name(), tool_use.input()) {
    ///             Ok(output) => output,
    ///             Err(e) => json!({
    ///                 "error": e.to_string(),
    ///                 "tool": tool_use.name()
    ///             })
    ///         };
    ///
    ///         client.add_tool_result(tool_use.id(), result)?;
    ///         client.send("").await?;
    ///     }
    /// }
    ///
    /// # fn execute_tool(name: &str, input: &serde_json::Value) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    /// #     Ok(json!({}))
    /// # }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Multiple Tool Calls
    ///
    /// ```rust,no_run
    /// use open_agent::{Client, AgentOptions, ContentBlock};
    /// use serde_json::json;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut client = Client::new(AgentOptions::default())?;
    /// client.send("Calculate 2+2 and 3+3").await?;
    ///
    /// let mut tool_calls = Vec::new();
    ///
    /// // Collect all tool calls
    /// while let Some(block) = client.receive().await? {
    ///     if let ContentBlock::ToolUse(tool_use) = block {
    ///         tool_calls.push(tool_use);
    ///     }
    /// }
    ///
    /// // Execute and add results for all tools
    /// for tool_call in tool_calls {
    ///     let result = json!({"result": 42}); // Execute tool
    ///     client.add_tool_result(tool_call.id(), result)?;
    /// }
    ///
    /// // Continue conversation
    /// client.send("").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_tool_result(&mut self, tool_use_id: &str, content: serde_json::Value) -> Result<()> {
        use crate::types::ToolResultBlock;

        // Flush any buffered manual-mode blocks first so the assistant's
        // tool_calls message appears in history before this tool result.
        if !self.manual_receive_buffer.is_empty() {
            let blocks = std::mem::take(&mut self.manual_receive_buffer);
            self.history.push(Message::assistant(blocks));
        }

        // Create a tool result block with the given ID and content
        let result_block = ToolResultBlock::new(tool_use_id, content);

        // Add to history as a tool message
        // Note: ToolResultBlock is properly serialized in build_api_request()
        // as a separate message with role="tool" and tool_call_id set
        let serialized = serde_json::to_string(result_block.content())
            .map_err(|e| Error::config(format!("Failed to serialize tool result: {}", e)))?;

        self.history.push(Message::new(
            MessageRole::Tool,
            vec![ContentBlock::Text(TextBlock::new(serialized))],
        ));

        Ok(())
    }

    /// Looks up a registered tool by name.
    ///
    /// This method provides access to the tool registry for manual execution scenarios.
    /// It searches the tools registered in `AgentOptions` and returns a reference to
    /// the matching tool if found.
    ///
    /// # Parameters
    ///
    /// - `name`: The tool name to search for (case-sensitive)
    ///
    /// # Returns
    ///
    /// - `Some(&Tool)`: Tool found
    /// - `None`: No tool with that name
    ///
    /// # Use Cases
    ///
    /// - Manual tool execution in response to `ToolUseBlock`
    /// - Validating tool availability before offering features
    /// - Inspecting tool metadata (name, description, schema)
    ///
    /// # Examples
    ///
    /// ## Execute Tool Manually
    ///
    /// ```rust,no_run
    /// use open_agent::{Client, AgentOptions, ContentBlock};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut client = Client::new(AgentOptions::default())?;
    /// # client.send("test").await?;
    /// while let Some(block) = client.receive().await? {
    ///     if let ContentBlock::ToolUse(tool_use) = block {
    ///         if let Some(tool) = client.get_tool(tool_use.name()) {
    ///             // Execute the tool
    ///             let result = tool.execute(tool_use.input().clone()).await?;
    ///             client.add_tool_result(tool_use.id(), result)?;
    ///             client.send("").await?;
    ///         } else {
    ///             println!("Unknown tool: {}", tool_use.name());
    ///         }
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Check Tool Availability
    ///
    /// ```rust
    /// # use open_agent::{Client, AgentOptions};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = Client::new(AgentOptions::default())?;
    /// if client.get_tool("calculator").is_some() {
    ///     println!("Calculator is available");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_tool(&self, name: &str) -> Option<&crate::tools::Tool> {
        // Search registered tools by name
        self.options
            .tools()
            .iter()
            .find(|t| t.name() == name)
            .map(|t| t.as_ref())
    }
}
