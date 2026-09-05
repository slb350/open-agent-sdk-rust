impl Client {
    fn discard_pending_output(&mut self) {
        self.current_stream = None;
        self.auto_output = None;
        self.manual_receive_buffer.clear();
    }

    /// Clears conversation history and abandons pending output, retaining options.
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.discard_pending_output();
    }

    // Join adjacent streamed fragments before replaying the assistant turn.
    pub(crate) fn push_assistant(&mut self, blocks: &[ContentBlock]) {
        self.history
            .push(Message::assistant(coalesce_text_blocks(blocks)));
    }

    /// Adds a manual tool result after flushing the pending assistant tool call.
    /// Continue the conversation with `send("")` after supplying the results.
    pub fn add_tool_result(&mut self, tool_use_id: &str, content: serde_json::Value) -> Result<()> {
        use crate::types::ToolResultBlock;

        if !self.manual_receive_buffer.is_empty() {
            let blocks = std::mem::take(&mut self.manual_receive_buffer);
            self.push_assistant(&blocks);
        }

        // Preserve the call ID until the protocol-specific wire conversion.
        self.history.push(Message::new(
            MessageRole::Tool,
            vec![ContentBlock::ToolResult(ToolResultBlock::new(
                tool_use_id,
                content,
            ))],
        ));

        Ok(())
    }

    /// Looks up a registered tool by name.
    pub fn get_tool(&self, name: &str) -> Option<&crate::tools::Tool> {
        self.options
            .tools()
            .iter()
            .find(|t| t.name() == name)
            .map(|t| t.as_ref())
    }
}
