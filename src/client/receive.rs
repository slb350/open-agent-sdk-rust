impl Client {
    /// Receives the next content block, or `None` when the request has finished.
    ///
    /// Manual mode records the complete assistant turn at EOF; errors and abandoned
    /// streams discard partial history. Auto mode executes tool rounds before yielding
    /// final text. Read [`Self::finish_reason`] and [`Self::reasoning`] after draining.
    pub async fn receive(&mut self) -> Result<Option<ContentBlock>> {
        if self.options.auto_execute_tools() {
            if self.interrupted.load(Ordering::SeqCst) {
                self.discard_pending_output();
                return Ok(None);
            }
            if self.auto_output.is_none() {
                self.auto_output = Some(self.auto_execute_loop().await?.into_iter());
            }
            Ok(self.auto_output.as_mut().and_then(Iterator::next))
        } else {
            match self.receive_one().await? {
                Some(block) => {
                    self.manual_receive_buffer.push(block.clone());
                    Ok(Some(block))
                }
                None => {
                    if self.interrupted.load(Ordering::SeqCst) && self.current_stream.is_some() {
                        self.discard_pending_output();
                    } else if !self.manual_receive_buffer.is_empty() {
                        let blocks = std::mem::take(&mut self.manual_receive_buffer);
                        self.push_assistant(&blocks);
                    }
                    Ok(None)
                }
            }
        }
    }

    /// Requests cancellation between stream events, tool calls, and hook phases.
    /// Pending network reads still wait for data or the HTTP timeout; a running
    /// tool or hook future finishes first. A new send resets the flag.
    pub fn interrupt(&self) {
        self.interrupted.store(true, Ordering::SeqCst);
    }

    /// Shares the cancellation flag for use from another task.
    pub fn interrupt_handle(&self) -> Arc<AtomicBool> {
        self.interrupted.clone()
    }

    /// Returns conversation history, excluding the separately configured system prompt.
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Allows explicit history edits, such as [`crate::truncate_messages`].
    pub fn history_mut(&mut self) -> &mut Vec<Message> {
        &mut self.history
    }

    /// Returns the client's configuration.
    pub fn options(&self) -> &AgentOptions {
        &self.options
    }

    /// Why the last completed generation stopped, or `MaxToolIterations` when the
    /// auto loop reached its limit. Reset when a new request starts.
    pub fn finish_reason(&self) -> Option<&FinishReason> {
        self.last_finish_reason.as_ref()
    }

    /// Captured reasoning across all rounds of the latest request, when opted in.
    /// Reasoning never enters conversation history.
    pub fn reasoning(&self) -> Option<&str> {
        (!self.last_reasoning.is_empty()).then_some(self.last_reasoning.as_str())
    }
}
