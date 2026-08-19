impl Client {
    async fn receive_one(&mut self) -> Result<Option<ContentBlock>> {
        // Check interrupt flag before attempting to receive
        // Uses SeqCst to ensure we see the latest value from any thread
        if self.interrupted.load(Ordering::SeqCst) {
            // Return None but leave current_stream intact so callers can
            // distinguish "interrupted a live stream" (current_stream is Some)
            // from "interrupt after stream already ended" (current_stream is None).
            return Ok(None);
        }

        // Drain non-content events (finish reason, reasoning) into client state and keep
        // polling, so `receive()` continues to hand the caller content blocks only.
        loop {
            // No active stream
            let Some(stream) = &mut self.current_stream else {
                return Ok(None);
            };

            match stream.next().await {
                Some(Ok(StreamEvent::Block(block))) => return Ok(Some(block)),
                Some(Ok(StreamEvent::Reasoning(reasoning))) => {
                    // Recorded for `reasoning()`, deliberately never pushed to history.
                    // Traces run to tens of kilobytes, so take the buffer by move when there
                    // is nothing to append to — which is every stream outside the auto loop.
                    if self.last_reasoning.is_empty() {
                        self.last_reasoning = reasoning;
                    } else {
                        self.last_reasoning.push_str(&reasoning);
                    }
                }
                Some(Ok(StreamEvent::Finish(reason))) => {
                    self.last_finish_reason = Some(reason);
                }
                Some(Err(e)) => return Err(e),
                None => {
                    // Natural EOF — mark stream as fully consumed
                    self.current_stream = None;
                    return Ok(None);
                }
            }
        }
    }

    /// Collects all blocks from the current stream into a vector.
    ///
    /// Internal helper for auto-execution mode. This method buffers the entire
    /// response in memory, which is necessary to determine if the response contains
    /// tool calls before returning anything to the caller.
    ///
    /// # Returns
    ///
    /// - `Ok(vec)`: Successfully collected all blocks
    /// - `Err(e)`: Error during collection or interrupted
    ///
    /// # Memory Usage
    ///
    /// This buffers the entire response, which can be large for long completions.
    /// Consider the memory implications when using auto-execution mode.
    ///
    /// # Interruption
    ///
    /// Checks interrupt flag during collection and returns error if interrupted.
    async fn collect_all_blocks(&mut self) -> Result<Vec<ContentBlock>> {
        let mut blocks = Vec::new();

        // Consume entire stream into vector
        while let Some(block) = self.receive_one().await? {
            // Check interrupt during collection for responsiveness
            if self.interrupted.load(Ordering::SeqCst) {
                self.current_stream = None;
                return Err(Error::other(
                    "Operation interrupted during block collection",
                ));
            }

            blocks.push(block);
        }

        Ok(blocks)
    }

    /// Executes a tool by name with the given input.
    ///
    /// Internal helper for auto-execution mode. Looks up the tool in the registered
    /// tools list and executes it with the provided input.
    ///
    /// # Parameters
    ///
    /// - `tool_name`: Name of the tool to execute
    /// - `input`: JSON value containing tool parameters
    ///
    /// # Returns
    ///
    /// - `Ok(result)`: Tool executed successfully, returns result as JSON
    /// - `Err(e)`: Tool not found or execution failed
    ///
    /// # Error Handling
    ///
    /// If the tool is not found in the registry, returns a ToolError.
    /// If execution fails, the error from the tool is propagated.
    async fn execute_tool_internal(
        &self,
        tool_name: &str,
        input: serde_json::Value,
    ) -> Result<serde_json::Value> {
        // Find tool in registered tools by name
        let tool = self
            .options
            .tools()
            .iter()
            .find(|t| t.name() == tool_name)
            .ok_or_else(|| Error::tool(format!("Tool '{}' not found", tool_name)))?;

        // Execute the tool's async function
        tool.execute(input).await
    }

    /// Auto-execution loop that handles tool calls automatically.
    ///
    /// This is the core implementation of automatic tool execution mode. It:
    ///
    /// 1. Collects all blocks from the current stream
    /// 2. Separates text blocks from tool use blocks
    /// 3. If there are tool blocks:
    ///    - Executes PreToolUse hooks (can modify/block)
    ///    - Executes each tool via its registered function
    ///    - Executes PostToolUse hooks (can modify result)
    ///    - Adds results to history
    ///    - Continues conversation with send("")
    /// 4. Repeats until text-only response or max iterations
    /// 5. Returns all final text blocks
    ///
    /// # Returns
    ///
    /// - `Ok(blocks)`: Final text blocks after all tool iterations
    /// - `Err(e)`: Error during execution, stream processing, or interruption
    ///
    /// # Iteration Limit
    ///
    /// The loop is bounded by `options.max_tool_iterations` to prevent infinite loops.
    /// When the limit is reached, the loop stops and returns whatever text blocks
    /// have been collected so far.
    ///
    /// # Hook Integration
    ///
    /// Hooks are executed for each tool call:
    /// - **PreToolUse**: Can modify input or block execution entirely
    /// - **PostToolUse**: Can modify the result before it's added to history
    ///
    /// If a hook blocks execution, a JSON error response is used as the tool result.
    ///
    /// # State Management
    ///
    /// The loop maintains history by adding:
    /// - Assistant messages with text + tool use blocks
    /// - User messages with tool result blocks
    ///
    /// This creates a proper conversation flow that the model can follow.
    ///
    /// # Error Recovery
    ///
    /// If a tool execution fails, the error is converted to a JSON error response
    /// and added as the tool result. This allows the conversation to continue
    /// and lets the model handle the error.
    async fn auto_execute_loop(&mut self) -> Result<Vec<ContentBlock>> {
        use crate::types::ToolResultBlock;

        // Track iterations to prevent infinite loops
        let mut iteration = 0;
        let max_iterations = self.options.max_tool_iterations();

        loop {
            // ========================================================================
            // STEP 1: Collect all blocks from current stream
            // ========================================================================
            // Buffer the entire response to determine if it contains tool calls
            let blocks = self.collect_all_blocks().await?;

            // Empty response means stream ended or was interrupted
            if blocks.is_empty() {
                return Ok(Vec::new());
            }

            // ========================================================================
            // STEP 2: Separate text blocks from tool use blocks
            // ========================================================================
            // The model can return a mix of text and tool calls in one response
            let mut text_blocks = Vec::new();
            let mut tool_blocks = Vec::new();

            for block in blocks {
                match block {
                    ContentBlock::Text(_) => text_blocks.push(block),
                    ContentBlock::ToolUse(_) => tool_blocks.push(block),
                    ContentBlock::ToolResult(_) | ContentBlock::Image(_) => {} // Ignore ToolResult and Image variants
                }
            }

            // ========================================================================
            // STEP 3: Check if we're done (no tool calls)
            // ========================================================================
            // If the response contains no tool calls, we've reached the final answer
            if tool_blocks.is_empty() {
                // Add assistant's final text response to history
                if !text_blocks.is_empty() {
                    let assistant_msg = Message::assistant(text_blocks.clone());
                    self.history.push(assistant_msg);
                }
                // Return text blocks to caller via buffered receive()
                return Ok(text_blocks);
            }

            // ========================================================================
            // STEP 4: Check iteration limit BEFORE executing tools
            // ========================================================================
            // Increment counter and check if we've hit the max
            iteration += 1;
            if iteration > max_iterations {
                // Max iterations reached - stop execution and return what we have
                // This prevents infinite tool-calling loops.
                //
                // The SDK, not the model, ended this operation. The last stream's reason was
                // `ToolCalls` — true of that generation, but misleading as the answer to
                // "why did this stop?", which is the question `finish_reason()` exists to
                // answer. Report the reason we are responsible for.
                self.last_finish_reason = Some(FinishReason::MaxToolIterations);
                if !text_blocks.is_empty() {
                    let assistant_msg = Message::assistant(text_blocks.clone());
                    self.history.push(assistant_msg);
                }
                return Ok(text_blocks);
            }

            // ========================================================================
            // STEP 5: Add assistant message to history
            // ========================================================================
            // The assistant message includes BOTH text and tool use blocks
            // This preserves the full context for future turns
            let mut all_blocks = text_blocks.clone();
            all_blocks.extend(tool_blocks.clone());
            let assistant_msg = Message::assistant(all_blocks);
            self.history.push(assistant_msg);

            // ========================================================================
            // STEP 6: Execute all tools and collect results
            // ========================================================================
            for block in tool_blocks {
                if let ContentBlock::ToolUse(tool_use) = block {
                    let has_pre_hooks = !self.options.hooks().pre_tool_use.is_empty();
                    let has_post_hooks = !self.options.hooks().post_tool_use.is_empty();
                    let mut history_snapshot = if has_pre_hooks || has_post_hooks {
                        Some(serialize_history_snapshot(&self.history)?)
                    } else {
                        None
                    };

                    // ============================================================
                    // Execute PreToolUse hooks
                    // ============================================================
                    use crate::hooks::PreToolUseEvent;
                    // Track whether to execute and what input to use
                    let mut tool_input = tool_use.input().clone();
                    let mut should_execute = true;
                    let mut block_reason = None;

                    if has_pre_hooks {
                        let pre_history = if has_post_hooks {
                            history_snapshot
                                .as_ref()
                                .expect("hook history snapshot should exist")
                                .clone()
                        } else {
                            history_snapshot
                                .take()
                                .expect("hook history snapshot should exist")
                        };
                        let pre_event = PreToolUseEvent::new(
                            tool_use.name().to_string(),
                            tool_use.input().clone(),
                            tool_use.id().to_string(),
                            pre_history,
                        );

                        if let Some(decision) =
                            self.options.hooks().execute_pre_tool_use(pre_event).await
                        {
                            if !decision.continue_execution() {
                                should_execute = false;
                                block_reason = decision.reason().map(ToString::to_string);
                            } else if let Some(modified) = decision.modified_input() {
                                tool_input = modified.clone();
                            }
                        }
                    }

                    // ============================================================
                    // Execute tool (or create error result if blocked)
                    // ============================================================
                    let result = if should_execute {
                        // Actually execute the tool
                        match self
                            .execute_tool_internal(tool_use.name(), tool_input.clone())
                            .await
                        {
                            Ok(res) => res, // Success - use the result
                            Err(e) => {
                                // Tool execution failed - convert to JSON error
                                // This allows the conversation to continue
                                serde_json::json!({
                                    "error": e.to_string(),
                                    "tool": tool_use.name(),
                                    "id": tool_use.id()
                                })
                            }
                        }
                    } else {
                        // Tool blocked by PreToolUse hook - create error result
                        serde_json::json!({
                            "error": "Tool execution blocked by hook",
                            "reason": block_reason.unwrap_or_else(|| "No reason provided".to_string()),
                            "tool": tool_use.name(),
                            "id": tool_use.id()
                        })
                    };

                    // ============================================================
                    // Execute PostToolUse hooks
                    // ============================================================
                    let mut final_result = result;
                    if has_post_hooks {
                        use crate::hooks::PostToolUseEvent;
                        let pending_result = Message::user_with_blocks(vec![
                            ContentBlock::ToolResult(ToolResultBlock::new(
                                tool_use.id(),
                                final_result.clone(),
                            )),
                        ]);
                        let mut post_history_snapshot = history_snapshot
                            .take()
                            .expect("hook history snapshot should exist");
                        post_history_snapshot.push(serialize_history_message(&pending_result)?);
                        let post_event = PostToolUseEvent::new(
                            tool_use.name().to_string(),
                            tool_input,
                            tool_use.id().to_string(),
                            final_result.clone(),
                            post_history_snapshot,
                        );

                        if let Some(decision) = self
                            .options
                            .hooks()
                            .execute_post_tool_use(post_event)
                            .await
                        {
                            // modified_input is historical naming for result replacement.
                            if let Some(modified) = decision.modified_input() {
                                final_result = modified.clone();
                            }
                        }
                    }

                    // ============================================================
                    // Add tool result to history
                    // ============================================================
                    // Tool results are added as user messages (per OpenAI convention)
                    let tool_result = ToolResultBlock::new(tool_use.id(), final_result);
                    let tool_result_msg =
                        Message::user_with_blocks(vec![ContentBlock::ToolResult(tool_result)]);
                    self.history.push(tool_result_msg);
                }
            }

            // ========================================================================
            // STEP 7: Continue conversation to get next response
            // ========================================================================
            // Send empty string to continue - the history contains all context.
            //
            // `send()` starts a new stream, which resets the per-stream observations. That is
            // right for the finish reason (the last round's is the one that matters) but wrong
            // for reasoning: the deliberation that chose these tools is exactly the part worth
            // keeping, so carry it across the boundary and let the next stream append to it.
            let carried_reasoning = std::mem::take(&mut self.last_reasoning);
            self.send("").await?;
            self.last_reasoning = carried_reasoning;

            // Loop continues to collect and process the next response
            // This will either be more tool calls or the final text answer
        }
    }

}
