impl Client {
    /// Starts a request from a text prompt, applying UserPromptSubmit hooks first.
    /// The prompt enters history before HTTP I/O, even on transport failure.
    /// Empty prompts are retained for tool continuation. Starting a new request
    /// discards pending output from the previous one.
    pub async fn send(&mut self, prompt: &str) -> Result<()> {
        use crate::hooks::UserPromptSubmitEvent;

        self.interrupted.store(false, Ordering::SeqCst);

        self.discard_pending_output();

        let mut final_prompt = prompt.to_string();
        if !self.options.hooks().user_prompt_submit.is_empty() {
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

        self.history.push(Message::user(final_prompt));

        self.start_request().await
    }
}
