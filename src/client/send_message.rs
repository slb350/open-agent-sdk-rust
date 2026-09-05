impl Client {
    /// Starts a request with a pre-built message, including image content.
    /// UserPromptSubmit hooks apply only to [`Self::send`], which accepts text.
    /// The message enters history before HTTP I/O; pending output is discarded.
    pub async fn send_message(&mut self, message: Message) -> Result<()> {
        self.interrupted.store(false, Ordering::SeqCst);

        self.discard_pending_output();

        self.history.push(message);

        self.start_request().await
    }
}
