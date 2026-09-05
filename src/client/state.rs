/// A reusable conversation client with optional automatic tool execution.
///
/// Manual mode streams blocks as they arrive. Auto mode runs tool rounds first,
/// then yields the final response. Use [`Client::interrupt_handle`] for cancellation
/// from another task without locking the client across an await.
pub struct Client {
    options: AgentOptions,
    history: Vec<Message>,
    current_stream: Option<EventStream>,
    http_client: reqwest::Client,
    interrupted: Arc<AtomicBool>,
    // None means this request has not run the auto loop; an exhausted iterator
    // records completion without retaining or cloning delivered blocks.
    auto_output: Option<std::vec::IntoIter<ContentBlock>>,
    manual_receive_buffer: Vec<ContentBlock>,
    last_finish_reason: Option<FinishReason>,
    last_reasoning: String,
}
