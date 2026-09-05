/// An asynchronous pre-tool handler shared through [`Arc`].
///
/// Use [`Hooks::add_pre_tool_use`] to wrap a closure automatically. The future is
/// boxed for dynamic dispatch and must be `Send`; the closure is `Send + Sync`.
pub type PreToolUseHandler = Arc<
    dyn Fn(PreToolUseEvent) -> Pin<Box<dyn Future<Output = Option<HookDecision>> + Send>>
        + Send
        + Sync,
>;

/// An asynchronous post-tool handler. See [`PreToolUseHandler`] for its shared
/// closure and future requirements.
pub type PostToolUseHandler = Arc<
    dyn Fn(PostToolUseEvent) -> Pin<Box<dyn Future<Output = Option<HookDecision>> + Send>>
        + Send
        + Sync,
>;

/// An asynchronous prompt handler. See [`PreToolUseHandler`] for its shared
/// closure and future requirements.
pub type UserPromptSubmitHandler = Arc<
    dyn Fn(UserPromptSubmitEvent) -> Pin<Box<dyn Future<Output = Option<HookDecision>> + Send>>
        + Send
        + Sync,
>;
