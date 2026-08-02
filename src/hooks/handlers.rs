/// Type alias for PreToolUse hook handler functions.
///
/// This complex type signature enables powerful async hook functionality while maintaining
/// thread safety and zero-cost abstraction. Let's break it down:
///
/// # Type Breakdown
///
/// ```text
/// Arc<                                  // Reference counting for thread-safe sharing
///     dyn Fn(PreToolUseEvent)           // Function taking the event
///         -> Pin<Box<                    // Heap-allocated, pinned future
///             dyn Future<Output = Option<HookDecision>>  // Async result
///                 + Send                 // Can be sent across threads
///         >>
///         + Send + Sync                  // The function itself is thread-safe
/// >
/// ```
///
/// # Why This Design?
///
/// - **`Arc`**: Enables zero-cost cloning when passing hooks between threads or agent instances.
///   Multiple agents can share the same hook without duplicating memory.
///
/// - **`dyn Fn`**: Allows any function or closure to be used as a hook, as long as it matches
///   the signature. This is trait object type erasure.
///
/// - **`Pin<Box<dyn Future>>`**: Async functions in Rust return opaque Future types. We need
///   to box them for dynamic dispatch and pin them because futures may contain self-references.
///
/// - **`Send + Sync`**: Ensures the hook can be safely called from multiple threads. Essential
///   for async runtimes like Tokio that may schedule tasks on different threads.
///
/// # Return Value
///
/// Hook handlers return `Option<HookDecision>`:
/// - `None`: "I don't care, continue normally or let next hook decide"
/// - `Some(HookDecision)`: "I'm taking control" - blocks remaining hooks from running
///
/// # Example Usage
///
/// You don't typically construct these types directly. Instead, use the builder methods:
///
/// ```rust
/// use open_agent::{Hooks, PreToolUseEvent, HookDecision};
///
/// let hooks = Hooks::new().add_pre_tool_use(|event| async move {
///     // Your async logic here
///     if event.tool_name == "dangerous" {
///         Some(HookDecision::block("Not allowed"))
///     } else {
///         None
///     }
/// });
/// ```
///
/// The builder automatically wraps your closure in `Arc<...>` and handles the `Pin<Box<...>>`.
pub type PreToolUseHandler = Arc<
    dyn Fn(PreToolUseEvent) -> Pin<Box<dyn Future<Output = Option<HookDecision>> + Send>>
        + Send
        + Sync,
>;

/// Type alias for PostToolUse hook handler functions.
///
/// Identical in structure to `PreToolUseHandler` but receives `PostToolUseEvent` instead.
/// See [`PreToolUseHandler`] for detailed explanation of the type signature.
///
/// # Common Usage Pattern
///
/// PostToolUse hooks typically don't modify execution (they return `None`) but are used
/// for observation, logging, and metrics:
///
/// ```rust
/// use open_agent::{Hooks, PostToolUseEvent, HookDecision};
///
/// let hooks = Hooks::new().add_post_tool_use(|event| async move {
///     // Log tool execution for audit trail
///     println!("Tool {} completed with result: {:?}",
///              event.tool_name, event.tool_result);
///
///     // Send metrics to monitoring system
///     // metrics::counter!("tool_executions", 1, "tool" => event.tool_name);
///
///     None // Don't interfere with execution
/// });
/// ```
pub type PostToolUseHandler = Arc<
    dyn Fn(PostToolUseEvent) -> Pin<Box<dyn Future<Output = Option<HookDecision>> + Send>>
        + Send
        + Sync,
>;

/// Type alias for UserPromptSubmit hook handler functions.
///
/// Identical in structure to `PreToolUseHandler` but receives `UserPromptSubmitEvent` instead.
/// See [`PreToolUseHandler`] for detailed explanation of the type signature.
///
/// # Common Usage Pattern
///
/// UserPromptSubmit hooks are often used for content moderation and prompt enhancement:
///
/// ```rust
/// use open_agent::{Hooks, UserPromptSubmitEvent, HookDecision};
///
/// let hooks = Hooks::new().add_user_prompt_submit(|event| async move {
///     // Block inappropriate content
///     if event.prompt.to_lowercase().contains("banned_word") {
///         return Some(HookDecision::block("Content policy violation"));
///     }
///
///     // Or enhance prompts with context
///     let enhanced = format!("{}\n\nContext: Session ID 12345", event.prompt);
///     Some(HookDecision::modify_prompt(enhanced, "Added session context"))
/// });
/// ```
pub type UserPromptSubmitHandler = Arc<
    dyn Fn(UserPromptSubmitEvent) -> Pin<Box<dyn Future<Output = Option<HookDecision>> + Send>>
        + Send
        + Sync,
>;
