/// Container for registering and managing lifecycle hooks.
///
/// The `Hooks` struct stores collections of hook handlers for different lifecycle events.
/// It provides a builder pattern for registering hooks and executor methods for running them.
///
/// # Design Principles
///
/// - **Builder Pattern**: Hooks can be chained during construction using `.add_*()` methods
/// - **Multiple Hooks**: You can register multiple hooks for the same event type
/// - **Execution Order**: Hooks execute in the order they were registered (FIFO)
/// - **First Wins**: The first hook returning `Some(HookDecision)` determines the outcome
/// - **Thread Safe**: The struct is `Clone` and all handlers are `Arc`-wrapped for sharing
///
/// # Example: Building a Hooks Collection
///
/// ```rust
/// use open_agent::{Hooks, PreToolUseEvent, PostToolUseEvent, HookDecision};
///
/// let hooks = Hooks::new()
///     // First: Security gate (highest priority)
///     .add_pre_tool_use(|event| async move {
///         if event.tool_name == "dangerous" {
///             return Some(HookDecision::block("Security violation"));
///         }
///         None
///     })
///     // Second: Rate limiting
///     .add_pre_tool_use(|event| async move {
///         // Check rate limits...
///         None
///     })
///     // Audit logging (happens after execution)
///     .add_post_tool_use(|event| async move {
///         println!("Tool '{}' executed", event.tool_name);
///         None
///     });
/// ```
///
/// # Fields
///
/// - `pre_tool_use`: Handlers invoked before tool execution
/// - `post_tool_use`: Handlers invoked after tool execution
/// - `user_prompt_submit`: Handlers invoked before processing user prompts
///
/// All fields are public, allowing direct manipulation if needed, though the builder
/// methods are the recommended approach.
#[derive(Clone, Default)]
pub struct Hooks {
    /// Collection of PreToolUse hook handlers, executed in registration order
    pub pre_tool_use: Vec<PreToolUseHandler>,

    /// Collection of PostToolUse hook handlers, executed in registration order
    pub post_tool_use: Vec<PostToolUseHandler>,

    /// Collection of UserPromptSubmit hook handlers, executed in registration order
    pub user_prompt_submit: Vec<UserPromptSubmitHandler>,
}

impl Hooks {
    /// Creates a new, empty `Hooks` container.
    ///
    /// Use this as the starting point for building a hooks collection using the builder pattern.
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::Hooks;
    ///
    /// let hooks = Hooks::new()
    ///     .add_pre_tool_use(|event| async move { None });
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a PreToolUse hook handler using the builder pattern.
    ///
    /// This method takes ownership of `self` and returns it back, allowing method chaining.
    /// The handler is wrapped in `Arc` and added to the collection of PreToolUse hooks.
    ///
    /// # Parameters
    ///
    /// - `handler`: An async function or closure that takes `PreToolUseEvent` and returns
    ///   `Option<HookDecision>`. Must be `Send + Sync + 'static` for thread safety.
    ///
    /// # Type Parameters
    ///
    /// - `F`: The function/closure type
    /// - `Fut`: The future type returned by the function
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::{Hooks, HookDecision};
    ///
    /// let hooks = Hooks::new()
    ///     .add_pre_tool_use(|event| async move {
    ///         println!("About to execute: {}", event.tool_name);
    ///         None
    ///     })
    ///     .add_pre_tool_use(|event| async move {
    ///         // This runs second (if first returns None)
    ///         if event.tool_name == "blocked" {
    ///             Some(HookDecision::block("Not allowed"))
    ///         } else {
    ///             None
    ///         }
    ///     });
    /// ```
    pub fn add_pre_tool_use<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(PreToolUseEvent) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<HookDecision>> + Send + 'static,
    {
        // Wrap the user's function in Arc and Box::pin for type erasure and heap allocation
        self.pre_tool_use
            .push(Arc::new(move |event| Box::pin(handler(event))));
        self
    }

    /// Registers a PostToolUse hook handler using the builder pattern.
    ///
    /// Identical to `add_pre_tool_use` but for PostToolUse events. See [`Self::add_pre_tool_use`]
    /// for detailed documentation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::Hooks;
    ///
    /// let hooks = Hooks::new()
    ///     .add_post_tool_use(|event| async move {
    ///         // Audit log all tool executions
    ///         println!("Tool '{}' completed: {:?}",
    ///                  event.tool_name, event.tool_result);
    ///         None // Don't interfere with execution
    ///     });
    /// ```
    pub fn add_post_tool_use<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(PostToolUseEvent) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<HookDecision>> + Send + 'static,
    {
        // Wrap the user's function in Arc and Box::pin for type erasure and heap allocation
        self.post_tool_use
            .push(Arc::new(move |event| Box::pin(handler(event))));
        self
    }

    /// Registers a UserPromptSubmit hook handler using the builder pattern.
    ///
    /// Identical to `add_pre_tool_use` but for UserPromptSubmit events. See [`Self::add_pre_tool_use`]
    /// for detailed documentation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::{Hooks, HookDecision};
    ///
    /// let hooks = Hooks::new()
    ///     .add_user_prompt_submit(|event| async move {
    ///         // Content moderation
    ///         if event.prompt.contains("forbidden") {
    ///             Some(HookDecision::block("Content violation"))
    ///         } else {
    ///             None
    ///         }
    ///     });
    /// ```
    pub fn add_user_prompt_submit<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(UserPromptSubmitEvent) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<HookDecision>> + Send + 'static,
    {
        // Wrap the user's function in Arc and Box::pin for type erasure and heap allocation
        self.user_prompt_submit
            .push(Arc::new(move |event| Box::pin(handler(event))));
        self
    }

    /// Executes all registered PreToolUse hooks in order and returns the first decision.
    ///
    /// This method implements the **"first non-None wins"** execution model:
    ///
    /// 1. Iterates through hooks in registration order (FIFO)
    /// 2. Calls each hook with the same event snapshot
    /// 3. If a hook returns `Some(decision)`, immediately returns that decision
    /// 4. Remaining hooks are **not executed**
    /// 5. If all hooks return `None`, returns `None`
    ///
    /// # Parameters
    ///
    /// - `event`: The PreToolUseEvent to pass to each hook
    ///
    /// # Returns
    ///
    /// - `Some(HookDecision)`: A hook made a decision (block, modify, or continue)
    /// - `None`: All hooks returned `None` (continue normally)
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::{Hooks, PreToolUseEvent, HookDecision};
    /// use serde_json::json;
    ///
    /// # async fn example() {
    /// let hooks = Hooks::new()
    ///     .add_pre_tool_use(|e| async move { None }) // Runs first
    ///     .add_pre_tool_use(|e| async move {
    ///         Some(HookDecision::block("Blocked")) // Runs second, blocks
    ///     })
    ///     .add_pre_tool_use(|e| async move {
    ///         None // NEVER runs because previous hook returned Some
    ///     });
    ///
    /// let event = PreToolUseEvent::new(
    ///     "test".to_string(),
    ///     json!({}),
    ///     "id".to_string(),
    ///     vec![]
    /// );
    ///
    /// let decision = hooks.execute_pre_tool_use(event).await;
    /// assert!(decision.is_some());
    /// assert!(!decision.unwrap().continue_execution());
    /// # }
    /// ```
    pub async fn execute_pre_tool_use(&self, event: PreToolUseEvent) -> Option<HookDecision> {
        let (last, preceding) = self.pre_tool_use.split_last()?;
        for handler in preceding {
            let decision = handler(event.clone()).await;
            if decision.is_some() {
                return decision;
            }
        }
        last(event).await
    }

    /// Executes all registered PostToolUse hooks in order and returns the first decision.
    ///
    /// Identical in behavior to [`Self::execute_pre_tool_use`] but for PostToolUse events.
    /// See that method for detailed documentation of the execution model.
    ///
    /// # Note
    ///
    /// PostToolUse hooks rarely return decisions in practice. They're primarily used for
    /// observation (logging, metrics) and typically always return `None`.
    pub async fn execute_post_tool_use(&self, event: PostToolUseEvent) -> Option<HookDecision> {
        let (last, preceding) = self.post_tool_use.split_last()?;
        for handler in preceding {
            let decision = handler(event.clone()).await;
            if decision.is_some() {
                return decision;
            }
        }
        last(event).await
    }

    /// Executes all registered UserPromptSubmit hooks in order and returns the first decision.
    ///
    /// Identical in behavior to [`Self::execute_pre_tool_use`] but for UserPromptSubmit events.
    /// See that method for detailed documentation of the execution model.
    pub async fn execute_user_prompt_submit(
        &self,
        event: UserPromptSubmitEvent,
    ) -> Option<HookDecision> {
        let (last, preceding) = self.user_prompt_submit.split_last()?;
        for handler in preceding {
            let decision = handler(event.clone()).await;
            if decision.is_some() {
                return decision;
            }
        }
        last(event).await
    }
}

/// Custom Debug implementation for Hooks.
///
/// Since hook handlers are closures (which don't implement Debug), we provide a custom
/// implementation that shows the number of registered handlers instead of trying to
/// debug-print the closures themselves.
///
/// # Example Output
///
/// ```text
/// Hooks {
///     pre_tool_use: 3 handlers,
///     post_tool_use: 1 handlers,
///     user_prompt_submit: 2 handlers
/// }
/// ```
impl std::fmt::Debug for Hooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hooks")
            .field(
                "pre_tool_use",
                &format!("{} handlers", self.pre_tool_use.len()),
            )
            .field(
                "post_tool_use",
                &format!("{} handlers", self.post_tool_use.len()),
            )
            .field(
                "user_prompt_submit",
                &format!("{} handlers", self.user_prompt_submit.len()),
            )
            .finish()
    }
}

/// String constant for the PreToolUse hook event name.
///
/// This constant can be used for logging, metrics, or when you need a string
/// representation of the hook type. It's primarily used internally but is exposed
/// as part of the public API for consistency.
pub const HOOK_PRE_TOOL_USE: &str = "pre_tool_use";

/// String constant for the PostToolUse hook event name.
///
/// See [`HOOK_PRE_TOOL_USE`] for usage details.
pub const HOOK_POST_TOOL_USE: &str = "post_tool_use";

/// String constant for the UserPromptSubmit hook event name.
///
/// See [`HOOK_PRE_TOOL_USE`] for usage details.
pub const HOOK_USER_PROMPT_SUBMIT: &str = "user_prompt_submit";
