# Open Agent SDK (Rust)

## Project Description

A lightweight Rust SDK (v0.11.3 source) for building AI agents with local or cloud LLMs. Speaks two wire protocols: OpenAI chat completions and Anthropic messages, selected per endpoint with `ApiProtocol`. Rust port of the Python open-agent-sdk. Published to crates.io as `open-agent-sdk`.

## Repository Structure

```text
open-agent-sdk-rust/
├── src/
│   ├── client.rs      # Public client module docs/imports and fragment orchestration
│   ├── client/        # Query, send, request assembly, streaming, receive, history, state, and tests
│   ├── config.rs      # Provider enum, get_base_url(), get_model() helpers
│   ├── context.rs     # estimate_tokens(), is_approaching_limit(), truncate_messages()
│   ├── error.rs       # Error type and Result alias
│   ├── hooks.rs       # Public hooks module orchestration
│   ├── hooks/         # Events, decisions, handlers, registry, and tests
│   ├── lib.rs         # Public exports and prelude module
│   ├── retry.rs       # RetryConfig, retry_with_backoff, retry_with_backoff_conditional, is_retryable_error
│   ├── retry/         # Retry unit tests
│   ├── tools.rs       # Public tool module orchestration
│   ├── tools/         # Tool, schema, builder, handler, factory, and tests
│   ├── types.rs       # Public core-type module orchestration
│   ├── types/         # Options, messages, images, wire types (openai.rs + openai_stream.rs,
│   │                  # anthropic.rs + anthropic_stream.rs), protocol.rs (ApiProtocol),
│   │                  # validated newtypes, HTTP-header handling, stream_event.rs (real mods);
│   │                  # tests/ subdir
│   ├── utils.rs       # Declares the utils submodules and re-exports their public items
│   └── utils/         # coalesce.rs (text-block joining for history),
│                      # accumulator.rs (StreamAccumulator) and anthropic_accumulator.rs
│                      # (AnthropicAccumulator), both decoding into buffers.rs
│                      # (StreamBuffers, the shared drain), driver.rs (EventAccumulator +
│                      # drive) and sse.rs (both SSE parsers), all real mods
├── examples/
│   ├── simple_query.rs              # Minimal streaming query
│   ├── anthropic_query.rs           # Anthropic messages endpoint via ApiProtocol
│   ├── calculator_tools.rs          # Manual tool execution
│   ├── auto_execution_demo.rs       # Automatic tool execution
│   ├── multi_tool_agent.rs          # Production agent: 5 tools + hooks
│   ├── hooks_example.rs             # Hook patterns (security gate, audit log)
│   ├── context_management.rs        # History management patterns
│   ├── interrupt_demo.rs            # Timeout and cancellation patterns
│   ├── git_commit_agent.rs          # Production: git diff → commit message
│   ├── log_analyzer_agent.rs        # Production: log file analysis
│   ├── advanced_patterns.rs         # Retry logic, concurrent requests
│   ├── vision_example.rs            # Multimodal: URLs, local files, base64
│   ├── vision_api_demo.rs           # Vision API walkthrough
│   └── test_tool_serialization.rs   # Tool call serialization verification
├── benches/
│   └── performance.rs               # Criterion benchmarks
├── tests/                         # Loopback protocol, lifecycle, validation, and infrastructure tests
│   ├── regression_client_lifecycle_test.rs # Repeated/abandoned requests, interruption, history reset
│   ├── hooks_integration_test.rs   # Real auto-tool and hook outcomes
│   ├── send_message_test.rs        # Text/image request-body and failed-send history checks
│   └── common/mod.rs              # Shared loopback-server and stream helpers
├── scripts/
│   ├── mutants-ci-scope.sh          # Classifies mutation work for complete CI diffs
│   ├── mutants-common.sh            # The one definition of the results directory
│   ├── mutants-run.sh               # Owns the verdict (missed.txt); called by the hook and CI
│   ├── mutants-remote.sh            # rsync + ssh to strix, mirrors results back, falls back loudly
│   └── mutants-staged.sh            # Staged-diff scope, through mutants-remote.sh
├── .githooks/
│   └── pre-commit                   # fmt, clippy, tests, and a --in-diff cargo-mutants sweep
├── .github/
│   ├── dependabot.yml               # Grouped weekly Cargo dependency updates
│   └── workflows/
│       ├── ci.yml                   # GitHub CI (fmt, clippy, MSRV, Linux/macOS stable + beta matrix, security audit, mutation sweep, docs, LLVM Tarpaulin coverage, benchmarks)
│       └── scheduled-audit.yml      # Scheduled dependency audit
├── .markdownlint.json               # Markdown lint rules (disable MD013, allow duplicate sibling headings)
├── Cargo.toml
├── Cargo.lock
├── CHANGELOG.md
└── README.md
```

## Tech Stack

| Component | Technology |
| ----------- | ------------ |
| **Language** | Rust 1.85+ |
| **Async runtime** | Tokio 1.50+ |
| **HTTP** | reqwest 0.12.28+ |
| **Serialization** | serde, serde_json 1.0 |
| **Streams** | futures 0.3.32+ |
| **SSE Parsing** | eventsource-stream 0.2 |
| **Error handling** | thiserror 2.0 |
| **Logging** | log 0.4.29+ |
| **Retry jitter** | rand 0.10 |
| **Base64** | base64 0.23 (no `simd-unsafe` feature) |
| **Benchmarks** | criterion |
| **HTTP mocking (dev)** | wiremock =0.6.4 (0.6.5 breaks MSRV) |
| **Virtual clock (dev)** | tokio `test-util` |
| **Mutation testing** | cargo-mutants 27.1.0 |
| **Automation** | Dependabot (grouped weekly Cargo and Actions updates; wiremock 0.6.5 ignored for MSRV) |

## Common Commands

```bash
# Build
cargo build

# Run tests (all: unit + integration + doctests)
cargo test

# Run tests with output
cargo test -- --nocapture

# Run a specific test
cargo test test_agent_options_builder

# Run an example
cargo run --example git_commit_agent
cargo run --example vision_example

# Run benchmarks
cargo bench

# Mutation sweep (offloaded to strix.local; falls back locally with a warning)
./scripts/mutants-remote.sh              # full sweep
./scripts/mutants-staged.sh              # staged diff only, what the hook runs
DREP_MUTANTS_REMOTE=0 ./scripts/mutants-run.sh   # force it onto this machine

# Lint
cargo clippy -- -D warnings

# Format
cargo fmt
```

## Public API

### AgentOptions (builder pattern)

```rust
use open_agent::AgentOptions;

let options = AgentOptions::builder()
    .system_prompt("You are a helpful assistant")  // optional
    .model("qwen2.5-32b-instruct")                 // required
    .base_url("http://localhost:1234/v1")           // required
    .tool(my_tool)                                  // add one tool
    .tools(vec![tool_a, tool_b])                    // add multiple tools
    .hooks(hooks)                                   // lifecycle hooks
    .auto_execute_tools(true)                       // auto-execute (default: false)
    .max_tool_iterations(10)                        // safety limit (auto mode)
    .max_tokens(4096)                               // tokens to generate (unset: server decides)
    .include_reasoning(true)                        // emit StreamEvent::Reasoning (default: false)
    .temperature(0.7)                               // sampling temperature
    .timeout(60)                                    // request timeout (seconds)
    .api_key("not-needed")                          // API key (default: "not-needed")
    .header("X-Title", "Example agent")             // caller HTTP header; repeat to replace
    .build()?;
```

### query() — single-turn

Yields `StreamEvent`, not `ContentBlock`. Every stream ends with exactly one
`StreamEvent::Finish`.

```rust
use open_agent::{query, ContentBlock, FinishReason, StreamEvent};
use futures::StreamExt;

let mut stream = query("prompt", &options).await?;
while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::Block(ContentBlock::Text(t)) => print!("{}", t.text),
        StreamEvent::Finish(FinishReason::Length) => eprintln!("truncated at the token cap"),
        _ => {}
    }
}
```

`StreamEvent` variants: `Block(ContentBlock)`, `Reasoning(String)`, `Finish(FinishReason)`.
`FinishReason` variants: `Stop`, `Length`, `ToolCalls`, `ContentFilter`, `Other(String)`,
`MaxToolIterations`, `Unspecified`. Both are `#[non_exhaustive]`. `StreamEvent::into_block()` is the one-line
migration from the pre-0.8.0 `ContentStream` (now `EventStream`).

### Client — multi-turn

```rust
use open_agent::{Client, ContentBlock};

let mut client = Client::new(options)?;
client.send("What's 2 + 2?").await?;

while let Some(block) = client.receive().await? {
    match block {
        ContentBlock::Text(t) => println!("{}", t.text),
        ContentBlock::ToolUse(tu) => { /* handle or let auto-execute handle it */ }
        _ => {}
    }
}

// Additional Client methods
client.send_message(msg).await?;   // send a pre-built Message (e.g. with images)
client.clear_history();            // clear conversation and pending output; retain options
client.options();                  // access AgentOptions
client.get_tool("name");           // look up a registered tool
client.interrupt();                // cancel in-flight stream (idempotent)
client.interrupt_handle();         // Arc<AtomicBool> for cross-task cancellation
client.history();                  // &[Message] — read conversation history
client.history_mut();              // &mut Vec<Message> — mutate (e.g. truncate)
client.finish_reason();            // Option<&FinishReason> — why the last stream stopped
client.reasoning();                // Option<&str> — reasoning from the last stream (opt-in)
```

`finish_reason()` and `reasoning()` are cleared by `start_request()`, so read them after the
receive loop drains, not during it. In auto-execution mode `finish_reason()` reports the final
generation of the tool loop, or `MaxToolIterations` when the loop itself stopped the run;
`reasoning()` accumulates across every round (`auto_execute_loop` carries it over the internal
`send("")`).

For cancellation from another task, share `client.interrupt_handle()` and set the
atomic flag. Never place `Client` behind a synchronous mutex whose guard can be
held across `receive().await`; that pattern deadlocks the cancellation task.

### tool() builder

```rust
use open_agent::tool;
use serde_json::json;

let my_tool = tool("add", "Add two numbers")
    .param("a", "number")
    .param("b", "number")
    .build(|args| async move {
        let result = args["a"].as_f64().unwrap_or(0.0) + args["b"].as_f64().unwrap_or(0.0);
        Ok(json!({"result": result}))
    });

// Full JSON Schema control:
let my_tool = tool("name", "desc")
    .schema(json!({ "type": "object", "properties": { "x": { "type": "number" } }, "required": ["x"] }))
    .build(|args| async move { Ok(json!({})) });
```

### Hooks

```rust
use open_agent::{Hooks, HookDecision};

let hooks = Hooks::new()
    .add_pre_tool_use(|event| async move {
        if event.tool_name == "dangerous_op" {
            return Some(HookDecision::block("requires approval"));
        }
        Some(HookDecision::continue_())
    })
    .add_post_tool_use(|event| async move {
        println!("ran: {} -> {:?}", event.tool_name, event.tool_result);
        None
    })
    .add_user_prompt_submit(|event| async move {
        // Modify or block the prompt before it's sent
        Some(HookDecision::continue_())
    });
```

`HookDecision` variants: `continue_()`, `block(reason)`, `modify_input(json, reason)`, `modify_prompt(text, reason)`.

Hooks run sequentially; first non-None decision wins. Hook name constants: `HOOK_PRE_TOOL_USE`, `HOOK_POST_TOOL_USE`, `HOOK_USER_PROMPT_SUBMIT`.

Every hook event carries `event.history: Vec<serde_json::Value>` — a structured JSON snapshot of the conversation at that lifecycle point. Pre-tool snapshots reflect history before the call; post-tool snapshots include the completed call and its unmodified result.

### Multimodal Vision

```rust
use open_agent::{Message, ContentBlock, ImageBlock, ImageDetail};

// From URL
let msg = Message::user_with_image("Describe this", "https://example.com/photo.jpg")?;

// From local file (auto-encodes to base64; supports jpg, png, gif, webp, bmp, svg)
let msg = Message::new(MessageRole::User, vec![
    ContentBlock::Text(TextBlock::new("Describe this photo")),
    ContentBlock::Image(ImageBlock::from_file_path("/path/to/photo.jpg")?),
]);

// From base64
let msg = Message::user_with_base64_image("Describe this", base64_data, "image/png")?;

// Control detail level
let msg = Message::user_with_image_detail("Analyze", url, ImageDetail::High)?;
```

`ImageBlock` sources: `from_url(url)`, `from_file_path(path)`, `from_base64(data, mime)`. Detail levels: `Low`, `High`, `Auto`.

### Context Management

```rust
use open_agent::{estimate_tokens, is_approaching_limit, truncate_messages};

let tokens = estimate_tokens(client.history());
if is_approaching_limit(client.history(), 32000, 0.8) {
    let truncated = truncate_messages(client.history(), 10, true); // keep last 10, preserve system
    *client.history_mut() = truncated;
}
```

Context management is **opt-in** — the SDK never silently mutates history.

### Retry

```rust
use open_agent::retry::{RetryConfig, retry_with_backoff, retry_with_backoff_conditional, is_retryable_error};

let config = RetryConfig::default()  // 3 attempts, exponential backoff
    .max_attempts(5)
    .initial_delay_ms(100)
    .max_delay_ms(5000)
    .backoff_multiplier(2.0);

let result = retry_with_backoff(config, || async { some_op().await }).await?;
```

### Provider Helpers

```rust
use open_agent::{Provider, get_base_url, get_model};

let url = get_base_url(Some(Provider::LMStudio), None);  // http://localhost:1234/v1
let url = get_base_url(Some(Provider::Ollama), None);    // http://localhost:11434/v1
let model = get_model(None, true);                        // reads OPEN_AGENT_MODEL env var
```

### Error and Result Types

```rust
use open_agent::{Error, Result};
```

### Prelude

```rust
use open_agent::prelude::*;  // imports the most commonly used types
```

## Supported Providers

Two protocols, selected per endpoint with `AgentOptions::builder().protocol(..)`.

`ApiProtocol::OpenAiChat` (default) — `POST {base_url}/chat/completions`, bearer auth:

- LM Studio: `http://localhost:1234/v1`
- Ollama: `http://localhost:11434/v1`
- llama.cpp server (OpenAI-compatible mode)
- vLLM, Text Generation WebUI
- Local gateways proxying cloud models (e.g. kimi-k2, deepseek, qwen3-coder via Ollama)
- Cloud vendors: OpenAI, OpenRouter, z.ai (`https://api.z.ai/api/coding/paas/v4`)

`ApiProtocol::Anthropic` — `POST {base_url}/messages`, `x-api-key` plus `anthropic-version`:

- Anthropic
- Moonshot Kimi for Coding: `https://api.kimi.com/coding/v1`
- MiniMax: `https://api.minimax.io/anthropic/v1`

Third-party Anthropic-compatible endpoints are not uniform. `api.kimi.com/coding/v1`
**requires** `max_tokens` and answers a bare `invalid_request_error` 400 without it, and
rejects any `temperature` but 1. Both are the caller's to set; the SDK invents neither.

## Validation

Run `cargo test` for current unit, integration, and doctest results. Tests should
exercise observable behavior: captured HTTP requests, stream events, hook outcomes,
and real script effects. Do not add construction-only tests that claim integration
coverage, copied implementations, or source-substring checks for executable policy.

```bash
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
cargo test --test mutation_ci_scope_test
./scripts/mutants-remote.sh
```

## Development Rules

- Client pending output has one reset path used by `send`, `send_message`, and `clear_history`. Auto output uses an owning iterator; exhausted output must never suppress a later request or survive interruption.
- `src/client/request.rs` is a real module shared by query and Client request assembly so this logic is visible to mutation testing. Keep protocol translation at `stream_request`.
- Client stream errors close pending transport and discard partial history in both modes; later receives cannot resume a broken turn. Automatic tool loops honor cancellation from hooks and tools before starting further work or continuation requests. Completed tool results remain in history, and skipped calls receive explicit cancellation results so the next caller-initiated send has valid call/result pairs.
- `add_tool_result` retains structured `ToolResultBlock` history, including the call ID; never convert it to a plain text block before request translation.
- `max_turns` is a retained, inert compatibility value. It does not limit Client conversations or tool rounds; `max_tool_iterations` controls automatic tool rounds. Do not silently implement a new turn limit or remove the public setter/getter in a cleanup.
- Library Tokio features are minimal; test and example runtime features belong in dev-dependencies. Do not reintroduce unused direct dependencies.
- Package verification is a repository-only test: exclude `tests/package_manifest_test.rs` from crate archives. Validate an unpacked archive's test suite with a separate target directory so Cargo cannot reuse binaries carrying another checkout's `CARGO_MANIFEST_DIR`.
- Token estimates count tool JSON through its existing `Display` serializer into a byte counter. Preserve JSON escaping and UTF-8 byte lengths without allocating strings solely to measure them.

- **TDD**: Write failing tests first, implement, refactor, commit
- **All tests must pass** before committing — `cargo test` must be green
- **Clippy clean**: `cargo clippy -- -D warnings` must pass with zero warnings
- **Formatted**: `cargo fmt --check` must pass
- **No breaking changes** to `AgentOptions` field order or builder method signatures
- **`auto_execute_tools` defaults to `false`** (backwards compatibility)
- **Context management is opt-in** — never silently mutate history
- **`add_tool_result()` is sync** — no await needed
- Hook event histories are structured JSON snapshots of internal `Message` values; never substitute placeholder objects. `PostToolUseEvent` includes the pending unmodified tool result.
- SSE parsing must buffer across arbitrary reqwest transport chunks and emit every complete event through `eventsource-stream`; never parse `bytes_stream()` chunks as self-contained SSE messages.
- The stream driver must signal end-of-transport to `StreamAccumulator` and call `finalize()`. `finish_reason` is optional in practice — llama.cpp, vLLM, and several local gateways never send it — and without the end-of-stream drain their content is discarded with no error. `flush()` is idempotent after a `finish_reason` flush, so it never double-emits.
- Exactly one `StreamEvent::Finish` is emitted per stream, and it is the last event. `process_chunk` records the first `finish_reason` it sees but never emits the event; only `finalize()` does. A server that never reports one yields `FinishReason::Unspecified`, which must stay distinct from `FinishReason::Stop` — conflating them claims knowledge the SDK does not have.
- Reasoning deltas (`reasoning_content` on DeepSeek, `reasoning` on OpenRouter) are read through `OpenAIDelta::reasoning_delta()` and must never reach `text_buffer`. `include_reasoning` decides only whether they are buffered for `StreamEvent::Reasoning` or discarded on arrival, never whether they can merge into content. Reasoning never enters conversation history.
- Parallel tool calls are emitted in ascending API index order. They are held in a `BTreeMap` so the ordering is a property of the container; do not swap in a `HashMap`, whose iteration order varies between runs.
- `FinishReason::MaxToolIterations` is the only variant the SDK originates rather than the server. `from_wire` must never produce it, and it must never appear in a `StreamEvent::Finish` — it exists so `Client::finish_reason()` answers "why did this operation stop?" rather than echoing the last generation's `ToolCalls`.
- `Client::reasoning()` accumulates across an auto-execution tool loop; `finish_reason()` is overwritten per stream. The two fields are reset together in `start_request()`, so any new internal continuation must carry reasoning across the boundary the way `auto_execute_loop` does.
- `Error::Api` carries `status: Option<u16>` as structured data; never embed a status code in the message and parse it back out. Build HTTP-derived API errors with `Error::api_status(status, body)` — `Error::api(msg)` yields `status: None`, which classifies as non-retryable no matter what the message says. Retry classification reads `Error::status_code()`; the retryable set is 408, 429, 500, 502, 503, 504, 529.
- `max_tokens` is not defaulted. Unset means `None`, which omits the field from the wire request so the server applies its own limit; a client-imposed cap truncates long-context and reasoning models mid-response.
- Caller HTTP headers are plain `String` pairs held in a `BTreeMap`, validated during `AgentOptionsBuilder::build()`, and applied only in `stream_request`. The caller wins over SDK defaults case-insensitively; defaults not named remain, an empty `api_key` omits the SDK auth header, and a repeated caller name replaces rather than appends. Never expose reqwest header types or include the header map in `Debug`, because it may carry credentials.
- Model-request HTTP clients never follow redirects, including same-origin redirects. `base_url` names the exact destination: a `30x` stays on the configured endpoint and enters the existing API-status error path, so caller headers and protocol credentials cannot be replayed to a response-selected target. Keep timeout and redirect policy centralized in `model_http_client_builder` while preserving each public path's existing client-build error mapping.
- Mutation testing is mandatory: the local pre-commit hook runs a staged Rust diff sweep. Ordinary CI classifies the complete event diff with `scripts/mutants-ci-scope.sh`: changed inline tests scope to their owning source files; integration tests, fixtures, snapshots, deletions and ambiguous ownership trigger a full sweep; production-only changes skip mutation. Manual dispatch and the fifteenth-day monthly schedule always run the full sweep. Failed runs retain bounded diagnostics for the shared `Monthly Mutation Repair` automation. Fix survivors through discriminating tests or deletion of unobservable code, never exclusions.
- **The verdict is `missed.txt`, not the exit code.** `scripts/mutants-run.sh` owns that rule and both the hook and CI call it, so the two cannot disagree. cargo-mutants reports exit 3 (Timeout) in preference to exit 2 (FoundProblems), so a run with one hanging mutant and one genuine survivor also exits 3; a timeout is a detection and passes, a survivor fails.
- **The local sweep runs on `strix.local`, not on this machine.** `scripts/mutants-staged.sh` calls `scripts/mutants-remote.sh`, which creates one unique remote checkout per invocation, runs `mutants-run.sh` over SSH and mirrors that run into its own `target/mutants/runs/` directory. A new run prunes completed same-host result directories while preserving live owners, so normal use retains the latest diagnostics instead of growing without bound. An unreachable host falls back to a local run **with a warning on stderr** — never a silent skip. A mirror failure fails the gate and retains the exact remote checkout for recovery. CI keeps calling `mutants-run.sh` directly, because a GitHub runner cannot reach the LAN. Knobs: `DREP_MUTANTS_HOST` (default `strix.local`), `DREP_MUTANTS_DIR`, `DREP_MUTANTS_TMPDIR`, `DREP_MUTANTS_REMOTE=0` to force local, `MUTANTS_JOBS`, `MUTANTS_LOCAL_JOBS`.
- More jobs can be slower, and `-j 4` on a 32-thread box is not a typo: concurrent source copies, compilation and linking made measured runs I/O-bound at higher job counts. cargo-mutants excludes the checkout's top-level `target/` by default; do not claim a shared target cache warms its per-mutant copies. Measure before raising `MUTANTS_JOBS`.
- **Scratch copies live in per-run namespaces beside the checkout, never in `/tmp`.** `mutants-run.sh` gives each invocation a unique directory under `<checkout>.mutants-tmp` (override with `DREP_MUTANTS_TMPDIR`), sweeps only dead same-host owner namespaces, and removes only its own namespace on exit while preserving the original status. cargo-mutants deletes its per-job tree copies only on a clean exit, and Strix's `/tmp` is a tmpfs, so copies stranded by a cancelled or timed-out run must remain a disk cost rather than a RAM leak. `mutants-staged.sh` likewise owns a unique diff file; concurrent hooks must never share cleanup targets.
- `cargo-mutants` walks `mod` declarations but does not expand `include!`, so code in an `include!` fragment is invisible to the mutation gate. New modules holding executable logic must be declared with `mod` and re-exported (see `src/utils.rs` and `src/types.rs`), not added as fragments. The pre-existing `client/`, `hooks/`, `retry/`, `tools/`, and most `types/` fragments remain uncovered by the sweep, and cannot be converted without weakening encapsulation: they reach into module-private fields of types their siblings define (`Client` in `client/state.rs`, `AgentOptions` in `types/agent_options.rs`), which sibling modules could only reach if those fields were widened to `pub(crate)`. Converting them is a real refactor with a real cost, not a mechanical move.
- The include-backed `client/`, `hooks/`, `retry/`, `tools/`, and `types/` fragments preserve their parent public module paths; keep production source files below 600 lines and retain the repository-wide 800-line Rust architecture guard.
- `ApiProtocol` selects the request path, the auth header, the request body and the streaming vocabulary together, in one place: `stream_request`. `OpenAIRequest` stays the single internal request representation and both call sites build one; the Anthropic translation happens at the transport boundary. Branching earlier means two request builders that drift.
- Text and reasoning fragments are forwarded to the caller as they arrive; `StreamBuffers` holds neither. `push_text` and `push_reasoning` return the event carrying the fragment (`None` for an empty one), and `flush` drains only the assembled tool calls, whose arguments are not valid JSON until the last fragment lands. Re-buffering either channel restores the defect 0.10.0 fixed: a stream that advertises token-by-token delivery and hands over the whole response at the end.
- Conversation history stores one text block per assistant turn. `utils::coalesce_text_blocks` joins the fragments at every site that builds a `Message::assistant` from streamed blocks, so what is replayed to the server is unchanged by the delivery granularity. It joins *adjacent* text only — text on either side of a tool call stays separate, because the call happened between them.
- The end-of-transport sentinel, accumulator threading and batch flattening live once, in `utils::drive`, behind the `EventAccumulator` trait. That machinery fixes a real defect — content stranded in the buffers when a server never reports why it stopped — so a per-protocol transcription of it is a second place for that defect to return.
- The accumulators own only their wire decoding. Everything they do with the results — the text and reasoning buffers, the tool-call map, the first-seen finish reason, and the drain order — lives once in `utils::buffers::StreamBuffers`, which is where four invariants are decided rather than restated per protocol: exactly one `Finish` and it is last, `Unspecified` distinct from `Stop`, reasoning with no path into the text buffer, and ascending tool-call order. The two copies this replaced had already drifted in their tool-argument error text.
- `StreamBuffers::push_reasoning` owns the `capture_reasoning` check, so no decoder can route reasoning into the text buffer by forgetting it. `tool_call()` opens a call on first mention (OpenAI never announces one) and `open_tool_call()` refuses to (Anthropic always sends `content_block_start` first, so a fragment for an index that never opened has nothing to attach to). Keep both; they are different policies, not an accessor and its convenience wrapper.
- Each accumulator carries its own `impl EventAccumulator`, so `utils::driver` names no protocol. An accumulator error is yielded in band as an `Err` item and does not close the stream; callers propagate it with `?`.
- Anthropic deltas are routed by **the delta's own tag**, never by the kind of the block they arrive on. `thinking_delta` reaches the reasoning channel and `text_delta` reaches content, which makes the reasoning separation a property of the parser rather than of bookkeeping a missing `content_block_start` could defeat.
- `anthropic_finish_reason()` maps Anthropic stop reasons; `FinishReason::from_wire` is OpenAI-shaped and files every Anthropic spelling under `Other`, so a caller branching on `Length` would never see a truncation. `model_context_window_exceeded` maps to `Length` (a token ceiling, same caller response); `pause_turn` keeps its own name, because it is resumable and no existing variant means that.
- A mid-stream Anthropic `error` event has no HTTP status of its own — the response already returned 200 — so the two transient kinds are mapped onto the statuses they would have carried had they arrived earlier (`overloaded_error` → 529, `rate_limit_error` → 429, `api_error` → 500). Retry classification reads `Error::status_code()`, so anything else stays a non-retryable stream error.
- `temperature` is not defaulted, for the same reason `max_tokens` is not. Unset means `None`, which omits the field so the server applies its own value. Anthropic's accepted range stops at 1.0 and several current models reject the parameter entirely, so a client-invented default turns a working request into a hard 400.
- The builder validates `temperature` through `Temperature::new`, the exported newtype that exists for exactly that range and message. Do not re-inline the check; the two copies this replaced spelled the same bound and the same error string.
- `parse_data_uri` (Anthropic translation) and `ImageBlock::from_url` must agree on where a `data:` URI's media type ends: at the first `;`, with anything up to `;base64,` being a parameter such as `charset`. They disagreed, so a URI the SDK accepted reached Anthropic with a media type it rejects.
- The Anthropic tool-result merge does **not** check the message role, and adding that check would be unfalsifiable code: a content array opening with a `tool_result` is only ever built by `push_tool_result`, which always writes `role: "user"`. The mutation gate found the redundant check as a survivor.
- `RetryConfig::max_delay` is a hard ceiling: jitter is applied after the exponential cap, so the jittered result must be clamped to `max_delay` before it becomes a `Duration`.
- Commit format: `type(scope): description` (feat, fix, docs, test, refactor, chore)
- Dependency updates: Dependabot runs weekly (grouped Cargo updates) — resolve security advisories promptly
- Reqwest compatibility: keep reqwest on 0.12.x while `Error::Http` publicly wraps `reqwest::Error`; Dependabot ignores only semver-major reqwest updates until a release that reworks that public boundary. v0.7.0 is not that release — it carries the `max_tokens` and `Error::Api` breaks instead.
- Wiremock compatibility: keep the dev-dependency pinned to 0.6.4 on Rust 1.85. Dependabot ignores exactly 0.6.5 because its let-chains do not compile on the MSRV; later releases remain eligible and must pass the all-target MSRV job before adoption.
- Base64 safety: use base64 0.23 without its default `simd-unsafe` feature unless a measured need justifies enabling it
- GitHub Actions: pin every third-party action to an immutable full commit SHA with a version comment; Dependabot maintains the pins
- Mutation installer: `taiki-e/install-action` is pinned at v2.87.1. The workflow-policy test asserts the exact SHA and version comment, so update the workflow and assertion together.
- Workflow permissions: default to `contents: read` and grant additional permissions only to the job that requires them
- GitHub is the canonical CI and release host. Linux and macOS jobs both run on GitHub Actions runners; no external runner host is involved. Do not add CI configuration that routes test or coverage jobs to non-GitHub runners.
- Audit workflows: install and verify stable Rust before the audit step, then run `cargo audit --deny warnings` directly (not the `actions-rust-lang/audit@` action) so vulnerabilities, yanked crates, unmaintained crates, and unsoundness warnings all fail CI
- PR benchmarks: compare Criterion results directly against the base commit with a shared target directory; do not restore the obsolete `boa-dev/criterion-compare-action`
- Coverage reports: use the exact cargo-tarpaulin 0.37.2 LLVM engine in unprivileged containers, but do not import its upstream lockfile while that lock contains vulnerable anyhow 1.0.102; retain the XML with the latest compatible immutable-SHA-pinned `actions/upload-artifact` release (currently v7.0.1) and fail CI when the report is missing

## Security Advisories (resolved in v0.6.5, current v0.11.3)

RUSTSEC-2026-0190 and RUSTSEC-2026-0204 resolved:

- `anyhow` raised to `1.0.103` minimum
- `reqwest` pinned to `0.12.28`
- `log` raised to `0.4.29`
- `futures` raised to `0.3.32`

## Current Version

**v0.11.3**. Fixes stale client output, failed-stream cleanup, cancellation during automatic
tool rounds, and manual tool-result call IDs. Image validation and logging handle non-ASCII
input safely. Token estimation counts serialized tool JSON without allocating temporary
strings. The cleanup removes duplicate tests, unused dependencies, and repeated documentation;
public signatures and defaults remain unchanged. See CHANGELOG.md.

**v0.11.2**. Model requests deliberately reject all HTTP redirects, including same-origin
redirects, so custom headers and other credentials are sent only to the exact configured
origin. Redirect responses retain their original status and use the existing API-status error
path. No public API changed. See CHANGELOG.md. Published to crates.io and GitHub on 2026-08-31
from merge commit `691f0667913576794af5b9d4b08300fcbecd2040` with annotated tag `v0.11.2`.

**v0.11.1**. Release-source maintenance hardens published package verification and mutation
infrastructure concurrency, removes the yanked `chacha20` 0.10.1 lockfile entry and refreshes
the immutable mutation-installer pin. No public API changed. See CHANGELOG.md.

**v0.11.0**. Callers can attach validated HTTP headers to every model request. Caller values
replace SDK defaults case-insensitively without removing unrelated defaults, an empty
`api_key` suppresses SDK auth, and repeated names replace rather than append. See CHANGELOG.md.
Published to crates.io and GitHub on 2026-08-26 from merge commit
`9901490db5e3f582b06e2ce2a47504e107cd3920` with annotated tag `v0.11.0`.

**v0.10.0**. Streaming that streams: text and reasoning reach the caller fragment by
fragment rather than as one block at the end. History is unchanged — the fragments are joined
before the assistant turn is written. See CHANGELOG.md.

**v0.9.1**. Documentation only, published so docs.rs carries the two-protocol surface
instead of 0.9.0's OpenAI-only description.

**v0.9.0**. A second wire protocol. `ApiProtocol` selects OpenAI chat completions (the
default, and what every endpoint supported before this release speaks) or Anthropic
messages; `AgentOptions::temperature()` returns `Option<f32>` and unset now genuinely means
unset, because a growing number of models reject the parameter outright. See CHANGELOG.md.

**v0.8.0**. `query()` now yields `StreamEvent` rather than `ContentBlock` — the item type had
no room for `finish_reason`, which is not content — so this could not ship as a 0.7.x patch.
`ContentStream` was renamed `EventStream`. `ContentBlock` itself is unchanged: no new variant,
no wire-shape change, so downstream exhaustive matches and its `serde` representation still
work. `Client::receive()` is unaffected and records the reason on the client instead.

New in v0.8.0: `FinishReason` surfaced on every stream (with `Unspecified` kept distinct from
`Stop`), `Client::finish_reason()`, explicit reasoning-channel parsing and separation with
opt-in `include_reasoning` / `StreamEvent::Reasoning` / `Client::reasoning()`, index-ordered
parallel tool call emission, and real-module layout for new code so the mutation gate actually
covers it.

v0.7.1 was docs-only on top of v0.7.0, which dropped the implicit 4096 `max_tokens` default
and restructured `Error::Api` into `Api { status, message }`, and added end-of-stream flushing
for servers that never send `finish_reason`, status-based retry classification, and the
mandatory mutation-testing gate.

v0.6.9 delivered transport-boundary-safe SSE streaming, complete structured
hook history, shared client request construction, source-size architecture
guards, and GitHub-canonical CI, while retaining Rust 1.85 compatibility
and the hardened dependency baseline.

Features: multimodal vision (URLs, local files, base64), manual-mode history fix (v0.6.2 regression), retry module, interrupt capability, lifecycle hooks, automatic tool execution, context management utilities, provider helpers, prelude module, OpenAI wire type exports.
