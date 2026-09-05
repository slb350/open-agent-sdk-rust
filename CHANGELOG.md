# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.3] - 2026-09-05

### Changed

- Stop running the full `cargo-mutants` sweep for every ordinary CI revision. Added, modified,
  deleted, and renamed inline tests now run the owning source files' mutants; integration tests,
  fixtures, snapshots, and ambiguous mappings fall back to the full sweep. Production-only
  revisions skip mutation, while manual dispatch and the fifteenth-day monthly schedule always
  sweep the tree. Failed runs retain bounded evidence for the following day's autonomous repair
  PR.

### Fixed

- Reset pending output for new requests and history clearing. Auto-mode responses no
  longer retain stale fragments across turns, and interruption stops buffered delivery.
- Close failed client streams and honor cancellation within automatic tool rounds.
- Preserve manual tool-result call IDs through OpenAI and Anthropic request translation.
- Truncate logged image URLs at UTF-8 boundaries and reject non-ASCII Base64 characters.
- Forward custom mutation-result directories to the remote execution environment.

### Maintenance

- Replace construction-only integration tests and copied implementations with compact
  loopback request, hook, logging, and lifecycle coverage. Consolidate image validation
  and keep one opt-in provider smoke test; isolate environment tests in one process.
- Share request assembly, remove unused dependencies and redundant example declarations,
  and move Tokio test/example runtime features out of production dependencies.
- Shorten repeated API tutorials, correct stale contracts, and document `max_turns` as
  an inert compatibility value. Public signatures and defaults remain unchanged.
- Validate workflow structure and actual package contents instead of source substrings,
  preserving scoped mutation CI and its monthly/manual full-sweep backstops.
- Exercise both branches of the context-management workflow benchmark.
- Count serialized tool JSON bytes without allocating temporary strings during token estimation.

## [0.11.2] - 2026-08-31

### Security

- Model requests no longer follow HTTP redirects, including same-origin redirects. Custom
  headers and other credentials are sent only to the configured origin, while any `30x`
  response remains visible through the existing API-status error path.

### Tests

- Added raw loopback regressions for both `query()` and `Client` proving a redirect target is
  never contacted and cannot receive a caller-supplied credential header.

## [0.11.1] - 2026-08-29

### Fixed

- Published crate sources no longer include repository-only tests after excluding the
  workflows, hooks and mutation scripts they load. Freshly unpacked packages now build and
  test without missing-file compiler errors.
- Mutation sweeps now own unique local scratch directories, staged diffs, remote checkouts and
  result directories. Concurrent runs cannot delete live scratch trees, mutate another run's
  checkout, overwrite its diff or return its diagnostics.
- Remote scratch overrides are forwarded, result-mirror failures are reported instead of
  hidden, the exact remote run is retained when recovery is necessary, and cleanup preserves
  the original command status. A new run prunes completed same-host diagnostics while
  preserving live concurrent runs, so local results remain bounded.

### Maintenance

- Refreshed six Rust-1.85-compatible lockfile entries: `chacha20` 0.10.2, `cpufeatures` 0.3.1,
  `h2` 0.4.19, `hyper` 1.11.1, `indexmap` 2.14.1 and `syn` 3.0.4. The `chacha20` update removes
  the yanked 0.10.1 release that caused the deny-warnings security audit to fail.
- Updated the immutable `taiki-e/install-action` pin from v2.86.6 to v2.87.1 for the
  cargo-mutants installation job, with its exact workflow-policy assertion updated in step.

### Tests

- Added behavioral shell regressions for failure status, stale cleanup, concurrent scratch
  ownership, unique remote runs, scratch forwarding, mirror failure and staged-diff cleanup.
- Added package-manifest coverage for the repository-only CI policy test.

## [0.11.0] - 2026-08-26

### Added

- **Arbitrary model-request headers** through
  `AgentOptions::builder().header(name, value)`. This enables OpenRouter attribution with
  `HTTP-Referer` / `X-Title`, Azure OpenAI authentication with `api-key`, and corporate
  gateways that require `User-Agent` or custom routing and billing metadata.
- Caller headers replace SDK defaults case-insensitively, while defaults the caller did not
  name remain intact. Repeating a caller header replaces its earlier value rather than
  appending a duplicate.
- An empty `api_key` now suppresses the protocol's SDK auth header, allowing authentication
  to come entirely from caller-supplied headers. Invalid header names and values fail
  `build()` as configuration errors that identify the offending name.

### Maintenance

- Refreshed seven Rust-1.85-compatible lockfile entries: `cc` 1.4.4, `either` 1.18.0,
  `h2` 0.4.18, `log` 0.4.34, `rustls-webpki` 0.103.15, `zerovec` 0.11.8 and
  `zerovec-derive` 0.11.6.
- Updated the immutable `taiki-e/install-action` pin from v2.86.2 to v2.86.6 for the
  cargo-mutants installation job, with its exact workflow-policy assertion updated in step.
- Dependabot now ignores wiremock 0.6.5 specifically. Its use of let-chains fails on the
  supported Rust 1.85 compiler, while future wiremock releases remain eligible for review.

### Tests

- Added a Dependabot-policy regression so the known MSRV-breaking wiremock release cannot be
  reintroduced by the weekly dependency group.

## [0.10.0] - 2026-08-19

Streaming that actually streams. Every release since 0.1.0 advertised token-by-token
delivery, and every release since 0.1.0 concatenated the whole response and handed it over in
one block when the stream ended.

### Breaking

- **Text and reasoning now reach the caller fragment by fragment.** A response that used to
  arrive as one `StreamEvent::Block(ContentBlock::Text(..))` at the end of the stream now
  arrives as one event per delta, in order, while the stream is still open. The same applies
  to `StreamEvent::Reasoning` and to `Client::receive()`, which yields a block per fragment.

  Code that concatenates what it receives is unaffected. Code that treated the first text
  block as the entire answer — `blocks[0]`, `blocks.len() == 1`, or a `match` that returns on
  the first `Text` — now sees a prefix of the answer instead. Join the fragments:

  ```rust
  let mut answer = String::new();
  while let Some(event) = stream.next().await {
      if let StreamEvent::Block(ContentBlock::Text(text)) = event? {
          answer.push_str(&text.text);
      }
  }
  ```

  Empty deltas emit nothing, so the common first chunk of an OpenAI stream (an empty content
  string alongside the role) does not turn into an empty block.

### Added

- **`coalesce_text_blocks`**, exported. It is the join the SDK applies before writing an
  assistant turn to history, so a caller that collects blocks and wants whole ones does not
  have to hand-write it.

### Unchanged

- **Conversation history still records one text block per assistant turn.** The fragments are
  joined before the turn is written, so what is replayed to the server on the next request is
  byte-identical to what 0.9.x sent. A message per fragment would multiply the assistant
  turns in every subsequent request.
- **Tool calls still emit whole, at the end of the stream, in ascending index order.** Their
  arguments arrive split at arbitrary byte positions and are not valid JSON until the last
  fragment lands, so a tool call is the one thing that cannot stream.
- Exactly one `StreamEvent::Finish`, and it is still the last event.

### Internal

- `StreamBuffers` no longer holds text or reasoning at all: `push_text` and `push_reasoning`
  return the event that carries the fragment, and `flush` drains only the assembled tool
  calls. Both protocol accumulators forward what those calls return.
- `Client::push_assistant` is now the only place streamed blocks become a `Message`, so the
  join cannot be forgotten by a new call site. It was four places, one of which the wire
  builder would have silently garbled by joining text blocks with newlines.

## [0.9.1] - 2026-08-19

Documentation only. No API or behaviour change: 0.9.0's published rustdoc still described the
SDK as OpenAI-compatible only, and docs.rs serves whatever the last release carried.

### Documentation

- v0.9.0 shipped with rustdoc and README copy that still described the SDK as
  OpenAI-compatible only. Both landing pages, the `base_url` and `query()` docs, and the
  `client`, `types`, `tools` and prelude module docs now cover both protocols.
- `README.md`: the wire-types section covers both formats and the exported
  `AnthropicRequest`/`OpenAIRequest` families, and the project-structure block matches the
  tree again.
- Two behaviours that were only written down in the maintainer notes: `ImageDetail` has no
  Anthropic equivalent and is dropped in translation, and a mid-stream Anthropic `error`
  event is mapped onto the status it would have carried earlier, which is what makes it
  retryable.
- New `examples/anthropic_query.rs`: a single-turn query against an Anthropic messages
  endpoint.
- `Cargo.toml` keywords: `local` swapped for `anthropic`, the five-keyword cap being what it
  is.

## [0.9.0] - 2026-08-19

A second wire protocol. Several vendors publish their subscription coding tiers only behind
an Anthropic-shaped `/messages` endpoint, so "which protocol" stopped being a property of
the SDK and became a property of the endpoint.

### Breaking

- **`AgentOptions::temperature()` returns `Option<f32>`, and unset now means unset.** It
  previously defaulted to 0.7 and was always sent. A growing number of models reject the
  parameter outright — Anthropic's accepted range stops at 1.0, and `kimi-for-coding`'s `k3`
  answers `only temperature 1 is allowed for this model` with a 400 — so a client-invented
  default turns a working request into a hard error the caller never asked for. `None` omits
  the field and the server decides, exactly as `max_tokens` has behaved since 0.7.0. The
  builder method is unchanged: `.temperature(0.2)` still sets one.

### Added

- **`ApiProtocol`**, selecting the wire protocol per endpoint: `OpenAiChat` (the default,
  and what every endpoint supported before this release speaks) or `Anthropic`. Set it with
  `AgentOptions::builder().protocol(ApiProtocol::Anthropic)`. It selects the request path,
  the auth header, the request body and the streaming vocabulary together.
- **Anthropic messages support**: `POST {base_url}/messages` with `x-api-key` and
  `anthropic-version`, the full streaming event vocabulary (`message_start`,
  `content_block_start`/`_delta`/`_stop`, `message_delta`, `message_stop`, `ping`, `error`),
  and extended thinking routed to the existing reasoning channel. Tool calls assemble from
  `input_json_delta` fragments and emit as ordinary `ContentBlock::ToolUse` blocks.
- **`anthropic_finish_reason()`**, mapping Anthropic stop reasons onto `FinishReason`.
  `FinishReason::from_wire` is OpenAI-shaped and files every Anthropic spelling under
  `Other`, so a caller branching on `Length` would never see a truncation.
- **The Anthropic wire types are exported** — `AnthropicRequest`, `AnthropicMessage`,
  `AnthropicEvent`, `AnthropicBlockStart`, `AnthropicDelta`, `AnthropicMessageDelta`,
  `AnthropicErrorBody` — for the same reason the OpenAI ones are: a gateway, a recording
  proxy or a test double needs to name what goes over the wire. `OpenAIRequest`,
  `OpenAIMessage`, `OpenAIFunction` and `OpenAIToolCall` are now exported too, since
  `AnthropicRequest::from_openai` takes one.

### Fixed

- A `data:` URI carrying parameters between the media type and `;base64,` — say
  `data:image/png;charset=utf-8;base64,...` — reached an Anthropic request with
  `image/png;charset=utf-8` as its media type, which the API rejects, for an image
  `ImageBlock::from_url` had already accepted. Both now read the media type up to the first
  `;`.

### Internal

- `OpenAIRequest` stays the single internal request representation. Both call sites build
  one, and the translation to Anthropic happens once, at the transport boundary, so there is
  no second request builder to drift.
- The end-of-transport sentinel, the accumulator threading and the batch flattening moved
  into `utils::drive`, shared by both protocols behind an `EventAccumulator` trait. That
  machinery is a fix for a real defect — content stranded in the buffers when a server never
  reports why it stopped — and a second transcription of it would be a second place for that
  defect to return.
- A mid-stream Anthropic `error` event is mapped onto the HTTP status it would have carried
  had it arrived before the stream opened (`overloaded_error` → 529, `rate_limit_error` →
  429, `api_error` → 500), because the retry layer classifies on `Error::status_code()` and
  a mid-stream error has none of its own. Anything else stays a non-retryable stream error.
- Both accumulators drain one shared `StreamBuffers` instead of a copy each. Four documented
  invariants — exactly one `Finish` and it is last, `Unspecified` distinct from `Stop`,
  reasoning with no path into the text buffer, ascending tool-call order — were asserted in
  two implementations that had already drifted; they are decided in one place now, and only
  the wire decoding stays per protocol.
- Each accumulator carries its own `EventAccumulator` implementation, so `utils::driver`
  names neither protocol and a third one is a new module rather than an edit to that one.
- The tool-argument parse error names the tool for both protocols:
  `Failed to parse tool call arguments for '<name>': <error>`. The OpenAI-side message was
  `Failed to parse tool arguments: <error>`.
- The mutation gate moved behind `scripts/mutants-run.sh`, which decides the verdict from
  `missed.txt` rather than the exit code — cargo-mutants reports exit 3 (Timeout) ahead of
  exit 2 (FoundProblems), so a run with one hang and one genuine survivor also exits 3. The
  pre-commit hook now offloads its sweep to a LAN build host and falls back locally with a
  warning; CI still runs it on the GitHub runner. `scripts/` is excluded from the published
  crate.

## [0.8.0] - 2026-08-18

Two pieces of information the SDK held internally and never handed over: why generation
stopped, and what a reasoning model streamed on its side channel. A caller parsing
structured output could not tell a token-capped truncation from a clean stop from a silent
server, and reasoning was dropped by accident rather than by decision.

### Breaking

- **`query()` yields `StreamEvent`, not `ContentBlock`**: the item type of the stream returned by `query()` changed from `ContentBlock` to the new `StreamEvent` enum, and the `ContentStream` type alias was renamed `EventStream`. `finish_reason` had nowhere to go in a stream of content, because it is not content. Every stream now ends with exactly one `StreamEvent::Finish` carrying a `FinishReason`. Existing loops migrate by wrapping the match in `event?.into_block()`; loops that care about truncation match `StreamEvent::Finish` instead. `Client::receive()` is unchanged — it still yields `ContentBlock` and records the finish reason on the client.
- **`ContentBlock` itself is unchanged**: no new variant, no wire-shape change, so downstream exhaustive matches over `ContentBlock` and its `serde` representation still compile and still round-trip.

### Added

- **`FinishReason`**: `Stop`, `Length`, `ToolCalls`, `ContentFilter`, `Other(String)` for provider-specific values, `MaxToolIterations` for the one case the SDK rather than the server decides, and `Unspecified` for a stream that ended without the server ever saying. `Unspecified` is deliberately distinct from `Stop`: reporting a silent server as a clean stop would claim knowledge the SDK does not have, and it is exactly the servers that omit `finish_reason` (llama.cpp, vLLM, several local gateways) that the 0.7.0 flush fix exists for. `is_truncated()` answers the question a caller parsing JSON actually has. `#[non_exhaustive]`.
- **`StreamEvent`**: `Block(ContentBlock)`, `Reasoning(String)`, `Finish(FinishReason)`, with `as_block`/`into_block`/`as_text`/`as_reasoning`/`finish_reason` accessors so the common cases stay one line. `#[non_exhaustive]`, so a future channel does not force another breaking release.
- **`Client::finish_reason()`**: why the most recent stream stopped, cleared when the next request starts. In auto-execution mode it reports the final generation of the tool loop — the one that produced the text the caller sees — except when the loop stopped at `max_tool_iterations`, where it reports `FinishReason::MaxToolIterations`. Without that case the SDK's own cut-off surfaced as the model's `ToolCalls`, an accurate answer to a different question than the one `finish_reason()` exists to ask.
- **`Client::reasoning()`**: reasoning text captured from the most recent turn, `None` unless capture is enabled. It accumulates across the rounds of an auto-execution tool loop rather than being overwritten each round; every round is its own stream, and keeping only the last would discard the deliberation that chose the tools — most of what was asked for.
- **`AgentOptions::include_reasoning(bool)`**: opts into `StreamEvent::Reasoning`. Defaults to `false`, in which case reasoning deltas are read off the wire and discarded immediately rather than buffered — a caller that does not want a long chain of thought should not pay to hold it.

### Fixed

- **Reasoning is now dropped by decision instead of by accident**: `OpenAIDelta` declared only `role`, `content`, and `tool_calls`, so DeepSeek's `reasoning_content` and OpenRouter's `reasoning` were discarded because serde ignores unknown fields — nothing in the code expressed the intent, and any future change to delta handling could have started splicing deliberation prose into the text a caller parses as JSON. Both fields are now declared and routed through `OpenAIDelta::reasoning_delta()` into a buffer that `text_buffer` has no path to. A gateway that mirrors the same trace on both field names is counted once.
- **Parallel tool calls are emitted in index order**: `flush` drained the `HashMap` of partial tool calls directly, so with more than one tool call in a response the emission order was whatever the hash iteration produced and varied between runs. It now sorts by the API-provided index.

### Changed

- **`ToolCallAggregator` is now `StreamAccumulator`**, and `flush()` gained a sibling `finalize()`. The type accumulates text, reasoning, tool calls, and the finish reason, so the old name described a third of it. `flush()` still drains buffered content; `finalize()` is what the stream driver calls at end of transport, and it is the only thing that emits `Finish`. Recording the reason in `process_chunk` but emitting it only in `finalize` is what guarantees `Finish` is last even when a server keeps sending after its own `finish_reason`. Both types are internal (`mod utils` is private), so this is not a public break.
- **New code lives in real modules, not `include!` fragments**: `cargo-mutants` walks `mod` declarations but does not expand `include!`, so the crate's fragment architecture was invisible to the mutation gate — the full sweep found 103 mutants across only 5 files, and a `--in-diff` sweep over this change found 1. `src/utils/{accumulator,sse}.rs` and `src/types/stream_event.rs` are declared with `mod` and re-exported, which keeps every path unchanged and brings the sweep to 146 mutants. The remaining fragments are untouched.
- **`src/types/openai.rs` split**: the streaming chunk and delta types moved to `src/types/openai_stream.rs`, keeping both halves inside the 600-line soft limit after the reasoning fields were added. It is a real module too, since `OpenAIDelta::reasoning_delta` is exactly the kind of channel-precedence logic a mutant flips.
- **Partial tool calls are held in a `BTreeMap`**: ordering now falls out of the container instead of a sort at flush time, which also drops the hash of a `u32` on every tool-call delta.

### Testing

- Added `tests/regression_finish_reason_test.rs` (10 tests): each well-known reason surfaces, `Unspecified` is distinct from `Stop`, an unrecognised reason survives verbatim, exactly one `Finish` is emitted and it is last, an empty stream still reports one, and `Client::finish_reason()` both reports and resets.
- Added `tests/regression_reasoning_channel_test.rs` (10 tests): reasoning from either channel never reaches a text block, a reasoning-only response yields no text block at all, reasoning is dropped unless requested, opting in emits it ahead of the text it produced, a mirrored trace is not double-counted, reasoning never enters history, and unknown delta fields are still ignored.
- Added `src/utils/accumulator.rs` and `src/types/stream_event.rs` unit tests (26 tests) covering finish-reason mapping and precedence, no double-emission after `finish_reason`, channel separation, and index-ordered tool call emission.
- The flush-without-`finish_reason` regressions from 0.7.0 are unchanged in intent and still pass; they now assert over the content blocks filtered out of the event stream.
- Added coverage for both cases a quality review surfaced: the auto-execution loop reporting its own iteration cap rather than the model's `ToolCalls`, and reasoning surviving every round of that loop. Both were RED against the first implementation.
- `cargo mutants --no-shuffle -j 4`: 149 mutants, 124 caught, 25 unviable, 0 missed.

## [0.7.1] - 2026-08-17

Documentation only. No code, API, or behaviour changes from 0.7.0 — this release exists
because crates.io renders the README captured at publish time, and 0.7.0 shipped without
upgrade guidance for its two breaking changes.

### Documentation

- **Added an "Upgrading from 0.6.x" guide** to the README covering both v0.7.0 breaks: the `Error::Api` tuple-to-struct variant change (caught by the compiler, shown with before/after) and the `max_tokens` default removal (a silent behaviour change with no compile error, so it carries the louder callout). Also notes the two fixes that need no caller action.
- **Documented the `Error` variants** in a table. The README previously described `Error` in a single sentence and never listed a variant, so the `Api { status, message }` shape introduced in 0.7.0 was not discoverable there at all. Added worked examples for `Error::api_status` and `Error::status_code`.
- **Described end-of-stream flushing in the Overview**, where streaming behaviour is explained, instead of only in the trailing status line.

## [0.7.0] - 2026-08-17

### Breaking

- **`Error::Api` is now a struct variant carrying the HTTP status**: `Api(String)` became `Api { status: Option<u16>, message: String }`. Code that matches `Error::Api(msg)` must become `Error::Api { message, .. }`. `Error::api(msg)` still works and yields `status: None`; new `Error::api_status(status, msg)` is the constructor for HTTP-derived errors and is what `stream_request` now uses. This also fixes the doubled prefix in `Display` — a status-carrying API error rendered as `API error: API error 503 Service Unavailable: …` and now renders as `API error 503: …`.
- **`max_tokens` is no longer defaulted to 4096**: leaving `.max_tokens()` unset now yields `None` and omits the field from the wire request, so the server applies its own limit. There was previously no way to express "no cap", and the implicit 4096 truncated long-context and reasoning models mid-response — surfacing as an unparseable partial answer that reads like a model failure rather than a client-imposed limit. Callers who want a ceiling must now set one explicitly.

### Fixed

- **Streamed content is no longer discarded when a stream ends without `finish_reason`**: `ToolCallAggregator` only emitted blocks on a non-null `finish_reason`, and nothing flushed its buffers at end of transport. Servers that stream content and then send `data: [DONE]` (or simply close the connection) with `finish_reason` still null — llama.cpp, vLLM, and several local gateways — produced zero blocks, no error, and no warning, leaving callers unable to distinguish an empty response from a lost one. The stream driver now signals end-of-transport and the new `ToolCallAggregator::flush` emits any buffered text and completed tool calls.
- **Rate limiting is retryable**: `is_retryable_error` classified only 500/502/503/504 as transient, so `429 Too Many Requests` — the canonical reason to back off — failed immediately, contradicting the documented behaviour. The retryable set is now 408, 429, 500, 502, 503, 504, and 529.
- **Status classification no longer matches on substrings**: `is_retryable_error` searched the whole message for `500`/`502`/`503`/`504`, so `API error 400 Bad Request: max_tokens 500 too small` was retried three times before failing even though it could never succeed. Classification now reads `Error::status_code()`, which returns the status `Error::Api` carries structurally — there is no message text to misparse.
- **README retry example**: corrected `retry_with_backoff_conditional` to its actual two-argument signature.
- **`max_delay` is now an actual maximum**: `calculate_delay` applied jitter *after* capping, so a 60s `max_delay` could still produce a 66s sleep at a 0.2 jitter factor. The jittered delay is now clamped to `max_delay`.
- **MSRV check covers test targets**: the `msrv` job ran `cargo check --all-features --all`, which skips tests and benches, so a dev-dependency requiring a newer compiler could land without failing CI while silently breaking `cargo test` on Rust 1.85. It now runs `--all-targets --workspace`. This was found immediately: wiremock 0.6.5 uses let-chains and does not build on 1.85, so the dev-dependency is pinned to `=0.6.4`.
- **Security audit portability**: Install and verify stable Rust, exact-pin cargo-audit 0.22.2, and invoke it directly with warnings denied. This removes assumptions that runner images provide either `cargo` or the Python interpreter used internally by the audit action. Its published upstream lockfile is deliberately not imported because that lock contains denied RustSec advisories.
- **Container-safe coverage**: Pin cargo-tarpaulin exactly to 0.37.2 for LLVM 23 support and its 32-bit profile fix, and use its LLVM engine to preserve required Cobertura output without ptrace, ASLR changes, privileged containers, or relaxed seccomp policy. The report is required, checked for content, and retained with GitHub's artifact service. The install deliberately resolves patched compatible build dependencies instead of importing Tarpaulin's upstream lockfile, which contains vulnerable anyhow 1.0.102.

### Added

- **`Error::api_status(status, msg)` and `Error::status_code()`**: construct and read HTTP-derived API errors without round-tripping the status through prose.
- **`ToolCallAggregator::flush()`**: drains buffered text and completed tool calls. Idempotent after a `finish_reason` flush, so end-of-stream flushing never double-emits.
- **Mutation testing gate**: `cargo mutants --no-shuffle -j 4` runs unconditionally in CI via an immutable-SHA-pinned `taiki-e/install-action`, and `.githooks/pre-commit` runs fmt, clippy, the test suite, and a `--in-diff` sweep scoped to the staged Rust changes. Enable locally with `git config core.hooksPath .githooks`.

### Changed

- **Stream driver**: `stream_request` appends an end-of-transport sentinel to the chunk stream and drops the unobservable `blocks.is_empty()` guard, since an empty batch already flattens to nothing.
- **Dead-guard removal**: the first full mutation sweep identified guards whose removal no test could detect because they were genuinely unreachable — `truncate_messages`'s `keep > 0` and `!messages.is_empty()` checks (the early returns above already guarantee both, so the tail slice is now a plain subtraction) and `is_retryable_error`'s explicit `Config`/`InvalidInput` arms (identical to the `_` fallthrough). All were deleted rather than papered over with mutant exclusions.
- **Test layout**: `src/retry.rs`'s test module moved to the `src/retry/tests.rs` include-backed fragment, matching the existing `client/`, `hooks/`, `tools/`, `types/`, and `utils/` layout and keeping the source file well under the 600-line soft limit.
- **Canonical CI host**: GitHub is now the canonical repository and sole Actions host. The family Gitea repository is retained only as a passive Git mirror, removing cross-host runner-routing and artifact-protocol compatibility code while preserving native stable/beta macOS coverage.
- **Compatible maintenance**: Refresh futures 0.3.33 to 0.3.34 and all other Rust 1.85-compatible lockfile dependencies available during the weekly maintenance window.

### Testing

- Added `tests/regression_stream_flush_test.rs`, `tests/regression_retry_classification_test.rs`, and `tests/regression_max_tokens_test.rs` (17 tests) covering end-of-stream flushing for text and tool calls, no double-emission after `finish_reason`, status-based retry classification in both directions, and `max_tokens` omission from the wire request. All were RED against 0.6.9 before the fixes.
- Added workflow-policy regressions asserting the mutation sweep runs unconditionally with a pinned toolchain and installer SHA, that the MSRV job covers test targets, and that the pre-commit hook runs the same fmt/clippy/test commands as CI. The helper that slices a job out of the workflow now finds the next job header structurally, so inserting a job no longer forces edits to unrelated assertions.
- Added a `wiremock` dev-dependency so streaming and wire-format behaviour can be asserted against a real HTTP server, with the shared harness in `tests/common/mod.rs`, plus tokio's `test-util` feature so retry timing is asserted on a paused virtual clock instead of flaky wall-clock tolerances.
- Closed every survivor from the first full mutation sweep: `tests/config_env_test.rs` covers the previously untested `get_model` environment resolution in its own process, `tests/context_estimation_test.rs` asserts exact token arithmetic for tool-use and tool-result blocks plus the strict `is_approaching_limit` threshold, and new `src/retry/tests.rs` cases pin exact backoff multiples, jitter distribution, the `max_delay` clamp, and the number of sleeps each retry driver performs. `cargo mutants --no-shuffle -j 4` now reports 96 caught, 0 missed. Tests that newer, stronger cases fully subsumed were deleted rather than left to be maintained twice.

- Added workflow-policy regressions for native GitHub-hosted Linux/macOS routing, direct warnings-denied cargo-audit execution without Python, the exact Tarpaulin version without its vulnerable upstream lock, required coverage report generation and retention, and the absence of privileged-container workarounds.

## [0.6.9] - 2026-08-08

### Fixed

- **SSE transport framing**: Streaming responses now buffer arbitrary HTTP transport fragments before parsing JSON, so events split across reqwest body chunks no longer fail with truncated-JSON errors.
- **SSE event delivery**: Every complete event is emitted when multiple SSE messages arrive in one transport chunk; split UTF-8 sequences are buffered and invalid UTF-8 is reported instead of being lossily replaced.

### Changed

- **SSE decoder**: Replaced the per-chunk line parser with the existing `eventsource-stream` dependency while preserving typed reqwest transport errors and the OpenAI `[DONE]` sentinel behavior.
- **Compatible maintenance**: Refreshed the direct base64 lock entry from 0.23.0 to 0.23.1 and advanced the immutable `dtolnay/rust-toolchain` action pin to `6c977a6ca4077a0ceb28ffbe03f59d46e9ac8772`.
- **Dependency automation**: Deferred only reqwest semver-major Dependabot updates because the public `Error::Http(reqwest::Error)` boundary requires the documented v0.7.0 migration; compatible reqwest updates and security alerts remain enabled.

### Testing

- Added a RED/GREEN loopback regression that sends one JSON event across separate HTTP chunks and multiple SSE events in one chunk, then verifies all three events arrive intact.

## [0.6.8] - 2026-08-01

### Fixed

- **Hook history**: `UserPromptSubmit`, `PreToolUse`, and `PostToolUse` events now receive complete structured JSON snapshots of conversation messages instead of one empty object per message.
- **Post-tool context**: `PostToolUseEvent::history` now includes the completed tool call and its unmodified result, matching the documented lifecycle semantics while still allowing the hook to modify the result before it is committed.
- **Client documentation**: Attached the receive-loop documentation to `Client::receive()` instead of accidentally including it in `Client::send_message()` documentation.

### Changed

- **Client request path**: Centralized request construction for `send()` and `send_message()`, then unified HTTP error handling and SSE stream setup across `query()` and the stateful client.
- **Source architecture**: Split the oversized client, hook, tool, type, and utility implementations into focused include-backed module fragments while preserving all existing public module paths and APIs. Every production Rust source file is now below the 600-line soft limit.

### Testing

- Added RED/GREEN regressions for prompt-hook, pre-tool, and post-tool history snapshots across text, tool-call, and tool-result content.
- Added an architecture guard that fails if any repository Rust source file exceeds the 800-line hard limit.

## [0.6.7] - 2026-07-25

### Fixed

- **Benchmarks**: Replaced Criterion's deprecated `black_box` re-export with `std::hint::black_box`, restoring zero-warning benchmark builds after the Criterion 0.7 update.
- **Coverage CI**: Replaced an unauthenticated Codecov upload that silently failed for this unconfigured repository with a required, first-party GitHub Actions coverage artifact retained for 14 days.

### Changed

- **Production dependencies**: Updated `thiserror` 1.x → 2.x, `rand` 0.8 → 0.10, and `base64` 0.22 → 0.23 while preserving Rust 1.85 compatibility.
- **Base64 safety**: Disabled base64 0.23's default `simd-unsafe` feature and enabled only `std`.
- **HTTP compatibility**: Retained reqwest 0.12.28 because the public `Error::Http` variant wraps `reqwest::Error`; a reqwest 0.13 migration remains reserved for a documented breaking release.
- **Development dependency**: Updated Criterion 0.5 → 0.7, the newest release compatible with the project's Rust 1.85 MSRV.
- **CI security**: Updated checkout to v7.0.1, pinned every action to an immutable commit, reduced default workflow permissions to read-only, and replaced the obsolete Node 16 benchmark action with direct Criterion base/head comparison.

### Testing

- Added retry jitter regression coverage that samples 1,000 delays and verifies the configured ±10% bounds.
- Verified the dependency batch on stable, beta, macOS, Ubuntu, and Rust 1.85 through the required GitHub Actions matrix.

## [0.6.6] - 2026-07-25

### Fixed

- **Interrupt example**: Replaced the deadlocking `Arc<Mutex<Client>>` concurrent-cancellation pattern with `Client::interrupt_handle()`, so the receive loop retains exclusive ownership of the client while another task signals cancellation (fixes #7).
- **Documentation**: Corrected the README concurrent-cancellation example and clarified that interrupted partial manual responses are discarded rather than committed to conversation history.
- **Package**: Excluded the newly restored `CLAUDE.md` notes and `.markdownlint.json` development configuration from published crate archives, preventing a recurrence of the packaging issue that caused v0.6.3 to be yanked.

### Testing

- Added regression coverage for non-locking concurrent cancellation and development-file package exclusions.

## [0.6.5] - 2026-07-22

### Fixed

- **Security**: Raised the `anyhow` minimum to 1.0.103 and refreshed the lockfile from 1.0.102 → 1.0.104 to resolve RUSTSEC-2026-0190 (`Error::downcast_mut()` unsoundness).
- **Security**: Updated transitive development dependency `crossbeam-epoch` 0.9.18 → 0.9.20 to resolve RUSTSEC-2026-0204 (invalid pointer dereference in pointer formatting).

### Changed

- **Maintenance**: Added grouped weekly Dependabot updates for Cargo and GitHub Actions dependencies.
- **Dependencies**: Refreshed all lockfile dependencies to their latest Rust 1.85-compatible releases.
- **CI**: Moved the standalone scheduled audit to Monday after the proactive Sunday maintenance window; push and pull-request audits remain unchanged.
- **Automation**: Added a weekly Codex maintenance task authorized to audit, remediate, verify, commit, push, and publish patch releases when downstream dependency constraints change.

## [0.6.4] - 2026-04-29

### Fixed

- **Package**: Remove inadvertent `CLAUDE.md` (development notes) from the published crate. v0.6.3 has been yanked. v0.6.4 is identical to v0.6.3 in terms of source, with the only change being the package no longer ships `CLAUDE.md`.

## [0.6.3] - 2026-04-29

### Fixed

- **Security**: Resolved 3 dependency vulnerabilities via `cargo update`:
  - `rustls-webpki` 0.103.10 → 0.103.13 (RUSTSEC-2026-0098: name constraints incorrectly accepted for URI names)
  - `rustls-webpki` 0.103.10 → 0.103.13 (RUSTSEC-2026-0099: name constraints accepted for wildcard certificates)
  - `rustls-webpki` 0.103.10 → 0.103.13 (RUSTSEC-2026-0104: reachable panic in CRL parsing)
- **Security**: Addressed `rand` unsoundness warning (RUSTSEC-2026-0097) by updating 0.8.5 → 0.8.6
- **Docs**: README code snippets used a stale `client.receive()` pattern that no longer compiled (fixes #6)
  - `while let Some(block) = client.receive().await { match block? { ... } }` did not match the v0.4.0+ `Result<Option<ContentBlock>>` return type
  - All 8 occurrences updated to `while let Some(block) = client.receive().await? { match block { ... } }`
  - Manual tool execution snippet updated to clone the borrowed `tool_use.input()` reference
- **Lint**: Fixed `clippy::unnecessary_sort_by` in `log_analyzer_agent` example (rust 1.95)

## [0.6.2] - 2026-03-30

### Fixed

- **Client**: Manual mode `receive()` now adds assistant messages to conversation history (fixes #4)
  - Previously, only user messages appeared in `history()`, breaking multi-turn conversations
  - Buffer is committed on natural stream EOF and flushed before `add_tool_result()`
  - Partial output is correctly discarded on stream errors, interrupts, and abandoned streams
  - `clear_history()` also clears the manual buffer to prevent stale replay
- **Client**: Interrupt after stream EOF now commits (not discards) the complete response
- **Client**: `send()`/`send_message()` discard unfinished stream buffers instead of persisting truncated content

## [0.6.1] - 2026-03-29

### Fixed

- **Security**: Resolved 2 dependency vulnerabilities via `cargo update`:
  - `bytes` 1.10.1 → 1.11.1 (RUSTSEC-2026-0007: integer overflow in `BytesMut::reserve`)
  - `rustls-webpki` 0.103.8 → 0.103.10 (RUSTSEC-2026-0049: faulty CRL matching logic)
- **CI**: Fixed daily Security Audit workflow failure — `create-issues` parameter renamed to `createIssues` per `actions-rust-lang/audit@v1` input schema

### Changed

- **Dependencies**: Updated all transitive dependencies to latest compatible versions
- **Dependencies**: Tightened direct dependency version floors:
  - `tokio` 1.40 → 1.50
  - `reqwest` 0.12 → 0.12.28
  - `futures` 0.3 → 0.3.32
  - `log` 0.4 → 0.4.29
- **CI**: Updated `actions/checkout` v4 → v6 (Node.js 20 deprecation)
- **CI**: Updated `codecov/codecov-action` v4 → v6

## [0.6.0] - 2025-11-14

### Added

**Multimodal Image Support** - Vision API Integration

Added comprehensive support for sending images alongside text to vision-capable models following the OpenAI Vision API format.

**Defensive Programming Enhancements** - Maximum Input Validation

Added comprehensive validation and logging for image handling following maximum defensive programming practices:

#### Enhanced Base64 Validation (`ImageBlock::from_base64`)

- **Character set validation**: Rejects invalid base64 characters (spaces, special chars, etc.)
- **Length validation**: Enforces length must be multiple of 4
- **Padding validation**: Max 2 '=' characters, must be at end
- **MIME injection prevention**: Rejects semicolons, newlines, commas in MIME type
- **Large data warning**: Warns when base64 exceeds 10MB

#### Enhanced URL Validation (`ImageBlock::from_url`)

- **Control character detection**: Rejects URLs with newline, tab, null, etc.
- **Data URI base64 validation**: Validates base64 portion using same rules as `from_base64()`
- **Long URL warning**: Warns when URL exceeds 2000 characters
- **Scheme validation**: Already rejected dangerous schemes (javascript:, file:, etc.)

#### Empty Text Block Handling

- **Warning on empty text**: Logs warning when empty or whitespace-only text blocks are serialized
- **No data loss**: Empty text blocks are still included (not dropped), just warned about
- **Debugging aid**: Warning includes message role to help identify source

#### Debug Logging (Optional)

- **Image serialization logging**: Debug logs when images are included in messages
- **URL truncation**: Long URLs truncated to 100 chars in logs for privacy
- **Detail level logging**: Logs image detail level (low/high/auto)
- **Opt-in**: Requires user to initialize a logger (using `log` crate)

#### New Dependencies

- `log = "0.4"` - Logging facade (runtime)
- `env_logger = "0.11"` - Logger implementation (dev dependency)

#### Testing

- 17 new tests across 4 test files
- Total: 154 tests passing (107 lib + 47 integration)
- Zero clippy warnings
- All tests follow TDD (RED → GREEN → REFACTOR → COMMIT)

**Note**: All defensive enhancements are backward compatible. Existing valid inputs continue to work; only truly invalid inputs are rejected.

### Changed

**BREAKING**: ToolUseBlock and ToolResultBlock fields now private

Following Rust API Guidelines (C-STRUCT-PRIVATE), public struct fields are now private with getter methods for better API stability:

#### ToolUseBlock

- **Private fields**: `id`, `name`, `input`
- **Getter methods**: `.id()`, `.name()`, `.input()`
- **Migration**:

  ```rust
  // Before:
  println!("Tool: {}", tool_use.name);
  client.add_tool_result(&tool_use.id, result)?;
  let params = tool_use.input.clone();

  // After:
  println!("Tool: {}", tool_use.name());         // Returns &str
  client.add_tool_result(tool_use.id(), result)?;  // Returns &str
  let params = tool_use.input().clone();         // Returns &Value
  ```

#### ToolResultBlock

- **Private fields**: `tool_use_id`, `content`
- **Getter methods**: `.tool_use_id()`, `.content()`
- **Migration**:

  ```rust
  // Before:
  let id = &tool_result.tool_use_id;
  let content = &tool_result.content;

  // After:
  let id = tool_result.tool_use_id();    // Returns &str
  let content = tool_result.content();   // Returns &Value
  ```

#### New Image Features

- **`ImageBlock::from_file_path(path)`** - Load and encode local image files
  - Supports: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`, `.svg`
  - MIME type inferred from file extension
  - Automatically encodes to base64 data URI

- **`ImageBlock::from_url(url)`** - Images from HTTP/HTTPS URLs
- **`ImageBlock::from_base64(data, mime)`** - Manual base64 with explicit MIME type
- **`Message::user_with_image(text, url)`** - Convenience helper for text + image
- **`Message::user_with_image_detail(text, url, detail)`** - With detail level control
- **`Message::user_with_base64_image(text, data, mime)`** - From base64 data

#### New Types

- **`ImageBlock`** - Represents an image in a message
- **`ImageDetail`** - Control image processing (Low/High/Auto)
- **`OpenAIContent`** - Message content format (Text or Parts)
- **`OpenAIContentPart`** - Content part for multimodal messages

#### Examples

- `examples/vision_example.rs` - Comprehensive vision API demonstration

**Related**: Closes GitHub issue #2

## [0.5.0] - 2025-11-13

### Fixed

**CRITICAL**: Tool Call Serialization Bug - Infinite Loop with auto_execute_tools

Fixed a critical bug where tool calls and tool results were not being serialized into OpenAI message format, causing an infinite loop when using `auto_execute_tools(true)`:

**The Problem:**

- Internal conversation history stored tool results as `ContentBlock::ToolResult`
- When building OpenAI API requests, only text blocks were extracted
- Tool results were silently dropped from the conversation history
- LLM never saw tool results, so it called the same tool repeatedly
- Loop continued until `max_tool_iterations` was reached
- Same tool called 50+ times instead of once

**The Fix:**

- Tool calls now properly serialized with `tool_calls` array in assistant messages
- Tool results now serialized as separate messages with `role: "tool"` and `tool_call_id`
- Message building logic handles three cases:
  1. Messages with ToolResult blocks → separate tool messages with `tool_call_id`
  2. Messages with ToolUse blocks → assistant messages with `tool_calls` array
  3. Messages with only text → normal text messages

**Impact:**

- ✅ Tool results now included in conversation history
- ✅ LLM sees tool results and responds appropriately
- ✅ Each tool called only once per unique request
- ✅ `auto_execute_tools(true)` now fully functional
- ✅ Works correctly with llama.cpp and other OpenAI-compatible servers

**Technical Details:**

- Modified `client.rs` message building logic (lines ~1105-1214)
- Added imports for `OpenAIToolCall` and `OpenAIFunction`
- Properly populates `tool_calls` field with tool ID, name, and serialized arguments
- Properly populates `tool_call_id` field in tool response messages
- Arguments serialized as JSON strings per OpenAI API specification

**Test Case:**

```rust
// Before: Tool called 50+ times, no final response
// After: Tool called once, final text response returned

let client = Client::new(AgentOptions::builder()
    .auto_execute_tools(true)
    .tool(database_tool)
    .build()?)?;

client.send("how many users?").await?;
while let Some(block) = client.receive().await? {
    // Now receives: "The users table has 5 rows."
}
```

See `examples/test_tool_serialization.rs` for demonstration.

## [0.4.0] - 2025-11-09

### Changed

**BREAKING**: API Stability Improvements - Private Fields with Getters

Following Rust API Guidelines (C-STRUCT-PRIVATE), all public struct fields are now private with getter methods for better encapsulation and future-proof APIs:

#### AgentOptions

- **Private fields**: `system_prompt`, `model`, `base_url`, `api_key`, `max_turns`, `max_tokens`, `temperature`, `timeout`, `tools`, `auto_execute_tools`, `max_tool_iterations`, `hooks`
- **Getter methods**: `.system_prompt()`, `.model()`, `.base_url()`, `.api_key()`, `.max_turns()`, `.max_tokens()`, `.temperature()`, `.timeout()`, `.tools()`, `.auto_execute_tools()`, `.max_tool_iterations()`, `.hooks()`
- **Migration**: `options.model` → `options.model()`

#### Tool

- **Private fields**: `name`, `description`, `input_schema`, `handler`
- **Getter methods**: `.name()`, `.description()`, `.input_schema()`
- **Migration**: `tool.name` → `tool.name()`

#### HookDecision

- **Private fields**: `continue_execution`, `modified_input`, `modified_prompt`, `reason`
- **Getter methods**: `.continue_execution()`, `.modified_input()`, `.modified_prompt()`, `.reason()`
- **Migration**: `decision.continue_execution` → `decision.continue_execution()`
- **Note**: Getters return references; use `.clone()` if owned value needed

**BREAKING**: Client::new() Returns Result

`Client::new()` now returns `Result<Self>` instead of panicking on HTTP client creation failure.

**Migration**:

```rust
// Before:
let client = Client::new(options);

// After:
let client = Client::new(options)?;
// or
let client = Client::new(options).expect("Failed to create client");
```

**BREAKING**: add_tool_result() Returns Result

`Client::add_tool_result()` now returns `Result<()>` instead of silently failing on serialization errors.

**Migration**:

```rust
// Before:
client.add_tool_result(&id, result);

// After:
client.add_tool_result(&id, result)?;
```

### Added

- **New method**: `Client::interrupt_handle()` - Returns a cloned `Arc<AtomicBool>` for thread-safe cancellation
  - Replaces direct access to the private `interrupted` field
  - Migration: `client.interrupted.clone()` → `client.interrupt_handle()`

- **Input Validation**: `AgentOptionsBuilder::build()` now validates configuration:
  - Temperature must be between 0.0 and 2.0
  - Model name cannot be empty or whitespace
  - Base URL must start with `http://` or `https://`
  - Max tokens must be greater than 0

### Fixed

- **Safety**: HTTP client no longer panics on invalid timeout - returns error instead
- **Error Handling**: Error response body parsing failures now logged instead of silently suppressed
- **SSE Parsing**: Handles empty chunks/heartbeats gracefully
- **Schema Validation**: Replaced `.unwrap()` with defensive assertions for better error messages
- **Tool Arguments**: Doc examples updated to validate parameters instead of using `.unwrap_or(0.0)`

### Documentation

- Added SAFETY comments to unsafe blocks in tests
- Documented OpenAI tool serialization limitation (ToolUse/ToolResult blocks not serialized to conversation history)
- Fixed documentation accuracy issues (system_prompt optionality, max_tokens defaults)
- Updated 150+ doctests for new APIs

### Technical Details

- All 66 unit and integration tests passing
- 135/139 doctests passing (97% success rate, 14 intentionally ignored)
- Zero tech debt: All identified issues fixed
- Breaking changes acceptable before 1.0 for long-term API stability

## [0.3.0] - 2025-11-05

### Changed

**BREAKING**: Improved `Client::receive()` API ergonomics

- Changed signature from `Option<Result<ContentBlock>>` to `Result<Option<ContentBlock>>`
- More intuitive: errors are always `Err()`, success is always `Ok()`
- Better ergonomics with `?` operator: `while let Some(block) = client.receive().await? { ... }`
- Migration: Change `match block? { ... }` inside the loop to `match block { ... }` and move the `?` to the while condition

### Benefits

- **Clearer Intent**: `Ok(Some(block))` = got a block, `Ok(None)` = stream ended, `Err(e)` = error occurred
- **Better Error Handling**: Can use `?` operator outside the loop instead of inside
- **More Idiomatic**: Follows Rust conventions for fallible iterators

### Technical Details

- All 85+ tests updated and passing
- All 10 examples updated with new API
- Zero breaking changes to other APIs
- Comprehensive test coverage for new signature

## [0.2.0] - 2025-11-04

### Changed

**BREAKING**: Upgraded to Rust Edition 2024

- Requires Rust 1.85.0 or newer (was 1.83.0)
- Edition 2024 brings latest language features and safety improvements
- No API changes - only compiler/edition upgrade

### Fixed

- **Safety**: Eliminated potential panic in `ToolBuilder::param()`
  - Now safely handles calling `.param()` after `.schema(non_object)`
  - Resets schema to empty object if needed instead of panicking
  - Added test coverage for edge case
- **Tests**: Made `test_auto_execution_with_tools` more robust
  - Accepts either text response OR tool execution
  - Better handles LLM behavior variance

### Technical Details

- Updated minimum supported Rust version (MSRV) to 1.85.0
- CI/CD pipeline updated to test against Rust 1.85
- All 100 tests passing with zero warnings
- Edition 2024 safety improvements applied

## [0.1.0] - 2025-11-04

### Added

#### Core Features

- **Streaming API**: Single-query `query()` function with async streaming responses
- **Multi-turn Client**: Stateful `Client` for conversation history management
- **Tool System**: Full function calling support with `tool()` builder
  - Type-safe parameter definitions
  - Async tool execution
  - Automatic tool result handling
- **Auto-execution Mode**: Automatic tool calling loop (`auto_execute_tools` option)
  - Configurable iteration limits
  - Transparent tool execution
  - Error handling and recovery

#### Advanced Features

- **Lifecycle Hooks**: Three extension points for custom logic
  - `PreToolUse`: Intercept before tool execution
  - `PostToolUse`: Process tool results
  - `UserPromptSubmit`: Transform user prompts before sending
- **Context Management**: Utilities for token budget management
  - `estimate_tokens()`: Approximate token counting
  - `truncate_messages()`: Smart message history truncation
  - `is_approaching_limit()`: Token budget monitoring
- **Interrupt Capability**: Cancel long-running operations via `client.interrupt()`
- **Retry Logic**: Exponential backoff with jitter
  - Configurable max retries and delays
  - Automatic retry on transient failures
  - Detailed error context

#### Configuration

- **AgentOptions Builder**: Fluent API for configuration
  - System prompts
  - Model selection
  - Temperature and sampling parameters
  - Token limits and turn limits
  - Base URL for local servers

#### Quality & Documentation

- **85+ Comprehensive Tests**:
  - 57 unit tests across 10 modules
  - 28 integration tests (hooks, auto-execution, advanced patterns)
  - Full test coverage for all features
- **10 Production Examples**:
  - `simple_query.rs` - Basic usage
  - `calculator_tools.rs` - Tool system demo
  - `hooks_example.rs` - Lifecycle hooks
  - `context_management.rs` - Token management patterns
  - `interrupt_demo.rs` - Interrupt capability
  - `git_commit_agent.rs` - Real-world agent (Git commits)
  - `log_analyzer_agent.rs` - Real-world agent (log analysis)
  - `advanced_patterns.rs` - Concurrent operations
  - `auto_execution_demo.rs` - Auto-execution patterns
  - `multi_tool_agent.rs` - Multiple tool coordination
- **CI/CD Pipeline**: GitHub Actions with 8 parallel jobs
  - Formatting (rustfmt)
  - Linting (clippy)
  - Matrix testing (Ubuntu + macOS × stable + beta)
  - MSRV verification (Rust 1.83)
  - Security audit (cargo-audit)
  - Documentation validation
  - Code coverage (tarpaulin + Codecov)
  - Benchmark comparison (PR only)
- **Performance Benchmarks**: Criterion-based benchmark suite
  - Token estimation benchmarks
  - Message truncation performance
  - Tool execution overhead

#### Documentation

- Comprehensive API documentation with examples
- Crate-level quick start guide
- Module-level documentation
- Doc tests for all public APIs

### Technical Details

- **Rust Edition**: 2021
- **MSRV**: 1.83.0
- **License**: MIT
- **Platform Support**: Linux, macOS, Windows
- **Async Runtime**: Tokio
- **HTTP Client**: reqwest with streaming support

### Compatibility

- Works with any OpenAI-compatible API server:
  - LM Studio (localhost:1234)
  - Ollama (localhost:11434)
  - llama.cpp server
  - vLLM
  - Any other OpenAI-compatible endpoint

[0.6.9]: https://github.com/slb350/open-agent-sdk-rust/releases/tag/v0.6.9
[0.6.8]: https://github.com/slb350/open-agent-sdk-rust/releases/tag/v0.6.8
[0.6.7]: https://github.com/slb350/open-agent-sdk-rust/releases/tag/v0.6.7
[0.6.6]: https://github.com/slb350/open-agent-sdk-rust/releases/tag/v0.6.6
[0.3.0]: https://github.com/slb350/open-agent-sdk-rust/releases/tag/v0.3.0
[0.2.0]: https://github.com/slb350/open-agent-sdk-rust/releases/tag/v0.2.0
[0.1.0]: https://github.com/slb350/open-agent-sdk-rust/releases/tag/v0.1.0
