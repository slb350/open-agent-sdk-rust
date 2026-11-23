//! Consolidated SDK integration tests.
//!
//! Compiling stateless SDK integration tests in one crate avoids linking 20 separate
//! ~15 MB executables on every test build and every mutation run. Tests requiring
//! process-global isolation (`config_env_test.rs`, `debug_logging_test.rs`) remain
//! standalone binaries.

mod common;

mod integration {
    mod anthropic_protocol;
    mod auto_execution;
    mod context_estimation;
    mod custom_headers;
    mod endpoint_tolerance;
    mod hooks_history_snapshot;
    mod hooks_integration;
    mod redirect_policy;
    mod regression_chunk_metadata;
    mod regression_client_lifecycle;
    mod regression_finish_reason;
    mod regression_incremental_streaming;
    mod regression_max_tokens;
    mod regression_reasoning_channel;
    mod regression_retry_classification;
    mod regression_stream_error;
    mod regression_stream_flush;
    mod regression_tool_call_index;
    mod send_message;
    mod tool_call_content;
}
