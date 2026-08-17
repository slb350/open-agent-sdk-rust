//! Coverage for the environment-dependent half of the provider helpers.
//!
//! These live in their own integration test binary because they mutate process-global
//! environment state. Cargo runs each integration test file in a separate process, and the
//! single test function below sequences every case, so no other test can observe a partially
//! applied `OPEN_AGENT_MODEL`.

use open_agent::get_model;

const MODEL_VAR: &str = "OPEN_AGENT_MODEL";

/// Sets `OPEN_AGENT_MODEL`, or removes it when `value` is `None`.
///
/// # Safety
///
/// `set_var`/`remove_var` are unsound if another thread reads the environment concurrently.
/// This binary's only test is single-threaded and no spawned threads read the environment.
fn set_model_var(value: Option<&str>) {
    unsafe {
        match value {
            Some(model) => std::env::set_var(MODEL_VAR, model),
            None => std::env::remove_var(MODEL_VAR),
        }
    }
}

#[test]
fn get_model_resolves_env_and_fallback_in_priority_order() {
    set_model_var(None);

    // No environment, no fallback: nothing to resolve.
    assert_eq!(get_model(None, true), None);
    assert_eq!(get_model(None, false), None);

    // No environment: the explicit fallback is used regardless of the preference flag.
    assert_eq!(get_model(Some("llama3:8b"), true), Some("llama3:8b".into()));
    assert_eq!(
        get_model(Some("llama3:8b"), false),
        Some("llama3:8b".into())
    );

    set_model_var(Some("qwen2.5-32b-instruct"));

    // Environment preferred: it wins over the fallback, and stands in when there is none.
    assert_eq!(
        get_model(Some("llama3:8b"), true),
        Some("qwen2.5-32b-instruct".into())
    );
    assert_eq!(get_model(None, true), Some("qwen2.5-32b-instruct".into()));

    // Environment not preferred: it is ignored entirely, even when set.
    assert_eq!(
        get_model(Some("llama3:8b"), false),
        Some("llama3:8b".into())
    );
    assert_eq!(get_model(None, false), None);

    set_model_var(None);
}
