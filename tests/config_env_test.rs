//! Environment-dependent provider helpers run in this dedicated test process.
//!
//! Keep all environment mutations in the single test below: splitting these cases
//! into independently scheduled tests would expose shared environment state.

use open_agent::{Provider, get_base_url, get_model};

const MODEL_VAR: &str = "OPEN_AGENT_MODEL";
const URL_VAR: &str = "OPEN_AGENT_BASE_URL";

/// Sets or removes one provider environment variable.
///
/// # Safety
///
/// `set_var`/`remove_var` are unsound if another thread reads the environment concurrently.
/// This binary's only test is single-threaded and no spawned threads read the environment.
fn set_env_var(key: &str, value: Option<&str>) {
    unsafe {
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }
}

#[test]
fn provider_helpers_resolve_env_and_fallback_in_priority_order() {
    set_env_var(MODEL_VAR, None);

    set_env_var(URL_VAR, None);
    assert_eq!(
        get_base_url(Some(Provider::Ollama), None),
        "http://localhost:11434/v1"
    );
    assert_eq!(
        get_base_url(None, Some("http://custom:8080/v1")),
        "http://custom:8080/v1"
    );

    // No environment, no fallback: nothing to resolve.
    assert_eq!(get_model(None, true), None);
    assert_eq!(get_model(None, false), None);

    // No environment: the explicit fallback is used regardless of the preference flag.
    assert_eq!(get_model(Some("llama3:8b"), true), Some("llama3:8b".into()));
    assert_eq!(
        get_model(Some("llama3:8b"), false),
        Some("llama3:8b".into())
    );

    set_env_var(MODEL_VAR, Some("qwen2.5-32b-instruct"));

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

    set_env_var(MODEL_VAR, None);
}
