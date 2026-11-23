//! Repository-only policy, packaging, architecture, and mutation-script tests.
//!
//! Excluded from published crate archives (see `Cargo.toml`) and skipped by
//! `.cargo/mutants.toml` because none of these tests exercise `src/` library code.

#[cfg(unix)]
#[path = "support/mutation.rs"]
mod mutation;
#[cfg(unix)]
#[path = "support/process.rs"]
mod process;

#[path = "repo_policy/ci_workflow_policy.rs"]
mod ci_workflow_policy;
#[cfg(unix)]
#[path = "repo_policy/mutation_ci_scope.rs"]
mod mutation_ci_scope;
#[cfg(unix)]
#[path = "repo_policy/mutation_locks.rs"]
mod mutation_locks;
#[cfg(unix)]
#[path = "repo_policy/mutation_scripts.rs"]
mod mutation_scripts;
#[path = "repo_policy/package_manifest.rs"]
mod package_manifest;
#[path = "repo_policy/source_file_size.rs"]
mod source_file_size;
