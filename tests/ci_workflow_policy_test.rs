const CI_WORKFLOW: &str = include_str!("../.github/workflows/ci.yml");
const SCHEDULED_AUDIT_WORKFLOW: &str = include_str!("../.github/workflows/scheduled-audit.yml");

fn between<'a>(workflow: &'a str, start: &str, end: &str) -> &'a str {
    workflow
        .split_once(start)
        .unwrap_or_else(|| panic!("missing workflow section: {start}"))
        .1
        .split_once(end)
        .unwrap_or_else(|| panic!("missing workflow section boundary: {end}"))
        .0
}

#[test]
fn macos_tests_run_only_on_github_hosted_runners() {
    let linux = between(CI_WORKFLOW, "  test-linux:\n", "  test-macos:\n");
    let macos = between(CI_WORKFLOW, "  test-macos:\n", "  msrv:\n");

    assert!(linux.contains("runs-on: ubuntu-latest"));
    assert!(!linux.contains("macos-latest"));
    assert!(macos.contains("runs-on: macos-latest"));
    assert!(
        macos.contains("if: github.server_url == 'https://github.com'"),
        "the macOS job must be skipped before Gitea tries to assign a runner"
    );
}

#[test]
fn audit_jobs_install_and_verify_rust_before_running() {
    let push_audit = between(CI_WORKFLOW, "  security:\n", "  docs:\n");

    for workflow in [push_audit, SCHEDULED_AUDIT_WORKFLOW] {
        let toolchain = workflow
            .find("uses: dtolnay/rust-toolchain@")
            .expect("audit workflow must install Rust");
        let verification = workflow
            .find("name: Verify Rust toolchain")
            .expect("audit workflow must verify the Rust installation");
        let audit = workflow
            .find("uses: actions-rust-lang/audit@")
            .expect("audit workflow must run cargo-audit");

        assert!(toolchain < verification && verification < audit);
        assert!(workflow.contains("toolchain: stable"));
        assert!(workflow.contains("rustc --version\n          cargo --version\n"));
        assert!(workflow.contains("denyWarnings: true"));
        assert!(workflow.contains("createIssues: false"));
    }
}

#[test]
fn coverage_uses_pinned_tarpaulin_with_the_unprivileged_llvm_engine() {
    let coverage = between(CI_WORKFLOW, "  coverage:\n", "  benchmarks:\n");

    assert!(coverage.contains("cargo install cargo-tarpaulin --version =0.37.0"));
    assert!(!coverage.contains("cargo install --locked cargo-tarpaulin"));
    assert!(
        coverage.contains("cargo tarpaulin --engine llvm --out xml --all-features --workspace")
    );
    assert!(coverage.contains("if-no-files-found: error"));
    assert!(!coverage.contains("seccomp=unconfined"));
    assert!(!coverage.contains("--privileged"));
}
