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
fn test_matrix_uses_native_github_hosted_runners() {
    let linux = between(CI_WORKFLOW, "  test-linux:\n", "  test-macos:\n");
    let macos = between(CI_WORKFLOW, "  test-macos:\n", "  msrv:\n");

    assert!(linux.contains("runs-on: ubuntu-latest"));
    assert!(!linux.contains("macos-latest"));
    assert!(macos.contains("runs-on: macos-latest"));
    assert!(!macos.contains("github.server_url"));
}

#[test]
fn audit_jobs_install_and_verify_rust_before_running_cargo_audit_directly() {
    let push_audit = between(CI_WORKFLOW, "  security:\n", "  docs:\n");

    for workflow in [push_audit, SCHEDULED_AUDIT_WORKFLOW] {
        let toolchain = workflow
            .find("uses: dtolnay/rust-toolchain@")
            .expect("audit workflow must install Rust");
        let verification = workflow
            .find("name: Verify Rust toolchain")
            .expect("audit workflow must verify the Rust installation");
        let install = workflow
            .find("cargo install cargo-audit --version '=0.22.2' --no-default-features")
            .expect("audit workflow must install the exact cargo-audit release");
        let audit = workflow
            .find("cargo audit --deny warnings")
            .expect("audit workflow must deny every cargo-audit warning");

        assert!(toolchain < verification && verification < install && install < audit);
        assert!(workflow.contains("toolchain: stable"));
        assert!(workflow.contains("rustc --version\n          cargo --version\n"));
        assert!(!workflow.contains("actions-rust-lang/audit@"));
        assert!(!workflow.contains("cargo install --locked cargo-audit"));
        assert!(!workflow.contains("shell: python"));
    }
}

#[test]
fn coverage_uses_pinned_tarpaulin_with_the_unprivileged_llvm_engine() {
    let coverage = between(CI_WORKFLOW, "  coverage:\n", "  benchmarks:\n");

    assert!(coverage.contains("cargo install cargo-tarpaulin --version =0.37.2"));
    assert!(!coverage.contains("cargo install --locked cargo-tarpaulin"));
    assert!(
        coverage.contains("cargo tarpaulin --engine llvm --out xml --all-features --workspace")
    );
    assert!(coverage.contains("if-no-files-found: error"));
    let report_check = coverage
        .find("test -s cobertura.xml")
        .expect("coverage must fail when its XML report is missing or empty");
    let artifact_upload = coverage
        .find("uses: actions/upload-artifact@")
        .expect("GitHub coverage must retain the report as an artifact");
    assert!(report_check < artifact_upload);
    assert!(!coverage.contains("github.server_url"));
    assert!(!coverage.contains("seccomp=unconfined"));
    assert!(!coverage.contains("--privileged"));
}
