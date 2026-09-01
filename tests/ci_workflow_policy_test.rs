const CI_WORKFLOW: &str = include_str!("../.github/workflows/ci.yml");
const SCHEDULED_AUDIT_WORKFLOW: &str = include_str!("../.github/workflows/scheduled-audit.yml");
const DEPENDABOT_CONFIG: &str = include_str!("../.github/dependabot.yml");
const PRE_COMMIT_HOOK: &str = include_str!("../.githooks/pre-commit");
const MUTANTS_RUN: &str = include_str!("../scripts/mutants-run.sh");
const MUTANTS_COMMON: &str = include_str!("../scripts/mutants-common.sh");
const MUTANTS_REMOTE: &str = include_str!("../scripts/mutants-remote.sh");
const MUTANTS_STAGED: &str = include_str!("../scripts/mutants-staged.sh");

/// Extracts the body of the named job from a workflow.
///
/// The slice runs from the job's header to the next top-level job header, found structurally
/// rather than by naming whichever job currently happens to follow. Slicing by the *next*
/// job's name couples every assertion to job ordering, so inserting a job forces edits to
/// unrelated tests — and a mis-edit silently widens what an assertion covers instead of
/// failing.
fn job<'a>(workflow: &'a str, name: &str) -> &'a str {
    let body = workflow
        .split_once(&format!("\n  {name}:\n"))
        .unwrap_or_else(|| panic!("missing workflow job: {name}"))
        .1;

    let end = body
        .match_indices('\n')
        .find(|(index, _)| starts_with_job_header(&body[index + 1..]))
        .map_or(body.len(), |(index, _)| index);

    &body[..end]
}

/// True when `text` begins with a top-level job header line.
///
/// A job header is exactly two spaces of indent, then a name with no spaces, then a colon —
/// which distinguishes it from both the four-space step keys inside a job and the two-space
/// `# comment` lines that introduce each one.
fn starts_with_job_header(text: &str) -> bool {
    let line = text.split('\n').next().unwrap_or_default().trim_end();

    line.starts_with("  ")
        && !line.starts_with("   ")
        && line.ends_with(':')
        && !line[2..].contains(' ')
}

#[test]
fn test_matrix_uses_native_github_hosted_runners() {
    let linux = job(CI_WORKFLOW, "test-linux");
    let macos = job(CI_WORKFLOW, "test-macos");

    assert!(linux.contains("runs-on: ubuntu-latest"));
    assert!(!linux.contains("macos-latest"));
    assert!(macos.contains("runs-on: macos-latest"));
    assert!(!macos.contains("github.server_url"));
}

#[test]
fn audit_jobs_install_and_verify_rust_before_running_cargo_audit_directly() {
    let push_audit = job(CI_WORKFLOW, "security");

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
fn msrv_check_covers_test_targets_and_dev_dependencies() {
    let msrv = job(CI_WORKFLOW, "msrv");

    // Without --all-targets the check skips tests and benches, so a dev-dependency requiring a
    // newer compiler would land unnoticed and break `cargo test` on the MSRV.
    assert!(msrv.contains("cargo check --all-features --all-targets --workspace"));
    assert!(msrv.contains("toolchain: \"1.85\""));
}

#[test]
fn dependabot_skips_the_wiremock_release_that_breaks_the_msrv() {
    assert!(
        DEPENDABOT_CONFIG
            .contains("- dependency-name: wiremock\n        versions:\n          - \"0.6.5\"")
    );
}

#[test]
fn mutation_sweep_is_scoped_for_test_changes_and_full_for_the_monthly_schedule() {
    let policy = job(CI_WORKFLOW, "mutation-policy");
    let mutants = job(CI_WORKFLOW, "mutants");

    assert!(CI_WORKFLOW.contains("schedule:\n    - cron: '37 9 15 * *'"));
    assert!(CI_WORKFLOW.contains("workflow_dispatch:"));
    assert!(policy.contains("id: test-policy"));
    assert!(policy.contains("./scripts/mutants-ci-scope.sh"));
    assert!(policy.contains("outputs:\n      run:"));
    assert!(policy.contains("mode: ${{ steps.test-policy.outputs.mode }}"));
    assert!(policy.contains("base: ${{ steps.test-policy.outputs.base }}"));
    assert!(mutants.contains("needs: mutation-policy"));
    assert!(mutants.contains("needs.mutation-policy.outputs.run == 'true'"));
    assert!(mutants.contains("mapfile -t mutation_scope"));
    assert!(mutants.contains("./scripts/mutants-run.sh \"${mutation_args[@]}\""));
    assert!(mutants.contains("name: Retain mutation repair evidence"));
    assert!(mutants.contains("if [[ \"$status\" -eq 2 && -s \"$missed\" ]]"));
    assert!(mutants.contains("${{ runner.temp }}/mutation-repair/missed.txt"));
    assert!(mutants.contains("actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"));
    // Exact tool version, and an immutable SHA pin with a version comment for the installer.
    assert!(mutants.contains("tool: cargo-mutants@27.1.0"));
    assert!(mutants.contains(
        "uses: taiki-e/install-action@742a3317eac7bd62f91cd888b4eead5e784ba833 # v2.87.1"
    ));
    assert!(mutants.contains("toolchain: stable"));
}

#[test]
fn the_mutation_verdict_comes_from_missed_txt_not_the_exit_code() {
    // cargo-mutants reports exit 3 (Timeout) in preference to exit 2 (FoundProblems), so a run
    // with one hanging mutant and one genuine survivor also exits 3. Deciding on the exit code
    // alone waves that survivor through. A timeout is a detection and must pass; a survivor
    // must fail.
    assert!(MUTANTS_RUN.contains("missed.txt"));
    assert!(
        MUTANTS_RUN.contains(r#"if [ -s "$MISSED" ]"#),
        "a non-empty missed.txt must fail the run"
    );
    assert!(
        MUTANTS_RUN.contains(r#"if [ "$status" -eq 3 ]"#),
        "a timeout with nothing missed must pass"
    );
}

#[test]
fn mutation_scratch_copies_stay_off_the_tmpfs() {
    // cargo-mutants creates one source-tree copy per job beneath $TMPDIR and removes those
    // copies only on a clean exit. The default root is checkout-adjacent so a killed run costs
    // disk, not Strix's tmpfs-backed /tmp. Every invocation owns a unique namespace, otherwise
    // one concurrent run's startup or EXIT cleanup can erase another run's live source tree.
    assert!(
        MUTANTS_RUN.contains(r#"SCRATCH_ROOT="${DREP_MUTANTS_TMPDIR:-${ROOT}.mutants-tmp}""#),
        "the default scratch root must stay beside the checkout"
    );
    assert!(
        MUTANTS_RUN.contains("mktemp -d \"$SCRATCH_ROOT/run_${HOST_ID}_$$_XXXXXX\""),
        "each run must own a unique scratch namespace"
    );
    assert!(
        MUTANTS_COMMON.contains(r#"kill -0 "$owner_pid""#),
        "stale cleanup must preserve a live owner"
    );
    assert!(
        MUTANTS_RUN.contains("trap cleanup_owned_scratch EXIT"),
        "normal, failed and trapped exits must clean only the owned namespace"
    );
    assert!(
        !MUTANTS_RUN.lines().any(|line| line.contains("rm -rf")),
        "the mutation runner must honor the repository deletion policy"
    );
}

#[test]
fn the_remote_sweep_never_silently_skips_itself() {
    // An unreachable host must fall back to a local run and say so. A commit gate that quietly
    // disables itself because the LAN blipped is worse than a slow one.
    assert!(MUTANTS_REMOTE.contains("is unreachable"));
    assert!(MUTANTS_REMOTE.contains("run_local"));
    // The transport does not re-derive the verdict; it invokes the shared script and
    // propagates its exit code.
    assert!(MUTANTS_REMOTE.contains("./scripts/mutants-run.sh"));
    assert!(MUTANTS_REMOTE.contains(r#"exit "$status""#));
    assert!(MUTANTS_REMOTE.contains("REMOTE_RUN_DIR="));
    assert!(MUTANTS_REMOTE.contains("LOCAL_RESULTS="));
    assert!(MUTANTS_REMOTE.contains("DREP_MUTANTS_TMPDIR="));
    assert!(
        !MUTANTS_REMOTE.contains("2>/dev/null || true"),
        "result-mirroring failures must never be hidden"
    );
    // The staged diff lands under target/, the one directory the sync excludes, so it has to
    // be named as a file the run needs.
    assert!(MUTANTS_STAGED.contains("MUTANTS_EXTRA_FILE"));
    assert!(MUTANTS_STAGED.contains("--in-diff"));
    assert!(MUTANTS_STAGED.contains("mktemp"));
}

#[test]
fn pre_commit_hook_runs_the_same_checks_as_ci() {
    // A hook weaker than CI lets a commit pass locally and fail remotely — the failure it
    // exists to prevent. Only the mutation scope may differ (staged diff vs full sweep).
    for command in [
        "cargo fmt --all -- --check",
        "cargo clippy --all-targets --all-features -- -D warnings",
        "cargo test --all-features --all",
    ] {
        assert!(
            PRE_COMMIT_HOOK.contains(command),
            "pre-commit hook must run `{command}` exactly as CI does"
        );
        assert!(
            CI_WORKFLOW.contains(command),
            "CI must run `{command}`; update the hook if this moved"
        );
    }

    // The cargo-mutants version is pinned in both places and must not drift.
    assert!(PRE_COMMIT_HOOK.contains("CARGO_MUTANTS_VERSION=\"27.1.0\""));
    // Scoped to the staged diff, through the same verdict script CI uses.
    assert!(PRE_COMMIT_HOOK.contains("./scripts/mutants-staged.sh"));
    assert!(MUTANTS_STAGED.contains("./scripts/mutants-remote.sh"));
}

#[test]
fn coverage_uses_pinned_tarpaulin_with_the_unprivileged_llvm_engine() {
    let coverage = job(CI_WORKFLOW, "coverage");

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
