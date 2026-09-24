#![cfg(unix)]
//! The mutation scripts are drep's (`~/dev/drep/scripts/`), copied with this
//! repository's ai-1 role, lock and workspace. These tests pin what the copy
//! must keep: ai-1 as the only offload host, one host lock shared by hosted and
//! laptop runs, results proven fresh, scratch beside the checkout, a sandboxed
//! transport, and a verdict read from `missed.txt` rather than the exit code.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

#[path = "support/process.rs"]
mod process;
use process::{bash_with_fakes, repo_root, write_executable};

const ROLE: &str = "open-agent-sdk-rust-mutants";

fn without_comments(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
}

struct Harness {
    temp: TempDir,
}

impl Harness {
    fn new() -> Self {
        let temp = tempfile::Builder::new()
            .prefix("open agent mutants ")
            .tempdir()
            .expect("create harness directory");
        fs::create_dir_all(temp.path().join("bin")).expect("create fake binary directory");
        Self { temp }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.temp.path().join(name)
    }

    /// Runs `scripts/mutants-run.sh` with a fake cargo that writes `missed.txt`.
    fn run_script(&self, scratch: &Path) -> Command {
        write_executable(
            &self.path("bin").join("cargo"),
            r#"#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "$FAKE_ARGS"
mkdir -p "$TMPDIR/cargo-mutants-trap.tmp/nested"
mkdir -p "$MUTANTS_OUT_DIR/mutants.out"
printf '%s' "${FAKE_MISSED:-}" > "$MUTANTS_OUT_DIR/mutants.out/missed.txt"
exit "${FAKE_EXIT:-0}"
"#,
        );
        let mut command = bash_with_fakes(&self.path("bin"));
        command
            .arg(repo_root().join("scripts/mutants-run.sh"))
            .env("DREP_MUTANTS_TMPDIR", scratch)
            .env("MUTANTS_OUT_DIR", self.path("mutation output"))
            .env("FAKE_ARGS", self.path("cargo-args"))
            .env_remove("DREP_MUTANTS_HOST_LOCK")
            .env_remove("DREP_MUTANTS_HOST_LOCK_WAIT_SECONDS")
            .env_remove("DREP_MUTANTS_RESULT_TOKEN");
        command
    }
}

#[test]
fn remote_mutation_defaults_to_this_repositorys_ai1_role() {
    let script = without_comments("scripts/mutants-remote.sh");
    for expected in [
        "HOST=\"${DREP_MUTANTS_HOST:-steve@192.168.68.88}\"".to_owned(),
        format!("REMOTE_DIR=\"${{DREP_MUTANTS_DIR:-.cache/{ROLE}/$(basename \"$PWD\")}}\""),
        format!("DREP_MUTANTS_REMOTE_HOST_LOCK:-/srv/ci/fleet/{ROLE}/home/host.lock"),
        format!("AI1_CI_ROLE={ROLE}"),
    ] {
        assert!(
            script.contains(&expected),
            "developer offload must default to this repository's ai-1 role: {expected}"
        );
    }
    let transport = without_comments("scripts/mutants-ai1-transport.sh");
    assert!(
        transport.contains(&format!("  {ROLE}) return 0 ;;")),
        "the transport must admit exactly this repository's role"
    );
    for retired in ["strix", "homelab-1.", "homelab-2", "legion"] {
        assert!(
            !script.to_lowercase().contains(retired),
            "every mutation workload runs on ai-1; {retired} must never return as a default"
        );
    }
}

/// A no-argument full sweep must remain a genuinely empty cargo-mutants scope.
#[test]
fn remote_full_mutation_sweep_passes_no_phantom_argument() {
    let script = without_comments("scripts/mutants-remote.sh");
    assert!(
        script.contains("for remote_arg in")
            && script.contains("shift 6")
            && script.contains("./scripts/mutants-run.sh \"$@\""),
        "the remote wrapper must preserve an empty post-transport argument vector"
    );
    assert!(
        !script.contains("$(printf '%q ' \"$@\")"),
        "empty positional parameters must not be formatted into a literal empty argument"
    );
}

#[test]
fn remote_mutation_session_owns_sync_run_and_fresh_result_mirroring() {
    let script = without_comments("scripts/mutants-remote.sh");
    assert!(
        script.contains("exec 9>\"$host_lock\"")
            && script.contains("flock -E 75 -w \"$wait_seconds\" 9"),
        "developer and hosted mutation must share the ai-1 host lock"
    );
    assert!(
        script.contains("DREP_MUTANTS_HOST_LOCK_WAIT_SECONDS")
            && script.contains("DREP_MUTANTS_RSYNC_TIMEOUT_SECONDS")
            && script.contains("--timeout=\"$RSYNC_IO_TIMEOUT_SECONDS\""),
        "remote lock and transfer waits must remain explicitly bounded"
    );
    assert!(
        script.contains("mkfifo \"$CONTROL_IN\" \"$CONTROL_OUT\"")
            && script.contains("mutants-lock-ready:$RUN_TOKEN")
            && script.contains("mutants-run-finished:$RUN_TOKEN")
            && script.contains("DREP_MUTANTS_RESULT_TOKEN")
            && script.contains(".run-token")
            && script.contains("printf 'mirrored\\n'"),
        "one remote lock session must prove that mirrored results belong to the current run"
    );
    assert!(
        script.contains("kill \"$REMOTE_SESSION_PID\"")
            && script.contains("wait \"$REMOTE_SESSION_PID\""),
        "abnormal local exit must terminate and reap the remote lock session"
    );
    let session_start = script
        .find("REMOTE_SESSION_PID=$!")
        .expect("remote session PID assignment must exist");
    let source_sync = script
        .find("rsync -a --delete")
        .expect("source synchronization must exist");
    assert!(
        session_start < source_sync,
        "the host lock must be acquired before source synchronization begins"
    );
}

#[test]
fn mutation_runner_holds_the_configured_host_lock() {
    let script = without_comments("scripts/mutants-run.sh");
    assert!(
        script.contains("DREP_MUTANTS_HOST_LOCK")
            && script.contains("validate_mutants_host_lock_wait_seconds mutants-run")
            && script.contains("flock -w")
            && script.contains("exec 9>\"$HOST_LOCK\""),
        "a configured mutation host must serialize GitHub and laptop-offloaded sweeps"
    );
    assert!(
        script.contains("DREP_MUTANTS_RESULT_TOKEN")
            && script.contains("$OUT_DIR/mutants.out")
            && script.contains("$OUT_DIR/.run-token"),
        "each remote run must clear stale output and publish its own freshness token"
    );
}

#[test]
fn mutation_host_lock_wait_policy_has_one_definition() {
    let common = without_comments("scripts/mutants-common.sh");
    assert!(
        common.contains(
            "MUTANTS_HOST_LOCK_WAIT_SECONDS=\"${DREP_MUTANTS_HOST_LOCK_WAIT_SECONDS:-1800}\""
        ) && common.contains("validate_mutants_host_lock_wait_seconds()"),
        "the shared mutation layer must own the host-lock wait default and validation"
    );
    for name in ["mutants-remote", "mutants-run"] {
        let script = without_comments(&format!("scripts/{name}.sh"));
        assert!(
            script.contains(&format!("validate_mutants_host_lock_wait_seconds {name}")),
            "{name} must invoke the shared host-lock wait validator"
        );
        assert!(
            !script.contains(
                "HOST_LOCK_WAIT_SECONDS=\"${DREP_MUTANTS_HOST_LOCK_WAIT_SECONDS:-1800}\""
            ),
            "{name} must not redefine the shared host-lock wait policy"
        );
    }
}

#[test]
fn ai1_transport_fails_closed_without_bypassing_the_sandbox() {
    let output = Command::new("bash")
        .arg("tests/ai1-transport.sh")
        .current_dir(repo_root())
        .output()
        .expect("transport contract must execute");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Mutation scratch copies live beside the checkout, never in the system temp dir.
#[test]
fn mutation_scratch_copies_stay_off_the_tmpfs() {
    let script = without_comments("scripts/mutants-run.sh");
    assert!(
        script.contains("export TMPDIR=\"${DREP_MUTANTS_TMPDIR:-${ROOT}.mutants-tmp}\""),
        "the shared runner must place scratch copies beside the checkout"
    );
    assert!(
        !script
            .lines()
            .any(|line| line.contains("TMPDIR=") && line.contains("/tmp")),
        "scratch copies must never default to the system temp dir"
    );
    assert!(
        !script.contains("rm ") && !script.contains("rmdir "),
        "mutation cleanup must never invoke rm or rmdir"
    );
    assert!(
        script.contains("trap cleanup_mutation_scratch EXIT"),
        "the run must remove its own copies on exit"
    );
}

/// The cleanup expression is destructive only inside its known prefix.
#[test]
fn mutation_scratch_cleanup_preserves_adjacent_state() {
    let harness = Harness::new();
    let scratch = harness.path("scratch root");
    let stale = scratch.join("cargo-mutants-stale.tmp/nested");
    let adjacent = scratch.join("cargo-mutants-stale.tmp.keep");
    let outside = harness.path("outside");
    fs::create_dir_all(&stale).expect("stale scratch tree");
    fs::write(stale.join("file"), "stale").expect("stale scratch file");
    fs::create_dir_all(&adjacent).expect("adjacent directory");
    fs::write(adjacent.join("keep"), "keep").expect("adjacent file");
    fs::create_dir_all(&outside).expect("outside directory");
    fs::write(outside.join("keep"), "keep").expect("outside file");
    std::os::unix::fs::symlink(&outside, scratch.join("cargo-mutants-link.tmp"))
        .expect("scratch symlink");

    let output = harness
        .run_script(&scratch)
        .output()
        .expect("run mutation wrapper with fake cargo");

    assert!(output.status.success(), "{output:?}");
    assert!(!scratch.join("cargo-mutants-stale.tmp").exists());
    assert!(!scratch.join("cargo-mutants-trap.tmp").exists());
    assert!(!scratch.join("cargo-mutants-link.tmp").exists());
    assert!(adjacent.join("keep").exists());
    assert!(outside.join("keep").exists());

    let args = fs::read_to_string(harness.path("cargo-args")).expect("captured cargo arguments");
    let timeout_values = args
        .lines()
        .collect::<Vec<_>>()
        .windows(2)
        .filter(|pair| pair[0] == "--minimum-test-timeout")
        .map(|pair| pair[1])
        .collect::<Vec<_>>();
    assert_eq!(
        timeout_values,
        ["120"],
        "the executed mutation command needs one exact test-timeout floor"
    );
}

#[test]
fn mutation_verdict_prioritizes_survivors_over_timeouts() {
    for (status, missed, expected) in [
        (0, "", 0),
        (3, "", 0),
        (3, "survivor", 2),
        (0, "survivor", 2),
        (7, "", 7),
    ] {
        let harness = Harness::new();
        let output = harness
            .run_script(&harness.path("scratch root"))
            .env("FAKE_EXIT", status.to_string())
            .env("FAKE_MISSED", missed)
            .output()
            .expect("run mutation verdict");
        assert_eq!(
            output.status.code(),
            Some(expected),
            "status={status}, missed={missed}: {output:?}"
        );
    }
}

#[test]
fn pre_commit_executes_checks_and_stops_at_a_failed_check() {
    let harness = Harness::new();
    let scripts = harness.path("scripts");
    fs::create_dir_all(&scripts).unwrap();
    let log = harness.path("checks.log");
    let shim = r#"#!/usr/bin/env bash
set -euo pipefail
printf '%s %s\n' "${0##*/}" "$*" >> "$FAKE_CHECKS"
if [ "${FAKE_FAIL:-}" = "$1" ]; then exit 7; fi
"#;
    write_executable(&harness.path("bin").join("cargo"), shim);
    write_executable(&harness.path("bin").join("cargo-mutants"), shim);
    write_executable(
        &scripts.join("mutants-staged.sh"),
        "#!/usr/bin/env bash\nprintf 'mutants\n' >> \"$FAKE_CHECKS\"\n",
    );
    for (failure, expected) in [
        ("fmt", vec!["cargo fmt --all -- --check"]),
        (
            "",
            vec![
                "cargo fmt --all -- --check",
                "cargo clippy --all-targets --all-features -- -D warnings",
                "cargo test --all-features --all",
                "mutants",
            ],
        ),
    ] {
        fs::write(&log, "").unwrap();
        let output = bash_with_fakes(&harness.path("bin"))
            .arg(repo_root().join(".githooks/pre-commit"))
            .current_dir(harness.temp.path())
            .env("FAKE_CHECKS", &log)
            .env("FAKE_FAIL", failure)
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(if failure.is_empty() { 0 } else { 7 }),
            "{output:?}"
        );
        assert_eq!(
            fs::read_to_string(&log)
                .unwrap()
                .lines()
                .collect::<Vec<_>>(),
            expected
        );
    }
}

/// A command that must not see the Git environment of a hook this suite may be running under.
fn isolated(program: &str, repository: &Path) -> Command {
    let mut command = Command::new(program);
    command.current_dir(repository);
    for variable in [
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_WORK_TREE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_COMMON_DIR",
    ] {
        command.env_remove(variable);
    }
    command
}

fn git(repository: &Path, arguments: &[&str]) -> String {
    let output = isolated("git", repository)
        .args(arguments)
        .output()
        .expect("run git");
    assert!(output.status.success(), "git {arguments:?}: {output:?}");
    String::from_utf8(output.stdout).expect("git output is UTF-8")
}

#[test]
fn staged_gate_refuses_unstaged_or_untracked_inputs_and_dispatches_a_matching_index() {
    let harness = Harness::new();
    let repository = harness.path("repository");
    let events = harness.path("events");
    fs::create_dir_all(repository.join("scripts")).unwrap();
    for name in ["mutants-common.sh", "mutants-staged.sh"] {
        fs::copy(
            repo_root().join("scripts").join(name),
            repository.join("scripts").join(name),
        )
        .unwrap();
    }
    write_executable(
        &repository.join("scripts/mutants-remote.sh"),
        "#!/usr/bin/env bash\nprintf 'remote:%s extra:%s\\n' \"$*\" \"$MUTANTS_EXTRA_FILES\" >> \"$FAKE_EVENTS\"\n",
    );
    fs::write(repository.join(".gitignore"), "target/\n").unwrap();
    fs::write(repository.join("source.rs"), "fn source() {}\n").unwrap();
    git(&repository, &["init", "-q", "-b", "main"]);
    git(&repository, &["add", "."]);
    git(
        &repository,
        &[
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ],
    );
    fs::write(repository.join("source.rs"), "fn staged() {}\n").unwrap();
    git(&repository, &["add", "source.rs"]);
    let staged = || {
        isolated("bash", &repository)
            .arg("scripts/mutants-staged.sh")
            .env("FAKE_EVENTS", &events)
            .output()
            .expect("run staged gate")
    };

    for (kind, path) in [
        ("unstaged", repository.join("source.rs")),
        ("untracked", repository.join("new_test.rs")),
    ] {
        fs::write(&path, "#[test]\nfn unstaged_test() {}\n").unwrap();
        let output = staged();
        assert_ne!(output.status.code(), Some(0), "{kind}: {output:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("index"),
            "{kind}: {output:?}"
        );
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            "#[test]\nfn unstaged_test() {}\n"
        );
        assert_eq!(
            git(&repository, &["show", ":source.rs"]),
            "fn staged() {}\n"
        );
        assert!(!events.exists(), "{kind}: refused before any remote run");
        if kind == "unstaged" {
            fs::write(&path, "fn staged() {}\n").unwrap();
        } else {
            fs::remove_file(&path).unwrap();
        }
    }

    let output = staged();
    assert!(output.status.success(), "{output:?}");
    let diff = "target/mutants/staged.diff";
    assert!(
        fs::read_to_string(repository.join(diff))
            .unwrap()
            .contains("+fn staged() {}")
    );
    assert_eq!(
        fs::read_to_string(&events).unwrap(),
        format!("remote:--in-diff {diff} extra:{diff}\n")
    );
}
