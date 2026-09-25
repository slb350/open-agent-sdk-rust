#![cfg(unix)]
//! The mutation scripts are drep's (`~/dev/drep/scripts/`), copied with this repository's ai-1 role. These tests pin what the copy must keep: ai-1 as the only offload host, one host lock shared by hosted and laptop runs, results proven fresh, scratch beside the checkout, a sandboxed transport, and a verdict read from `missed.txt` rather than the exit code. The locks' behavior is exercised in `mutation_locks_test.rs`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[path = "support/mutation.rs"]
mod mutation;
#[path = "support/process.rs"]
mod process;
use mutation::{Harness, ROLE, away_from_outer_git, git, without_comments};
use process::{bash_with_fakes, repo_root, write_executable};

#[test]
fn remote_mutation_defaults_to_this_repositorys_ai1_role() {
    let script = without_comments("scripts/mutants-remote.sh");
    for expected in [
        "HOST=\"${DREP_MUTANTS_HOST:-steve@192.168.68.88}\"".to_owned(),
        format!("AI1_CI_ROLE={ROLE}"),
        "REMOTE_DIR=\"$(remote_checkout_dir \"$AI1_CI_ROLE\")\"".to_owned(),
    ] {
        assert!(
            script.contains(&expected),
            "developer offload must default to this repository's ai-1 role: {expected}"
        );
    }
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
            && script.contains("shift 5")
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
        script.contains("exec 9>>\"${DREP_MUTANTS_HOST_LOCK:?")
            && script.contains("flock -E 75 -w \"$wait_seconds\" 9")
            && !script.contains("unset DREP_MUTANTS_HOST_LOCK"),
        "the offloaded run must take the role's host lock, the one hosted sweeps take, and hand it to the run on descriptor 9"
    );
    assert!(
        script.contains("\"$MUTANTS_HOST_LOCK_WAIT_SECONDS\"")
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
            && script.contains("wait \"$REMOTE_SESSION_PID\"")
            && script.contains("trap 'exit 74' PIPE"),
        "abnormal local exit, a dead session included, must terminate and reap the remote lock session"
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

/// The checkout lock is taken before the host is probed, so a run that waited for it does not act on a probe that is half an hour old.
#[test]
fn remote_mutation_takes_the_checkout_lock_before_probing_the_host() {
    let script = without_comments("scripts/mutants-remote.sh");
    let lock = script
        .find("acquire_checkout_lock mutants-remote")
        .expect("checkout lock");
    let probe = script
        .find("ssh -o BatchMode=yes -o ConnectTimeout=5")
        .expect("host probe");
    assert!(lock < probe);
}

/// The source sync mirrors this checkout with --delete, so its remote directory is named for this machine and the checkout's path: two checkouts never share one, and holding the checkout lock is all it takes to own it.
#[test]
fn checkouts_with_one_name_get_their_own_remote_directories() {
    let harness = Harness::new();
    write_executable(
        &harness.path("bin").join("hostname"),
        "#!/bin/sh\necho other-machine\n",
    );
    let remote_dir_on = |parent: &str, name: &str, machine: bool| {
        let scripts = harness.path(parent).join(name).join("scripts");
        fs::create_dir_all(&scripts).expect("scripts directory");
        fs::copy(
            repo_root().join("scripts/mutants-common.sh"),
            scripts.join("mutants-common.sh"),
        )
        .expect("copy mutation script");
        let mut command = Command::new("bash");
        command
            .args([
                "-c",
                &format!(". \"$1/mutants-common.sh\" && remote_checkout_dir {ROLE}"),
                "remote-dir-test",
            ])
            .arg(&scripts);
        if machine {
            command.env("PATH", process::prepend_path(&harness.path("bin")));
        }
        let output = command.output().expect("derive the remote directory");
        assert!(output.status.success(), "{output:?}");
        String::from_utf8(output.stdout).expect("utf-8 directory")
    };
    let remote_dir = |parent: &str, name: &str| remote_dir_on(parent, name, false);

    let first = remote_dir("one", "sdk");
    let second = remote_dir("two", "sdk");
    let odd = remote_dir("three", "my repo;x");

    let cache = format!(".cache/{ROLE}/");
    for dir in [&first, &second, &odd] {
        let name = dir
            .strip_prefix(&cache)
            .unwrap_or_else(|| panic!("{dir} must sit in the role's cache"));
        assert!(
            !name.is_empty()
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"-_.".contains(&byte))
                && name != "."
                && name != "..",
            "{dir} must be one plain directory name"
        );
    }
    assert!(first.starts_with(&format!("{cache}sdk-")));
    assert!(odd.starts_with(&format!("{cache}myrepox-")));
    assert_ne!(
        first, second,
        "same-named checkouts must not share a directory"
    );
    assert_eq!(
        first,
        remote_dir("one", "sdk"),
        "a checkout keeps its directory"
    );
    assert_ne!(
        first,
        remote_dir_on("one", "sdk", true),
        "the same path on another machine must not share a directory"
    );
}

#[test]
fn mutation_runner_holds_the_configured_host_lock() {
    let script = without_comments("scripts/mutants-run.sh");
    assert!(
        script.contains("HOST_LOCK=\"${DREP_MUTANTS_HOST_LOCK:-}\"")
            && script.contains("hold_lock 9 \"$HOST_LOCK\" mutants-run"),
        "a configured mutation host must serialize GitHub and laptop-offloaded sweeps"
    );
    assert!(
        script.contains("DREP_MUTANTS_RESULT_TOKEN")
            && script.contains("$OUT_DIR/mutants.out")
            && script.contains("$OUT_DIR/.run-token"),
        "each remote run must clear stale output and publish its own freshness token"
    );
    assert!(
        script.contains("\"$@\" 6<&- 9<&- && status=0"),
        "cargo-mutants and its fixtures must not inherit the checkout or host lock"
    );
}

#[test]
fn mutation_host_lock_wait_policy_has_one_definition() {
    let common = without_comments("scripts/mutants-common.sh");
    assert!(
        common.contains(
            "MUTANTS_HOST_LOCK_WAIT_SECONDS=\"${DREP_MUTANTS_HOST_LOCK_WAIT_SECONDS:-1800}\""
        ) && common.contains("validate_mutants_host_lock_wait_seconds()")
            && common.contains("validate_mutants_host_lock_wait_seconds \"$caller\" || return"),
        "the shared mutation layer must own the lock wait default and check it before every lock"
    );
    for name in ["mutants-remote", "mutants-run"] {
        let script = without_comments(&format!("scripts/{name}.sh"));
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
        script.contains("RUN_SCRATCH=\"$MUTANTS_SCRATCH_ROOT/run\"")
            && without_comments("scripts/mutants-common.sh").contains(
                "MUTANTS_SCRATCH_ROOT=\"${DREP_MUTANTS_TMPDIR:-${MUTANTS_ROOT}.mutants-tmp}\""
            )
            && script.contains("export TMPDIR=\"$RUN_SCRATCH\""),
        "the run must place its scratch copies in its directory beside the checkout"
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
        script.contains("trap 'remove_tree \"$RUN_SCRATCH\"' EXIT"),
        "the run must remove its own copies on exit"
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
            .run_script()
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

/// A committed fixture repository for the pre-commit hook, with a fake `mutants-staged.sh`, and fakes for cargo and cargo-mutants; every check appends its name to `checks.log`.
fn hook_fixture(harness: &Harness) -> PathBuf {
    let repository = harness.path("repository");
    fs::create_dir_all(repository.join("scripts")).unwrap();
    let shim = r#"#!/usr/bin/env bash
set -euo pipefail
printf '%s %s\n' "${0##*/}" "$*" >> "$FAKE_CHECKS"
if [ "${FAKE_FAIL:-}" = "$1" ]; then exit 7; fi
"#;
    write_executable(&harness.path("bin").join("cargo"), shim);
    write_executable(&harness.path("bin").join("cargo-mutants"), shim);
    write_executable(
        &repository.join("scripts/mutants-staged.sh"),
        "#!/usr/bin/env bash\nprintf 'mutants\n' >> \"$FAKE_CHECKS\"\n",
    );
    fs::write(repository.join("lib.rs"), "fn one() {}\n").unwrap();
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
    repository
}

/// Runs the pre-commit hook in `repository`, failing the check named by `failure`; returns its output and the checks it ran.
fn run_hook(harness: &Harness, repository: &Path, failure: &str) -> (Output, Vec<String>) {
    let log = harness.path("checks.log");
    fs::write(&log, "").unwrap();
    let mut command = bash_with_fakes(&harness.path("bin"));
    away_from_outer_git(&mut command, repository);
    let output = command
        .arg(repo_root().join(".githooks/pre-commit"))
        .env("FAKE_CHECKS", &log)
        .env("FAKE_FAIL", failure)
        .output()
        .unwrap();
    let checks = fs::read_to_string(&log)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect();
    (output, checks)
}

#[test]
fn pre_commit_executes_checks_and_stops_at_a_failed_check() {
    let harness = Harness::new();
    let repository = hook_fixture(&harness);
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
        let (output, checks) = run_hook(&harness, &repository, failure);
        assert_eq!(
            output.status.code(),
            Some(if failure.is_empty() { 0 } else { 7 }),
            "{output:?}"
        );
        assert_eq!(checks, expected);
    }
}

/// fmt, clippy and the tests read the working tree, so the hook refuses one that differs from the index before running any of them.
#[test]
fn pre_commit_refuses_a_working_tree_that_differs_from_the_index() {
    let harness = Harness::new();
    let repository = hook_fixture(&harness);
    for (kind, path) in [
        ("unstaged", repository.join("lib.rs")),
        ("untracked", repository.join("new.rs")),
    ] {
        fs::write(&path, "fn changed() {}\n").unwrap();
        let (output, checks) = run_hook(&harness, &repository, "");
        assert_eq!(output.status.code(), Some(1), "{kind}: {output:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("index"),
            "{kind}: {output:?}"
        );
        assert!(checks.is_empty(), "{kind}: no check may run: {checks:?}");
        if kind == "unstaged" {
            fs::write(&path, "fn one() {}\n").unwrap();
        } else {
            fs::remove_file(&path).unwrap();
        }
    }
}

/// The staged run hands the wrapper a snapshot of the index: the sync ships that tree and a local fallback builds it.
#[test]
fn remote_mutation_builds_the_source_it_is_given() {
    let script = without_comments("scripts/mutants-remote.sh");
    assert!(script.contains("SOURCE=\"${MUTANTS_SOURCE_DIR:-.}\""));
    assert!(script.contains("\"$SOURCE/\" \"$REMOTE/\""));
    assert!(script.contains("exec ./scripts/mutants-run.sh --dir \"$SOURCE\" \"$@\""));
}
