#![cfg(unix)]
//! How the mutation scripts share a checkout and the ai-1 host: one run per
//! checkout at a time, a host lock the remote session hands to the run it
//! starts, scratch that never reaches beyond the run's own directory, and a
//! staged hook that dispatches only a working tree matching the index.

use std::fs;
use std::path::PathBuf;

#[path = "support/mutation.rs"]
mod mutation;
#[path = "support/process.rs"]
mod process;
use mutation::{Harness, LOCK_PROBE, git, hold_lock, isolated, lock_is_free};
use process::{repo_root, write_executable};

/// The sweep is destructive only inside the run directory, and the run removes
/// that directory, test debris included, when it ends.
#[test]
fn mutation_scratch_cleanup_preserves_adjacent_state() {
    let harness = Harness::new();
    let scratch = harness.scratch();
    let adjacent = scratch.join("adjacent");
    fs::create_dir_all(scratch.join("run/cargo-mutants-killed.tmp/nested"))
        .expect("stale run directory");
    fs::create_dir_all(&adjacent).expect("adjacent directory");
    fs::write(adjacent.join("keep"), "keep").expect("adjacent file");

    let output = harness
        .run_script()
        .output()
        .expect("run mutation wrapper with fake cargo");

    assert!(output.status.success(), "{output:?}");
    assert!(
        !scratch.join("run").exists(),
        "the run must remove its directory, stale copies and test debris included"
    );
    assert!(adjacent.join("keep").exists());
    assert!(
        lock_is_free(&harness.checkout_lock()),
        "the run must release the checkout lock"
    );

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

/// A run directory that is a symlink is removed as a link; what it points at
/// is left alone.
#[test]
fn a_symlinked_run_directory_is_not_followed() {
    let harness = Harness::new();
    let outside = harness.path("outside");
    fs::create_dir_all(harness.scratch()).expect("scratch");
    fs::create_dir_all(&outside).expect("outside");
    fs::write(outside.join("keep"), "keep").expect("outside file");
    std::os::unix::fs::symlink(&outside, harness.scratch().join("run")).expect("run symlink");

    let output = harness.run_script().output().expect("run mutation wrapper");

    assert!(output.status.success(), "{output:?}");
    assert!(outside.join("keep").exists());
    assert!(!harness.scratch().join("run").exists());
}

/// A second run in the same checkout waits for the first rather than deleting
/// its results or copies, and gives up with 75 instead of running.
#[test]
fn a_second_run_in_one_checkout_waits_for_the_first() {
    let harness = Harness::new();
    let mut first = hold_lock(&harness.checkout_lock(), "30");
    let results = harness.output().join("mutants.out");
    fs::create_dir_all(&results).expect("first run's results");
    fs::write(results.join("missed.txt"), "first run's survivor\n").expect("first run's verdict");

    let output = harness
        .run_script()
        .env("DREP_MUTANTS_HOST_LOCK_WAIT_SECONDS", "0")
        .output()
        .expect("run mutation wrapper");
    let _ = first.kill();
    let _ = first.wait();

    assert_eq!(output.status.code(), Some(75), "{output:?}");
    assert!(
        !harness.path("cargo-args").exists(),
        "the second run must not start cargo-mutants"
    );
    assert_eq!(
        fs::read_to_string(results.join("missed.txt")).expect("first run's verdict"),
        "first run's survivor\n",
        "the second run must not touch the first run's results"
    );
}

/// A run that is allowed to wait takes the lock once its holder lets go.
#[test]
fn a_waiting_run_starts_once_the_lock_is_released() {
    let harness = Harness::new();
    let mut first = hold_lock(&harness.checkout_lock(), "0.2");

    let output = harness
        .run_script()
        .env("DREP_MUTANTS_HOST_LOCK_WAIT_SECONDS", "30")
        .output()
        .expect("run mutation wrapper");
    let _ = first.wait();

    assert!(output.status.success(), "{output:?}");
    assert!(harness.path("cargo-args").exists());
}

/// The kernel drops the lock with its holder, so a lock file a killed run left
/// behind holds nothing.
#[test]
fn a_leftover_lock_file_holds_nothing() {
    let harness = Harness::new();
    fs::write(harness.checkout_lock(), "").expect("leftover lock file");

    let output = harness.run_script().output().expect("run mutation wrapper");

    assert!(output.status.success(), "{output:?}");
    assert!(harness.path("cargo-args").exists(), "the run must go ahead");
    assert!(
        lock_is_free(&harness.checkout_lock()),
        "the run must release the lock"
    );
}

/// The remote session holds the host lock on descriptor 9 and starts the run
/// with it open: the run carries on under that lock instead of waiting on it.
#[test]
fn a_run_started_by_the_host_lock_holder_reuses_its_lock() {
    let harness = Harness::new();

    let output = harness
        .run_script_after(
            "exec 9>>\"$DREP_MUTANTS_HOST_LOCK\" && perl -MFcntl=:flock -e 'open(my $l, \">&=\", 9) or exit 2; flock($l, LOCK_EX) or exit 1' && ",
        )
        .env("DREP_MUTANTS_HOST_LOCK", harness.path("host.lock"))
        .env("DREP_MUTANTS_HOST_LOCK_WAIT_SECONDS", "0")
        .output()
        .expect("run mutation wrapper");

    assert!(output.status.success(), "{output:?}");
    assert!(harness.path("cargo-args").exists());
}

/// Another sweep's host lock makes the run wait, here for no time at all, and
/// give up with 75 before it starts cargo-mutants.
#[test]
fn a_run_waits_for_a_host_lock_another_sweep_holds() {
    let harness = Harness::new();
    let mut other = hold_lock(&harness.path("host.lock"), "30");

    let output = harness
        .run_script()
        .env("DREP_MUTANTS_HOST_LOCK", harness.path("host.lock"))
        .env("DREP_MUTANTS_HOST_LOCK_WAIT_SECONDS", "0")
        .output()
        .expect("run mutation wrapper");
    let _ = other.kill();
    let _ = other.wait();

    assert_eq!(output.status.code(), Some(75), "{output:?}");
    assert!(!harness.path("cargo-args").exists());
}

/// A fixture repository with the staged wrapper, a fake remote that records how
/// it was called, and one committed Rust file.
fn staged_fixture(harness: &Harness) -> PathBuf {
    let repository = harness.path("repository");
    fs::create_dir_all(repository.join("scripts")).unwrap();
    for name in ["mutants-common.sh", "mutants-staged.sh"] {
        fs::copy(
            repo_root().join("scripts").join(name),
            repository.join("scripts").join(name),
        )
        .unwrap();
    }
    // Records whether another process is refused the checkout lock, and whether
    // this process, started by the lock's holder, gets it without waiting.
    write_executable(
        &repository.join("scripts/mutants-remote.sh"),
        "#!/usr/bin/env bash\nperl -MFcntl=:flock -e \"$LOCK_PROBE\" target/mutants.lock; refused=$?\n. scripts/mutants-common.sh\nMUTANTS_HOST_LOCK_WAIT_SECONDS=0\nacquire_checkout_lock fake-remote; inherited=$?\nprintf 'remote:%s extra:%s inherited:%s refused:%s\\n' \"$*\" \"$MUTANTS_EXTRA_FILES\" \"$inherited\" \"$refused\" >> \"$FAKE_EVENTS\"\n",
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
    repository
}

fn run_staged(harness: &Harness, repository: &std::path::Path) -> std::process::Output {
    isolated("bash", repository)
        .arg("scripts/mutants-staged.sh")
        .env("FAKE_EVENTS", harness.path("events"))
        .env("LOCK_PROBE", LOCK_PROBE)
        .env("DREP_MUTANTS_HOST_LOCK_WAIT_SECONDS", "0")
        .output()
        .expect("run staged gate")
}

#[test]
fn staged_gate_refuses_unstaged_or_untracked_inputs_and_dispatches_a_matching_index() {
    let harness = Harness::new();
    let repository = staged_fixture(&harness);
    let events = harness.path("events");
    fs::write(repository.join("source.rs"), "fn staged() {}\n").unwrap();
    git(&repository, &["add", "source.rs"]);

    for (kind, path) in [
        ("unstaged", repository.join("source.rs")),
        ("untracked", repository.join("new_test.rs")),
    ] {
        fs::write(&path, "#[test]\nfn unstaged_test() {}\n").unwrap();
        let output = run_staged(&harness, &repository);
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

    let output = run_staged(&harness, &repository);
    assert!(output.status.success(), "{output:?}");
    let diff = "target/mutants/staged.diff";
    assert!(
        fs::read_to_string(repository.join(diff))
            .unwrap()
            .contains("+fn staged() {}")
    );
    assert_eq!(
        fs::read_to_string(&events).unwrap(),
        format!("remote:--in-diff {diff} extra:{diff} inherited:0 refused:1\n"),
        "the remote run must inherit the checkout lock, which refuses anyone else"
    );
    assert!(
        lock_is_free(&repository.join("target/mutants.lock")),
        "the lock must be free once the staged run returns"
    );
}

/// A commit with no Rust changes has nothing to mutate, so it leaves at once
/// even while a sweep holds this checkout's lock.
#[test]
fn a_commit_without_rust_changes_does_not_wait_for_a_running_sweep() {
    let harness = Harness::new();
    let repository = staged_fixture(&harness);
    fs::write(repository.join("notes.md"), "notes\n").unwrap();
    git(&repository, &["add", "notes.md"]);
    fs::create_dir_all(repository.join("target")).unwrap();
    let mut sweep = hold_lock(&repository.join("target/mutants.lock"), "30");

    let output = run_staged(&harness, &repository);
    let _ = sweep.kill();
    let _ = sweep.wait();

    assert!(output.status.success(), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("no staged Rust changes"),
        "{output:?}"
    );
    assert!(
        !harness.path("events").exists(),
        "nothing may reach the remote"
    );
}
