//! Fixtures shared by the mutation-script tests.
#![allow(dead_code)]

use std::fs;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use tempfile::TempDir;

use super::process::{bash_with_fakes, repo_root, write_executable};

pub(crate) const ROLE: &str = "open-agent-sdk-rust-mutants";

pub(crate) fn without_comments(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) struct Harness {
    pub(crate) temp: TempDir,
}

impl Harness {
    pub(crate) fn new() -> Self {
        let temp = tempfile::Builder::new()
            .prefix("open agent mutants ")
            .tempdir()
            .expect("create harness directory");
        fs::create_dir_all(temp.path().join("bin")).expect("create fake binary directory");
        Self { temp }
    }

    pub(crate) fn path(&self, name: &str) -> PathBuf {
        self.temp.path().join(name)
    }

    /// The scratch root `run_script` hands the run.
    pub(crate) fn scratch(&self) -> PathBuf {
        self.path("scratch root")
    }

    /// The results directory `run_script` hands the run.
    pub(crate) fn output(&self) -> PathBuf {
        self.path("mutation output")
    }

    /// The checkout lock the run takes, beside its results directory.
    pub(crate) fn checkout_lock(&self) -> PathBuf {
        let mut lock = self.output().into_os_string();
        lock.push(".lock");
        lock.into()
    }

    /// Runs `scripts/mutants-run.sh` with a fake cargo that writes `missed.txt`.
    pub(crate) fn run_script(&self) -> Command {
        self.run_script_after("")
    }

    /// As `run_script`, with `prelude` run first in the shell that starts the
    /// script, so it can hand the script an open descriptor.
    pub(crate) fn run_script_after(&self, prelude: &str) -> Command {
        write_executable(
            &self.path("bin").join("cargo"),
            r#"#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "$FAKE_ARGS"
mkdir -p "$TMPDIR/cargo-mutants-trap.tmp/nested" "$TMPDIR/.tmp-test-debris"
mkdir -p "$MUTANTS_OUT_DIR/mutants.out"
printf '%s' "${FAKE_MISSED:-}" > "$MUTANTS_OUT_DIR/mutants.out/missed.txt"
exit "${FAKE_EXIT:-0}"
"#,
        );
        let mut command = bash_with_fakes(&self.path("bin"));
        command
            .args(["-c", &format!("{prelude}exec bash \"$1\""), "mutants-run"])
            .arg(repo_root().join("scripts/mutants-run.sh"))
            .env("DREP_MUTANTS_TMPDIR", self.scratch())
            .env("MUTANTS_OUT_DIR", self.output())
            .env("FAKE_ARGS", self.path("cargo-args"))
            .env_remove("DREP_MUTANTS_HOST_LOCK")
            .env_remove("DREP_MUTANTS_HOST_LOCK_WAIT_SECONDS")
            .env_remove("DREP_MUTANTS_RESULT_TOKEN");
        command
    }
}

/// perl that exits 0 when the lock file named by its argument could be taken
/// now and 1 while another process holds it: the same kernel flock the
/// mutation scripts take through perl, since macOS has no flock(1).
pub(crate) const LOCK_PROBE: &str =
    "open(my $f, '>>', $ARGV[0]) or exit 2; exit(flock($f, LOCK_EX | LOCK_NB) ? 0 : 1)";

/// Whether another process could take the lock at `path` right now.
pub(crate) fn lock_is_free(path: &Path) -> bool {
    Command::new("perl")
        .args(["-MFcntl=:flock", "-e", LOCK_PROBE])
        .arg(path)
        .status()
        .expect("probe the lock")
        .success()
}

/// A process holding the lock at `path` until it is killed or `seconds` pass;
/// returns once the lock is held.
pub(crate) fn hold_lock(path: &Path, seconds: &str) -> Child {
    let mut holder = Command::new("perl")
        .args([
            "-MFcntl=:flock",
            "-MTime::HiRes=sleep",
            "-e",
            "open(my $f, '>>', $ARGV[0]) or die; flock($f, LOCK_EX) or die; $| = 1; print \"locked\\n\"; sleep $ARGV[1]",
        ])
        .arg(path)
        .arg(seconds)
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn the lock holder");
    let mut ready = String::new();
    std::io::BufReader::new(holder.stdout.take().expect("holder stdout"))
        .read_line(&mut ready)
        .expect("the holder reports the lock");
    assert_eq!(ready, "locked\n");
    holder
}

/// A command that must not see the Git environment of a hook this suite may be running under.
pub(crate) fn isolated(program: &str, repository: &Path) -> Command {
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

pub(crate) fn git(repository: &Path, arguments: &[&str]) -> String {
    let output = isolated("git", repository)
        .args(arguments)
        .output()
        .expect("run git");
    assert!(output.status.success(), "git {arguments:?}: {output:?}");
    String::from_utf8(output.stdout).expect("git output is UTF-8")
}
