#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::thread;
use std::time::{Duration, Instant};

use tempfile::TempDir;

#[path = "support/process.rs"]
mod process;
use process::{prepend_path, repo_root, write_executable};

const START_TIMEOUT: Duration = Duration::from_secs(5);

struct Harness {
    _temp: TempDir,
    bin: PathBuf,
    scratch: PathBuf,
    output: PathBuf,
}

impl Harness {
    fn new() -> Self {
        let temp = tempfile::Builder::new()
            .prefix("open agent mutants ")
            .tempdir()
            .expect("create harness directory");
        let bin = temp.path().join("bin");
        let scratch = temp.path().join("scratch root");
        let output = temp.path().join("mutation output");
        fs::create_dir_all(&bin).expect("create fake binary directory");
        write_executable(
            &bin.join("cargo"),
            r#"#!/usr/bin/env bash
set -euo pipefail
run_dir="$TMPDIR/cargo-mutants-${FAKE_ID}.tmp"
mkdir -p "$run_dir"
: > "$run_dir/owner-${FAKE_ID}"
printf '%s\n' "$TMPDIR" > "$FAKE_OBSERVED"
if [ -n "${FAKE_READY:-}" ]; then
  : > "$FAKE_READY"
fi
if [ -n "${FAKE_WAIT:-}" ]; then
  while [ ! -e "$FAKE_WAIT" ]; do
    sleep 0.01
  done
  test -e "$run_dir/owner-${FAKE_ID}"
fi
exit "${FAKE_EXIT:-0}"
"#,
        );
        Self {
            _temp: temp,
            bin,
            scratch,
            output,
        }
    }

    fn command(&self, id: &str, observed: &Path) -> Command {
        let mut command = Command::new("bash");
        command
            .arg(repo_root().join("scripts/mutants-run.sh"))
            .env("PATH", prepend_path(&self.bin))
            .env("DREP_MUTANTS_TMPDIR", &self.scratch)
            .env("MUTANTS_OUT_DIR", &self.output)
            .env("FAKE_ID", id)
            .env("FAKE_OBSERVED", observed);
        command
    }
}

#[test]
fn mutation_runner_owns_its_scratch_and_preserves_the_cargo_status() {
    let harness = Harness::new();
    let observed = harness._temp.path().join("observed path");
    let stale = stale_run(&harness.scratch);
    let sentinel = harness.scratch.join("keep-me");
    fs::create_dir_all(stale.join("nested")).expect("create stale scratch");
    fs::write(stale.join("nested/file"), "stale").expect("seed stale scratch");
    fs::write(&sentinel, "keep").expect("seed non-run sentinel");

    let output = harness
        .command("failure", &observed)
        .env("FAKE_EXIT", "7")
        .output()
        .expect("run mutation wrapper");

    assert_eq!(output.status.code(), Some(7), "{output:?}");
    let owned = PathBuf::from(read_trimmed(&observed));
    assert_eq!(owned.parent(), Some(harness.scratch.as_path()));
    assert_ne!(owned, harness.scratch);
    assert!(
        owned
            .file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with("run_"))
    );
    assert!(!owned.exists(), "the failed run must clean its own scratch");
    assert!(!stale.exists(), "a dead prior run must be swept");
    assert!(sentinel.exists(), "unowned entries must be preserved");
}

#[test]
fn concurrent_mutation_runs_never_delete_each_others_scratch() {
    let harness = Harness::new();
    let first_observed = harness._temp.path().join("first observed");
    let second_observed = harness._temp.path().join("second observed");
    let first_ready = harness._temp.path().join("first ready");
    let release_first = harness._temp.path().join("release first");

    let mut first = harness
        .command("first", &first_observed)
        .env("FAKE_READY", &first_ready)
        .env("FAKE_WAIT", &release_first)
        .spawn()
        .expect("spawn first mutation run");
    wait_for(&first_ready, &mut first);

    let second = harness
        .command("second", &second_observed)
        .output()
        .expect("run second mutation wrapper");
    assert!(second.status.success(), "{second:?}");
    assert_ne!(
        read_trimmed(&first_observed),
        read_trimmed(&second_observed)
    );

    fs::write(&release_first, "go").expect("release first run");
    let first_status = first.wait().expect("wait for first mutation run");
    assert!(first_status.success(), "{first_status:?}");
}

fn stale_run(root: &Path) -> PathBuf {
    root.join(format!("run_{}_4294967295_stale", host_id()))
}

fn host_id() -> String {
    let output = Command::new("hostname")
        .output()
        .expect("read local hostname");
    assert!(output.status.success());
    String::from_utf8(output.stdout)
        .expect("hostname is UTF-8")
        .trim()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '.' {
                character
            } else {
                '-'
            }
        })
        .collect()
}

fn read_trimmed(path: &Path) -> String {
    fs::read_to_string(path)
        .expect("read observed path")
        .trim()
        .to_owned()
}

fn wait_for(path: &Path, child: &mut Child) {
    let deadline = Instant::now() + START_TIMEOUT;
    while Instant::now() < deadline {
        if path.exists() {
            return;
        }
        if let Some(status) = child.try_wait().expect("poll child") {
            panic!("mutation wrapper exited before ready: {status:?}");
        }
        thread::sleep(Duration::from_millis(10));
    }
    panic!("timed out waiting for {}", path.display());
}
