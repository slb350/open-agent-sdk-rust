#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::thread;
use std::time::{Duration, Instant};

use tempfile::TempDir;

#[path = "support/process.rs"]
mod process;
use process::{bash_with_fakes, repo_root, write_executable};

const START_TIMEOUT: Duration = Duration::from_secs(5);

struct HeldRun {
    child: Child,
    release: PathBuf,
}

impl Drop for HeldRun {
    fn drop(&mut self) {
        // Release even after an assertion fails, before TempDir removes the marker.
        let _ = fs::write(&self.release, "go");
        let _ = self.child.wait();
    }
}

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
    if [ "$SECONDS" -ge 15 ]; then exit 99; fi
    sleep 0.01
  done
  test -e "$run_dir/owner-${FAKE_ID}"
fi
mkdir -p "$MUTANTS_OUT_DIR/mutants.out"
printf '%s' "${FAKE_MISSED:-}" > "$MUTANTS_OUT_DIR/mutants.out/missed.txt"
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
        let mut command = bash_with_fakes(&self.bin);
        command
            .arg(repo_root().join("scripts/mutants-run.sh"))
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
    assert!(!owned.exists(), "the failed run must clean its own scratch");
    assert!(!stale.exists(), "a dead prior run must be swept");
    assert!(sentinel.exists(), "unowned entries must be preserved");
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
            .command("verdict", &harness._temp.path().join("observed"))
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
fn concurrent_mutation_runs_never_delete_each_others_scratch() {
    let harness = Harness::new();
    let first_observed = harness._temp.path().join("first observed");
    let second_observed = harness._temp.path().join("second observed");
    let first_ready = harness._temp.path().join("first ready");
    let release_first = harness._temp.path().join("release first");

    let child = harness
        .command("first", &first_observed)
        .env("FAKE_READY", &first_ready)
        .env("FAKE_WAIT", &release_first)
        .spawn()
        .expect("spawn first mutation run");
    let mut first = HeldRun {
        child,
        release: release_first.clone(),
    };
    wait_for(&first_ready, &mut first.child);

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
    let first_status = first.child.wait().expect("wait for first mutation run");
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

#[test]
fn pre_commit_executes_checks_and_stops_at_a_failed_check() {
    let harness = Harness::new();
    let scripts = harness._temp.path().join("scripts");
    fs::create_dir_all(&scripts).unwrap();
    let log = harness._temp.path().join("checks.log");
    let shim = r#"#!/usr/bin/env bash
set -euo pipefail
printf '%s %s\n' "${0##*/}" "$*" >> "$FAKE_CHECKS"
if [ "${FAKE_FAIL:-}" = "$1" ]; then exit 7; fi
"#;
    write_executable(&harness.bin.join("cargo"), shim);
    write_executable(&harness.bin.join("cargo-mutants"), shim);
    write_executable(&harness.bin.join("python3"), shim);
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
                "python3 -B scripts/test_mutants_ci_scope.py",
                "mutants",
            ],
        ),
    ] {
        fs::write(&log, "").unwrap();
        let output = bash_with_fakes(&harness.bin)
            .arg(repo_root().join(".githooks/pre-commit"))
            .current_dir(harness._temp.path())
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
