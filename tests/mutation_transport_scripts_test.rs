#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use tempfile::TempDir;

#[path = "support/process.rs"]
mod process;
use process::{bash_with_fakes, repo_root, write_executable};

struct RemoteHarness {
    _temp: TempDir,
    repo: PathBuf,
    bin: PathBuf,
    remote: PathBuf,
    runs: PathBuf,
}

impl RemoteHarness {
    fn new() -> Self {
        let temp = tempfile::Builder::new()
            .prefix("open agent remote mutants ")
            .tempdir()
            .expect("create remote harness directory");
        let repo = temp.path().join("checkout with spaces");
        let scripts = repo.join("scripts");
        let bin = temp.path().join("fake bin");
        fs::create_dir_all(&scripts).expect("create scripts directory");
        fs::create_dir_all(&bin).expect("create fake binary directory");
        assert!(
            Command::new("git")
                .args(["init", "-q"])
                .arg(&repo)
                .status()
                .unwrap()
                .success()
        );
        copy_script("mutants-common.sh", &scripts);
        copy_script("mutants-remote.sh", &scripts);

        let remote = temp.path().join("remote home");
        fs::create_dir_all(&remote).unwrap();
        let runs = temp.path().join("runs.log");
        write_executable(
            &scripts.join("mutants-run.sh"),
            r#"#!/usr/bin/env bash
set -euo pipefail
: "${MUTANTS_OUT_DIR:?output directory was not forwarded}"
printf '%s\n' "$PWD" >> "$FAKE_RUNS"
mkdir -p "$MUTANTS_OUT_DIR/mutants.out"
printf '%s\n' "$DREP_MUTANTS_TMPDIR" > "$MUTANTS_OUT_DIR/mutants.out/scratch.txt"
printf '%s\n' "$@" > "$MUTANTS_OUT_DIR/mutants.out/arguments.txt"
exit "${FAKE_SWEEP_EXIT:-0}"
"#,
        );
        write_executable(
            &bin.join("ssh"),
            r#"#!/usr/bin/env bash
set -euo pipefail
while [ "$1" = "-o" ]; do shift 2; done
test "$1" = worker.example
shift
if [ "$1" = true ]; then exit "${FAKE_PROBE_EXIT:-0}"; fi
if [ "$1" = bash ]; then payload=$(cat); else payload=$1; fi
payload=${payload//\$HOME/\$FAKE_REMOTE_ROOT}
# SSH does not inherit the calling shell's mutation settings.
unset MUTANTS_OUT_DIR DREP_MUTANTS_TMPDIR
exec bash -c "$payload"
"#,
        );
        write_executable(
            &bin.join("rsync"),
            r#"#!/usr/bin/env bash
set -euo pipefail
arguments=("$@")
source=${arguments[${#arguments[@]}-2]}
destination=${arguments[${#arguments[@]}-1]}
if [[ "$source" == worker.example:* ]]; then
  if [ "${FAKE_MIRROR_FAILURE:-0}" = "1" ]; then exit 23; fi
  source="$FAKE_REMOTE_ROOT/${source#*:}"
fi
if [[ "$destination" == worker.example:* ]]; then
  destination="$FAKE_REMOTE_ROOT/${destination#*:}"
fi
mkdir -p "$destination"
cp -R "${source%/}/." "$destination"
"#,
        );

        Self {
            _temp: temp,
            repo,
            bin,
            remote,
            runs,
        }
    }

    fn command(&self) -> Command {
        let mut command = bash_with_fakes(&self.bin);
        command
            .arg(self.repo.join("scripts/mutants-remote.sh"))
            .current_dir(&self.repo)
            .env("FAKE_REMOTE_ROOT", &self.remote)
            .env("FAKE_RUNS", &self.runs)
            .env("DREP_MUTANTS_HOST", "worker.example")
            .env("DREP_MUTANTS_DIR", "ci/open-agent-sdk-rust")
            .env("DREP_MUTANTS_TMPDIR", "/var/tmp/mutants scratch")
            .env("MUTANTS_OUT_DIR", "out");
        command
    }
}

#[test]
fn remote_mutation_runs_use_unique_checkouts_and_forward_the_scratch_override() {
    let harness = RemoteHarness::new();

    let first = harness
        .command()
        .args(["--file", "src/file with spaces.rs"])
        .output()
        .expect("run first remote sweep");
    assert_success(&first);
    let first_results = result_path(&first);
    let artifacts = harness.repo.join(&first_results).join("mutants.out");
    assert_eq!(
        fs::read_to_string(artifacts.join("scratch.txt")).unwrap(),
        "/var/tmp/mutants scratch\n"
    );
    assert_eq!(
        fs::read_to_string(artifacts.join("arguments.txt")).unwrap(),
        "--file\nsrc/file with spaces.rs\n"
    );
    let second = harness.command().output().expect("run second remote sweep");
    assert_success(&second);

    let second_results = result_path(&second);
    assert_ne!(first_results, second_results);
    assert!(
        !harness.repo.join(first_results).exists(),
        "the next run must prune completed diagnostics"
    );
    assert!(harness.repo.join(second_results).is_dir());

    let runs = fs::read_to_string(&harness.runs).unwrap();
    let paths: Vec<_> = runs.lines().map(Path::new).collect();
    assert_eq!(paths.len(), 2);
    assert_ne!(paths[0], paths[1]);
    assert!(
        paths
            .iter()
            .all(|path| path.starts_with(&harness.remote) && !path.exists())
    );
}

#[test]
fn a_result_mirror_failure_is_reported_and_retains_the_remote_run() {
    let harness = RemoteHarness::new();
    let output = harness
        .command()
        .env("FAKE_MIRROR_FAILURE", "1")
        .output()
        .expect("run remote sweep with mirror failure");

    assert_eq!(output.status.code(), Some(74), "{output:?}");
    let stderr = String::from_utf8(output.stderr).expect("stderr is UTF-8");
    assert!(stderr.contains("failed to mirror results"), "{stderr}");
    assert!(stderr.contains("remote run retained"), "{stderr}");

    let runs = fs::read_to_string(&harness.runs).unwrap();
    let retained = Path::new(runs.trim());
    assert!(retained.starts_with(&harness.remote));
    assert!(retained.join("out/mutants.out/scratch.txt").is_file());
}

#[test]
fn staged_mutation_runs_pass_unique_diffs_and_remove_them_afterward() {
    let temp = tempfile::Builder::new()
        .prefix("open agent staged mutants ")
        .tempdir()
        .expect("create staged harness directory");
    let repo = temp.path().join("checkout with spaces");
    let scripts = repo.join("scripts");
    let bin = temp.path().join("fake bin");
    let paths_log = temp.path().join("paths.log");
    fs::create_dir_all(&scripts).expect("create scripts directory");
    fs::create_dir_all(&bin).expect("create fake binary directory");
    copy_script("mutants-common.sh", &scripts);
    copy_script("mutants-staged.sh", &scripts);
    write_executable(
        &bin.join("git"),
        r#"#!/usr/bin/env bash
set -euo pipefail
test "$1" = "diff"
printf '%s\n' "$FAKE_DIFF"
"#,
    );
    write_executable(
        &scripts.join("mutants-remote.sh"),
        r#"#!/usr/bin/env bash
set -euo pipefail
test -f "$MUTANTS_EXTRA_FILE"
printf '%s|%s\n' "$MUTANTS_EXTRA_FILE" "$(wc -c < "$MUTANTS_EXTRA_FILE")" >> "$FAKE_PATHS_LOG"
test "$1" = "--in-diff"
test "$2" = "$MUTANTS_EXTRA_FILE"
"#,
    );

    for diff in ["first staged diff", "second staged diff"] {
        let output = bash_with_fakes(&bin)
            .arg(scripts.join("mutants-staged.sh"))
            .current_dir(&repo)
            .env("MUTANTS_OUT_DIR", "out")
            .env("FAKE_DIFF", diff)
            .env("FAKE_PATHS_LOG", &paths_log)
            .output()
            .expect("run staged mutation wrapper");
        assert_success(&output);
    }

    let paths = fs::read_to_string(paths_log).expect("read staged paths");
    let records: Vec<_> = paths.lines().collect();
    assert_eq!(records.len(), 2, "{paths}");
    let first = records[0].split_once('|').expect("first path record");
    let second = records[1].split_once('|').expect("second path record");
    assert_ne!(first.0, second.0);
    assert_ne!(first.1, "0");
    assert_ne!(second.1, "0");
    assert!(!repo.join(first.0).exists());
    assert!(!repo.join(second.0).exists());
    assert_eq!(
        fs::read_dir(repo.join("out/staged"))
            .expect("read staged output directory")
            .count(),
        0
    );
}

fn copy_script(name: &str, destination: &Path) {
    fs::copy(
        repo_root().join("scripts").join(name),
        destination.join(name),
    )
    .unwrap_or_else(|error| panic!("copy {name}: {error}"));
}

fn assert_success(output: &Output) {
    assert!(output.status.success(), "{output:?}");
}

fn result_path(output: &Output) -> PathBuf {
    let stdout = std::str::from_utf8(&output.stdout).expect("stdout is UTF-8");
    let marker = "results will mirror to ";
    let path = stdout
        .lines()
        .find_map(|line| line.split_once(marker).map(|(_, path)| path))
        .unwrap_or_else(|| panic!("missing result path in {stdout}"));
    PathBuf::from(path)
}
