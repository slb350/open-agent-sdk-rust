#![allow(dead_code)]

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

pub(crate) fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub(crate) fn prepend_path(bin: &Path) -> String {
    format!(
        "{}:{}",
        bin.display(),
        env::var("PATH").expect("PATH is set")
    )
}

pub(crate) fn bash_with_fakes(bin: &Path) -> Command {
    let mut command = Command::new("bash");
    command
        .args([
            "-c",
            r#"
for name in cargo ssh rsync git; do
  if [ -f "$FIXTURE_BIN/$name" ]; then
    printf -v definition '%s() { command bash "$FIXTURE_BIN/%s" "$@"; }' "$name" "$name"
    eval "$definition"
    export -f "$name"
  fi
done
exec bash "$@"
"#,
            "fixture",
        ])
        .env("FIXTURE_BIN", bin)
        .env("PATH", prepend_path(bin));
    command
}

/// Writes an executable stub from a child process, so no descriptor of this process can hold it open for writing when another test thread forks: Linux refuses to `exec` such a file (`Text file busy`).
pub(crate) fn write_executable(path: &Path, contents: &str) {
    let status = Command::new("/bin/sh")
        .arg("-c")
        .arg(r#"printf '%s' "$2" > "$1" && chmod +x "$1""#)
        .arg("sh")
        .arg(path)
        .arg(contents)
        .status()
        .expect("the writer process must start");
    assert!(
        status.success(),
        "writing {} failed: {status}",
        path.display()
    );
}
