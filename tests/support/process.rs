use std::env;
use std::fs;
use std::os::unix::fs::PermissionsExt;
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

pub(crate) fn write_executable(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write fake executable");
    let mut permissions = fs::metadata(path)
        .expect("stat fake executable")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions).expect("make fake executable executable");
}
