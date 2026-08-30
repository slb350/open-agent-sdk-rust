use std::env;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

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

pub(crate) fn write_executable(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write fake executable");
    let mut permissions = fs::metadata(path)
        .expect("stat fake executable")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions).expect("make fake executable executable");
}
