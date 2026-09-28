//! Every test reaches git through `support/mutation.rs`, whose helpers keep the Git environment of a hook this suite may run under away from a fixture. A bare spawn inherits it and acts on the repository being committed instead of the fixture's own.

use std::{fs, path::Path};

/// The one file allowed to spawn git directly.
const HELPER: &str = "support/mutation.rs";

/// A bare spawn, spelled so this file does not match itself.
const BARE_GIT: &str = concat!("Command::new(", "\"git\")");

fn scan(dir: &Path, root: &Path, offenders: &mut Vec<String>) {
    for entry in fs::read_dir(dir).expect("list test sources") {
        let path = entry.expect("read a test source entry").path();
        if path.is_dir() {
            scan(&path, root, offenders);
            continue;
        }
        let relative = path.strip_prefix(root).expect("path under tests");
        if path.extension().is_none_or(|extension| extension != "rs")
            || relative == Path::new(HELPER)
        {
            continue;
        }
        let text = fs::read_to_string(&path).expect("read a test source");
        for (number, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap_or_default();
            if code.contains(BARE_GIT) {
                offenders.push(format!("tests/{}:{}", relative.display(), number + 1));
            }
        }
    }
}

#[test]
fn tests_spawn_git_only_through_the_isolating_helper() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut offenders = Vec::new();
    scan(&root, &root, &mut offenders);
    assert!(
        offenders.is_empty(),
        "spawn git through support/mutation.rs (`isolated` or `git`) instead: {offenders:?}"
    );
}
