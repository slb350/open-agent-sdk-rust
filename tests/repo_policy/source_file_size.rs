//! Architecture guard for the repository's Rust source-file hard limit.

use std::fs;
use std::path::{Path, PathBuf};

const HARD_LINE_LIMIT: usize = 800;

fn rust_files_under(directory: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut directories = vec![directory.to_path_buf()];

    while let Some(current) = directories.pop() {
        for entry in fs::read_dir(&current).expect("source directory should be readable") {
            let path = entry.expect("source entry should be readable").path();
            if path.is_dir() {
                directories.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }

    files.sort();
    files
}

#[test]
fn rust_source_files_respect_hard_line_limit() {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"));
    let violations: Vec<String> = ["src", "tests", "examples", "benches"]
        .into_iter()
        .flat_map(|directory| rust_files_under(&repository.join(directory)))
        .filter_map(|path| {
            let line_count = fs::read_to_string(&path)
                .expect("Rust source should be readable")
                .lines()
                .count();
            (line_count > HARD_LINE_LIMIT).then(|| {
                format!(
                    "{}: {line_count} lines",
                    path.strip_prefix(env!("CARGO_MANIFEST_DIR"))
                        .expect("source path should be inside repository")
                        .display()
                )
            })
        })
        .collect();

    assert!(
        violations.is_empty(),
        "Rust source files exceed the {HARD_LINE_LIMIT}-line hard limit:\n{}",
        violations.join("\n")
    );
}
