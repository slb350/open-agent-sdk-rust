use std::process::Command;

#[test]
fn development_only_files_are_excluded_from_package() {
    let target = tempfile::tempdir().expect("create independent package target");
    let output = Command::new(env!("CARGO"))
        .args(["package", "--list", "--allow-dirty", "--offline"])
        .env("CARGO_TARGET_DIR", target.path())
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("list published package files");
    assert!(output.status.success(), "{output:?}");
    let listing = std::str::from_utf8(&output.stdout).expect("package paths are UTF-8");
    for path in listing.lines() {
        assert!(
            !matches!(path, "AGENTS.md" | "CLAUDE.md" | ".markdownlint.json")
                && ![
                    ".Codex/",
                    ".githooks/",
                    ".github/",
                    "mutants.out/",
                    "scripts/",
                    "tests/support/",
                    "tests/ci_workflow_policy_test.rs",
                    "tests/mutation_ci_scope_test.rs",
                    "tests/mutation_scripts_test.rs",
                    "tests/mutation_transport_scripts_test.rs",
                    "tests/package_manifest_test.rs"
                ]
                .iter()
                .any(|prefix| path.starts_with(prefix)),
            "development file included in published package: {path}"
        );
    }
    assert!(listing.lines().any(|path| path == "src/lib.rs"));
}
