#[test]
fn development_only_files_are_excluded_from_package() {
    let manifest = include_str!("../Cargo.toml");

    for path in [
        "CLAUDE.md",
        ".Codex/",
        ".markdownlint.json",
        ".githooks/",
        "mutants.out/",
        "scripts/",
        "tests/ci_workflow_policy_test.rs",
        "tests/mutation_scripts_test.rs",
        "tests/mutation_transport_scripts_test.rs",
        "tests/support/",
    ] {
        assert!(
            manifest.contains(&format!("    \"{path}\",")),
            "{path} must remain excluded from published crates"
        );
    }
}
