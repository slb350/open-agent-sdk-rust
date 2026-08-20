#[test]
fn development_only_files_are_excluded_from_package() {
    let manifest = include_str!("../Cargo.toml");

    for path in [
        "CLAUDE.md",
        ".markdownlint.json",
        ".githooks/",
        "mutants.out/",
        "scripts/",
    ] {
        assert!(
            manifest.contains(&format!("    \"{path}\",")),
            "{path} must remain excluded from published crates"
        );
    }
}
