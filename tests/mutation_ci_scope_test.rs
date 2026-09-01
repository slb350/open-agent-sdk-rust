#![cfg(unix)]

use std::{fs, path::Path, process::Command};

use tempfile::TempDir;

const SCOPE_SCRIPT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/mutants-ci-scope.sh");

struct Repository {
    root: TempDir,
}

impl Repository {
    fn new() -> Self {
        let root = tempfile::Builder::new()
            .prefix("open-agent-mutation-scope-")
            .tempdir()
            .expect("create temporary repository");
        let repository = Self { root };
        repository.git(&["init", "--quiet"]);
        repository.git(&["config", "user.email", "ci@example.invalid"]);
        repository.git(&["config", "user.name", "CI"]);
        repository
    }

    fn path(&self) -> &Path {
        self.root.path()
    }

    fn write(&self, path: &str, contents: &str) {
        let path = self.path().join(path);
        fs::create_dir_all(path.parent().expect("fixture path has a parent"))
            .expect("create fixture parent");
        fs::write(path, contents).expect("write fixture");
    }

    fn commit(&self, message: &str) -> String {
        self.git(&["add", "."]);
        self.git(&["commit", "--quiet", "-m", message]);
        self.git_output(&["rev-parse", "HEAD"])
    }

    fn scope(&self, base: &str) -> Vec<String> {
        let output = Command::new("bash")
            .arg(SCOPE_SCRIPT)
            .arg(base)
            .arg("HEAD")
            .current_dir(self.path())
            .output()
            .expect("run mutation scope classifier");
        assert!(output.status.success(), "{output:?}");
        String::from_utf8(output.stdout)
            .expect("scope output is UTF-8")
            .lines()
            .map(str::to_owned)
            .collect()
    }

    fn git(&self, args: &[&str]) {
        let output = Command::new("git")
            .args(args)
            .current_dir(self.path())
            .output()
            .expect("run git");
        assert!(output.status.success(), "git {args:?}: {output:?}");
    }

    fn git_output(&self, args: &[&str]) -> String {
        let output = Command::new("git")
            .args(args)
            .current_dir(self.path())
            .output()
            .expect("run git");
        assert!(output.status.success(), "git {args:?}: {output:?}");
        String::from_utf8(output.stdout)
            .expect("git output is UTF-8")
            .trim()
            .to_owned()
    }
}

#[test]
fn modified_inline_test_scopes_mutants_to_its_owning_source_file() {
    let repository = Repository::new();
    repository.write(
        "src/lib.rs",
        "pub fn answer() -> u8 { 42 }\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn answer_is_stable() { assert_eq!(super::answer(), 42); }\n}\n",
    );
    let base = repository.commit("initial test");
    repository.write(
        "src/lib.rs",
        "pub fn answer() -> u8 { 42 }\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn answer_is_stable() { assert!(super::answer() > 40); }\n}\n",
    );
    repository.commit("modify inline test");

    assert_eq!(repository.scope(&base), ["files", "--file", "src/lib.rs"]);
}

#[test]
fn integration_test_changes_fall_back_to_the_complete_sweep() {
    let repository = Repository::new();
    repository.write("src/lib.rs", "pub fn answer() -> u8 { 42 }\n");
    repository.write(
        "tests/api.rs",
        "#[test]\nfn api() { assert_eq!(2 + 2, 4); }\n",
    );
    let base = repository.commit("initial integration test");
    repository.write(
        "tests/api.rs",
        "#[test]\nfn api() { assert!(2 + 2 > 3); }\n",
    );
    repository.commit("modify integration test");

    assert_eq!(repository.scope(&base), ["full"]);
}

#[test]
fn deleted_test_files_fall_back_to_the_complete_sweep() {
    let repository = Repository::new();
    repository.write("src/lib.rs", "pub fn answer() -> u8 { 42 }\n");
    repository.write(
        "tests/api.rs",
        "#[test]\nfn api() { assert_eq!(2 + 2, 4); }\n",
    );
    let base = repository.commit("initial integration test");
    repository.git(&["rm", "--quiet", "tests/api.rs"]);
    repository.commit("delete integration test");

    assert_eq!(repository.scope(&base), ["full"]);
}

#[test]
fn repository_test_modules_fall_back_to_the_complete_sweep() {
    let repository = Repository::new();
    repository.write("src/body_map.rs", "pub fn body_map() -> u8 { 1 }\n");
    repository.write(
        "src/body_map_tests.rs",
        "#[test]\nfn body_map_is_stable() { assert_eq!(2 + 2, 4); }\n",
    );
    let base = repository.commit("initial repository test module");
    repository.write(
        "src/body_map_tests.rs",
        "#[test]\nfn body_map_is_stable() { assert!(2 + 2 > 3); }\n",
    );
    repository.commit("modify repository test module");

    assert_eq!(repository.scope(&base), ["full"]);
}

#[test]
fn repository_test_support_changes_fall_back_to_the_complete_sweep() {
    let repository = Repository::new();
    repository.write("src/lib.rs", "pub fn answer() -> u8 { 42 }\n");
    repository.write(
        "src/test_support/doubles.rs",
        "pub fn response() -> u8 { 41 }\n",
    );
    let base = repository.commit("initial test support");
    repository.write(
        "src/test_support/doubles.rs",
        "pub fn response() -> u8 { 42 }\n",
    );
    repository.commit("modify test support");

    assert_eq!(repository.scope(&base), ["full"]);
}

#[test]
fn production_only_changes_do_not_start_mutation_ci() {
    let repository = Repository::new();
    repository.write(
        "src/lib.rs",
        "pub fn answer() -> u8 { 41 }\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn answer_is_stable() { assert!(super::answer() > 40); }\n}\n",
    );
    let base = repository.commit("initial production code");
    repository.write(
        "src/lib.rs",
        "pub fn answer() -> u8 { 42 }\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn answer_is_stable() { assert!(super::answer() > 40); }\n}\n",
    );
    repository.commit("modify production code");

    assert_eq!(repository.scope(&base), ["none"]);
}

#[test]
fn production_changes_after_a_test_gated_item_do_not_start_mutation_ci() {
    let repository = Repository::new();
    repository.write(
        "src/lib.rs",
        "pub async fn upload() -> u8 {\n    #[cfg(test)]\n    let delay = 1;\n    consume(delay);\n    41\n}\n",
    );
    let base = repository.commit("initial mixed source");
    repository.write(
        "src/lib.rs",
        "pub async fn upload() -> u8 {\n    #[cfg(test)]\n    let delay = 1;\n    consume(delay);\n    42\n}\n",
    );
    repository.commit("modify trailing production code");

    assert_eq!(repository.scope(&base), ["none"]);
}

#[test]
fn production_changes_after_an_inline_test_module_do_not_start_mutation_ci() {
    let repository = Repository::new();
    repository.write(
        "src/lib.rs",
        "#[cfg(test)]\nmod tests {\n    #[test]\n    fn arithmetic() { assert_eq!(2 + 2, 4); }\n}\n\npub fn answer() -> u8 { 41 }\n",
    );
    let base = repository.commit("initial test-first source");
    repository.write(
        "src/lib.rs",
        "#[cfg(test)]\nmod tests {\n    #[test]\n    fn arithmetic() { assert_eq!(2 + 2, 4); }\n}\n\npub fn answer() -> u8 { 42 }\n",
    );
    repository.commit("modify production after inline tests");

    assert_eq!(repository.scope(&base), ["none"]);
}
