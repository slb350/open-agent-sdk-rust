use serde_yaml_ng::Value;

const CI: &str = include_str!("../.github/workflows/ci.yml");
const AUDIT: &str = include_str!("../.github/workflows/scheduled-audit.yml");

fn workflow(source: &str) -> Value {
    serde_yaml_ng::from_str(source).expect("workflow must be valid YAML")
}

fn steps(job: &Value) -> &[Value] {
    job["steps"].as_sequence().expect("job must have steps")
}

fn command<'a>(job: &'a Value, prefix: &[&str]) -> Vec<&'a str> {
    steps(job)
        .iter()
        .filter_map(|step| step["run"].as_str())
        .flat_map(str::lines)
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
        .find(|words| words.starts_with(prefix))
        .unwrap_or_else(|| panic!("job must execute {}", prefix.join(" ")))
}

#[test]
fn workflows_keep_hosted_runners_read_only_permissions_and_immutable_actions() {
    for source in [CI, AUDIT] {
        let workflow = workflow(source);
        assert_eq!(workflow["permissions"]["contents"], "read");
        for job in workflow["jobs"].as_mapping().unwrap().values() {
            assert!(matches!(
                job["runs-on"].as_str(),
                Some("ubuntu-latest" | "macos-latest")
            ));
            for step in steps(job) {
                if let Some(action) = step["uses"].as_str() {
                    let (name, sha) = action.split_once('@').expect("action must be pinned");
                    assert_eq!(sha.len(), 40, "{action}");
                    assert!(sha.bytes().all(|byte| byte.is_ascii_hexdigit()), "{action}");
                    assert_ne!(name, "actions-rust-lang/audit");
                    assert_ne!(name, "boa-dev/criterion-compare-action");
                }
            }
        }
    }
    let ci = workflow(CI);
    assert_eq!(ci["jobs"]["test-linux"]["runs-on"], "ubuntu-latest");
    assert_eq!(ci["jobs"]["test-macos"]["runs-on"], "macos-latest");
}

#[test]
fn msrv_audit_and_coverage_keep_their_required_checks() {
    let ci = workflow(CI);
    let scheduled = workflow(AUDIT);
    let msrv = &ci["jobs"]["msrv"];
    assert!(
        steps(msrv)
            .iter()
            .any(|step| step["with"]["toolchain"] == "1.85")
    );
    let check = command(msrv, &["cargo", "check"]);
    for flag in ["--all-features", "--all-targets", "--workspace"] {
        assert!(check.contains(&flag));
    }
    for audit in [&ci["jobs"]["security"], &scheduled["jobs"]["audit"]] {
        let install = command(audit, &["cargo", "install", "cargo-audit"]);
        assert!(install.contains(&"'=0.22.2'"));
        assert!(install.contains(&"--no-default-features"));
        assert!(!install.contains(&"--locked"));
        assert!(
            command(audit, &["cargo", "audit"])
                .windows(2)
                .any(|pair| pair == ["--deny", "warnings"])
        );
        let toolchain = steps(audit)
            .iter()
            .position(|step| step["with"]["toolchain"] == "stable")
            .unwrap();
        let verification = steps(audit)
            .iter()
            .position(|step| {
                step["run"]
                    .as_str()
                    .is_some_and(|run| run.lines().any(|line| line.trim() == "rustc --version"))
            })
            .unwrap();
        let execution = steps(audit)
            .iter()
            .position(|step| {
                step["run"]
                    .as_str()
                    .is_some_and(|run| run.starts_with("cargo audit "))
            })
            .unwrap();
        assert!(toolchain < verification && verification < execution);
    }
    let coverage = &ci["jobs"]["coverage"];
    let install = command(coverage, &["cargo", "install", "cargo-tarpaulin"]);
    assert!(install.contains(&"=0.37.2") && !install.contains(&"--locked"));
    let run = command(coverage, &["cargo", "tarpaulin"]);
    for required in [["--engine", "llvm"], ["--out", "xml"]] {
        assert!(run.windows(2).any(|pair| pair == required));
    }
    assert_eq!(command(coverage, &["test", "-s"])[2], "cobertura.xml");
    let upload = steps(coverage)
        .iter()
        .find(|step| {
            step["uses"]
                .as_str()
                .is_some_and(|action| action.starts_with("actions/upload-artifact@"))
        })
        .unwrap();
    assert_eq!(upload["with"]["path"], "cobertura.xml");
    assert_eq!(upload["with"]["if-no-files-found"], "error");
}

#[test]
fn mutation_sweep_uses_complete_event_scope_and_an_explicit_backstop() {
    let ci = workflow(CI);
    assert!(ci["on"]["workflow_dispatch"].is_null());
    assert!(
        ci["on"]
            .as_mapping()
            .unwrap()
            .contains_key(Value::from("workflow_dispatch"))
    );
    assert_eq!(ci["on"]["schedule"][0]["cron"], "0 16 15 * *");
    let policy = &ci["jobs"]["mutation-policy"];
    assert!(
        steps(policy)
            .iter()
            .any(|step| step["with"]["fetch-depth"] == 0)
    );
    command(
        policy,
        &["python3", "-B", "scripts/test_mutants_ci_scope.py"],
    );
    command(policy, &["python3", "-B", "scripts/mutants-ci-scope.py"]);
    assert_eq!(policy["outputs"]["run"], "${{ steps.scope.outputs.run }}");
    let mutants = &ci["jobs"]["mutants"];
    assert_eq!(mutants["needs"], "mutation-policy");
    assert_eq!(mutants["if"], "needs.mutation-policy.outputs.run == 'true'");
    assert_eq!(command(mutants, &["./scripts/mutants-run.sh"]).len(), 1);
    let installer = steps(mutants)
        .iter()
        .find(|step| step["with"]["tool"] == "cargo-mutants@27.1.0")
        .unwrap();
    assert_eq!(
        installer["uses"],
        "taiki-e/install-action@742a3317eac7bd62f91cd888b4eead5e784ba833"
    );
    // The exact installer pin/comment is a documented project requirement.
    assert!(CI.lines().any(|line| line.trim()
        == "- uses: taiki-e/install-action@742a3317eac7bd62f91cd888b4eead5e784ba833 # v2.87.1"));
}
