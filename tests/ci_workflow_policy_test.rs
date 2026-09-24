use serde_yaml_ng::Value;

const CI: &str = include_str!("../.github/workflows/ci.yml");
const AUDIT: &str = include_str!("../.github/workflows/scheduled-audit.yml");
const DEPENDABOT: &str = include_str!("../.github/dependabot.yml");
const MUTATION_INSTALL_ACTION: &str =
    "taiki-e/install-action@94c31af3204a9f15ab40b35ad084410b905bbc73";
const UPLOAD_ARTIFACT_ACTION: &str =
    "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a";
/// About five times the measured full sweep; see the mutation job in ci.yml.
const MUTATION_TIMEOUT_MINUTES: u64 = 45;
/// Every mutation workload runs on homelab-ai-1 as this repository's role.
const MUTATION_RUNNER: [&str; 5] = [
    "self-hosted",
    "linux",
    "x64",
    "homelab-ai-1",
    "open-agent-sdk-rust-mutants",
];

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
fn only_mutation_runs_self_hosted_and_every_workflow_keeps_read_only_pinned_actions() {
    for source in [CI, AUDIT] {
        let workflow = workflow(source);
        assert_eq!(workflow["permissions"]["contents"], "read");
        for (name, job) in workflow["jobs"].as_mapping().unwrap() {
            if source == CI && name == "mutants" {
                let labels = job["runs-on"]
                    .as_sequence()
                    .expect("the mutation job must name its runner labels")
                    .iter()
                    .map(|label| label.as_str().unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(labels, MUTATION_RUNNER);
            } else {
                assert!(
                    matches!(
                        job["runs-on"].as_str(),
                        Some("ubuntu-latest" | "macos-latest")
                    ),
                    "{name:?} must stay on a GitHub-hosted runner"
                );
            }
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
fn dependabot_excludes_known_msrv_breaking_releases() {
    let config = workflow(DEPENDABOT);
    let cargo = config["updates"]
        .as_sequence()
        .unwrap()
        .iter()
        .find(|update| update["package-ecosystem"] == "cargo")
        .unwrap();
    let ignored = cargo["ignore"].as_sequence().unwrap();

    for (dependency, version) in [("wiremock", "0.6.5"), ("yoke-derive", "0.8.3")] {
        let rule = ignored
            .iter()
            .find(|rule| rule["dependency-name"] == dependency)
            .unwrap_or_else(|| panic!("{dependency} must have an MSRV exception"));
        assert!(
            rule["versions"]
                .as_sequence()
                .unwrap()
                .contains(&Value::from(version))
        );
    }
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
    assert!(install.contains(&"=0.37.3") && !install.contains(&"--locked"));
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
    assert_eq!(upload["uses"], UPLOAD_ARTIFACT_ACTION);
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
    assert_eq!(ci["on"]["schedule"][0]["cron"], "37 9 15 * *");
    let policy = &ci["jobs"]["mutation-policy"];
    assert!(
        steps(policy)
            .iter()
            .any(|step| step["with"]["fetch-depth"] == 0)
    );
    for output in ["run", "mode", "base"] {
        assert_eq!(
            policy["outputs"][output],
            format!("${{{{ steps.test-policy.outputs.{output} }}}}")
        );
    }
    for job in ci["jobs"].as_mapping().unwrap().values() {
        if job != policy && job != &ci["jobs"]["mutants"] && job != &ci["jobs"]["benchmarks"] {
            assert_eq!(
                job["if"],
                "github.event_name != 'schedule' && github.event_name != 'workflow_dispatch'"
            );
        }
    }
    let mutants = &ci["jobs"]["mutants"];
    assert_eq!(mutants["needs"], "mutation-policy");
    // A public repository's fork pull requests must never execute on the self-hosted runner.
    assert_eq!(
        mutants["if"],
        "needs.mutation-policy.outputs.run == 'true' && (github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository)"
    );
    assert_eq!(mutants["timeout-minutes"], MUTATION_TIMEOUT_MINUTES);
    let mutant_steps = steps(mutants);
    let checkout = &mutant_steps[0];
    assert_eq!(checkout["with"]["fetch-depth"], 0);
    assert_eq!(checkout["with"]["persist-credentials"], false);
    assert!(
        !mutant_steps.iter().any(|step| step["uses"]
            .as_str()
            .is_some_and(|action| action.starts_with("Swatinem/rust-cache@"))),
        "cargo-mutants never reads target/, and on a persistent runner the cache action deletes host-installed cargo binaries"
    );
    assert_eq!(
        command(mutants, &["./scripts/mutants-run.sh"]),
        ["./scripts/mutants-run.sh", "\"${mutation_args[@]}\""]
    );
    let installer = mutant_steps
        .iter()
        .find(|step| step["with"]["tool"] == "cargo-mutants@27.1.0")
        .unwrap();
    assert_eq!(installer["uses"], MUTATION_INSTALL_ACTION);
    // The exact installer pin/comment is a documented project requirement.
    let expected_installer_line = format!("- uses: {MUTATION_INSTALL_ACTION} # v2.87.17");
    assert!(
        CI.lines()
            .any(|line| line.trim() == expected_installer_line.as_str())
    );
    let repair_upload = mutant_steps
        .iter()
        .find(|step| step["with"]["name"] == "mutation-repair")
        .unwrap();
    assert_eq!(repair_upload["uses"], UPLOAD_ARTIFACT_ACTION);
    assert_eq!(repair_upload["if"], "failure()");
}
