"""Admission checks against complete, isolated Git histories."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("mutants-ci-scope.py")


class MutationScopeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "Scope fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.base = self.commit("README.md", "base\n")

    def git(self, *arguments):
        return subprocess.check_output(
            ["git", *arguments], cwd=self.root, text=True, stderr=subprocess.PIPE
        ).strip()

    def commit(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        ignored = subprocess.run(
            ["git", "check-ignore", "--", name],
            cwd=self.root,
            capture_output=True,
            check=False,
        )
        self.assertEqual(ignored.returncode, 1)
        self.git("add", "--", name)
        self.git(
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        )
        return self.git("rev-parse", "HEAD")

    def scope(self, event, payload):
        event_file = self.root / "event.json"
        event_file.write_text(json.dumps(payload))
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), event, str(event_file)],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() == "run=true"

    def push(self, before, after):
        return self.scope("push", {"before": before, "after": after})

    def test_source_changes_distinguish_added_tests_from_examples_and_existing_tests(
        self,
    ):
        cases = [
            ("fn production() {}\n", False),
            ('const SAMPLE: &str = r#"#[test]\nfn fake() {}"#;\n', False),
            ("/* outer /* #[test] */ comment */\nfn production() {}\n", False),
            ("#[test]\nfn added() {}\n", True),
            (
                '#[\n tokio::test(flavor = "current_thread")\n]\nasync fn added() {}\n',
                True,
            ),
            (
                "/// ```no_run\n/// assert_eq!(1, 1);\n/// ```\nfn documented() {}\n",
                True,
            ),
        ]
        for index, (source, expected) in enumerate(cases):
            with self.subTest(source=source):
                before = self.git("rev-parse", "HEAD")
                after = self.commit(f"src/case_{index}.rs", source)
                self.assertEqual(self.push(before, after), expected)
        before = self.commit(
            "tests/existing.rs", "#[test]\nfn existing() { assert!(true); }\n"
        )
        after = self.commit(
            "tests/existing.rs", "#[test]\nfn existing() { assert_eq!(1, 1); }\n"
        )
        self.assertFalse(self.push(before, after))
        after = self.commit("tests/existing.rs", "// removed test\n")
        self.assertFalse(self.push(before, after))

    def test_complete_push_and_initial_push_include_tests_before_the_tip(self):
        self.commit("tests/added.rs", "#[test]\nfn added() {}\n")
        head = self.commit("README.md", "tip documentation\n")
        self.assertTrue(self.push(self.base, head))
        self.assertTrue(self.push("0" * 40, head))

    def test_pull_requests_compare_from_merge_base(self):
        self.git("checkout", "-qb", "feature")
        head = self.commit("src/feature.rs", "fn feature() {}\n")
        self.git("checkout", "-q", "main")
        base = self.commit("tests/main.rs", "#[test]\nfn on_main() {}\n")
        payload = {"pull_request": {"base": {"sha": base}, "head": {"sha": head}}}
        self.assertFalse(self.scope("pull_request", payload))
        self.git("checkout", "-q", "feature")
        self.commit("tests/feature.rs", "#[test]\nfn feature() {}\n")
        payload["pull_request"]["head"]["sha"] = self.commit("README.md", "tip docs\n")
        self.assertTrue(self.scope("pull_request", payload))

    def test_explicit_sweeps_and_invalid_events(self):
        for event in ("schedule", "workflow_dispatch"):
            self.assertTrue(self.scope(event, {}))
        self.assertFalse(self.scope("push", {"deleted": True}))
        with self.assertRaises(subprocess.CalledProcessError):
            self.scope("push", {"before": "missing", "after": self.base})


if __name__ == "__main__":
    unittest.main()
