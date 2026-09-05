"""Run a full mutation sweep when a complete GitHub diff adds a Rust test."""

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

NON_CODE = re.compile(
    r'//[^\n]*|/\*|(?:b|c)?r(?P<hashes>\#{0,255})"|"(?:\\.|[^"\\])*"'
    r"|'(?:\\(?:u\{[\da-fA-F_]+\}|x[\da-fA-F]{2}|.)|[^'\\\n])'",
    re.DOTALL,
)
COMMENT_DELIMITER = re.compile(r"/\*|\*/")
TEST = re.compile(
    r"#\s*\[\s*(?:\w+\s*::\s*)*(?:test|rstest|test_case)\b[^]]*\]"
    r"(?:\s*#\s*\[[^]]*\])*\s*(?:async\s+)?fn\s+(\w+)"
)
DOC_FENCE = re.compile(
    r"^[ \t]*(`{3,}|~{3,})([^\n]*)\n(.*?)^[ \t]*\1[ \t]*$", re.MULTILINE | re.DOTALL
)


def git(*arguments, input=None):
    return subprocess.check_output(["git", *arguments], input=input, text=True)


def test_items(source):
    # Mask strings and nested comments before reading attributes; keep doc comments.
    code, documentation, position = [], [], 0
    while match := NON_CODE.search(source, position):
        start, end = match.span()
        if match[0] == "/*":
            depth = 1
            while depth and (delimiter := COMMENT_DELIMITER.search(source, end)):
                depth += 1 if delimiter[0] == "/*" else -1
                end = delimiter.end()
            if depth:
                end = len(source)
        elif match["hashes"] is not None:
            terminator = '"' + match["hashes"]
            closing = source.find(terminator, end)
            end = len(source) if closing < 0 else closing + len(terminator)
        comment = source[start:end]
        if comment.startswith(("///", "//!")) and not comment.startswith("////"):
            documentation.append(comment[3:])
        elif comment.startswith(("/**", "/*!")) and not comment.startswith("/***"):
            documentation.append(re.sub(r"(?m)^\s*\* ?", "", comment[3:-2]))
        code.extend([source[position:start], re.sub(r"[^\n]", " ", source[start:end])])
        position = end
    code.append(source[position:])
    items = Counter(("test", name) for name in TEST.findall("".join(code)))
    for fence in DOC_FENCE.finditer("\n".join(documentation)):
        flags = re.split(r"[\s,]+", fence[2].strip())
        if all(
            flag in {"", "rust", "no_run", "should_panic", "compile_fail", "ignore"}
            or re.fullmatch(r"edition\d{4}|ignore-.+", flag)
            for flag in flags
        ):
            items[("doctest", fence[3])] += 1
    return items


def revision(value):
    if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", value):
        raise ValueError("event must contain complete Git object IDs")
    return value


def needs_sweep(event_name, event):
    if event_name in {"schedule", "workflow_dispatch"}:
        return True
    if event_name == "push":
        if event.get("deleted"):
            return False
        base, head = revision(event["before"]), revision(event["after"])
        if not base.strip("0"):
            base = git("hash-object", "-w", "-t", "tree", "--stdin", input="").strip()
    elif event_name == "pull_request":
        request = event["pull_request"]
        head = revision(request["head"]["sha"])
        base = git("merge-base", revision(request["base"]["sha"]), head).strip()
    else:
        raise ValueError(f"unsupported event: {event_name}")
    changes = git(
        "diff", "--name-status", "--no-renames", "-z", base, head, "--", "*.rs"
    ).split("\0")[:-1]
    for status, path in zip(changes[::2], changes[1::2]):
        before = "" if status == "A" else git("show", f"{base}:{path}")
        after = "" if status == "D" else git("show", f"{head}:{path}")
        if test_items(after) - test_items(before):
            return True
    return False


if __name__ == "__main__":
    result = needs_sweep(sys.argv[1], json.loads(Path(sys.argv[2]).read_text()))
    print(f"run={str(result).lower()}")
