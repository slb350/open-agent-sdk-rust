#!/usr/bin/env bash
#
# Mutation-test only the lines this commit changes.
#
# A passing test suite proves the tests run; it does not prove they would notice
# if the code were wrong. cargo-mutants perturbs the implementation and reports
# mutations no test catches - a surviving mutant IS a non-discriminating test.
#
# Scoped with --in-diff rather than running the whole tree, so the cost is
# proportional to the change. CI runs the full sweep through the same
# scripts/mutants-run.sh, which is where the pass/fail rule lives.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/mutants-common.sh
# shellcheck disable=SC1091
. "$SCRIPT_DIR/mutants-common.sh"
STAGED_DIR="$MUTANTS_OUT_DIR/staged"
mkdir -p "$STAGED_DIR"
DIFF="$(mktemp "$STAGED_DIR/staged.XXXXXX")"

# The diff is an input to the remote run, so it must live until that run has
# copied it. Give every invocation its own file and delete only that exact file
# on every exit; concurrent pre-commit hooks must never overwrite one another.
# Invoked indirectly by the EXIT trap.
# shellcheck disable=SC2329
cleanup_staged_diff() {
  local original_status=$?
  local cleanup_status
  trap - EXIT
  set +e
  mutants_delete_exact_path "$DIFF"
  cleanup_status=$?
  if [ "$cleanup_status" -ne 0 ]; then
    echo "mutants-staged: failed to clean staged diff $DIFF" >&2
  fi
  mutants_reconcile_cleanup_status "$original_status" "$cleanup_status"
  original_status=$?
  exit "$original_status"
}
trap cleanup_staged_diff EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

# pre-commit stashes unstaged changes before running hooks, so the working tree
# matches the index here - which is what makes diffing the index correct.
git diff --cached -- '*.rs' > "$DIFF"

if [ ! -s "$DIFF" ]; then
  echo "no staged Rust changes; nothing to mutate"
  exit 0
fi

# Through mutants-remote.sh, which offloads the run to a bigger machine and
# falls back to a local run when it cannot be reached. The verdict is
# scripts/mutants-run.sh's either way.
#
# The diff lands under the one directory the remote sync excludes - target/ -
# so it is named here as a file the run needs. The alternative was for the
# transport layer to scan the arguments for `--in-diff`, which is cargo-mutants
# grammar it has no business knowing.
MUTANTS_EXTRA_FILE="$DIFF" ./scripts/mutants-remote.sh --in-diff "$DIFF"
