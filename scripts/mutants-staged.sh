#!/usr/bin/env bash
# Mutation-test staged Rust changes through the shared remote/verdict runner.

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

git diff --cached -- '*.rs' > "$DIFF"

if [ ! -s "$DIFF" ]; then
  echo "no staged Rust changes; nothing to mutate"
  exit 0
fi

# Explicitly transfer the diff, which lives under excluded target/.
MUTANTS_EXTRA_FILE="$DIFF" bash ./scripts/mutants-remote.sh --in-diff "$DIFF"
