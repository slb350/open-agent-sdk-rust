#!/usr/bin/env bash
#
# Mutation-test only the lines this commit changes.
#
# A passing test suite proves the tests run; it does not prove they would notice
# if the code were wrong. cargo-mutants perturbs the implementation and reports
# mutations no test catches - a surviving mutant IS a non-discriminating test.
#
# The hook tests the staged diff; CI tests the complete pushed diff on main.
# Full sweeps run separately. scripts/mutants-run.sh owns every verdict.

set -euo pipefail

# shellcheck source=scripts/mutants-common.sh
. "$(dirname "$0")/mutants-common.sh"

# Diffing the index is only correct when the working tree matches it.
require_matching_index
if git diff --cached --quiet -- '*.rs'; then
  echo "no staged Rust changes; nothing to mutate"
  exit 0
fi

# The diff and the run it feeds belong to this checkout's one mutation run, so
# a manual sweep started meanwhile waits rather than overwriting it. A commit
# with nothing to mutate has already left without waiting.
acquire_checkout_lock mutants-staged || exit $?
DIFF="$MUTANTS_OUT_DIR/staged.diff"
mkdir -p "$MUTANTS_OUT_DIR"
git diff --cached -- '*.rs' > "$DIFF"

# Through mutants-remote.sh, which offloads the run to a bigger machine and
# falls back to a local run when it cannot be reached. The verdict is
# scripts/mutants-run.sh's either way.
#
# The diff lands under the one directory the remote sync excludes - target/ -
# so it is named here as a file the run needs. The alternative was for the
# transport layer to scan the arguments for `--in-diff`, which is cargo-mutants
# grammar it has no business knowing.
MUTANTS_EXTRA_FILES="$DIFF" exec ./scripts/mutants-remote.sh --in-diff "$DIFF"
