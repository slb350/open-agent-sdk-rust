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

if git diff --cached --quiet -- '*.rs'; then
  echo "no staged Rust changes; nothing to mutate"
  exit 0
fi

# The diff and the run it feeds belong to this checkout's one mutation run, so a manual sweep started meanwhile waits rather than overwriting it.
acquire_checkout_lock mutants-staged || exit $?

# The run tests the commit, not the working tree. The index is written to a tree now and the run builds a copy of that tree, so unstaged edits, untracked files and anything changed while the run waits for a lock never reach it. `git commit` reads the index again after this hook, so the index is compared with the tree once the run is over.
TREE="$(git write-tree)"
SNAPSHOT="$MUTANTS_SCRATCH_ROOT/index"
remove_tree "$SNAPSHOT"
trap 'remove_tree "$SNAPSHOT"; remove_tree "$SNAPSHOT.git-index"' EXIT
mkdir -p "$MUTANTS_SCRATCH_ROOT"
GIT_INDEX_FILE="$SNAPSHOT.git-index" git read-tree "$TREE"
GIT_INDEX_FILE="$SNAPSHOT.git-index" git checkout-index --all --prefix="$SNAPSHOT/"

DIFF="$MUTANTS_OUT_DIR/staged.diff"
mkdir -p "$MUTANTS_OUT_DIR"
BASE="$(git rev-parse --quiet --verify HEAD || git hash-object -t tree /dev/null)"
git diff "$BASE" "$TREE" -- '*.rs' > "$DIFF"

# Through mutants-remote.sh, which offloads the run to a bigger machine and
# falls back to a local run when it cannot be reached. The verdict is
# scripts/mutants-run.sh's either way.
#
# The diff lands under the one directory the remote sync excludes - target/ -
# so it is named here as a file the run needs. The alternative was for the
# transport layer to scan the arguments for `--in-diff`, which is cargo-mutants
# grammar it has no business knowing.
MUTANTS_SOURCE_DIR="$SNAPSHOT" MUTANTS_EXTRA_FILES="$DIFF" \
  ./scripts/mutants-remote.sh --in-diff "$DIFF" && status=0 || status=$?

if [ "$(git write-tree)" != "$TREE" ]; then
  echo "mutants-staged: the index changed during the run, so the commit is not what was tested; stage it again and retry" >&2
  exit 1
fi
exit "$status"
