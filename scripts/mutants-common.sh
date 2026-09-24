#!/usr/bin/env bash
#
# The one definition of where mutation results live, sourced by every script in
# this trio.
#
# mutants-run.sh reads `missed.txt` out of this directory to reach its verdict,
# mutants-staged.sh writes the staged diff into it, and mutants-remote.sh
# mirrors it back from the remote host. It was the same string literal in six
# places across three files, all of them silently wrong the day one of them
# changed: a stale copy does not error, it just stops finding missed.txt.
#
# target/ because it is already gitignored. Overridable so a caller with a
# different layout does not have to edit three scripts.
MUTANTS_OUT_DIR="${MUTANTS_OUT_DIR:-target/mutants}"

# One lock wait policy for both the remote transaction wrapper and direct/CI
# mutation runs. The two scripts acquire the lock at different boundaries, but
# accepting and defaulting the operator's value must not drift between them.
MUTANTS_HOST_LOCK_WAIT_SECONDS="${DREP_MUTANTS_HOST_LOCK_WAIT_SECONDS:-1800}"

validate_mutants_host_lock_wait_seconds() {
  local caller="$1"
  case "$MUTANTS_HOST_LOCK_WAIT_SECONDS" in
    ''|*[!0-9]*)
      echo "$caller: DREP_MUTANTS_HOST_LOCK_WAIT_SECONDS must be a non-negative integer" >&2
      return 64
      ;;
  esac
}

# The hook is a plain Git hook, so nothing stashes unstaged work, and the remote
# run syncs the working tree. Tests can read fixtures outside src/, so the whole
# tracked tree must match the index, with no untracked inputs.
require_matching_index() {
  if ! git diff --quiet -- || [ -n "$(git ls-files --others --exclude-standard)" ]; then
    echo 'mutation gate requires the working tree to match the index, without untracked inputs' >&2
    echo 'stage intended changes or isolate the commit; no files have been altered' >&2
    return 1
  fi
}
