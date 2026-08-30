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

mutants_host_id() {
  printf '%s' "$(hostname)" | tr -c '[:alnum:].-' '-'
}

mutants_delete_exact_path() {
  local target=$1
  if [ ! -e "$target" ] && [ ! -L "$target" ]; then
    return 0
  fi
  # `find` does not follow a symlink passed as its root. -depth removes children
  # before their parents and finally removes this exact owned root.
  find "$target" -depth -delete
}

mutants_reconcile_cleanup_status() {
  local original_status=$1
  local cleanup_status=$2
  if [ "$original_status" -eq 0 ] && [ "$cleanup_status" -ne 0 ]; then
    return 74
  fi
  return "$original_status"
}

mutants_sweep_stale_owned_paths() {
  local root=$1
  local host_id=$2
  local description=$3
  local candidate name prefix owner_host owner_pid suffix
  for candidate in "$root"/run_*; do
    if [ ! -e "$candidate" ] && [ ! -L "$candidate" ]; then
      continue
    fi
    name=${candidate##*/}
    IFS=_ read -r prefix owner_host owner_pid suffix <<EOF
$name
EOF
    if [ "$prefix" != "run" ] || [ "$owner_host" != "$host_id" ] || [ -z "$suffix" ]; then
      continue
    fi
    case "$owner_pid" in
      ''|*[!0-9]*) continue ;;
    esac
    if kill -0 "$owner_pid" 2>/dev/null; then
      continue
    fi
    mutants_delete_exact_path "$candidate" || {
      echo "mutants: failed to clean stale $description $candidate" >&2
      return 1
    }
  done
}
