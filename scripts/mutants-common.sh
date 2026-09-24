#!/usr/bin/env bash
#
# The one definition of where mutation results live and how a run holds its
# locks, sourced by every script in this trio.
#
# mutants-run.sh reads `missed.txt` out of the results directory to reach its
# verdict, mutants-staged.sh writes the staged diff into it, and
# mutants-remote.sh mirrors it back from the remote host. It was the same string
# literal in six places across three files, all of them silently wrong the day
# one of them changed: a stale copy does not error, it just stops finding
# missed.txt.
#
# Every script works from the checkout root, whatever the caller's cwd, so a
# relative results directory and the lock beside it name one place per
# checkout.
MUTANTS_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$MUTANTS_ROOT" || exit

# target/ because it is already gitignored. Overridable so a caller with a
# different layout does not have to edit three scripts.
MUTANTS_OUT_DIR="${MUTANTS_OUT_DIR:-target/mutants}"

# One wait policy for every lock a run takes: its checkout's, and the host lock
# a dedicated mutation machine shares between its hosted and offloaded runs.
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

# Remove one file or tree. `find` does not follow a symlink given as its root,
# and an entry that vanishes mid-walk is not a failure.
remove_tree() {
  find "$1" -depth -delete 2>/dev/null || true
}

# A repo-relative path: never absolute and never stepping outside through a . or
# .. component.
is_contained_path() {
  case "$1" in
  '' | /* | . | .. | ./* | ../* | */. | */.. | */./* | */../*) return 1 ;;
  esac
}

# hold_lock FD PATH CALLER takes an exclusive kernel flock on PATH through this
# shell's descriptor FD, waiting at most MUTANTS_HOST_LOCK_WAIT_SECONDS. The
# kernel drops the lock when the last process holding that descriptor exits, so
# a killed run never leaves one behind. perl takes it because macOS has no
# flock(1).
#
# A descriptor inherited already open on PATH is the holder's own: flock on it
# succeeds at once because it is the same open file, so a script started by the
# holder carries on instead of waiting on itself. A child that must not hold
# the lock is started with the descriptor closed.
hold_lock() {
  local fd="$1" path="$2" caller="$3"
  validate_mutants_host_lock_wait_seconds "$caller" || return
  # shellcheck disable=SC2016  # perl source, not shell.
  if ! perl -e '
    open(my $held, ">&=", $ARGV[0]) or exit 1;
    my @held = stat($held);
    my @path = stat($ARGV[1]) or exit 1;
    exit !($held[0] == $path[0] && $held[1] == $path[1]);
  ' "$fd" "$path"; then
    mkdir -p "$(dirname "$path")"
    eval "exec $fd>>\"\$path\""
  fi
  # shellcheck disable=SC2016  # perl source, not shell.
  if ! perl -MFcntl=:flock -e '
    open(my $lock, ">&=", $ARGV[0]) or exit 2;
    exit 0 if flock($lock, LOCK_EX | LOCK_NB);
    exit 1 unless $ARGV[1] > 0;
    $SIG{ALRM} = sub { exit 1 };
    alarm $ARGV[1];
    flock($lock, LOCK_EX) or exit 1;
  ' "$fd" "$MUTANTS_HOST_LOCK_WAIT_SECONDS"; then
    echo "$caller: another mutation run holds $path" >&2
    return 75
  fi
}

# One mutation run per checkout at a time. The hook, a local run and a remote
# transaction all write $MUTANTS_OUT_DIR and this checkout's scratch, so a
# second run waits instead of deleting the first run's results or copies.
acquire_checkout_lock() {
  hold_lock 6 "${MUTANTS_OUT_DIR}.lock" "$1"
}

# This checkout's own directory in an ai-1 role's cache. The source sync mirrors
# into it with --delete, so it is named for this machine and the checkout's path
# as well as its name: no other checkout shares it, even one at the same path on
# another machine, and holding the checkout lock is all it takes to own it. A
# renamed machine starts again from a fresh copy.
remote_checkout_dir() {
  printf '.cache/%s/%s-%s' "$1" \
    "$(basename "$MUTANTS_ROOT" | LC_ALL=C tr -cd 'A-Za-z0-9._-')" \
    "$(printf '%s:%s' "$(hostname)" "$MUTANTS_ROOT" | cksum | cut -d' ' -f1)"
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
