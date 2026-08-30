#!/usr/bin/env bash
#
# Run the mutation sweep on a bigger machine over SSH, then report its verdict
# here.
#
# Mutation testing is the most CPU-hungry gate in this repo: every mutant is a
# full build plus a full test run, and the hook fires on a laptop the developer
# is still using. Offloading keeps that work away from the interactive machine.
#
# More jobs is not always better, and -j 4 on a 32-thread box is not a typo.
# Measured runs slowed down at -j 8 and -j 16 because concurrent source copies,
# compilation and linking became I/O-bound. Treat MUTANTS_JOBS as a knob to
# measure, not a number to raise on principle.
#
# The verdict rule is NOT duplicated here. This script syncs, invokes
# scripts/mutants-run.sh on the remote, and propagates its exit code - so the
# hook, CI and the remote sweep cannot disagree about what counts as a failure.
#
# Falls back to a local run, loudly, when the host is unreachable. A commit gate
# that silently skips itself because the LAN blipped is worse than a slow one.
#
#   DREP_MUTANTS_HOST    ssh target (default: strix.local)
#   DREP_MUTANTS_DIR     remote base, $HOME-relative (default: ci/<repo name>)
#   DREP_MUTANTS_REMOTE  0 to force a local run
#   MUTANTS_JOBS         -j for the remote run (default: 4)
#   MUTANTS_LOCAL_JOBS   -j for a local or fallback run (default: 4)
#   MUTANTS_EXTRA_FILE   one repo-relative file this run needs that the sync
#                        would otherwise skip

set -euo pipefail

# `git rev-parse`, not `dirname "$0"/..`: the same answer install.sh already
# uses, and it does not care whether the script was reached through a symlink,
# a relative path or PATH.
cd "$(git rev-parse --show-toplevel)"
# shellcheck source=scripts/mutants-common.sh
# shellcheck disable=SC1091
. scripts/mutants-common.sh

HOST="${DREP_MUTANTS_HOST:-strix.local}"
REMOTE_BASE="${DREP_MUTANTS_DIR:-ci/${PWD##*/}}"
JOBS="${MUTANTS_JOBS:-4}"

case "$REMOTE_BASE" in
  ''|/*|*..*) echo "mutants-remote: DREP_MUTANTS_DIR must be a safe HOME-relative path" >&2
              exit 64 ;;
esac
case "$MUTANTS_OUT_DIR" in
  ''|/*|*..*) echo "mutants-remote: MUTANTS_OUT_DIR must be repo-relative" >&2
              exit 64 ;;
esac
case "$JOBS" in
  ''|*[!0-9]*|0) echo "mutants-remote: MUTANTS_JOBS must be a positive integer" >&2
                  exit 64 ;;
esac

LOCAL_HOST_ID="$(mutants_host_id)"
RUN_ID="run_${LOCAL_HOST_ID}_$$_$(date +%s)-${RANDOM:-0}"
REMOTE_RUN_DIR="$REMOTE_BASE/runs/$RUN_ID"
REMOTE="$HOST:$REMOTE_RUN_DIR"
LOCAL_RUNS_DIR="$MUTANTS_OUT_DIR/runs"
mkdir -p "$LOCAL_RUNS_DIR"
mutants_sweep_stale_owned_paths "$LOCAL_RUNS_DIR" "$LOCAL_HOST_ID" "mutation results"
LOCAL_RESULTS="$LOCAL_RUNS_DIR/$RUN_ID"
REMOVE_REMOTE_ON_EXIT=0

printf -v REMOTE_RUN_Q '%q' "$REMOTE_RUN_DIR"
if [ -n "${DREP_MUTANTS_TMPDIR:-}" ]; then
  printf -v REMOTE_SCRATCH_Q '%q' "$DREP_MUTANTS_TMPDIR"
  REMOTE_SCRATCH_COMMAND="export DREP_MUTANTS_TMPDIR=$REMOTE_SCRATCH_Q"
else
  printf -v REMOTE_SCRATCH_Q '%q' "$REMOTE_BASE/scratch"
  REMOTE_SCRATCH_COMMAND="export DREP_MUTANTS_TMPDIR=\"\$HOME\"/$REMOTE_SCRATCH_Q"
fi
printf -v MUTANTS_OUT_DIR_Q '%q' "$MUTANTS_OUT_DIR"

# Invoked indirectly by the EXIT trap.
# shellcheck disable=SC2329
cleanup_remote_run() {
  local original_status=$?
  local cleanup_status=0
  trap - EXIT
  set +e
  if [ "$REMOVE_REMOTE_ON_EXIT" -eq 1 ]; then
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" \
      "find \"\$HOME\"/$REMOTE_RUN_Q -depth -delete"
    cleanup_status=$?
    if [ "$cleanup_status" -ne 0 ]; then
      echo "mutants-remote: failed to clean remote run $HOST:~/$REMOTE_RUN_DIR" >&2
    fi
  fi
  mutants_reconcile_cleanup_status "$original_status" "$cleanup_status"
  original_status=$?
  exit "$original_status"
}
trap cleanup_remote_run EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

run_local() {
  MUTANTS_JOBS="${MUTANTS_LOCAL_JOBS:-4}" exec ./scripts/mutants-run.sh "$@"
}

if [ "${DREP_MUTANTS_REMOTE:-1}" = "0" ]; then
  run_local "$@"
fi

# A bare probe rather than letting the first rsync fail and reading its exit
# code: "the host is down, here is what I am doing instead" is the message the
# developer needs, and inferring it from an rsync failure would also swallow a
# full disk or an unwritable directory as "unreachable". One handshake, ~145ms,
# against a run measured in minutes.
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" true 2>/dev/null; then
  echo "warning: $HOST is unreachable - running the mutation sweep locally instead." >&2
  echo "         This will use this machine's CPU for the duration." >&2
  run_local "$@"
fi

echo "mutants: running on $HOST (-j $JOBS), results will mirror to $LOCAL_RESULTS"

# Every invocation syncs into a unique remote checkout. That removes the race
# where a second rsync changed a shared tree beneath a running mutation sweep,
# and it lets each run mirror only its own diagnostics without a remote flock.
# cargo-mutants 27.1 excludes the top-level target/ from its source copies by
# default, so preserving a shared checkout target never warmed those copies.
REMOVE_REMOTE_ON_EXIT=1
rsync -a --mkpath \
  --exclude target --exclude 'mutants.out*' \
  --exclude .git --exclude venv --exclude node_modules \
  --exclude .mypy_cache --exclude .pytest_cache --exclude .ruff_cache \
  --exclude '*.egg-info' --exclude dist --exclude build --exclude .drep \
  --exclude '.tokens' --exclude '.env*' \
  ./ "$REMOTE/"

# Files the run needs that the sync above skipped - in practice the staged diff,
# which mutants-staged.sh writes under the excluded target/. Named by the caller
# rather than recovered by scanning "$@" for cargo-mutants' own flags: what
# belongs at this layer is "move these bytes", not that layer's argument
# grammar. -R recreates each path under the remote root, directories included.
if [ -n "${MUTANTS_EXTRA_FILE:-}" ]; then
  case "$MUTANTS_EXTRA_FILE" in
    /*|../*|*/../*|*/..)
      echo "mutants-remote: MUTANTS_EXTRA_FILE must stay within the repo, got $MUTANTS_EXTRA_FILE" >&2
      exit 64 ;;
  esac
  if [ ! -e "$MUTANTS_EXTRA_FILE" ]; then
    echo "mutants-remote: required extra file does not exist: $MUTANTS_EXTRA_FILE" >&2
    exit 66
  fi
  rsync -aR --mkpath "$MUTANTS_EXTRA_FILE" "$REMOTE/"
fi

# `bash -s` rather than a quoted one-liner: the arguments are quoted with printf
# %q, which is bash's dialect, so the remote end must be bash whatever login
# shell the account uses.
# `printf '%q ' "$@"` still runs its format once when there are no arguments, so it
# emits a single empty token and the remote sweep is invoked as `mutants-run.sh ''`,
# which cargo-mutants rejects with "unexpected argument '' found". Only ever reached by
# an unscoped full sweep; the staged-diff path always passes --in-diff.
QUOTED_ARGS=""
if [ "$#" -gt 0 ]; then
  printf -v QUOTED_ARGS '%q ' "$@"
fi

status=0
# shellcheck disable=SC2087  # local expansion is the point: the unique remote
# dir, job count, scratch command and %q-quoted args are all known here.
ssh -o BatchMode=yes "$HOST" bash -s <<EOF || status=$?
set -euo pipefail
export PATH=\$HOME/.cargo/bin:\$PATH
cd "\$HOME"/$REMOTE_RUN_Q
$REMOTE_SCRATCH_COMMAND
mkdir -p $MUTANTS_OUT_DIR_Q
MUTANTS_JOBS=$JOBS ./scripts/mutants-run.sh $QUOTED_ARGS
EOF

# Mirror this run into its own local directory so concurrent invocations cannot
# overwrite one another's diagnostics. A mirror failure is part of the verdict:
# preserve the remote checkout for recovery and fail a mutation run that had
# otherwise passed instead of silently presenting stale local results.
mkdir -p "$LOCAL_RESULTS"
mirror_status=0
rsync -a --mkpath "$REMOTE/$MUTANTS_OUT_DIR/" "$LOCAL_RESULTS/" || mirror_status=$?
if [ "$mirror_status" -ne 0 ]; then
  REMOVE_REMOTE_ON_EXIT=0
  echo "mutants-remote: failed to mirror results; remote run retained at $HOST:~/$REMOTE_RUN_DIR" >&2
  if [ "$status" -eq 0 ]; then
    status=74
  fi
fi

exit "$status"
