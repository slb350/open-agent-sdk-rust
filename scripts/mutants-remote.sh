#!/usr/bin/env bash
# Offload mutation sweeps to SSH; unreachable hosts fall back locally.
# Measured compilation/linking contention made -j 4 faster than -j 8 or 16.
# The runner owns the verdict; transport failures must never hide it.
#
#   DREP_MUTANTS_HOST    ssh target (default: strix.local)
#   DREP_MUTANTS_DIR     remote base, $HOME-relative (default: ci/<repo name>)
#   DREP_MUTANTS_REMOTE  0 to force a local run
#   MUTANTS_JOBS         -j for the remote run (default: 4)
#   MUTANTS_LOCAL_JOBS   -j for a local or fallback run (default: 4)
#   MUTANTS_EXTRA_FILE   one repo-relative file this run needs that the sync
#                        would otherwise skip

set -euo pipefail

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
  MUTANTS_JOBS="${MUTANTS_LOCAL_JOBS:-4}" exec bash ./scripts/mutants-run.sh "$@"
}

if [ "${DREP_MUTANTS_REMOTE:-1}" = "0" ]; then
  run_local "$@"
fi

# Only an unreachable host permits fallback; rsync failures must fail the gate.
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" true 2>/dev/null; then
  echo "warning: $HOST is unreachable - running the mutation sweep locally instead." >&2
  echo "         This will use this machine's CPU for the duration." >&2
  run_local "$@"
fi

echo "mutants: running on $HOST (-j $JOBS), results will mirror to $LOCAL_RESULTS"

# Unique checkouts prevent an overlapping rsync from changing a running sweep.
REMOVE_REMOTE_ON_EXIT=1
rsync -a --mkpath \
  --exclude target --exclude 'mutants.out*' \
  --exclude .git --exclude venv --exclude node_modules \
  --exclude .mypy_cache --exclude .pytest_cache --exclude .ruff_cache \
  --exclude '*.egg-info' --exclude dist --exclude build --exclude .drep \
  --exclude '.tokens' --exclude '.env*' \
  ./ "$REMOTE/"

# The staged diff lives under excluded target/, so transfer it explicitly.
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

# %q requires bash. With no arguments it would emit one empty argument.
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
export MUTANTS_OUT_DIR=$MUTANTS_OUT_DIR_Q
mkdir -p $MUTANTS_OUT_DIR_Q
MUTANTS_JOBS=$JOBS bash ./scripts/mutants-run.sh $QUOTED_ARGS
EOF

# Preserve the remote checkout if its diagnostics cannot be mirrored.
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
