#!/usr/bin/env bash
#
# Run the mutation sweep on a bigger machine over SSH, then report its verdict
# here.
#
# Mutation testing is the most CPU-hungry gate in this repo: every mutant is a
# full build plus a full test run, and the hook fires on a laptop the developer
# is still using. Measured in drep on 12 mutants from src/docs/fence.rs, a local
# M5 Max at -j 4 took 1m54 with the machine pinned. The offload runs as this
# repository's CI role on ai-1 and costs this machine nothing.
#
# More jobs is not automatically better. Each job builds in its own copy of the
# tree, which cargo-mutants 27.1.0 makes without target/ unless copy_target is
# set, so every job cold-builds the dependencies before its first mutant. On the
# former 32-thread host, the same scope measured 38s at -j 4, 54s at -j 8, and
# 72s at -j 16.
# Keep the measured worker baseline until a complete sweep gives
# a measured reason to change it.
#
# The verdict rule is NOT duplicated here. This script syncs, invokes
# scripts/mutants-run.sh on the remote, and propagates its exit code - so the
# hook, CI and the remote sweep cannot disagree about what counts as a failure.
#
# Falls back to a local run, loudly, when the host is unreachable. A commit gate
# that silently skips itself because the LAN blipped is worse than a slow one.
#
#   DREP_MUTANTS_HOST    ssh target (default: steve@192.168.68.88)
#   DREP_MUTANTS_HOST_LOCK_WAIT_SECONDS
#                        wait for the checkout and host locks (default: 1800)
#   DREP_MUTANTS_RSYNC_TIMEOUT_SECONDS
#                        rsync I/O timeout while the host lock is held
#                        (default: 300)
#   DREP_MUTANTS_REMOTE  0 to force a local run
#   MUTANTS_JOBS         -j for the remote run (default: the role's MUTANTS_JOBS)
#   MUTANTS_LOCAL_JOBS   -j for a local or fallback run (default: 4)
#   MUTANTS_EXTRA_FILES  repo-relative paths this run needs that the sync
#                        would otherwise skip (space-separated, no spaces in
#                        the paths themselves)

set -euo pipefail

# `git rev-parse`, not `dirname "$0"/..`: the same answer install.sh already
# uses, and it does not care whether the script was reached through a symlink,
# a relative path or PATH.
cd "$(git rev-parse --show-toplevel)"
# shellcheck source=scripts/mutants-common.sh
. scripts/mutants-common.sh

# The ai-1 CI role this repository's mutation runs as, here and in its hosted
# sweeps. The role's unit also supplies the host lock and the job count.
AI1_CI_ROLE=open-agent-sdk-rust-mutants
HOST="${DREP_MUTANTS_HOST:-steve@192.168.68.88}"
REMOTE_DIR="$(remote_checkout_dir "$AI1_CI_ROLE")"
REMOTE="$HOST:$REMOTE_DIR"
JOBS="${MUTANTS_JOBS:-}"
RSYNC_IO_TIMEOUT_SECONDS="${DREP_MUTANTS_RSYNC_TIMEOUT_SECONDS:-300}"

case "$RSYNC_IO_TIMEOUT_SECONDS" in
0 | '' | *[!0-9]*)
  echo "mutants-remote: DREP_MUTANTS_RSYNC_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 64
  ;;
esac

run_local() {
  MUTANTS_JOBS="${MUTANTS_LOCAL_JOBS:-4}" exec ./scripts/mutants-run.sh "$@"
}

# The mirror writes this checkout's results, so it is this checkout's one run.
# Taken before anything else so a run that waits here probes the host once it
# has the lock, and a local fallback inherits it through exec.
acquire_checkout_lock mutants-remote || exit $?

if [ "${DREP_MUTANTS_REMOTE:-1}" = "0" ]; then
  run_local "$@"
fi

# A bare probe rather than letting the first rsync fail and reading its exit
# code: "the host is down, here is what I am doing instead" is the message the
# developer needs, and inferring it from an rsync failure would also swallow a
# full disk or an unwritable directory as "unreachable". One handshake, ~145ms,
# against a run measured in minutes.
# shellcheck source=scripts/mutants-ai1-transport.sh
if ! . scripts/mutants-ai1-transport.sh; then
  echo "warning: ai-1 transport unavailable; running mutation locally" >&2
  run_local "$@"
fi

if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" true 2>/dev/null; then
  echo "warning: $HOST is unreachable - running the mutation sweep locally instead." >&2
  echo "         This will use this machine's CPU for the duration." >&2
  run_local "$@"
fi

echo "mutants: running on $HOST as $AI1_CI_ROLE, results mirrored back to $MUTANTS_OUT_DIR"

# Keep one remote SSH process alive for the entire transaction. Its open file
# descriptor holds the host-wide lock while this process synchronizes source,
# runs mutation, and mirrors the matching result. Named pipes are used instead
# of Bash 4's `coproc` because macOS still ships Bash 3.2.
RUN_TOKEN="$$-$RANDOM-$RANDOM"
SESSION_DIR="$MUTANTS_OUT_DIR/remote-session-$RUN_TOKEN"
CONTROL_IN="$SESSION_DIR/control-in"
CONTROL_OUT="$SESSION_DIR/control-out"
mkdir -p "$SESSION_DIR"
mkfifo "$CONTROL_IN" "$CONTROL_OUT"

# shellcheck disable=SC2016  # This is the literal remote Bash program.
REMOTE_SCRIPT='set -euo pipefail
wait_seconds=$1
remote_dir=$2
out_dir=$3
jobs=$4
run_token=$5
shift 5
export PATH="$HOME/.cargo/bin:$PATH"
exec 9>>"${DREP_MUTANTS_HOST_LOCK:?the role unit names the host lock}"
flock -E 75 -w "$wait_seconds" 9
checkout="$HOME/$remote_dir"
mkdir -p "$checkout"
printf "mutants-lock-ready:%s\n" "$run_token"
IFS= read -r action
[[ $action == run ]] || exit 74
cd "$checkout"
mkdir -p "$out_dir"
set +e
DREP_MUTANTS_RESULT_TOKEN="$run_token" MUTANTS_JOBS="${jobs:-${MUTANTS_JOBS:-4}}" \
  ./scripts/mutants-run.sh "$@"
status=$?
set -e
result_token_file="$out_dir/.run-token"
[[ -f $result_token_file && ! -L $result_token_file ]] || exit 76
[[ $(<"$result_token_file") == "$run_token" ]] || exit 76
printf "mutants-run-finished:%s:%s\n" "$run_token" "$status"
IFS= read -r action
[[ $action == mirrored ]] || exit 74
exit "$status"'

REMOTE_COMMAND=
printf -v REMOTE_COMMAND 'bash -c %q bash' "$REMOTE_SCRIPT"
for remote_arg in \
  "$MUTANTS_HOST_LOCK_WAIT_SECONDS" \
  "$REMOTE_DIR" \
  "$MUTANTS_OUT_DIR" \
  "$JOBS" \
  "$RUN_TOKEN" \
  "$@"; do
  printf -v remote_arg_q '%q' "$remote_arg"
  REMOTE_COMMAND+=" $remote_arg_q"
done

ssh -o BatchMode=yes "$HOST" "$REMOTE_COMMAND" \
  <"$CONTROL_IN" >"$CONTROL_OUT" &
REMOTE_SESSION_PID=$!
exec 7>"$CONTROL_IN"
exec 8<"$CONTROL_OUT"
remote_session_open=1

# shellcheck disable=SC2317,SC2329  # Invoked by the EXIT trap (SC2317 on shellcheck before 0.10, SC2329 after).
cleanup_remote_session() {
  if [ "$remote_session_open" -eq 1 ]; then
    kill "$REMOTE_SESSION_PID" 2>/dev/null || true
    wait "$REMOTE_SESSION_PID" 2>/dev/null || true
  fi
  exec 7>&- 8<&-
  remove_tree "$SESSION_DIR"
}
trap cleanup_remote_session EXIT
# A session that has ended turns the next control write into SIGPIPE, which
# would kill this shell before the trap above could run.
trap 'exit 74' PIPE

exit_after_remote_session_failure() {
  if wait "$REMOTE_SESSION_PID"; then
    status=74
  else
    status=$?
  fi
  remote_session_open=0
  exit "$status"
}

if ! IFS= read -r ready <&8; then
  exit_after_remote_session_failure
fi
if [ "$ready" != "mutants-lock-ready:$RUN_TOKEN" ]; then
  echo "mutants-remote: unexpected remote lock handshake" >&2
  exit 74
fi

# The remote session created the destination before it reported the lock ready.
# REMOTE_DIR is this checkout's alone and the checkout lock is held, so nothing
# else writes it even if the session ends mid-transfer; the next control write
# then fails the run.
#
# --delete so a file deleted locally cannot linger and be mutated remotely.
# target/ is excluded: the run writes its results there after this sync, and
# the staged diff follows in the transfer below. The cache directories are excluded
# because they are 64MB of this checkout that no mutation run reads, re-diffed
# on every commit against a Rust payload of about 1MB. Credentials are excluded
# because nothing in the suite reads them and they have no business on another
# host.
# --delete alone leaves the remote tree stale, and the sweep then tests a tree
# the commit does not have. An excluded name *inside* a directory protects that
# directory from removal, so `docs/api/build/html` kept `docs/api` alive after
# the commit that deleted it, and a test asserting the directory is gone failed
# on the remote while passing here. --force is not enough - it deletes
# non-empty directories, not protected ones.
#
# So: --delete-excluded, which removes the excluded leftovers too.
rsync -a --delete --force --delete-excluded \
  --timeout="$RSYNC_IO_TIMEOUT_SECONDS" \
  --exclude target --exclude 'mutants.out*' \
  --exclude .git --exclude node_modules \
  --exclude dist --exclude build --exclude .drep \
  --exclude '.env*' \
  ./ "$REMOTE/"

# Files the run needs that the sync above skipped - in practice the staged diff,
# which mutants-staged.sh writes under the excluded target/. Named by the caller
# rather than recovered by scanning "$@" for cargo-mutants' own flags: what
# belongs at this layer is "move these bytes", not that layer's argument
# grammar. -R recreates each path under the remote root, directories included.
if [ -n "${MUTANTS_EXTRA_FILES:-}" ]; then
  for extra in ${MUTANTS_EXTRA_FILES}; do
    if ! is_contained_path "$extra"; then
      echo "mutants-remote: MUTANTS_EXTRA_FILES must be repo-relative without . or .. components, got $extra" >&2
      exit 64
    fi
  done
  # shellcheck disable=SC2086  # word splitting is the interface: it is a list
  rsync -aR --timeout="$RSYNC_IO_TIMEOUT_SECONDS" \
    ${MUTANTS_EXTRA_FILES} "$REMOTE/"
fi

printf 'run\n' >&7
finished_status=
while IFS= read -r remote_line <&8; do
  case "$remote_line" in
  "mutants-run-finished:$RUN_TOKEN:"*)
    finished_status=${remote_line##*:}
    break
    ;;
  *) printf '%s\n' "$remote_line" ;;
  esac
done

case "$finished_status" in
'' | *[!0-9]*)
  exit_after_remote_session_failure
  ;;
esac

# Mirror the results back so `missed.txt`, the logs and the diffs of surviving
# mutants can be read here, where the fix gets written. The remote process still
# owns the host lock here, and its unique `.run-token` proved this run reached
# cargo-mutants rather than exposing a previous result after an early failure.
rsync -a --timeout="$RSYNC_IO_TIMEOUT_SECONDS" \
  "$REMOTE/$MUTANTS_OUT_DIR/" "$MUTANTS_OUT_DIR/" 2>/dev/null ||
  echo "warning: mutation completed but its result mirror failed" >&2
printf 'mirrored\n' >&7

while IFS= read -r remote_line <&8; do
  printf '%s\n' "$remote_line"
done
if wait "$REMOTE_SESSION_PID"; then
  status=0
else
  status=$?
fi
remote_session_open=0

if [ "$status" -ne "$finished_status" ]; then
  echo "mutants-remote: remote status changed after the result handshake" >&2
  exit 74
fi

exit "$finished_status"
