#!/usr/bin/env bash
#
# Run cargo-mutants and decide the verdict from the results.
#
# Shared by the pre-commit hook (scoped to the staged diff) and by CI (full
# sweep), so the two cannot disagree about what counts as a failure. Any
# arguments given are passed through to cargo-mutants; the scope is the
# caller's business, the verdict is this script's.

set -euo pipefail

# --output pins the results directory because this script reads `missed.txt` out
# of it to reach its verdict, so it has to know where it is rather than inherit
# whatever the caller's cwd happened to be. The path itself is defined once, in
# mutants-common.sh, because all three scripts in this trio need it. This is not
# concurrency protection: two runs in the same checkout share this directory
# exactly as they shared a cwd-relative `mutants.out`.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/mutants-common.sh
# shellcheck disable=SC1091
. "$SCRIPT_DIR/mutants-common.sh"
OUT_DIR="$MUTANTS_OUT_DIR"
mkdir -p "$OUT_DIR"

# Scratch copies default beside the checkout, not in the system temp dir.
#
# cargo-mutants copies the source tree once per job beneath `$TMPDIR` and deletes
# those copies only on a clean exit. A process killed with SIGKILL can strand
# them, and Strix mounts `/tmp` as a tmpfs: five killed sweeps in another
# repository pinned 31 GiB of RAM. The adjacent default makes a stranded copy a
# disk cost instead, and the next run safely reclaims namespaces whose owner is
# no longer alive. An explicit DREP_MUTANTS_TMPDIR override remains authoritative.
#
# Every invocation gets an owned directory. A shared fixed TMPDIR is unsafe:
# startup and EXIT cleanup from one direct, fallback or concurrent local run can
# otherwise erase another run's live cargo-mutants tree. The owner hostname and
# PID are encoded in the name before it becomes visible, so stale cleanup can
# distinguish a dead local owner from a live one without a non-portable flock.
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRATCH_ROOT="${DREP_MUTANTS_TMPDIR:-${ROOT}.mutants-tmp}"
case "$SCRATCH_ROOT" in
  /*) ;;
  *) echo "mutants-run: DREP_MUTANTS_TMPDIR must be absolute: $SCRATCH_ROOT" >&2
     exit 64 ;;
esac
if [ "$SCRATCH_ROOT" = "/" ]; then
  echo "mutants-run: refusing to use / as the scratch root" >&2
  exit 64
fi
mkdir -p "$SCRATCH_ROOT"

HOST_ID="$(mutants_host_id)"

mutants_sweep_stale_owned_paths "$SCRATCH_ROOT" "$HOST_ID" "scratch namespace"
RUN_TMPDIR="$(mktemp -d "$SCRATCH_ROOT/run_${HOST_ID}_$$_XXXXXX")"
export TMPDIR="$RUN_TMPDIR"

# Invoked indirectly by the EXIT trap.
# shellcheck disable=SC2329
cleanup_owned_scratch() {
  local original_status=$?
  local cleanup_status
  trap - EXIT
  set +e
  mutants_delete_exact_path "$RUN_TMPDIR"
  cleanup_status=$?
  if [ "$cleanup_status" -ne 0 ]; then
    echo "mutants-run: failed to clean owned scratch $RUN_TMPDIR" >&2
  fi
  mutants_reconcile_cleanup_status "$original_status" "$cleanup_status"
  original_status=$?
  exit "$original_status"
}
trap cleanup_owned_scratch EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

# --minimum-test-timeout: cargo-mutants derives the per-mutant timeout from the
# unmutated baseline, which on a fast suite is a second or two. With -j running
# several full suites at once on a loaded machine, a healthy mutant can exceed
# that and be recorded as TIMEOUT. Give it real headroom so a timeout means what
# it should.
# MUTANTS_JOBS so the same script can be driven harder on a 32-thread box than
# on the laptop the hook runs on; see scripts/mutants-remote.sh.
cargo mutants -j "${MUTANTS_JOBS:-4}" --no-shuffle --minimum-test-timeout 60 \
  --output "$OUT_DIR" "$@" && status=0 || status=$?

MISSED="$OUT_DIR/mutants.out/missed.txt"

# The verdict comes from the results, not from the exit code alone.
#
# A timeout is NOT a failure: some mutations produce an infinite loop (`i += 1`
# becoming `i *= 1` never advances), and a suite that hangs has detected the
# mutant as surely as one that fails. But cargo-mutants reports exit 3
# (Timeout) in preference to exit 2 (FoundProblems) - see `Outcome::exit_code`,
# where `timeout > 0` is tested before `missed > 0` - so a run with one hanging
# mutant AND a genuine survivor also exits 3. Mapping 3 to success on the exit
# code alone would wave that survivor through.
#
# Both callers previously carried a comment asserting that cargo-mutants exits
# 0 for timeouts. It does not, and neither had ever seen a timeout to find out:
# the hook would have blocked on a hang that was really a detection, and CI
# would have done the same.
#
# So: any missed mutant fails, whatever the exit code says. Otherwise a timeout
# passes, and everything else (usage error, failing baseline, unparseable diff)
# fails with the code cargo-mutants chose.
if [ -s "$MISSED" ]; then
  echo "mutants survived - a surviving mutant is a test that cannot tell" >&2
  echo "correct behaviour from incorrect. Fix the test, never the mutant list." >&2
  cat "$MISSED" >&2
  exit 2
fi

if [ "$status" -eq 3 ]; then
  echo "note: mutants timed out with none missed; a hang is detection, not a failure"
  exit 0
fi

exit "$status"
