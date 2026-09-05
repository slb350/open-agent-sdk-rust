#!/usr/bin/env bash
#
# Run cargo-mutants and decide the verdict from the results.
#
# Shared by the pre-commit hook (scoped to the staged diff) and by CI (full
# sweep), so the two cannot disagree about what counts as a failure. Any
# arguments given are passed through to cargo-mutants; the scope is the
# caller's business, the verdict is this script's.

set -euo pipefail

# The runner and result reader must use the same output directory.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/mutants-common.sh
# shellcheck disable=SC1091
. "$SCRIPT_DIR/mutants-common.sh"
OUT_DIR="$MUTANTS_OUT_DIR"
mkdir -p "$OUT_DIR"

# Keep stranded copies off Strix's tmpfs. Each invocation owns a namespace;
# stale cleanup may remove only dead owners on this host.
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

# Concurrent builds need headroom beyond the short unmutated baseline.
cargo mutants -j "${MUTANTS_JOBS:-4}" --no-shuffle --minimum-test-timeout 60 \
  --output "$OUT_DIR" "$@" && status=0 || status=$?

MISSED="$OUT_DIR/mutants.out/missed.txt"

# cargo-mutants prioritizes timeout exit 3 over survivor exit 2. Check the
# actual survivor list first; a timeout alone counts as detection.
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
