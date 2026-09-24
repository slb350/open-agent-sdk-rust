#!/usr/bin/env bash
#
# Run cargo-mutants and decide the verdict from the results.
#
# Shared verdict for staged-diff, pushed-diff and full mutation runs.
# Arguments pass through to cargo-mutants; the caller chooses the scope.

set -euo pipefail

# --output pins the results directory because this script reads `missed.txt` out
# of it to reach its verdict, so it has to know where it is rather than inherit
# whatever the caller's cwd happened to be. The path itself is defined once, in
# mutants-common.sh, because all three scripts in this trio need it.
# shellcheck source=scripts/mutants-common.sh
. "$(dirname "$0")/mutants-common.sh"
OUT_DIR="$MUTANTS_OUT_DIR"
mkdir -p "$OUT_DIR"

acquire_checkout_lock mutants-run || exit $?

# A dedicated mutation host may serve both GitHub and laptop-offloaded runs in
# separate persistent workspaces. When its operator provides a shared lock,
# serialize those otherwise independent checkouts before either can clean
# scratch or start cargo-mutants. Local fallback runs leave the variable unset
# and retain checkout-local behavior.
HOST_LOCK="${DREP_MUTANTS_HOST_LOCK:-}"
if [ -n "$HOST_LOCK" ]; then
  case "$HOST_LOCK" in
    /*) ;;
    *)
      echo "mutants-run: DREP_MUTANTS_HOST_LOCK must be absolute" >&2
      exit 64
      ;;
  esac
  hold_lock 9 "$HOST_LOCK" mutants-run || exit $?
fi

# Scratch copies go beside the checkout, not in the system temp dir.
#
# cargo-mutants copies the tree into `$TMPDIR` once per job and deletes the
# copies only on a clean exit. A run that is cancelled or
# hits the job timeout strands them. The former Strix host mounted `/tmp` as a
# tmpfs; in another repository five such sweeps pinned 31 GiB of RAM with
# nothing else running. Here the copies sit on disk and a stale one costs
# storage instead of memory.
#
# A sibling of the checkout rather than a child: cargo-mutants copies the
# checkout, so scratch inside it would be copied into every later copy
# (target/ included, once copy_target is on).
RUN_SCRATCH="${DREP_MUTANTS_TMPDIR:-${MUTANTS_ROOT}.mutants-tmp}/run"

# The checkout lock means no other run is using this checkout's scratch, so
# everything a previous run left in the run directory - tree copies and the
# temporary files of tests it killed - can go. cargo-mutants never sees a
# SIGKILL, and the runner's cancellation ends in one, so the trap below is the
# common case and this is the backstop.
remove_tree "$RUN_SCRATCH"
mkdir -p "$RUN_SCRATCH"
export TMPDIR="$RUN_SCRATCH"
trap 'remove_tree "$RUN_SCRATCH"' EXIT

# A caller that mirrors results across machines needs proof that the output is
# from this invocation, not a previous sweep. Clear only the exact prior result
# tree, remove any old marker without following it, and publish the caller's
# unique token immediately before cargo-mutants starts.
remove_tree "$OUT_DIR/mutants.out"
RESULT_TOKEN_FILE="$OUT_DIR/.run-token"
remove_tree "$RESULT_TOKEN_FILE"
if [ -n "${DREP_MUTANTS_RESULT_TOKEN:-}" ]; then
  (umask 077; printf '%s\n' "$DREP_MUTANTS_RESULT_TOKEN" >"$RESULT_TOKEN_FILE")
fi

# --cap-lints: `[lints.rust] warnings = "deny"` in Cargo.toml applies to the
# mutated build too, and a mutant that replaces a function body leaves the
# arguments unread. `unused_variable` is then a hard error, the mutant is
# recorded UNVIABLE, and unviable is silently not a failure - so the gate passes
# having compiled the mutant and never run it. On the branch that found this, 16
# of 17 mutants in the diff were unviable for that reason and for one missing
# `use`, and the sweep still exited 0. Capping lints for the scratch build only
# took that to 1, which is a mutant whose return type has no `Default` and is
# genuinely unbuildable.
#
# --minimum-test-timeout: cargo-mutants derives the per-mutant timeout from the
# unmutated baseline, which on a fast suite is a second or two. With -j running
# several full suites at once on a loaded machine, a healthy mutant can exceed
# that and be recorded as TIMEOUT. Give it real headroom so a timeout means what
# it should.
# MUTANTS_JOBS so the same script can be driven harder on a 32-thread box than
# on the laptop the hook runs on; see scripts/mutants-remote.sh.
#
# 6<&- 9<&-: the checkout and host locks stay with this script. A test fixture
# that outlives its mutant must not inherit either and block the next run.
cargo mutants -j "${MUTANTS_JOBS:-4}" --no-shuffle --minimum-test-timeout 120 \
  --cap-lints true --output "$OUT_DIR" "$@" 6<&- 9<&- && status=0 || status=$?

MISSED="$OUT_DIR/mutants.out/missed.txt"
UNVIABLE="$OUT_DIR/mutants.out/unviable.txt"

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

# Unviable is reported rather than passed over in silence. It is not a failure -
# some mutants cannot be built at all, and there is nothing to fix in a test for
# one whose return type has no `Default` - but it is also not a verdict, because
# the mutant never ran. Left unprinted it reads as a pass: the summary line
# scrolls past, `missed.txt` is empty, and a sweep that built its whole scope and
# tested none of it exits 0. Naming the count is what makes "the gate has stopped
# covering this file" something a reader can notice.
if [ -s "$UNVIABLE" ]; then
  echo "note: $(wc -l <"$UNVIABLE" | tr -d ' ') mutant(s) did not build, so nothing was"
  echo "proven about them either way. A count that climbs means the gate is losing scope."
  cat "$UNVIABLE"
fi

if [ "$status" -eq 3 ]; then
  echo "note: mutants timed out with none missed; a hang is detection, not a failure"
  exit 0
fi

exit "$status"
