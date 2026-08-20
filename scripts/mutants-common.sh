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
