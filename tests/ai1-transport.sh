#!/usr/bin/env bash
# Exercise fail-closed parsing without SSH, rsync, credentials, or filesystem I/O.
set -euo pipefail
# Bash 3.2, which macOS ships, does not stop a set -e script when `[[ ]]` fails,
# so every assertion names its own failure instead of relying on errexit.
fail() {
  echo "ai1-transport: assertion failed at line ${BASH_LINENO[0]}" >&2
  exit 1
}
# The expected role is held apart from AI1_CI_ROLE, which the transport under test could overwrite.
expected_role=open-agent-sdk-rust-mutants
offload="sudo /usr/local/lib/ai-ci/offload.py $expected_role"
AI1_CI_ROLE=$expected_role
# shellcheck source=scripts/mutants-ai1-transport.sh
. "$(dirname "${BASH_SOURCE[0]}")/../scripts/mutants-ai1-transport.sh"
# shellcheck disable=SC2329 # Called indirectly by the sourced transport functions.
command() { printf '<%s>' "$@"; }
expect_refusal() {
  local status=0 output
  output=$("$@" 2>/dev/null) || status=$?
  [[ $status == 2 && -z $output ]] || fail
}
expect_refusal ai1_ssh -t true
expect_refusal ai1_ssh -o
expect_refusal ai1_ssh
output=$(ai1_ssh -o BatchMode=yes true)
[[ $output == "<ssh><-o><BatchMode=yes><steve@192.168.68.88><$offload true>" ]] || fail
for refused in "-e ssh" --rsh=other --rsync-path=sh --exclude; do
  expect_refusal ai1_push -a "$refused" source/ remote/
done
expect_refusal ai1_push -a source/
expect_refusal ai1_pull -a remote/ local/ extra/
output=$(ai1_push -a --exclude=target source/ remote/dir/)
[[ $output == "<rsync><--rsync-path=$offload rsync><-a><--exclude=target><./source/><steve@192.168.68.88:remote/dir/>" ]] || fail
# A local path holding a colon stays local, relative or absolute.
output=$(ai1_push -aR a:b/f /tmp/c:d/f remote/)
[[ $output == "<rsync><--rsync-path=$offload rsync><-aR><./a:b/f></tmp/c:d/f><steve@192.168.68.88:remote/>" ]] || fail
output=$(ai1_pull -a remote/out/ out/)
[[ $output == "<rsync><--rsync-path=$offload rsync><-a><steve@192.168.68.88:remote/out/><./out/>" ]] || fail
# Decode only this fixed test payload through a mocked sudo; never contact SSH.
command() {
  local last
  for last in "$@"; do :; done
  eval "$last"
}
sudo() {
  [[ $1 == /usr/local/lib/ai-ci/offload.py && $2 == "$expected_role" ]] || fail
  printf '%s' "$3"
}
payload='printf "%s" "literal $ text"'
output=$(ai1_ssh "$payload")
[[ $output == "$payload" ]] || fail
for AI1_CI_ROLE in 'bad; role' '' drep-linux Drep-mutants; do
  expect_refusal ai1_ssh true
  expect_refusal ai1_push source/ remote/
done
