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
# shellcheck disable=SC2329 # Called indirectly by the sourced transport wrappers.
command() { printf '<%s>' "$@"; }
expect_refusal() {
  local status=0 output
  output=$("$@" 2>/dev/null) || status=$?
  [[ $status == 2 && -z $output ]] || fail
}
expect_refusal ssh -t steve@192.168.68.88 true
expect_refusal ssh -o
expect_refusal ssh
expect_refusal ssh homelab-ai-1
expect_refusal rsync --rsync-path=sh source steve@192.168.68.88:dest
output=$(ssh -o BatchMode=yes steve@192.168.68.88 'printf "%s" "literal $ text"')
[[ $output == *"$offload"* ]] || fail
[[ $output == *'literal'* ]] || fail
output=$(ssh -o BatchMode=yes explicit-old-host true)
[[ $output == '<ssh><-o><BatchMode=yes><explicit-old-host><true>' ]] || fail
output=$(rsync -a source steve@192.168.68.88:dest)
[[ $output == *"<--rsync-path=$offload rsync>"* ]] || fail
output=$(rsync -a explicit-old-host:source dest)
[[ $output == '<rsync><-a><explicit-old-host:source><dest>' ]] || fail
output=$(ssh 192.168.68.88 true)
[[ $output == *"$offload"* ]] || fail
output=$(rsync homelab-ai-1.local:source dest)
[[ $output == *"<--rsync-path=$offload rsync>"* ]] || fail
expect_refusal rsync rsync://192.168.68.88/module dest
expect_refusal rsync 192.168.68.88::module dest
expect_refusal rsync homelab-ai-1.local::module dest
expect_refusal rsync steve@192.168.68.88::module dest
output=$(rsync rsync://legacy-host/module/homelab-ai-1 dest)
[[ $output == '<rsync><rsync://legacy-host/module/homelab-ai-1><dest>' ]] || fail
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
# shellcheck disable=SC2029 # Exercise local argument quoting through the mocked transport.
output=$(ssh 192.168.68.88 "$payload")
[[ $output == "$payload" ]] || fail
for AI1_CI_ROLE in 'bad; role' '' drep-linux Drep-mutants; do
  expect_refusal ssh steve@192.168.68.88 true
  expect_refusal rsync source steve@192.168.68.88:dest
done
