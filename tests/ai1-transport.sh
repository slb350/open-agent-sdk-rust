#!/usr/bin/env bash
# Exercise fail-closed parsing without SSH, rsync, credentials, or filesystem I/O.
set -euo pipefail
# Bash 3.2, which macOS ships, does not stop a set -e script when `[[ ]]` fails,
# so every assertion names its own failure instead of relying on errexit.
fail() {
  echo "ai1-transport: assertion failed at line $1" >&2
  exit 1
}
AI1_CI_ROLE=open-agent-sdk-rust-mutants
# shellcheck source=scripts/mutants-ai1-transport.sh
. "$(dirname "${BASH_SOURCE[0]}")/../scripts/mutants-ai1-transport.sh"
# shellcheck disable=SC2329 # Called indirectly by the sourced transport wrappers.
command() { printf '<%s>' "$@"; }
expect_refusal() {
  local status=0 output
  output=$("$@" 2>/dev/null) || status=$?
  [[ $status == 2 && -z $output ]]
}
expect_refusal ssh -t steve@192.168.68.88 true
expect_refusal ssh -o
expect_refusal ssh
expect_refusal ssh homelab-ai-1
expect_refusal rsync --rsync-path=sh source steve@192.168.68.88:dest
output=$(ssh -o BatchMode=yes steve@192.168.68.88 'printf "%s" "literal $ text"')
[[ $output == *"sudo /usr/local/lib/ai-ci/offload.py $AI1_CI_ROLE"* ]] || fail $LINENO
[[ $output == *'literal'* ]] || fail $LINENO
output=$(ssh -o BatchMode=yes explicit-old-host true)
[[ $output == '<ssh><-o><BatchMode=yes><explicit-old-host><true>' ]] || fail $LINENO
output=$(rsync -a source steve@192.168.68.88:dest)
[[ $output == *"<--rsync-path=sudo /usr/local/lib/ai-ci/offload.py $AI1_CI_ROLE rsync>"* ]] || fail $LINENO
output=$(rsync -a explicit-old-host:source dest)
[[ $output == '<rsync><-a><explicit-old-host:source><dest>' ]] || fail $LINENO
output=$(ssh 192.168.68.88 true)
[[ $output == *"sudo /usr/local/lib/ai-ci/offload.py $AI1_CI_ROLE"* ]] || fail $LINENO
output=$(rsync homelab-ai-1.local:source dest)
[[ $output == *"<--rsync-path=sudo /usr/local/lib/ai-ci/offload.py $AI1_CI_ROLE rsync>"* ]] || fail $LINENO
expect_refusal rsync rsync://192.168.68.88/module dest
expect_refusal rsync 192.168.68.88::module dest
expect_refusal rsync homelab-ai-1.local::module dest
expect_refusal rsync steve@192.168.68.88::module dest
output=$(rsync rsync://legacy-host/module/homelab-ai-1 dest)
[[ $output == '<rsync><rsync://legacy-host/module/homelab-ai-1><dest>' ]] || fail $LINENO
# Decode only this fixed test payload through a mocked sudo; never contact SSH.
command() {
  local last
  for last in "$@"; do :; done
  eval "$last"
}
sudo() {
  [[ $1 == /usr/local/lib/ai-ci/offload.py && $2 == "$AI1_CI_ROLE" ]] || return 1
  printf '%s' "$3"
}
payload='printf "%s" "literal $ text"'
# shellcheck disable=SC2029 # Exercise local argument quoting through the mocked transport.
output=$(ssh 192.168.68.88 "$payload")
[[ $output == "$payload" ]] || fail $LINENO
AI1_CI_ROLE='bad; role'
expect_refusal ssh steve@192.168.68.88 true
expect_refusal rsync source steve@192.168.68.88:dest
