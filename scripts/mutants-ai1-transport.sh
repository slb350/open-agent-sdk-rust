#!/usr/bin/env bash
# Source after AI1_CI_ROLE is assigned. Every command and transfer goes to homelab-ai-1 and runs there inside the installed CI sandbox as that role; any other host is refused, because mutation runs only on ai-1. This private adapter accepts only the -o option pairs used by its callers.
is_ai1_host() {
  case "${1##*@}" in
  192.168.68.88 | homelab-ai-1 | homelab-ai-1.local) return 0 ;;
  *) return 1 ;;
  esac
}

# offload.py decides which roles exist. This only keeps a missing or malformed name, or one that is not a mutation role, off the remote command line.
ai1_role_check() {
  case "${AI1_CI_ROLE:-}" in
  '' | *[![:lower:][:digit:]-]*) ;;
  *-mutants) return 0 ;;
  esac
  printf 'ai-1 transport: invalid or missing CI role\n' >&2
  return 2
}

ssh() {
  local options=() target command_text
  while [ "$#" -gt 0 ]; do
    case "$1" in
    -o)
      if [ "$#" -lt 2 ]; then
        printf 'ai-1 transport: -o requires a value\n' >&2
        return 2
      fi
      options+=("$1" "$2")
      shift 2
      ;;
    -*)
      printf 'ai-1 transport: unsupported SSH option %s\n' "$1" >&2
      return 2
      ;;
    *) break ;;
    esac
  done
  if [ "$#" -eq 0 ]; then
    printf 'ai-1 transport: destination required\n' >&2
    return 2
  fi
  target="$1"
  shift
  if ! is_ai1_host "$target"; then
    printf 'ai-1 transport: %s is not homelab-ai-1\n' "$target" >&2
    return 2
  fi
  if [ "$#" -eq 0 ]; then
    printf 'ai-1 transport: a sandboxed command is required\n' >&2
    return 2
  fi
  ai1_role_check || return $?
  if [ "$#" -eq 1 ]; then
    command_text="$1"
  else
    printf -v command_text '%q ' "$@"
  fi
  printf -v command_text 'sudo /usr/local/lib/ai-ci/offload.py %q %q' "$AI1_CI_ROLE" "$command_text"
  command ssh ${options[@]+"${options[@]}"} "$target" "$command_text"
}

rsync() {
  local argument ai1_transfer=0 rsync_path
  for argument in "$@"; do
    case "$argument" in
    rsync://* | *::*)
      printf 'ai-1 transport: daemon transfers are unsupported\n' >&2
      return 2
      ;;
    --rsync-path | --rsync-path=*)
      printf 'ai-1 transport: caller may not replace the remote execution path\n' >&2
      return 2
      ;;
    *:*)
      if ! is_ai1_host "${argument%%:*}"; then
        printf 'ai-1 transport: %s is not homelab-ai-1\n' "${argument%%:*}" >&2
        return 2
      fi
      ai1_transfer=1
      ;;
    esac
  done
  if [ "$ai1_transfer" -eq 0 ]; then
    printf 'ai-1 transport: a homelab-ai-1 endpoint is required\n' >&2
    return 2
  fi
  ai1_role_check || return $?
  printf -v rsync_path 'sudo /usr/local/lib/ai-ci/offload.py %q rsync' "$AI1_CI_ROLE"
  command rsync --rsync-path="$rsync_path" "$@"
}
