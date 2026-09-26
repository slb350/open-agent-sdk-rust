#!/usr/bin/env bash
# The only way the mutation scripts reach homelab-ai-1, where mutation runs. Source after AI1_CI_ROLE is assigned. Every command and transfer runs there inside the installed CI sandbox as that role, through root-owned offload.py, and each function accepts only the options its callers use.
AI1_HOST=steve@192.168.68.88
AI1_OFFLOAD=/usr/local/lib/ai-ci/offload.py

# offload.py decides which roles exist. This only keeps a missing or malformed name, or one that is not a mutation role, off the remote command line.
ai1_role_check() {
  case "${AI1_CI_ROLE:-}" in
  '' | *[![:lower:][:digit:]-]*) ;;
  *-mutants) return 0 ;;
  esac
  printf 'ai-1 transport: invalid or missing CI role\n' >&2
  return 2
}

# ai1_ssh [-o OPTION]... COMMAND...: run COMMAND on ai-1 as the role.
ai1_ssh() {
  local options=() command_text
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
      printf 'ai-1 transport: unsupported ssh option %s\n' "$1" >&2
      return 2
      ;;
    *) break ;;
    esac
  done
  if [ "$#" -eq 0 ]; then
    printf 'ai-1 transport: a command is required\n' >&2
    return 2
  fi
  ai1_role_check || return $?
  if [ "$#" -eq 1 ]; then
    command_text="$1"
  else
    printf -v command_text '%q ' "$@"
  fi
  printf -v command_text 'sudo %s %q %q' "$AI1_OFFLOAD" "$AI1_CI_ROLE" "$command_text"
  command ssh ${options[@]+"${options[@]}"} "$AI1_HOST" "$command_text"
}

# ai1_push [OPTION]... LOCAL... REMOTE and ai1_pull [OPTION]... REMOTE LOCAL copy between local paths and a path relative to the role's home on ai-1. The caller names which side is remote, so no operand is ever read as a host.
ai1_push() { ai1_rsync push "$@"; }
ai1_pull() { ai1_rsync pull "$@"; }

ai1_rsync() {
  local direction="$1" options=() operands=() operand rsync_path last
  shift
  for operand in "$@"; do
    case "$operand" in
    -a | -aR | --delete | --force | --delete-excluded | --no-times | --omit-dir-times | --timeout=* | --exclude=* | --filter=*) options+=("$operand") ;;
    -*)
      printf 'ai-1 transport: unsupported rsync option %s\n' "$operand" >&2
      return 2
      ;;
    # rsync reads a colon before any slash as a host; ./ keeps a relative path local.
    /*) operands+=("$operand") ;;
    *) operands+=("./$operand") ;;
    esac
  done
  if [ "${#operands[@]}" -lt 2 ] || { [ "$direction" = pull ] && [ "${#operands[@]}" -ne 2 ]; }; then
    printf 'ai-1 transport: %s needs local and remote paths\n' "ai1_$direction" >&2
    return 2
  fi
  ai1_role_check || return $?
  if [ "$direction" = push ]; then
    last=$((${#operands[@]} - 1))
    operands[last]="$AI1_HOST:${operands[last]#./}"
  else
    operands[0]="$AI1_HOST:${operands[0]#./}"
  fi
  printf -v rsync_path 'sudo %s %q rsync' "$AI1_OFFLOAD" "$AI1_CI_ROLE"
  command rsync --rsync-path="$rsync_path" ${options[@]+"${options[@]}"} "${operands[@]}"
}
