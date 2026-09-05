#!/usr/bin/env bash
#
# Classify mutation work for an ordinary CI diff.
#
# Output is line-oriented for Bash mapfile:
#   none
#   full
#   files
#   --file
#   path/to/owning-source.rs
#
# Test files and fixtures often exercise code outside a name-matched module, so
# ambiguous ownership deliberately expands to a full package/workspace sweep.
# A changed source file that contains inline tests is safe to map to itself.

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: mutants-ci-scope.sh BASE HEAD" >&2
  exit 64
fi

BASE="$1"
HEAD="$2"
TEST_SIGNAL_REGEX='#\[(cfg\(test\)|test|[[:alnum:]_]+::test|rstest|test_case|case|parameterized)([[:space:](]|])|(^|[[:space:]])mod[[:space:]]+tests([[:space:]]|\{|;)|//[/!][[:space:]]*```(rust)?|proptest!|rstest!'
git rev-parse --verify --quiet "$BASE^{commit}" >/dev/null || {
  echo "mutants-ci-scope: invalid base revision: $BASE" >&2
  exit 65
}
git rev-parse --verify --quiet "$HEAD^{commit}" >/dev/null || {
  echo "mutants-ci-scope: invalid head revision: $HEAD" >&2
  exit 65
}

content_has_tests() {
  local revision="$1"
  local path="$2"
  git show "$revision:$path" 2>/dev/null | grep -Eq "$TEST_SIGNAL_REGEX"
}

test_content() {
  local revision="$1"
  local path="$2"
  # Extract only test-gated items; a suffix from the first test marker would
  # misclassify later production changes in mixed Rust source files.
  { git show "$revision:$path" 2>/dev/null || true; } |
    awk '
      function brace_delta(source, opens, closes) {
        sub(/\/\/.*$/, "", source)
        gsub(/"([^"\\]|\\.)*"/, "", source)
        opens = gsub(/\{/, "{", source)
        closes = gsub(/\}/, "}", source)
        return opens - closes
      }

      function test_attribute(source) {
        return source ~ /^[[:space:]]*#\[(cfg\(test\)|test|[[:alnum:]_]+::test|rstest|test_case|case|parameterized)([[:space:](]|\])/
      }

      function test_module(source) {
        return source ~ /(^|[[:space:]])mod[[:space:]]+tests([[:space:]]|\{|;)/
      }

      function begin_item(source, delta, remainder, has_open) {
        print source
        delta = brace_delta(source)
        has_open = source ~ /\{/
        if (has_open && delta > 0) {
          depth = delta
          return
        }
        if (has_open) {
          pending = 0
          return
        }
        remainder = source
        sub(/^.*\][[:space:]]*/, "", remainder)
        pending = remainder == ""
      }

      depth > 0 {
        print
        depth += brace_delta($0)
        next
      }

      pending {
        print
        if ($0 ~ /^[[:space:]]*$/ || $0 ~ /^[[:space:]]*#\[/) {
          next
        }
        delta = brace_delta($0)
        if ($0 ~ /\{/ && delta > 0) {
          depth = delta
        }
        pending = 0
        next
      }

      test_attribute($0) || test_module($0) || $0 ~ /proptest!|rstest!/ {
        begin_item($0)
        next
      }
    '
}

test_content_changed() {
  local path="$1"
  ! cmp -s <(test_content "$BASE" "$path") <(test_content "$HEAD" "$path")
}

path_exists() {
  git cat-file -e "$1:$2" 2>/dev/null
}

add_file_scope() {
  local candidate="$1"
  local existing

  for existing in "${FILE_SCOPES[@]-}"; do
    [ "$existing" = "$candidate" ] && return
  done
  FILE_SCOPES+=("$candidate")
}

FULL=false
FILE_SCOPES=()
while IFS= read -r -d '' path; do
  case "$path" in
    *$'\n'*|*:* )
      FULL=true
      continue
      ;;
  esac

  case "$path" in
    tests/*|*/tests/*|test/*|*/test/*|test_support/*|*/test_support/*|test_support.rs|*/test_support.rs|*_test.rs|*_tests.rs|fixtures/*|*/fixtures/*|testdata/*|*/testdata/*|snapshots/*|*/snapshots/*|*.snap|*.snap.new|.cargo/mutants.toml)
      FULL=true
      continue
      ;;
  esac

  case "$path" in
    *.rs) ;;
    *) continue ;;
  esac

  if ! path_exists "$HEAD" "$path"; then
    if content_has_tests "$BASE" "$path"; then
      FULL=true
    fi
    continue
  fi

  case "$path" in
    */tests.rs|*/test.rs)
      parent="${path%/*}"
      if path_exists "$HEAD" "$parent.rs"; then
        add_file_scope "$parent.rs"
      elif path_exists "$HEAD" "$parent/mod.rs"; then
        add_file_scope "$parent/mod.rs"
      else
        FULL=true
      fi
      ;;
    *)
      if (content_has_tests "$HEAD" "$path" || content_has_tests "$BASE" "$path") &&
        test_content_changed "$path"; then
        add_file_scope "$path"
      fi
      ;;
  esac
done < <(git diff --no-renames --name-only --diff-filter=ACDMRTUXB -z "$BASE" "$HEAD" --)

if [ "$FULL" = true ]; then
  echo full
elif [ -n "${FILE_SCOPES[*]-}" ]; then
  echo files
  for path in "${FILE_SCOPES[@]}"; do
    echo --file
    printf '%s\n' "$path"
  done
else
  echo none
fi
