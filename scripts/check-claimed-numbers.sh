#!/usr/bin/env bash
# Consistency gate: numbers claimed in the documentation must match a real run.
#
# Why this exists: the README once claimed a test count that no longer matched
# what pytest reported, and later three rows of a table that the shipped code
# no longer produced. A rule nobody enforces drifts, so this enforces it.
#
# Usage:       bash scripts/check-claimed-numbers.sh [--slow]
#   default    the test suite, links, and every documented number that can be
#              read from a committed artifact or computed in seconds
#   --slow     also re-runs the scripts behind the held-out tables, the
#              sample-size table and the re-runnable attack entries (~1 hour)
# Exit status: 0 -- the claims hold; 1 -- a mismatch was found.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
fail=0

# Interpreter: $PYTHON if set, else the active virtualenv, else ./.venv, else python3.
PY="${PYTHON:-}"
[ -z "$PY" ] && [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
[ -z "$PY" ] && [ -x ".venv/bin/python" ] && PY=".venv/bin/python"
[ -z "$PY" ] && PY="python3"

echo "-> running the test suite (several minutes)"
out="$("$PY" -m pytest -q 2>&1 | tail -3)"
actual="$(printf '%s' "$out" | grep -oE '[0-9]+ passed' | head -1 | grep -oE '[0-9]+')"
xfail="$(printf '%s' "$out" | grep -oE '[0-9]+ xfailed' | head -1 | grep -oE '[0-9]+')"
xfail="${xfail:-0}"
if [ -z "${actual:-}" ]; then
  echo "x could not read the test count. Output:"; printf '%s\n' "$out"; exit 1
fi
if printf '%s' "$out" | grep -qE '[0-9]+ (failed|error)'; then
  echo "x the suite has failures:"; printf '%s\n' "$out"; fail=1
fi
echo "   actual: $actual passed, $xfail xfailed"

# Every "NNN passed" / "N xfailed" in a tracked Markdown file must equal the real count.
while IFS= read -r hit; do
  f="${hit%%:*}"; n="$(printf '%s' "$hit" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+')"
  if [ "$n" != "$actual" ]; then
    echo "x $f claims $n passed, the suite reports $actual"; fail=1
  fi
done < <(git grep -n -E '[0-9]+ passed' -- '*.md' 2>/dev/null)
while IFS= read -r hit; do
  f="${hit%%:*}"; n="$(printf '%s' "$hit" | grep -oE '[0-9]+ xfailed' | grep -oE '[0-9]+')"
  if [ "$n" != "$xfail" ]; then
    echo "x $f claims $n xfailed, the suite reports $xfail"; fail=1
  fi
done < <(git grep -n -E '[0-9]+ xfailed' -- '*.md' 2>/dev/null)

# Markdown links must stay inside the repository, or they 404 on the hosted copy.
if git grep -nE '\]\(\.\./\.\./' -- '*.md' >/dev/null 2>&1 \
   || grep -nE '\]\(\.\./' README.md >/dev/null 2>&1; then
  echo "x a Markdown link points outside the repository:"
  grep -nE '\]\(\.\./' README.md; git grep -nE '\]\(\.\./\.\./' -- '*.md'; fail=1
fi

echo "-> checking documented numbers against their sources"
"$PY" scripts/check_documented_numbers.py "$@" || fail=1

[ "$fail" = "0" ] && echo "ok: documented numbers and links are consistent" || echo "FAILED"
exit "$fail"
