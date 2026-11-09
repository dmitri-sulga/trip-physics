#!/usr/bin/env bash
# Consistency gate: numbers claimed in the documentation must match a real run.
#
# Why this exists: the README once claimed a test count that no longer matched
# what pytest actually reported, in the very table that tells a reader to check
# claims by running them. A rule nobody enforces drifts, so this enforces it.
#
# Usage:       bash scripts/check-claimed-numbers.sh
# Exit status: 0 -- the claims hold; 1 -- a mismatch was found.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
fail=0

PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"

echo "-> running the test suite (about 2 minutes)"
out="$("$PY" -m pytest -q 2>&1 | tail -3)"
actual="$(printf '%s' "$out" | grep -oE '[0-9]+ passed' | head -1 | grep -oE '[0-9]+')"
if [ -z "${actual:-}" ]; then
  echo "x could not read the test count. Output:"; printf '%s\n' "$out"; exit 1
fi
echo "   actual: $actual passed"

# Every "NNN passed" in a tracked Markdown file must equal the real count.
while IFS= read -r hit; do
  f="${hit%%:*}"; n="$(printf '%s' "$hit" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+')"
  if [ "$n" != "$actual" ]; then
    echo "x $f claims $n passed, the suite reports $actual"; fail=1
  fi
done < <(git grep -n -E '[0-9]+ passed' -- '*.md' 2>/dev/null)

# README links must stay inside the repository, or they 404 on the hosted copy.
if grep -nE '\]\(\.\./' README.md >/dev/null 2>&1; then
  echo "x README links outside the repository:"
  grep -nE '\]\(\.\./' README.md; fail=1
fi

[ "$fail" = "0" ] && echo "ok: documented numbers and links are consistent" || echo "FAILED"
exit "$fail"
