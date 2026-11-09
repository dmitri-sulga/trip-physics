# Security policy

## What this repository is

A measurement harness. It reads a shadow log, scores accounts, and writes a
report. It has no network calls, no server, no persistence beyond the files it
writes into `out/`, and one runtime dependency (numpy). The Docker image is
built to run with `--network=none`.

That shape removes most of the attack surface a security reviewer usually
looks for. What remains is worth stating plainly rather than leaving implied.

## Reporting a vulnerability

Email **europeworktallinn@gmail.com** with `SECURITY` in the subject line.
Include the version or commit hash, what you ran, and what happened. A
proof-of-concept is welcome but not required.

You will get an acknowledgement within 3 working days and an assessment within 10. If the issue is confirmed, the fix and the disclosure timeline are agreed
with you before anything is published. If it is not confirmed, you get the
reasoning, not silence.

Nothing here is bound by a bounty programme, so there is no payout and no
requirement to withhold anything from your own publication after a fix ships.

## Scope

In scope:

- Code execution or file writes outside `out/` triggered by a crafted input log
- A parsing path that reads outside the input file it was given
- Dependency issues that reach the runtime path (numpy)
- Any way the harness makes a network call, which it must never do
- A way to make the report claim a number the run did not produce

Out of scope:

- Findings that require modifying the source and then running it
- Resource exhaustion from an input the operator chose to feed it: the demo
  loads a population into memory on purpose, and its size is a flag
- The synthetic generator producing unrealistic fraud. That is a modelling
  argument, not a vulnerability, and it belongs in an issue

## Handling of data

The harness never uploads anything. When it runs on real logs it runs inside
the operator's perimeter, and the guide for that mode (`BYOS`) is written so
that only aggregates leave the container. If you find a path where a record
value can reach the report verbatim, that is a finding in scope, because the
report is the artefact that leaves the machine.

When the harness is run on someone else's logs, the intended shape is that
they run it inside their own perimeter and keep everything except
`out/byos.json`, which holds counts, rates and thresholds — no account id and
no score.
