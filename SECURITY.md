# Security notes

## What this repository is

A measurement harness. It reads trip logs or score files, scores accounts, and
writes a report. It makes no network calls, runs no server, persists nothing
beyond the files it writes into `out/`, and has one runtime dependency
(numpy). The Docker image is built to run with `--network=none`.

## Reporting a problem

Email **europeworktallinn@gmail.com** with `SECURITY` in the subject line.
Include the commit hash, what you ran, and what happened. This is a
single-author project with no support commitment, so there is no promised
response time; reports are read and, if confirmed, fixed in a public commit
that says what was wrong.

## What counts as a security problem here

- Code execution or file writes outside `out/` triggered by a crafted input
  file
- A parsing path that reads outside the input file it was given
- Dependency issues that reach the runtime path (numpy)
- Any way the harness makes a network call, which it must never do
- A way to make the report claim a number the run did not produce
- A path by which an account id or a raw score reaches `out/byos.json`, which
  is designed to hold aggregates only

Not in scope:

- Findings that require modifying the source and then running it
- Resource exhaustion from an input the user chose to feed it: the demo loads
  a population into memory on purpose, and its size is a flag
- The synthetic generator producing unrealistic fraud. That is a modelling
  argument, not a vulnerability
