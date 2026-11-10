# Bring your own scores

You already have risk scores. This is how to measure them with the trip-physics
harness without integrating anything, without exporting trip data, and without
any of this code touching your database.

Ten minutes, one command, two CSVs you can produce with one query each.

---

## Install

```bash
git clone https://github.com/dmitri-sulga/trip-physics.git
cd trip-physics
pip install -e .
```

Python 3.11 or newer; numpy is the only dependency pulled in. After this,
`python -m tripphysics.byos --help` prints the flags used below. Prefer a
virtualenv (`python3 -m venv .venv && source .venv/bin/activate`) if the
machine's Python is managed.

---

## What to export

**One file, `scores.csv`:**

```
account_id,score,label
a1f39c,0.812431,1
b7d02e,0.104882,0
c93ba1,0.556190,0
```

- `account_id` — any stable identifier. It never leaves your machine and never
  appears in the output file, so an internal surrogate key is fine.
- `score` — your current risk score for that account. Any scale: calibrated
  probability, logit, rule count. Higher must mean more suspicious.
- `label` — optional. `1` if the account was confirmed shared in the window,
  `0` if it was reviewed and cleared. Accepts `1/0`, `true/false`, `yes/no`.

Comma-separated, UTF-8, `.` as the decimal point. A BOM is fine. Semicolons and
decimal commas (`0,73`) are rejected with a message naming the offending line
or file — a European Excel export writing `0,73` would otherwise truncate
every score to `0` and score your evaluation against zeros.

Labels can live in a separate `labels.csv` (`account_id,label`) instead, passed
with `--labels`. Do not supply both — two sources of ground truth that disagree
would silently mislabel the run, so it is refused.

**Which accounts to include.** All scored accounts in the window, not only the
confirmed cases. The false-positive budget is defined per 10,000 _honest_
accounts, so a file containing only positives cannot set a threshold at all.

**Which labels count as `0`.** Reviewed and cleared, not merely unreported. An
account nobody looked at is unknown, not honest; leave it out. Scored accounts
without a label are excluded automatically, with a count printed.

---

## The command

```bash
python -m tripphysics.byos --scores scores.csv --out out/byos
```

With labels in a separate file:

```bash
python -m tripphysics.byos --scores scores.csv --labels labels.csv --out out/byos
```

To get U, add your current production scores as a second file:

```bash
python -m tripphysics.byos \
  --scores candidate_scores.csv \
  --baseline-scores production_scores.csv \
  --out out/byos
```

Writes `out/byos.json` and `out/byos.md`. Exit code 0 on success, 2 on any
input problem.

---

## What each number means

**PR-AUC** — area under the precision-recall curve. Ranks the whole score
column against outcomes, independent of any threshold. Useful for comparing two
scorers; not an operating point.

**precision@100 / @500** — if analysts review the 100 or 500 highest-scored
accounts, what fraction are actually shared. This is the number that predicts
queue quality.

**Budget / threshold / flagged / recall** — the budget is how many honest
accounts you accept flagging per 10,000 honest accounts (50/10k = 0.5%). For
each budget the harness picks the tightest threshold staying within it, then
reports how many accounts clear it and what fraction of confirmed-shared
accounts that catches. The sweep across {15, 30, 50, 100, 200} exists so the
operating point is visible rather than chosen after seeing which one flatters
the result.

**R0 / R1 / U** — with two score columns: R0 is recall using your current
stack, R1 is recall using your current stack _stacked_ with the candidate
(`max` of the two in rank space, so the candidate can add alerts but never bury
one of yours), and U = (R1 − R0) / R0 is the relative uplift at unchanged
budget. U can be negative: the candidate's false alarms spend budget too.

**U 95% CI** — a percentile bootstrap over accounts, stratified to keep the
observed positive count, re-deriving the threshold on every resample so
threshold-choice variance lands inside the interval. If the lower bound sits
near zero, the point estimate is mostly sampling noise.

**Held-out FP check** — the threshold is normally fitted on the same honest
accounts it is then measured on, which makes the realized false-positive rate
match the budget by construction. That is a tautology. This check fits on half
the honest population and measures on the other half; the gap is how much
budget the operating point actually spends on accounts it was not calibrated
to.

**Promised vs realized budget, with an interval** — a single-scorer run with
`--budget` prints the head number of the whole exercise: the budget you
promised next to the budget the held-out half actually spent, as
`promised 50/10k -> realized Y/10k [95% CI low, high]`. The interval is a
percentile bootstrap over accounts, each half resampled separately with the
threshold re-fitted every time, so an optimistic split cannot hide in it. The
same block lands in the written `.md` and `.json` report.

---

## What the harness cannot tell you without labels

Run it without a label column and you get the score distribution, tie counts,
the threshold at each budget, and how many accounts each budget flags. That is
all, and the run says so on the first screen.

Not computable without outcomes: **recall, PR-AUC, precision@k, and U.** Every
one of them is a statement about whether the accounts you flag are actually
fraudulent. No property of the score distribution answers that. The harness
prints no placeholder in their place.

**U specifically needs two scorers.** U is an uplift of one scorer over
another. With a single score column there is no baseline recall to divide by,
so U is undefined — not zero, not "no uplift". A single-scorer run refuses to
print it and names the file that would make it computable. This matters because
a fabricated zero is the one number that would survive into a debrief unchecked.

---

## Two inputs that are refused

Both parse cleanly and would otherwise return an arithmetically correct zero:

- **Every score identical.** A constant column cannot rank anything, so every
  threshold flags nobody and every metric returns 0.0 with no error. It looks
  like "the candidate added nothing"; it means the export is broken — a
  collapsed model, a failed join, or a default value written to every row.
- **Every label honest.** With no confirmed positives, recall is a fraction of
  an empty set and R0, R1 and U are all 0.0. Nothing was measured; the zero is
  the absence of positives.

Both stop the run with exit code 2 before any metric is printed. A crash gets
fixed in ten minutes; a plausible zero gets quoted in a debrief.

Also refused, each with the file, the line and the fix: empty files, duplicate
`account_id`, NaN or infinite scores, decimal commas, ragged rows, unparseable
or blank label values, and label files whose account ids do not match the score
file (reported with the overlap counts and the first few missing ids). No
traceback is shown — a stack trace describes the harness's call graph, which is
not the thing that is broken.

If the population is too small for a budget to buy even one alert — 15/10k
needs 667 honest accounts — that budget row is marked "budget buys 0 honest
alerts here", so the threshold printed next to it cannot be read as an
operating point. Below 150 confirmed positives it prints the count and tells
you to read the output as a plumbing check.

---

## No network access

The code makes zero network calls. There is no telemetry, no license check, no
callback. numpy and the Python standard library are the only imports.

You do not have to take that on trust:

```bash
docker build -t trip-physics .
docker run --rm --network=none -v "$PWD:/data" trip-physics \
  python -m tripphysics.byos --scores /data/scores.csv --out /data/out/byos
```

`--network=none` removes the container's network stack entirely. If the harness
tried to reach anything, it would fail rather than succeed quietly — which is
the point: the claim is enforced by the runtime rather than promised in a
README.

The same property is pinned without Docker, if you would rather check it in
seconds than build an image: run the harness with Python's socket constructors
replaced by a function that raises. It completes normally and exits 0, because
no code path in it opens a socket.

The output file, `out/byos.json`, holds counts, rates and thresholds only. No
account id and no score appears in it. That is the artifact that can be
shared when the underlying data cannot move.

---

Dmitri Sulga · Tallinn · <europeworktallinn@gmail.com>
