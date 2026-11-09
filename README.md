# trip-physics — shadow-evaluation harness for profile-sharing detection

> **All data in this repository is synthetic.** It is produced by the generator
> in `src/tripphysics/synth.py` from fixed seeds; no real trip, driver or
> company data was used anywhere. Every number below describes that simulator,
> not anyone's production traffic, and the generator and the detector were
> written by the same person — on this data the detector is right by
> construction. The harness exists so the same measurement can be re-run on
> real shadow logs, where the result would mean something.

> Developed July–November 2026 and published here in November 2026 as a single
> snapshot commit.

One driver account, several people driving on it. This repo is a harness for
measuring whether a physics-first detector adds recall on top of an existing
fraud stack — at a fixed false-positive budget, on your own shadow logs, inside
your own perimeter. It ships with a deterministic synthetic-data mode so the
whole pipeline can be verified end-to-end before any real data is involved.

Python 3.11+, numpy as the only runtime dependency. PR-AUC, thresholding and
the headline metric are implemented by hand in `src/tripphysics/eval.py` — a
few dozen lines you can audit, not a library call you have to trust.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python examples/run_demo.py
```

Writes `./out/report.json` and `./out/report.md` and prints the metric table.
Everything is seeded; two runs produce byte-identical reports.

The demo sweeps several independent seeds and reports the **median** U, plus
spread and a bootstrap CI. That costs real time — feature extraction over
20,000 accounts per seed dominates:

| command                               | population | seeds | runtime\* | peak RSS\* |
| ------------------------------------- | ---------- | ----- | --------- | ---------- |
| `python examples/run_demo.py`         | 20,000     | 3     | 123 s     | 5.0 GiB    |
| `python examples/run_demo.py --full`  | 20,000     | 20    | 542 s     | not taken  |
| `python examples/run_demo.py --quick` | 2,000      | 3     | 17 s      | 0.55 GiB   |
| `pytest`                              | —          | —     | 132 s     | not taken  |

\* wall-clock from `/usr/bin/time`, measured **2026-08-04** on an idle 14-core
Apple-silicon laptop with 36 GiB of RAM, Python 3.14.5 and numpy 2.5.1, from a
clean `git archive HEAD` checkout into a fresh virtualenv. Every figure in the
table was timed; none is extrapolated. On a busy machine expect roughly double:
the seed sweep runs across processes, so it competes for cores with whatever
else you have open.

**Memory is the constraint, not time.** The sweep holds one 20,000-account
population per worker process, and peak resident memory for the default run
measured 5.0 GiB. On a machine with 8 GiB you will swap; use `--quick` or lower
`--accounts` there. `--quick` peaks at 0.55 GiB and runs anywhere.

**A note on these numbers, because they moved.** The same table, measured the
same way on the same laptop in July, read 989 s for the default run, 48 s for
`--quick`, and ~11 min for the suite — four to eight times higher than what it
measures today. The interpreter is not the explanation: Python 3.12 with the
same numpy runs `--quick` in 11.9 s. The most likely cause is that the July
measurement was taken while the machine was doing other work, which is exactly
the caveat the footnote warned about and exactly the kind of thing that is
invisible in a single reading. Timings are re-measured every release rather
than inherited, and both readings are kept here rather than the newer one
quietly replacing the older. Measure it in your own environment before planning
around either number.

`--full` is the quotable run. `--quick` is a smoke test only: at 2,000
accounts U is genuinely noisy (see [Sample size](#sample-size-why-20000)), so
do not read a result off it. `--accounts N` and `--seeds K` override the
population and sweep size for a custom run.

**If your number does not match mine, that is expected.** The figure quoted
elsewhere, +15.5%, is the median of a 20-seed sweep at 20,000 accounts, and it
comes from `--full`. The default command runs 3 seeds, so its median will land
somewhere else in the +12.9% … +19.9% range. Any single seed moves more than
that again: the worked example in the shipped report is seed 1 at +14.3%. Three
numbers, all correct, measuring different things. Compare like for like before
concluding anything, and only compare `--full` medians against the +15.5%.

Every run overwrites `out/report.json` and `out/report.md`, which are
committed here holding the quotable 20-seed sweep. After a `--quick` or custom
run, `git checkout -- out/` puts the shipped artifacts back.

The virtualenv is not ceremony: on macOS with Homebrew Python and on current
Debian/Ubuntu, a plain `pip install` into the system interpreter is refused
outright (PEP 668, `externally-managed-environment`). Create the venv and the
whole thing takes one command each.

Requires Python 3.11+; the demo and test suite above were last run end-to-end
on 3.14 (numpy resolves to 2.5 there, 1.24+ on 3.11) and the container build
pins 3.11. If `python3` on your machine is older than 3.11, point the first
command at a newer interpreter, e.g. `python3.11 -m venv .venv`.

Docker alternative — no Python setup on your side at all:

```bash
docker build -t trip-physics .
mkdir -p out
docker run --rm --network=none -v "$PWD/out:/app/out" trip-physics
```

`--network=none` is not decoration: the harness makes zero network calls at
runtime, and running it with egress blocked enforces that instead of promising
it.

## The U metric

Recall comparisons are meaningless without a fixed alert budget, so the budget
comes first:

- **Budget** — B flagged honest accounts per 10,000 honest accounts
  (default B = 50, i.e. 0.5% — a realistic driver-action review-queue
  tolerance; a weekly-active-driver analog). For each scorer, pick the
  tightest threshold whose expected honest false positives stay within B. The
  demo reports U across B ∈ {15, 30, 50, 100, 200} so the operating point is
  chosen openly, never cherry-picked.
- **R0** — recall on confirmed-shared accounts at that threshold, using the
  baseline (your current stack's scores).
- **R1** — same, using the pilot scorer: the baseline **stacked** with this
  detector.
- **U = (R1 − R0) / R0** — relative recall uplift at the same _realized_ FP
  spend. The budget names a maximum; both scorers are measured at the largest
  spend a strict threshold reaches on both of them, so a coarse baseline cannot
  underspend the budget and inflate the uplift. The report prints that realized
  spend and the budget utilization next to U.

### The pass bar, and what it does not apply to

**Pass bar for the pilot: U ≥ +20% at the same realized false-positive spend,
measured on your shadow log**, reported with a bootstrap 95% CI and a
pre-registered minimum of ≥150 confirmed positives in the window. Below that
bar, the detector is not worth the integration cost and the report says so.

That bar applies to **one** measurement: real traffic, real confirmed-shared
labels. It explicitly does **not** apply to the synthetic demo in this repo,
and the demo never renders a pass/fail verdict against it. Two different
numbers, which must not be compared:

|              | synthetic demo                                        | pilot                                            |
| ------------ | ----------------------------------------------------- | ------------------------------------------------ |
| data         | generated by `synth.py`                               | your shadow log                                  |
| labels       | true by construction                                  | confirmed cases, incomplete                      |
| what U means | the pipeline computes the metric correctly end-to-end | whether the detector adds recall on your traffic |
| verdict      | none — none is possible                               | pass/fail against U ≥ +20%                       |

I tuned the generator until the demo showed a plausible-but-imperfect result,
because a demo showing a perfect number would only tell you that I overfit my
own simulator. The generator and the detector were written by the same person;
on that data the detector is right by construction. The only number that means
anything is the one your shadow log produces.

Stacking rule (applied inside `evaluate()`):
`pilot = max(rank(baseline), rank(detector))` — each score is rank-normalized
within its own population (ascending rank / n, ties averaged), then the
per-account max is taken. Rank-normalizing makes the two scales comparable;
`max` means the detector can only add candidates to the queue, never bury a
baseline alert. The honest flip side: detector false positives spend budget
too, so stacking _can_ push R1 below R0. If it does, that is a real result,
not a harness artifact.

Also reported: hand-implemented PR-AUC (trapezoid over the precision-recall
curve) and precision@500 for both scorers, plus the chosen thresholds.

## How the demo reports U honestly

Three rules, each of which exists because breaking it is how offline
evaluations flatter themselves.

The shipped run (`--full`, committed as [`out/report.md`](out/report.md)):

```
median U = +15.5% across 20 seeds   (mean +16.0%, sd 2.0pp, range +12.9% … +19.9%)
worked example, seed 1:  U = +14.3%, 95% CI [+9.6%, +18.5%], 1000 resamples
PR-AUC 0.61 → 0.73      held-out FP: budget 50/10k → baseline 46.4/10k, pilot 70.1/10k
synthetic spread +12.9% … +19.9%  (the pilot bar sits above this range on purpose)
```

That last line is the point: on synthetic data the honest median does **not**
clear the pilot's +20% bar, and the demo says so rather than hunting for a
configuration that does. See [the pass bar](#the-pass-bar-and-what-it-does-not-apply-to)
for why those two numbers are not comparable in the first place.

**1. The headline is a median across seeds, never a single seed.** One seed is
one draw from a distribution of synthetic worlds. Report a single seed and you
are reporting luck alongside signal — and the reader cannot tell how much of
each. The demo sweeps N seeds, prints median / mean / sd / min / max, and
leads with the median. The seed shown in the detailed tables is simply the
first one, labelled as a worked example rather than a result.

**2. U carries a bootstrap 95% CI.** A point estimate says nothing about its
own precision. `eval.bootstrap_uplift_ci()` resamples accounts with
replacement, stratified so each replicate keeps the observed shared/honest
counts, and re-derives the threshold on every replicate — so threshold-choice
variance lands inside the interval instead of hiding behind it.

<a id="sample-size-why-20000"></a>

**3. The population is large enough for U to mean anything.** This section
used to describe a 2,000-account demo, which was below this repo's own stated
noise floor ("U becomes noisy below roughly 10k scored accounts") and below
the pilot's pre-registered 150-positive minimum — 2,000 accounts at a 3% base
rate is 60 positives, and the entire headline rested on ~8 accounts changing
hands. Measured across seeds, that is exactly as unstable as it sounds:

| accounts | positives | seeds | median U | sd of U   | range           |
| -------- | --------- | ----- | -------- | --------- | --------------- |
| 2,000    | 60        | 25    | +17.1%   | 8.2pp     | +2.5% … +41.7%  |
| 5,000    | 150       | 10    | +17.2%   | 6.5pp     | +8.3% … +29.4%  |
| 8,000    | 240       | 10    | +16.1%   | 3.9pp     | +6.1% … +20.9%  |
| 20,000   | 600       | 20    | +15.5%   | **2.0pp** | +12.9% … +19.9% |

The median barely moves across all four rows — it stays between +15.5% and +17.2%.
Only the spread collapses. That is the whole finding: **the sample size was
the story, not the detector.** At 2,000 accounts a single seed could return
anything from +2.5% to +41.7%, so any single-seed headline from that
configuration reported mostly luck. The demo therefore defaults to 20,000
accounts and leads with a median.

**On the FP budget.** The threshold is calibrated on the same honest
population it is measured on, so in-sample the realized FP rate matches the
budget by construction — that is a tautology, not evidence. Since the budget
is a _promise_ about how many honest drivers get flagged, the demo also runs
`eval.holdout_fp_check()`: threshold fitted on half the honest accounts, FP
rate measured on the other half.

The shipped 20,000-account run reports a single split (`seed=0`): 46.4/10k for
the baseline and 70.1/10k for the pilot, against a 50/10k budget. **That pair
is not a property of the scorers.** Re-running the same check across 20 split
seeds on the same population:

| scorer   | budget   | median held-out rate | range over 20 splits |
| -------- | -------- | -------------------- | -------------------- |
| baseline | 50 / 10k | 44.3 / 10k           | 30.9 … 69.1 / 10k    |
| pilot    | 50 / 10k | **49.5 / 10k**       | 24.7 … 70.1 / 10k    |

The pilot's median lands on the budget, not 40% over it, and the shipped
`seed=0` value of 70.1 is the **highest of the 20 splits** — the next one down
is 58.8. Re-drawing the population instead of the split tells the same story
over 8 population seeds: the shipped `seed=1` is again the maximum, the median
is 47.4/10k, and on five of the eight the pilot spends _no more_ than the
baseline — the sign of the effect flips. Nor is the gap separable:
`eval.holdout_fp_budget_ci()` puts a stratified bootstrap interval around each
rate, and they overlap on 53.6 … 74.2 — baseline [30.9, 74.2], pilot
[53.6, 93.8]. A claim that the pilot systematically overspends does not follow
from one split.

The noise is countable, not mysterious. The split leaves 9,700 honest accounts
in the measurement half, so a 50/10k budget expects 9,700 × 50/10,000 = 48.5
flags. Poisson counting noise alone is √48.5 / 48.5 = **±14%**, and on top of
that the threshold itself is fitted on the other 9,700 accounts, so
threshold-selection variance adds to it. Spread of this width is what the
measurement is made of. Unlike the uplift table above, this spread must not be
read as converging: the flag count stays small by design — a tight FP budget is
precisely a promise to produce few events — so the held-out rate remains
sensitive to the split at every population size tested here, and does not
collapse to a single quotable point the way the median U does.

This is the instrument working, not a defect in it: a budget promise read off
one split is exactly the kind of number this harness exists to refuse. Read
the held-out FP rate through `eval.holdout_fp_budget_ci()`, which returns the
interval, rather than through a point estimate. On real data, calibrate on a
window that _precedes_ the measurement window.

## Architecture: code goes to you, data never leaves

```
              ┌──────────────────────────────┐
              │    this repository (code)    │
              └──────────────┬───────────────┘
                             │  code ships to you
                             ▼
┌───────────────────────────────────────────────────────────┐
│ YOUR PERIMETER — data never leaves it                     │
│                                                           │
│  shadow log (JSONL)           production scores (CSV)     │
│        │                             │                    │
│        ▼                             │                    │
│  ┌───────────┐    ┌───────────┐      ▼                    │
│  │ features  │───▶│ detector  │───▶ evaluate ──▶ report   │
│  └───────────┘    └───────────┘   (budgeted     .json/.md │
│                                    recall, U)             │
│                                                           │
│  egress-blocked container (docker run --network=none)     │
└───────────────────────────────────────────────────────────┘
```

Nothing in the report is row-level: aggregate metrics only.

## Plugging in your shadow log

1. Export driver-side trip fixes as JSONL — ten fields, defined precisely in
   [`schema/input_schema.md`](schema/input_schema.md). No PII, no rider data,
   no payment data.
2. Export your current stack's per-account scores as `account_id,score` CSV.
   Comma-separated, UTF-8, `.` as the decimal point. A BOM is tolerated;
   semicolon/tab delimiters and decimal commas (`0,73`) are rejected with a
   message naming the line — a European Excel export that writes `0,73` would
   otherwise truncate every score to `0`, so the loader refuses it rather than
   scoring your pilot against zeros.
3. Collect window labels: which accounts were confirmed shared (manual review
   or case outcomes).
4. Run:

```python
from tripphysics.baseline import load_scores_csv
from tripphysics.detector import score_account
from tripphysics.eval import evaluate
from tripphysics.features import extract_features
from tripphysics.report import render_report

accounts = ...  # your JSONL, grouped per schema/input_schema.md
labels = ...    # {account_id: confirmed_shared_bool}

base = load_scores_csv("production_scores.csv")
det = {a["account_id"]: score_account(extract_features(a)) for a in accounts}

# evaluate() stacks base and det into the pilot scorer internally
result = evaluate(labels, base, det, budget_fp_per_10k=50.0)  # 50/10k = the default budget used everywhere in this repo
render_report(result, {"window": "W1"}, "out/report.json", "out/report.md")
```

`NaiveBaseline` in `src/tripphysics/baseline.py` is a deliberately crude
stand-in used only by the synthetic demo. Replace it with your real scores via
the CSV adapter; do not benchmark against it and call the result a lift.

## Bring your own scores

The section above asks you to integrate a detector before you learn anything.
This one does not. If you already have risk scores — any scorer, ours or not —
you can put them through the same budgeted-recall arithmetic in about ten
minutes, on your machine, with no trip data and no code of ours near your
database.

Export one CSV of `account_id,score` and, if you have outcomes, a `label`
column (`1`/`0`, `true`/`false`, `yes`/`no`) alongside it. Then:

```bash
python -m tripphysics.byos --scores examples/byos_demo_scores.csv --out out/byos
```

Labels can live in a second file instead — `--labels labels.csv` with columns
`account_id,label` — and are omitted entirely if you have none.

**U needs two scorers.** U = (R1 − R0) / R0 compares a candidate against a
baseline. With one score column there is no R0 to divide by, so the run says
U is undefined and names the flag that would make it computable, rather than
printing a zero you might quote. Add your current production scores as a
second file to get it:

```bash
python -m tripphysics.byos \
  --scores examples/byos_demo_scores.csv \
  --baseline-scores examples/byos_demo_baseline.csv \
  --out out/byos
```

which prints:

```
========================================================================
  trip-physics — measuring your scores, on your machine
========================================================================

Baseline (current stack): examples/byos_demo_baseline.csv
Candidate (your new scores): examples/byos_demo_scores.csv
Population: 4000 accounts (160 confirmed shared, 3840 honest)

  At budget 50 false positives per 10,000 honest accounts:
    R0 (baseline recall):   0.2562
    R1 (stacked recall):    0.3563
    U (relative uplift):    +39.0%
    thresholds:             baseline 0.703405, stacked 0.98675
    realized FP spend:      19 of the 19 honest flags the budget allows (49.5/10k, utilization 100%)

    PR-AUC:        0.4895 -> 0.6150
    precision@500: 0.2440 -> 0.2580

    U 95% CI: [+1.5%, +74.2%] over 1000 resamples

    Held-out FP check (threshold fitted on half the honest accounts,
    measured on the other half): budget 50/10k -> baseline 68/10k, stacked 52/10k.
    The gap is how much of the uplift is bought with budget the
    in-sample threshold does not admit to spending.

  U across budgets (the operating point is not chosen after the fact):

    budget/10k        R0        R1         U
         15      0.1688    0.2562      +51.9%
         30      0.2000    0.3125      +56.2%
         50      0.2562    0.3563      +39.0%
        100      0.3500    0.4938      +41.1%
        200      0.4437    0.6250      +40.8%

Written: out/byos.json and out/byos.md (aggregate metrics only — no account id and no score is in either)
```

Those example files are synthetic and deliberately unflattering: the two
scorers are strongly correlated, the CI's lower bound barely clears zero
(+1.5% at 160 positives), and the point of shipping them is that the command
runs from a clean checkout — not that +39% means anything.

Without labels the run reports the score distribution, the threshold at each
budget and how many accounts each budget flags, then states plainly that
recall, PR-AUC and U are not computable. It never substitutes a placeholder
for them.

Two inputs are refused outright, before any metric is printed, because both
otherwise return an arithmetically correct zero that reads like a result:

- **a constant score column** — a collapsed model or a failed join. Every
  threshold flags nobody, so U comes out 0.0 with no error.
- **an all-honest label column** — no confirmed positives at all. Recall is a
  fraction of the shared population, so R0, R1 and U are all 0.0 and nothing
  has been measured.

Malformed files exit 2 with one sentence naming the file, the line and the
fix; no traceback reaches you. `out/byos.json` holds counts, rates and
thresholds only — no account id and no score — so it is the artifact you can
share when the data cannot leave.

A fraud engineer's walkthrough, including what to export and what each number
means, is in [`docs/byos-guide.md`](docs/byos-guide.md).

## Honest limits

- **Synthetic data proves the harness works, NOT that the detector works on
  your traffic — that is what the shadow window measures.** The synthetic U
  printed by the demo is a plumbing check, nothing more.
- The synthetic world and the detector were built by the same hands, so the
  detector's assumptions are true by construction there. Synthetic U is
  optimistic by design. Never quote it as expected production uplift.
- R0/R1/U are only as good as the labels. Sparse or noisy confirmed-shared
  labels bias both recalls; the report states label counts so this is visible.
- The detector is physics-first and interpretable, which also bounds it: a
  sharing pattern that is physically clean — same household, phone handed over
  at home, similar driving styles — will not trip it. It targets the
  physically implausible tail, not all sharing.
- With small honest populations the FP budget quantizes hard (15/10k of 5,000
  accounts is 7 alerts); U becomes noisy below roughly 10k scored accounts.
  This is measured, not asserted — see the sd-vs-sample-size table above — and
  it is why the demo defaults to 20,000 accounts and reports a median across
  seeds rather than a single run.
- The threshold is calibrated on the same honest population it is measured on,
  which makes the in-sample FP rate optimistic by construction. The demo sizes
  that gap with a held-out check instead of assuming it away.
- One detector, one abuse type. This is not a fraud platform; it is a
  measurement harness plus one focused signal.
- **Known evasion vectors, stated because you would find them anyway.** These
  are properties of the current scoring, not hypotheticals, and each is cheap
  for an attacker:
  - _Volume dilution._ `handover_gap_score` is the **mean** over consecutive
    inter-trip gaps, so appending geometrically clean filler trips lowers it
    without removing the original handover. Measured on 3,000 accounts: 33/50
    shared accounts caught above the honest p99 with no padding, 19/50 after
    25 filler trips, 17/50 after 100. `night_share`, `mock_rate` and
    `cell_mismatch_rate` are rates and dilute the same way. A max or a
    high quantile would resist this; the mean is what ships today.
  - _Client-reported fields._ `max_speed_kmh` and `accel_p99_ms2` come from
    the reported `speed_mps` field and are never cross-checked against
    position over time, so self-consistent falsified speeds defeat both.
    Likewise `is_mock` and `cell_id` are taken verbatim from the client.
    In a real deployment `cell_id` must come from an operator-side tower
    join; in this synthetic world it is a pure function of x/y, so it is
    trivially forgeable here.
  - _Bridging._ Handover excess is scored only between _consecutive_ trips,
    so inserting one plausible intermediate trip splits a large spatial jump
    into two explainable legs.
  - _Silence._ An account with no usable trips yields the all-zero vector and
    is mapped to a hard 0.0, the minimum score. That is deliberate ("absence
    of evidence is not evidence of innocence"), but it does mean that in pure
    ranking terms, submitting nothing sorts below every honest account.
    Empty accounts belong in their own queue, not at the bottom of this one.

## Repository layout

```
src/tripphysics/     synth, features, detector, baseline, eval, report
schema/              input_schema.md — the 10-field data contract
examples/run_demo.py end-to-end synthetic run
examples/byos_*.csv  synthetic score and label files for the BYOS command
attacks/             registry.json + generated ATTACKS.md — named evasions and
                     the measured damage each does, including the ones that win
docs/byos-guide.md   bring-your-own-scores walkthrough
out/                 the committed 20-seed report (report.md / report.json)
scripts/             check-claimed-numbers.sh — doc numbers vs a real test run
tests/               determinism, hand-computed eval math, schema round-trip
.github/workflows/   CI: suite on 3.11 and 3.14, determinism, dependency audit
CHANGELOG.md         defects this harness had, and what each did to the numbers
SECURITY.md          how to report a vulnerability, and what is in scope
NOTICE               licence boundary and data provenance
```

## What a reviewer can check without taking anything on trust

| Claim                             | How to check it yourself                                                |
| --------------------------------- | ----------------------------------------------------------------------- |
| The suite passes                  | `python -m pytest -q` — 555 passed, 1 xfailed                           |
| The pipeline is seeded end to end | run `--quick` twice, compare `shasum -a 256 out/report.*`               |
| The headline is reproducible      | `--full` prints median +15.5% across 20 seeds, matching `out/report.md` |
| No network access at runtime      | `docker run --rm --network=none …` completes normally                   |
| The metric is auditable           | `src/tripphysics/eval.py` — PR-AUC and thresholding written out by hand |
| Failure modes are disclosed       | `## Honest limits` above, and CHANGELOG.md for defects already found    |

CI runs the first two on every push: the suite on both the version the container
pins and the version the documented runs were taken on, and the `--quick`
determinism check. The 20-seed `--full` sweep is too slow for CI and is checked
by running it. The golden anchors assert to 1e-9, so that matrix doubles as the
portability check: if a numpy build moves an anchored metric, the job goes red
rather than the number moving quietly.

## Tests

With the venv from the quickstart active (`[dev]` already installs pytest):

```bash
python -m pytest -q
```

If you skipped the venv, use `.venv/bin/python -m pytest -q` — `python -m
pytest` is deliberate rather than a bare `pytest`, so the tests always run
against the interpreter the package was installed into.

`tests/test_eval_math.py` doubles as documentation: an 8-account fixture with
hand-computed R0/R1/U asserted exactly.

---

Dmitri Sulga · Tallinn · <europeworktallinn@gmail.com> · MIT
