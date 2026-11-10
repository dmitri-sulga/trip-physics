# Changelog

Every entry below is a defect this harness had, found by testing it rather
than by a reader, and what it did to the numbers before it was fixed. A
measurement tool with no visible history of being wrong asks to be trusted on
nothing.

This repository starts from a snapshot published in November 2026, so its git
log does not contain these fixes. Dates are those of the fix in the private
working repository, to the day where the day is known and to the month where
it is not. Figures describe the code before the fix; where a test pins the
case, the entry names it. Metric changes are stated in the units the report
prints.

The tests label six defects:

| label    | what                                                              | status | test                                                                  |
| -------- | ----------------------------------------------------------------- | ------ | --------------------------------------------------------------------- |
| DEFECT-1 | `pr_auc` with every score tied                                    | open   | `tests/test_metric_edge_cases.py` (strict xfail)                      |
| DEFECT-2 | `precision_at_k` at k = 0 or below                                | fixed  | `tests/test_metric_edge_cases.py`                                     |
| DEFECT-3 | one non-finite coordinate erased the in-trip teleport signal      | fixed  | `tests/test_feature_geometry.py`                                      |
| DEFECT-4 | one NaN coordinate erased the between-trip handover signal        | fixed  | `tests/test_pipeline_invariants.py`                                   |
| DEFECT-5 | the style-bimodality null was a flat constant                     | fixed  | `tests/test_feature_calibration.py`                                   |
| DEFECT-6 | `style_bimodality` over-weighted                                   | fixed  | `tests/test_feature_calibration.py`, `tests/test_golden_metrics.py`   |

## Open — DEFECT-1: `pr_auc` over-credits a mixed top group

The curve is anchored at (recall 0, precision 1) and integrated by trapezoid.
When the highest-scoring group of tied scores contains honest accounts, the
first segment credits precision no threshold achieves; with every score tied
the result is (1 + base rate) / 2 (0.55 at a 0.10 base rate) instead of the
base rate. The strict xfail in `tests/test_metric_edge_cases.py` asserts the
correct value, so the suite reports it as the one expected failure. Not
patched for the constant case alone, because the proper fix changes every
value the function returns. It does not affect the committed report, whose
top groups are pure; the bring-your-own-scores command refuses a constant
score column.

## 2026-11-10 — published claims that the repository could not back

Found by reviewers reading the published snapshot. All fixed in this
repository's own git log.

- **CI was red from the first push.** `pip-audit --strict` fails on the
  package itself, which is installed from the checkout and is not on PyPI, so
  the static scan after it never ran. The audit now skips the editable
  package; no vulnerability had been reported.
- **Three rows of the sample-size table did not reproduce.** The 2,000-,
  5,000- and 8,000-account rows were stale; the 2,000-account range read
  +2.5% … +41.7% where the shipped code gives +2.5% … +37.5%. The table now
  comes from `examples/sample_size_table.py`. The 20,000-account row and the
  +15.5% headline were unaffected.
- **A paragraph about the held-out false-positive check was wrong.** It said
  the split-to-split spread does not shrink with population size. It does, as
  1/√N: 52.8% at 2,425 honest accounts down to 18.1% at 19,400
  (`examples/holdout_spread_vs_n.py`). The held-out figures it summarized had
  no command behind them; `examples/holdout_split_sweep.py` and
  `examples/holdout_population_sweep.py` now produce them, unchanged.
- **The attack registry read as runnable attacks.** No code in the repository
  applied any of them. `tripphysics.attack_measure` now re-runs the 11 entries
  whose manipulation can be written unambiguously, each reproducing its
  recorded count, and the registry says the rest are a record.

## 2026-08-03 — the baseline could underspend the budget, and the uplift ate the difference

**Found by deliberately trying to break the evaluation code.**

`U` compares recall between two scorers "at the same false-positive budget".
The budget names a maximum, not a spend. Two scorers of different granularity,
thresholded independently, do not spend the same number of false positives: a
coarse baseline with rounded scores has ties sitting exactly on its threshold,
those ties go unflagged, and it therefore buys fewer false positives than it was
allowed. Its recall is depressed by the shortfall, and every point of that
shortfall lands in `U` as if the stacked scorer had earned it.

Size of the error on data built to trigger it (a baseline rounded to 0.5): the
coarse scorer spent **28.9 false positives per 10,000** out of a budget of 50,
the stacked scorer spent **50.0**, and the median uplift read **+147.6% instead of
+68.5%**. The phrase "at equal FP spend" in the README and in the report was
false on any real stack with rounded probabilities or rule-counter scores —
which is what an eight-year-old production baseline usually is.

Fixed by holding both scorers to the largest spend a strict threshold reaches
on both ladders, so equal spend is a printed integer identity rather than a
claim. The report now prints the realized spend and the budget utilization next
to `U`, refuses the headline when a coarse baseline cannot reach the floor, and
refuses the bootstrap CI when too many resamples go degenerate.

The default synthetic demo did not move: at full population the honest ladder
has no boundary ties, utilization is 100%, and the 20-seed median stays +15.5%.
That is the uncomfortable part worth stating — the demo looked fine throughout,
and only data resembling a real baseline exposed it.

**Two further holes in the fix itself**, found by a second attempt to break it
and closed the same day:

- A `-inf` honest score with a budget permitting everyone to be flagged made the
  two scorers spend different amounts again. The threshold is `-inf`, nothing is
  strictly above it, but the spend shortcut assumed the whole honest population
  had been spent. That produced an unequal realized spend and a fabricated
  infinite uplift, with utilization at 1.0 so the granularity gate stayed quiet.
  Now the code counts what the threshold actually flags.
- A negative budget stopped being rejected. The old guard lived in
  `threshold_at_fp_budget`, which the new realized-spend path never calls, so
  `evaluate` and the bootstrap returned a negative allowance and a realized
  spend larger than it. Both refuse it again.

**Five ways the BYOS report could still mislead**, closed in the same series:
the constant-column guard is re-checked after narrowing to the labelled subset
(a file that varies overall can be flat within the labelled accounts, which are
the ones scored); a baseline file whose own label column contradicts the labels
in use is refused instead of dropped silently; a NaN or infinite budget is
rejected rather than misdiagnosed later; `precision@k` is labelled with the `k`
actually used, since the metric truncates to the population and a smaller export
was being reported under a name that overstated the queue; and a sweep now
reports how many seeds it dropped for an infinite uplift, so `n_seeds` describes
what was summarized rather than what was run.

## July 2026 — one NaN GPS point could lower a teleporting account's score

Two defects with one cause (DEFECT-3 and DEFECT-4). `np.hypot` returned NaN for
any step that touched a non-finite coordinate, the running maximum and the
mean over gaps became NaN, and a final "finite or 0.0" guard rewrote the
poisoned aggregate to 0.0 — the least suspicious value there is. An account
could lower its own score by emitting garbage. Measured at the time, in the
test comments: a 500 km in-trip jump scored 0.69 and 0.27 with one NaN fix in
the same trip; a 40 km jump between trips moved `handover_gap_score` from 1.0
to 0.0 and the score from 0.2935 to 0.1215.

Fixed by dropping non-finite fixes before any differencing and reporting the
share dropped as its own feature, `nonfinite_rate`, so garbage raises the
score rather than lowering it. The final guard now raises instead of
substituting 0.0. Regression tests:
`tests/test_feature_geometry.py` (`test_nan_injection_does_not_lower_suspicion`,
`test_no_non_finite_flavour_lowers_suspicion` for NaN, +inf and -inf,
`test_the_surviving_fixes_still_show_the_in_trip_teleport`) and
`tests/test_pipeline_invariants.py`
(`test_injecting_a_nan_fix_never_lowers_the_handover_signal`,
`test_the_teleport_survives_nan_injection_intact`,
`test_emitting_garbage_raises_suspicion_rather_than_lowering_it`). Three
registry entries (`nonfinite-flood`, `nonfinite-trickle`,
`nonfinite-targeted-endpoints`) record the attack against the fixed code.

## July 2026 — the headline was one lucky seed on a population too small to measure

The demo used to report U from a single seed: seed 42 at 2,000 accounts, U =
+25.8%, which is 8 more caught accounts out of 60 shared ones (R0 31/60, R1
39/60). Swept across seeds, the same configuration ranges from +2.5% to
+37.5% (`python examples/sample_size_table.py`, 25 seeds), so the figure
reported mostly luck, and 60 positives sat below the harness's own stated
minimum.

Replaced by a 20-seed sweep at 20,000 accounts (600 positives) whose median is
the headline: +15.5%, range +12.9% … +19.9% (`python examples/run_demo.py
--full`). The bootstrap CI and the budget curve stayed; the detailed tables
now come from the first seed, labelled as a worked example. Seed 42 at 2,000
accounts is still the golden regression configuration in
`tests/test_golden_metrics.py`, which pins U = 0.2580645 as a value, not as a
result.

## 2026-07-27 — the runtime table was off by four times

The default run was advertised at about 4 minutes. Measured with
`/usr/bin/time -p` from a clean `git archive` checkout on an idle 14-core
laptop, it took **989 s**. `--quick` was advertised at 30 s and took 48 s. The
test suite was not listed at all.

Anyone handed this repository checks timings by running them, not by reading
them, so a figure off by four times reads as a number nobody measured. Times
now carry the conditions they were taken under, and `--full` is marked as
extrapolated rather than timed.

## 2026-07-27 — a single-scorer run would have printed a zero uplift

`U = (R1 − R0) / R0` compares two scorers. Given one score column there is no
`R0` to divide by, so `U` is undefined — not zero, and not "no uplift". A zero
there is the one number a reader carries into a debrief. A single-scorer run now
says `U` is undefined and names the file that would make it computable.

Two inputs that parse cleanly are rejected before any metric prints, because
both otherwise return an arithmetically correct zero: every score identical (a
collapsed model is indistinguishable from a detector that added nothing), and
every label honest (recall is a fraction of a shared population that does not
exist).

## 2026-07-27 — the detector was grading its own homework

Until this point the harness reported only the score it achieved on its own
corpus. The adversarial half was added: named attacks, each with the measured
damage it does to the detector's own numbers, so the failure modes ship with the
tool instead of being discovered by whoever runs it against a real adversary.
The scripts that took those measurements were not kept with the registry; see
the 2026-11-10 entry.

## 2026-07-26 — a feature was over-weighted, and the anchor moved

`style_bimodality` separates at roughly AUC 0.54. It carried weight 2.0. Cut to
0.3 after recalibration against a size-aware null. Stacked PR-AUC moved
0.7809983 → 0.7849702. Verified before re-anchoring the golden test: `R0`, `R1`
and `U` unchanged to the last bit, the baseline arm untouched, inverting every
label leaves the detector scores bit-identical, and the non-finite rate is
identically zero on this corpus — so the gain is a ranking improvement in the
tail rather than leakage.

## 2026-07-26 — a CSV export could be scored against zeros

Exports that would have been silently scored against missing values are
rejected instead. A pass-rate count that was disclaimed in prose is asserted
absent in the artifact, because a caveat under a number does not travel with
the number.
