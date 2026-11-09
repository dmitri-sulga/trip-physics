# Changelog

Every entry below is a defect this harness had, found here rather than by a
reader, and what it did to the numbers before it was fixed. The list exists
because a measurement tool with no visible history of being wrong is asking to
be trusted on nothing, and because the alternative — a reviewer finding these
themselves in the git log — is strictly worse.

Dates are commit dates. Metric changes are stated in the units the report
prints.

## 2026-08-03 — the baseline could underspend the budget, and the uplift ate the difference

**Found by an adversarial pass over our own evaluation code.**

`U` compares recall between two scorers "at the same false-positive budget".
The budget names a maximum, not a spend. Two scorers of different granularity,
thresholded independently, do not spend the same number of false positives: a
coarse baseline with rounded scores has ties sitting exactly on its threshold,
those ties go unflagged, and it therefore buys fewer false positives than it was
allowed. Its recall is depressed by the shortfall, and every point of that
shortfall lands in `U` as if the pilot had earned it.

Size of the error on data built to trigger it (a baseline rounded to 0.5): the
coarse scorer spent **28.9 false positives per 10,000** out of a budget of 50,
the pilot spent **50.0**, and the median uplift read **+147.6% instead of
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

**Two further holes in the fix itself**, found by a second adversarial pass and
closed the same day:

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

## 2026-07-26 — a feature was over-weighted, and the anchor moved

`style_bimodality` separates at roughly AUC 0.54. It carried weight 2.0. Cut to
0.3 after recalibration against a size-aware null. Pilot PR-AUC moved
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
