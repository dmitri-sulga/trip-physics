"""Uncertainty reporting: the bootstrap CI and the multi-seed sweep.

A point estimate of U says nothing about how much of it is sampling noise.
Two pieces of machinery answer that, and both are load-bearing for the
claims the harness makes in writing:

* :func:`bootstrap_uplift_ci` -- percentile bootstrap over accounts,
  stratified by label, recomputing the threshold on every resample. It
  answers "how wide is this number given the size of the confirmed-positive
  set?"
* :func:`summarize_across_seeds` -- the distribution of U over independent
  synthetic worlds. It answers "how much of this number is the seed?"

What is tested here is the *statistics*, not the detector: coverage of the
true value, determinism under a fixed seed, ordering and range of the
interval bounds, and behaviour on the degenerate inputs a sweep actually
hits (one seed, identical seeds, a scorer that catches nothing).

Deliberately absent: an anchor on the exact CI width for the demo run. The
width is a property of the corpus and of the detector's calibration, both of
which move for legitimate reasons; pinning it here would turn every honest
recalibration into a spurious failure in a file about methodology. The
demo's numeric anchors live in test_golden_metrics.py.
"""

import math

import numpy as np
import pytest

from tripphysics.baseline import NaiveBaseline
from tripphysics.detector import score_account
from tripphysics.eval import (
    DEFAULT_BUDGET_FP_PER_10K,
    bootstrap_uplift_ci,
    evaluate,
    summarize_across_seeds,
)
from tripphysics.features import extract_features
from tripphysics.synth import generate_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _corpus(n_accounts=500, shared_fraction=0.08, seed=4):
    accounts = generate_dataset(n_accounts=n_accounts,
                                shared_fraction=shared_fraction,
                                seed=seed)["accounts"]
    labels = {a["account_id"]: a["is_shared"] for a in accounts}
    naive = NaiveBaseline()
    baseline = {a["account_id"]: naive.score(a) for a in accounts}
    detector = {a["account_id"]: score_account(extract_features(a))
                for a in accounts}
    return labels, baseline, detector


@pytest.fixture(scope="module")
def corpus():
    return _corpus()


@pytest.fixture(scope="module")
def ci(corpus):
    labels, baseline, detector = corpus
    return bootstrap_uplift_ci(labels, baseline, detector,
                               n_resamples=300, seed=0)


# ---------------------------------------------------------------------------
# bootstrap_uplift_ci: shape and well-formedness
# ---------------------------------------------------------------------------

def test_ci_returns_the_documented_keys(ci):
    assert set(ci) == {"u_point", "ci_low", "ci_high", "n_resamples",
                       "confidence", "n_degenerate", "degeneracy_rate",
                       "ci_valid"}


def test_ci_values_are_plain_builtins(ci):
    # The dict is rendered into the shipped JSON artifact; a numpy scalar
    # here breaks json.dump at report time rather than at test time. bool is
    # allowed (ci_valid): it is a JSON-native type and a subclass of int.
    import json

    for key, value in ci.items():
        assert type(value) in (float, int, bool), f"{key} is {type(value)}"
    json.dumps(ci)


def test_ci_bounds_are_ordered(ci):
    assert ci["ci_low"] <= ci["ci_high"], (
        f"inverted interval: [{ci['ci_low']}, {ci['ci_high']}]")


def test_ci_brackets_the_point_estimate(ci):
    # A percentile bootstrap interval is not *guaranteed* to contain the
    # point estimate in pathological cases, but on a well-behaved corpus a
    # point estimate outside its own interval means the resampling is wrong
    # (e.g. resampling the wrong unit, or reusing a stale threshold).
    assert ci["ci_low"] <= ci["u_point"] <= ci["ci_high"], (
        f"U = {ci['u_point']} lies outside its own CI "
        f"[{ci['ci_low']}, {ci['ci_high']}]")


def test_ci_point_estimate_matches_evaluate(corpus, ci):
    # u_point must be the same number evaluate() reports, not a bootstrap
    # mean. A bootstrap mean would be biased and would silently disagree
    # with the headline U quoted everywhere else.
    labels, baseline, detector = corpus
    assert ci["u_point"] == evaluate(labels, baseline, detector)["U"]


def test_ci_bounds_are_finite_and_in_a_sane_range(ci):
    for key in ("ci_low", "ci_high", "u_point"):
        assert math.isfinite(ci[key]), f"{key} is not finite"
    # U is a relative uplift: bounded below by -1 (recall cannot fall past
    # zero) and, on this corpus, nowhere near a 100x improvement.
    assert ci["ci_low"] >= -1.0
    assert ci["ci_high"] < 100.0


def test_ci_reports_its_own_configuration(corpus):
    labels, baseline, detector = corpus
    result = bootstrap_uplift_ci(labels, baseline, detector,
                                 n_resamples=120, confidence=0.90, seed=3)
    assert result["n_resamples"] == 120
    assert result["confidence"] == 0.90


def test_ci_counts_degenerate_replicates_rather_than_hiding_them(ci):
    # Replicates where baseline recall is zero make U undefined. They must be
    # excluded from the percentiles *and* counted, so a CI computed mostly
    # from discarded replicates cannot pass for a clean one.
    assert ci["n_degenerate"] >= 0
    assert isinstance(ci["n_degenerate"], int)


# ---------------------------------------------------------------------------
# bootstrap_uplift_ci: determinism
# ---------------------------------------------------------------------------

def test_ci_is_deterministic_for_a_fixed_seed(corpus):
    labels, baseline, detector = corpus
    first = bootstrap_uplift_ci(labels, baseline, detector,
                                n_resamples=150, seed=11)
    second = bootstrap_uplift_ci(labels, baseline, detector,
                                 n_resamples=150, seed=11)
    assert first == second


def test_ci_differs_between_seeds(corpus):
    # Guards the mirror-image bug: a seed that never reaches the RNG would
    # make the interval look reassuringly stable for the wrong reason.
    labels, baseline, detector = corpus
    first = bootstrap_uplift_ci(labels, baseline, detector,
                                n_resamples=150, seed=1)
    second = bootstrap_uplift_ci(labels, baseline, detector,
                                 n_resamples=150, seed=2)
    assert (first["ci_low"], first["ci_high"]) != (second["ci_low"],
                                                   second["ci_high"])


def test_ci_does_not_consume_global_numpy_entropy(corpus):
    labels, baseline, detector = corpus
    np.random.seed(0)
    first = bootstrap_uplift_ci(labels, baseline, detector,
                                n_resamples=100, seed=5)
    np.random.seed(0)
    np.random.random(2000)
    second = bootstrap_uplift_ci(labels, baseline, detector,
                                 n_resamples=100, seed=5)
    assert first == second


def test_ci_does_not_mutate_its_inputs(corpus):
    labels, baseline, detector = corpus
    labels_copy = dict(labels)
    baseline_copy = dict(baseline)
    detector_copy = dict(detector)
    bootstrap_uplift_ci(labels, baseline, detector, n_resamples=50, seed=0)
    assert labels == labels_copy
    assert baseline == baseline_copy
    assert detector == detector_copy


# ---------------------------------------------------------------------------
# bootstrap_uplift_ci: statistical behaviour
# ---------------------------------------------------------------------------

def test_wider_confidence_gives_a_wider_interval(corpus):
    # Basic sanity on the percentile arithmetic: 99% must not be narrower
    # than 80%. An inverted tail calculation would show up exactly here.
    labels, baseline, detector = corpus
    narrow = bootstrap_uplift_ci(labels, baseline, detector,
                                 n_resamples=400, confidence=0.80, seed=7)
    wide = bootstrap_uplift_ci(labels, baseline, detector,
                               n_resamples=400, confidence=0.99, seed=7)
    narrow_width = narrow["ci_high"] - narrow["ci_low"]
    wide_width = wide["ci_high"] - wide["ci_low"]
    assert wide_width >= narrow_width, (
        f"99% CI ({wide_width}) narrower than 80% CI ({narrow_width})")


def test_ci_covers_the_true_uplift_of_a_constructed_population():
    # Coverage check on a population whose U is known by construction rather
    # than measured: 100 shared and 900 honest accounts, a baseline that
    # catches a fixed half of the shared tail and a detector that catches
    # more. The observed U is the estimand; a correctly implemented
    # stratified bootstrap must bracket it.
    labels = {f"s{i}": True for i in range(100)}
    labels.update({f"h{i}": False for i in range(900)})
    # Baseline: shared accounts split into a caught half and a missed half.
    baseline = {a: (0.9 if (a.startswith("s") and int(a[1:]) < 50) else 0.1)
                for a in labels}
    # Detector: catches 80 of the 100, including the baseline's 50.
    detector = {a: (0.9 if (a.startswith("s") and int(a[1:]) < 80) else 0.1)
                for a in labels}

    observed = evaluate(labels, baseline, detector,
                        budget_fp_per_10k=DEFAULT_BUDGET_FP_PER_10K)["U"]
    result = bootstrap_uplift_ci(labels, baseline, detector, n_resamples=400,
                                 seed=0)
    assert result["ci_low"] <= observed <= result["ci_high"]


def test_ci_reports_no_interval_when_every_replicate_is_degenerate():
    # A flat scorer flags nobody at any budget, so R0 = 0 in every replicate
    # and U is 0/0 -- undefined, not zero. The honest output is NaN bounds
    # plus a degenerate count equal to the number of replicates. Returning a
    # confident [0, 0] here would be the failure mode: an interval that looks
    # tight precisely because the estimate does not exist.
    labels = {f"s{i}": True for i in range(30)}
    labels.update({f"h{i}": False for i in range(270)})
    flat = {a: 0.5 for a in labels}
    result = bootstrap_uplift_ci(labels, flat, flat, n_resamples=100, seed=0)
    assert result["n_degenerate"] == 100
    assert math.isnan(result["ci_low"])
    assert math.isnan(result["ci_high"])


def test_ci_collapses_to_a_point_when_the_uplift_is_identically_zero():
    # The neighbouring case that *is* well defined: the baseline catches a
    # fixed set, the detector adds nothing, so U = 0 on every resample and
    # the interval is a genuine point at zero.
    labels = {f"s{i}": True for i in range(40)}
    labels.update({f"h{i}": False for i in range(360)})
    scores = {a: (0.9 if a.startswith("s") else 0.1) for a in labels}
    result = bootstrap_uplift_ci(labels, scores, scores, n_resamples=100, seed=0)
    assert result["n_degenerate"] == 0
    assert result["ci_low"] == pytest.approx(0.0)
    assert result["ci_high"] == pytest.approx(0.0)


def test_ci_survives_a_single_resample(corpus):
    # n_resamples=1 is degenerate but legal; the percentiles of a one-element
    # sample are that element. It must not raise or return NaN.
    labels, baseline, detector = corpus
    result = bootstrap_uplift_ci(labels, baseline, detector,
                                 n_resamples=1, seed=0)
    assert result["n_resamples"] == 1
    assert result["ci_low"] == pytest.approx(result["ci_high"])


def test_ci_handles_a_population_with_one_shared_account():
    # The smallest non-degenerate positive set. Stratified resampling keeps
    # exactly one positive per replicate, so U is defined throughout.
    labels = {"s0": True}
    labels.update({f"h{i}": False for i in range(199)})
    baseline = {a: (0.9 if a == "s0" else 0.1) for a in labels}
    detector = dict(baseline)
    result = bootstrap_uplift_ci(labels, baseline, detector,
                                 n_resamples=50, seed=0)
    assert math.isfinite(result["ci_low"])
    assert math.isfinite(result["ci_high"])


# ---------------------------------------------------------------------------
# bootstrap_uplift_ci: input validation
# ---------------------------------------------------------------------------

def test_ci_rejects_non_positive_resample_counts(corpus):
    labels, baseline, detector = corpus
    for bad in (0, -5):
        with pytest.raises(ValueError, match="n_resamples"):
            bootstrap_uplift_ci(labels, baseline, detector, n_resamples=bad)


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.2])
def test_ci_rejects_out_of_range_confidence(corpus, bad):
    labels, baseline, detector = corpus
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_uplift_ci(labels, baseline, detector,
                            n_resamples=10, confidence=bad)


def test_ci_rejects_a_population_with_no_shared_accounts():
    labels = {f"h{i}": False for i in range(10)}
    scores = {a: 0.5 for a in labels}
    with pytest.raises(ValueError, match="no shared accounts"):
        bootstrap_uplift_ci(labels, scores, scores, n_resamples=10)


def test_ci_rejects_a_population_with_no_honest_accounts():
    labels = {f"s{i}": True for i in range(10)}
    scores = {a: 0.5 for a in labels}
    with pytest.raises(ValueError, match="no honest accounts"):
        bootstrap_uplift_ci(labels, scores, scores, n_resamples=10)


# ---------------------------------------------------------------------------
# summarize_across_seeds
# ---------------------------------------------------------------------------

def test_sweep_summary_returns_the_documented_keys():
    summary = summarize_across_seeds([0.1, 0.2, 0.3])
    assert set(summary) == {"median", "mean", "sd", "min", "max", "n_seeds",
                            "n_seeds_supplied", "n_seeds_dropped_nonfinite",
                            "target", "n_at_or_above_target",
                            "fraction_at_or_above_target"}


def test_sweep_summary_computes_the_median_by_hand():
    # Odd n: the middle element. Even n: the mean of the two middle ones.
    assert summarize_across_seeds([0.5, 0.1, 0.3])["median"] == pytest.approx(0.3)
    assert summarize_across_seeds([0.1, 0.2, 0.3, 0.4])["median"] == pytest.approx(0.25)


def test_sweep_summary_is_invariant_under_input_order():
    ordered = summarize_across_seeds([0.1, 0.2, 0.3, 0.9])
    shuffled = summarize_across_seeds([0.9, 0.3, 0.1, 0.2])
    assert ordered == shuffled


def test_sweep_summary_uses_the_sample_standard_deviation():
    # sd must divide by (n-1), not n. On [0, 2] the sample sd is sqrt(2)
    # while the population sd is 1 -- a divisor bug is unmistakable here.
    assert summarize_across_seeds([0.0, 2.0])["sd"] == pytest.approx(math.sqrt(2.0))


def test_sweep_summary_reports_zero_spread_for_a_single_seed():
    # n=1 leaves the sample sd undefined; reporting 0.0 is the documented
    # choice. What must never happen is a ZeroDivisionError in a report run.
    summary = summarize_across_seeds([0.42])
    assert summary["n_seeds"] == 1
    assert summary["sd"] == 0.0
    assert summary["median"] == pytest.approx(0.42)
    assert summary["min"] == summary["max"] == pytest.approx(0.42)


def test_sweep_summary_reports_zero_spread_for_identical_seeds():
    summary = summarize_across_seeds([0.3, 0.3, 0.3, 0.3])
    assert summary["sd"] == pytest.approx(0.0)
    assert summary["min"] == summary["max"] == pytest.approx(0.3)


def test_sweep_summary_counts_the_pass_rate_against_the_target():
    summary = summarize_across_seeds([0.1, 0.2, 0.25, 0.5], target=0.20)
    # 0.2, 0.25 and 0.5 clear the bar; the boundary value counts as a pass.
    assert summary["n_at_or_above_target"] == 3
    assert summary["fraction_at_or_above_target"] == pytest.approx(0.75)
    assert summary["target"] == pytest.approx(0.20)


def test_sweep_summary_target_boundary_is_inclusive():
    assert summarize_across_seeds([0.20], target=0.20)["n_at_or_above_target"] == 1


def test_sweep_summary_min_and_max_bracket_everything():
    uplifts = [0.4, -0.1, 0.9, 0.0, 0.25]
    summary = summarize_across_seeds(uplifts)
    assert summary["min"] == pytest.approx(min(uplifts))
    assert summary["max"] == pytest.approx(max(uplifts))
    assert summary["min"] <= summary["median"] <= summary["max"]
    assert summary["min"] <= summary["mean"] <= summary["max"]


def test_sweep_summary_drops_non_finite_uplifts():
    # U is +inf when the baseline caught nothing. Such a seed must not
    # poison the median or make the sd NaN; it is dropped from the summary.
    summary = summarize_across_seeds([0.1, 0.3, math.inf, 0.5])
    assert summary["n_seeds"] == 3
    assert math.isfinite(summary["sd"])
    assert summary["median"] == pytest.approx(0.3)


def test_sweep_summary_rejects_an_empty_sweep():
    with pytest.raises(ValueError, match="no finite uplift"):
        summarize_across_seeds([])


def test_sweep_summary_rejects_an_all_degenerate_sweep():
    with pytest.raises(ValueError, match="no finite uplift"):
        summarize_across_seeds([math.inf, math.nan])


def test_sweep_summary_handles_negative_uplifts():
    # U < 0 is a real outcome (the detector spent budget without adding
    # signal). It must survive the summary rather than being clipped.
    summary = summarize_across_seeds([-0.3, -0.1, 0.2], target=0.20)
    assert summary["min"] == pytest.approx(-0.3)
    assert summary["n_at_or_above_target"] == 1


# ---------------------------------------------------------------------------
# The sweep as it is actually used: independent seeds, real corpora
# ---------------------------------------------------------------------------

SWEEP_SEEDS = tuple(range(1, 13))


def _uplift_for_seed(seed, n_accounts=300, shared_fraction=0.06):
    labels, baseline, detector = _corpus(n_accounts=n_accounts,
                                         shared_fraction=shared_fraction,
                                         seed=seed)
    return evaluate(labels, baseline, detector)["U"]


@pytest.fixture(scope="module")
def sweep_uplifts():
    return [_uplift_for_seed(seed) for seed in SWEEP_SEEDS]


def test_the_seed_set_is_a_fixed_deterministic_sequence():
    # The seeds must be a declared constant, not drawn at runtime: a sweep
    # over randomly chosen seeds cannot be reproduced by a reader, and a
    # sweep whose seed set drifts is a sweep that can be re-rolled until it
    # flatters the result.
    assert SWEEP_SEEDS == tuple(range(1, 13))
    assert len(set(SWEEP_SEEDS)) == len(SWEEP_SEEDS)


def test_the_sweep_is_reproducible(sweep_uplifts):
    again = [_uplift_for_seed(seed) for seed in SWEEP_SEEDS]
    assert again == sweep_uplifts


def test_every_seed_yields_a_finite_uplift(sweep_uplifts):
    for seed, uplift in zip(SWEEP_SEEDS, sweep_uplifts):
        assert math.isfinite(uplift), f"seed {seed} gave U = {uplift}"


def test_the_median_uplift_is_positive_across_independent_worlds(sweep_uplifts):
    # The claim the harness actually makes. Stated on the median over
    # independent seeds rather than on any single run, and directionally
    # (U > 0) rather than at a pinned value -- the magnitude is anchored in
    # test_golden_metrics.py, where a recalibration is supposed to move it.
    summary = summarize_across_seeds(sweep_uplifts)
    assert summary["median"] > 0.0, (
        f"median U over {summary['n_seeds']} seeds is {summary['median']}")


def test_the_uplift_is_not_carried_by_a_single_lucky_seed(sweep_uplifts):
    # If the headline held on one seed out of twelve it would be noise. A
    # clear majority of independent worlds must show positive uplift.
    positive = sum(1 for u in sweep_uplifts if u > 0.0)
    assert positive > len(sweep_uplifts) / 2, (
        f"only {positive}/{len(sweep_uplifts)} seeds gave positive uplift")


def test_the_sweep_spread_is_reported_and_finite(sweep_uplifts):
    summary = summarize_across_seeds(sweep_uplifts)
    assert math.isfinite(summary["sd"])
    assert summary["sd"] >= 0.0
    assert summary["n_seeds"] == len(SWEEP_SEEDS)
