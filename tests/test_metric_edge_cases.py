"""Degenerate inputs to the metric layer: empty, singleton, one-class, all-tied.

The hand-computed fixture in test_eval_math.py pins the *normal* case. This
module pins the boundaries, where evaluation code usually fails silently:
an empty slice, a single account, a population with no positives, or a
scorer that returns the same number for everyone. Each of those is a real
shape a shadow pipeline hits on day one (a quiet market, a filtered cohort,
a scorer that failed open and emits a constant).

One assertion here is marked xfail: it documents DEFECT-1, where pr_auc
returns a degenerate value with all scores tied. DEFECT-2 (precision_at_k
raising ZeroDivisionError at k=0) is fixed, and its test now pins the fixed
behaviour. Both document defects found in the production code rather than
accepting current behaviour as correct.
"""

import math

import pytest

from tripphysics.eval import (
    evaluate,
    pr_auc,
    precision_at_k,
    rank_normalize,
    recall_at_threshold,
    threshold_at_fp_budget,
)


# ---------------------------------------------------------------------------
# pr_auc: empty / one-class / singleton populations
# ---------------------------------------------------------------------------

def test_pr_auc_empty_population_is_zero():
    # No accounts at all -> no positives -> the documented 0.0 fallback.
    assert pr_auc({}, {}) == 0.0


def test_pr_auc_no_positives_is_zero():
    # A cohort with zero confirmed-shared accounts: recall is undefined, and
    # the contract says 0.0 rather than a ZeroDivisionError.
    labels = {"h1": False, "h2": False, "h3": False}
    scores = {"h1": 0.1, "h2": 0.5, "h3": 0.9}
    assert pr_auc(labels, scores) == 0.0


def test_pr_auc_all_positives_is_one():
    # Every account is shared: precision is 1.0 at every operating point,
    # so the curve is the constant P=1 line and the area is exactly 1.
    labels = {"s1": True, "s2": True, "s3": True}
    scores = {"s1": 0.1, "s2": 0.5, "s3": 0.9}
    assert pr_auc(labels, scores) == pytest.approx(1.0)


def test_pr_auc_single_positive_account():
    # n=1 is the smallest non-degenerate input: one point at (R=1, P=1),
    # trapezoid against the (0, 1) anchor gives 1.0.
    assert pr_auc({"s1": True}, {"s1": 0.42}) == pytest.approx(1.0)


def test_pr_auc_single_negative_account():
    assert pr_auc({"h1": False}, {"h1": 0.42}) == 0.0


def test_pr_auc_perfect_separation_is_one():
    # Every shared account scores above every honest one -> AUC exactly 1.
    labels = {f"s{i}": True for i in range(10)}
    labels.update({f"h{i}": False for i in range(90)})
    scores = {a: (1.0 if is_shared else 0.0) for a, is_shared in labels.items()}
    assert pr_auc(labels, scores) == pytest.approx(1.0)


def test_pr_auc_inverted_scorer_is_poor_but_bounded():
    # A scorer that ranks shared accounts *below* honest ones must land well
    # under the base rate, and must still be a valid area in [0, 1].
    labels = {f"s{i}": True for i in range(10)}
    labels.update({f"h{i}": False for i in range(90)})
    scores = {a: (0.0 if is_shared else 1.0) for a, is_shared in labels.items()}
    auc = pr_auc(labels, scores)
    assert 0.0 <= auc <= 1.0
    assert auc < 0.10  # base rate; a perfectly inverted scorer cannot reach it


# ---------------------------------------------------------------------------
# pr_auc: tied scores
# ---------------------------------------------------------------------------

def test_pr_auc_ties_are_grouped_not_split():
    # Ties must be consumed as one group: a threshold can never separate
    # equal scores. If ties were split, ordering shared-before-honest inside
    # a tied block would inflate the area. Both orderings must agree.
    labels = {"s1": True, "h1": False, "s2": True, "h2": False}
    tied = {a: 0.5 for a in labels}
    auc_tied = pr_auc(labels, tied)

    # Same population, same tie value, different insertion order.
    relabelled = {"h2": False, "s2": True, "h1": False, "s1": True}
    assert pr_auc(relabelled, tied) == pytest.approx(auc_tied)


@pytest.mark.xfail(
    reason="DEFECT-1: with all scores tied, pr_auc returns "
           "(1 + base_rate) / 2 instead of the base rate. The (R=0, P=1) "
           "anchor is a fictitious operating point when no threshold can "
           "separate the population, and the trapezoid interpolates the "
           "whole curve down from it. Observed 0.55 at a 0.10 base rate.",
    strict=True,
)
def test_pr_auc_all_scores_tied_equals_base_rate():
    # A scorer that returns a constant carries zero information; the only
    # achievable operating point is "flag everyone", whose precision is the
    # base rate. A non-informative scorer must therefore score the base rate.
    labels = {f"s{i}": True for i in range(10)}
    labels.update({f"h{i}": False for i in range(90)})
    constant = {a: 1.0 for a in labels}
    assert pr_auc(labels, constant) == pytest.approx(0.10, abs=1e-9)


def test_pr_auc_all_scores_tied_current_behaviour():
    # Characterisation lock for DEFECT-1: pins today's value so the defect
    # cannot change magnitude unnoticed while it is open.
    labels = {f"s{i}": True for i in range(10)}
    labels.update({f"h{i}": False for i in range(90)})
    constant = {a: 1.0 for a in labels}
    assert pr_auc(labels, constant) == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

def test_precision_at_k_empty_population_is_zero():
    assert precision_at_k({}, {}) == 0.0


def test_precision_at_k_truncates_to_population_size():
    # Fewer accounts than k: take everyone, shrink the divisor.
    labels = {"s1": True, "h1": False, "h2": False, "h3": False}
    scores = {"s1": 0.9, "h1": 0.3, "h2": 0.2, "h3": 0.1}
    assert precision_at_k(labels, scores, k=500) == pytest.approx(1 / 4)


def test_precision_at_k_exact_top_k():
    labels = {"s1": True, "s2": True, "h1": False, "h2": False}
    scores = {"s1": 0.9, "s2": 0.8, "h1": 0.3, "h2": 0.1}
    assert precision_at_k(labels, scores, k=2) == pytest.approx(1.0)
    assert precision_at_k(labels, scores, k=4) == pytest.approx(0.5)


def test_precision_at_k_ties_broken_deterministically():
    # All scores equal: the account_id tiebreak must make the result stable
    # across dict orderings, not dependent on insertion sequence.
    labels_a = {"s1": True, "h1": False, "s2": True, "h2": False}
    labels_b = {"h2": False, "s2": True, "h1": False, "s1": True}
    tied = {a: 0.5 for a in labels_a}
    assert precision_at_k(labels_a, tied, k=2) == precision_at_k(labels_b, tied, k=2)


# DEFECT-2 (FIXED): precision_at_k used to raise ZeroDivisionError at k=0 and
# to return a *negative* "precision" at k<0 via Python slice semantics. It now
# validates k up front. The chosen contract is to raise rather than to return
# 0.0, consistent with the module's stated preference for failing loudly over
# silently degrading -- a review queue configured to size zero is a caller
# bug worth surfacing, not a number worth reporting.

def test_precision_at_k_zero_k_is_rejected_with_a_clear_message():
    labels = {"s1": True, "h1": False}
    scores = {"s1": 0.9, "h1": 0.1}
    with pytest.raises(ValueError, match=r"k must be >= 1"):
        precision_at_k(labels, scores, k=0)


def test_precision_at_k_negative_k_is_rejected_with_a_clear_message():
    labels = {"s1": True, "h1": False}
    scores = {"s1": 0.9, "h1": 0.1}
    with pytest.raises(ValueError, match=r"k must be >= 1"):
        precision_at_k(labels, scores, k=-1)


def test_precision_at_k_error_names_the_offending_value():
    # The message must carry the bad k, or a caller reading a traceback from
    # a batch report cannot tell which configuration produced it.
    with pytest.raises(ValueError, match=r"got -7"):
        precision_at_k({"s1": True}, {"s1": 0.5}, k=-7)


def test_precision_at_k_validates_k_before_the_empty_population_shortcut():
    # Ordering matters: an invalid k on an empty cohort is still an invalid
    # k. If the empty-population 0.0 shortcut ran first, the bug would come
    # back for exactly the cohorts that are hardest to notice.
    with pytest.raises(ValueError, match=r"k must be >= 1"):
        precision_at_k({}, {}, k=0)


def test_precision_at_k_accepts_the_smallest_valid_k():
    labels = {"s1": True, "h1": False}
    scores = {"s1": 0.9, "h1": 0.1}
    assert precision_at_k(labels, scores, k=1) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# rank_normalize
# ---------------------------------------------------------------------------

def test_rank_normalize_empty():
    assert rank_normalize({}) == {}


def test_rank_normalize_single_account_is_one():
    # n=1: the only account holds rank 1 of 1.
    assert rank_normalize({"a": -17.5}) == {"a": 1.0}


def test_rank_normalize_all_ties_share_average_rank():
    # Three tied values hold ranks 1,2,3; the average is 2, so each maps to
    # 2/3. Ties must stay ties -- otherwise dict order would leak into scores.
    result = rank_normalize({"a": 1.0, "b": 1.0, "c": 1.0})
    assert result == pytest.approx({"a": 2 / 3, "b": 2 / 3, "c": 2 / 3})


def test_rank_normalize_is_invariant_under_monotone_rescaling():
    # The stated design property: ranks are invariant under any strictly
    # increasing transform, which is why stacking with max() is meaningful
    # across scorers on incomparable scales.
    raw = {"a": 0.1, "b": 0.4, "c": 0.9, "d": 2.0}
    rescaled = {k: math.exp(3.0 * v) for k, v in raw.items()}
    assert rank_normalize(raw) == pytest.approx(rank_normalize(rescaled))


def test_rank_normalize_top_score_is_one():
    raw = {"a": 0.1, "b": 0.4, "c": 0.9}
    ranks = rank_normalize(raw)
    assert ranks["c"] == pytest.approx(1.0)
    assert ranks["a"] == pytest.approx(1 / 3)


def test_rank_normalize_independent_of_insertion_order():
    forward = rank_normalize({"a": 0.3, "b": 0.1, "c": 0.2})
    backward = rank_normalize({"c": 0.2, "b": 0.1, "a": 0.3})
    assert forward == backward


# ---------------------------------------------------------------------------
# threshold_at_fp_budget / recall_at_threshold
# ---------------------------------------------------------------------------

def test_threshold_rejects_empty_honest_population():
    # An FP budget is a promise about honest accounts; with none, there is
    # nothing to promise. Fail loudly rather than invent a threshold.
    with pytest.raises(ValueError, match="no honest accounts"):
        threshold_at_fp_budget([], 50.0)


def test_threshold_rejects_negative_budget():
    with pytest.raises(ValueError, match="must be >= 0"):
        threshold_at_fp_budget([0.1, 0.2], -1.0)


def test_threshold_zero_budget_pins_to_highest_honest_score():
    # Zero allowance -> threshold is the top honest score, and the strict
    # ">" flagging rule means no honest account is flagged.
    honest = [0.1, 0.4, 0.9, 0.2]
    assert threshold_at_fp_budget(honest, 0.0) == pytest.approx(0.9)


def test_threshold_saturating_budget_returns_negative_infinity():
    # A budget large enough to flag everyone: -inf flags the whole population
    # under the strict ">" rule.
    assert threshold_at_fp_budget([0.1, 0.2], 100_000.0) == -math.inf


def test_threshold_never_overspends_the_budget():
    # The core promise: at most `allowed_fp` honest accounts strictly exceed
    # the threshold, for every budget across the meaningful range.
    honest = [i / 1000.0 for i in range(1000)]
    for budget in (0.0, 10.0, 50.0, 200.0, 1000.0, 5000.0):
        threshold = threshold_at_fp_budget(honest, budget)
        allowed = math.floor(len(honest) * budget / 10000.0)
        flagged = sum(1 for s in honest if s > threshold)
        assert flagged <= allowed, f"overspent at budget {budget}"


def test_threshold_is_monotone_non_increasing_in_budget():
    # A larger budget must never raise the threshold -- this is what makes
    # recall-at-budget monotone downstream.
    honest = [i / 500.0 for i in range(500)]
    thresholds = [threshold_at_fp_budget(honest, b)
                  for b in (0.0, 20.0, 100.0, 400.0, 2000.0)]
    for higher_threshold, lower_threshold in zip(thresholds, thresholds[1:]):
        assert lower_threshold <= higher_threshold


def test_threshold_ties_at_the_boundary_underspend():
    # Documented convention: ties *at* the threshold are not flagged, so the
    # budget is under-spent rather than over-spent. Five identical honest
    # scores with an allowance of 1 must flag zero, not one.
    honest = [0.5] * 5
    threshold = threshold_at_fp_budget(honest, 4000.0)  # allowance = 2
    assert sum(1 for s in honest if s > threshold) == 0


def test_recall_at_threshold_no_shared_accounts_is_zero():
    scores = {"h1": 0.9, "h2": 0.8}
    labels = {"h1": False, "h2": False}
    assert recall_at_threshold(scores, labels, 0.0) == 0.0


def test_recall_at_threshold_uses_strict_inequality():
    # An account scoring exactly at the threshold is NOT flagged.
    scores = {"s1": 0.5, "s2": 0.6}
    labels = {"s1": True, "s2": True}
    assert recall_at_threshold(scores, labels, 0.5) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# evaluate(): input validation and degenerate populations
# ---------------------------------------------------------------------------

def test_evaluate_rejects_empty_population():
    with pytest.raises(ValueError):
        evaluate({}, {}, {})


def test_evaluate_rejects_mismatched_account_ids():
    # A partial overlap silently evaluated would bias every number reported.
    labels = {"a": True, "b": False}
    baseline = {"a": 0.5, "b": 0.5}
    pilot = {"a": 0.5}  # missing "b"
    with pytest.raises(ValueError, match="identical"):
        evaluate(labels, baseline, pilot)


def test_evaluate_rejects_extra_ids_in_scores():
    labels = {"a": True}
    baseline = {"a": 0.5, "ghost": 0.9}
    pilot = {"a": 0.5, "ghost": 0.9}
    with pytest.raises(ValueError, match="identical"):
        evaluate(labels, baseline, pilot)


def test_evaluate_requires_at_least_one_honest_account():
    # All-positive population: no honest accounts means no FP budget can be
    # set, which must surface as the documented ValueError.
    labels = {"s1": True, "s2": True}
    scores = {"s1": 0.9, "s2": 0.8}
    with pytest.raises(ValueError, match="no honest accounts"):
        evaluate(labels, scores, scores)


def test_evaluate_no_shared_accounts_gives_zero_recall_and_zero_uplift():
    # A cohort with no confirmed-shared accounts: R0 = R1 = 0, and U is the
    # documented 0.0 (not NaN, not inf) because neither scorer caught anything.
    labels = {f"h{i}": False for i in range(10)}
    baseline = {a: 0.1 * i for i, a in enumerate(labels)}
    pilot = {a: 0.2 * i for i, a in enumerate(labels)}
    result = evaluate(labels, baseline, pilot, budget_fp_per_10k=50.0)
    assert result["R0"] == 0.0
    assert result["R1"] == 0.0
    assert result["U"] == 0.0
    assert result["n_shared"] == 0


def test_evaluate_uplift_is_infinite_when_baseline_catches_nothing():
    # Documented convention: R0 == 0 and R1 > 0 -> U = +inf. report.py has a
    # dedicated rendering branch for this, so the value must really be inf.
    #
    # The budget must allow at least one honest FP. At a zero budget the
    # threshold pins to the top *stacked* honest rank, which is always 1.0
    # (the highest-ranked honest account under the baseline keeps rank 1.0
    # through the elementwise max), and the strict ">" rule then flags no one.
    labels = {f"h{i}": False for i in range(20)}
    labels["s1"] = True
    # Baseline ranks the shared account dead last; detector ranks it top.
    baseline = {a: (0.0 if a == "s1" else 0.5 + 0.01 * i)
                for i, a in enumerate(labels)}
    pilot = {a: (1.0 if a == "s1" else 0.01 * i) for i, a in enumerate(labels)}
    result = evaluate(labels, baseline, pilot, budget_fp_per_10k=1000.0)
    assert result["R0"] == 0.0
    assert result["R1"] > 0.0
    assert result["U"] == math.inf


def test_evaluate_all_scores_tied_yields_zero_recall_at_zero_budget():
    # A constant scorer cannot separate anyone. With a strict ">" rule and a
    # threshold pinned to the (constant) top honest score, nothing is flagged.
    labels = {f"h{i}": False for i in range(10)}
    labels.update({f"s{i}": True for i in range(3)})
    constant = {a: 0.5 for a in labels}
    result = evaluate(labels, constant, constant, budget_fp_per_10k=0.0)
    assert result["R0"] == 0.0
    assert result["R1"] == 0.0
    assert result["U"] == 0.0


def test_evaluate_single_honest_single_shared():
    # The smallest population that admits a threshold at all.
    labels = {"h1": False, "s1": True}
    baseline = {"h1": 0.1, "s1": 0.9}
    pilot = {"h1": 0.2, "s1": 0.8}
    result = evaluate(labels, baseline, pilot, budget_fp_per_10k=0.0)
    assert result["n_accounts"] == 2
    assert result["n_shared"] == 1
    assert result["R0"] == pytest.approx(1.0)
    assert result["R1"] == pytest.approx(1.0)


def test_evaluate_returns_json_safe_scalars():
    # report.py dumps this dict with allow_nan=False; every value must be a
    # plain int/float (never numpy, never bool) or that call raises.
    labels = {"h1": False, "h2": False, "s1": True}
    baseline = {"h1": 0.1, "h2": 0.2, "s1": 0.9}
    pilot = {"h1": 0.15, "h2": 0.25, "s1": 0.95}
    result = evaluate(labels, baseline, pilot)
    for key, value in result.items():
        assert type(value) in (int, float), f"{key} is {type(value)}"
        assert not isinstance(value, bool)
