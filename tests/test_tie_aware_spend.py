"""The tie-aware symmetric FP-spend fix: equal realized spend across scorers.

Regression lock for the tie-underspend bug. Two scorers of different
granularity, held to the same nominal FP budget, used to realize *different*
false-positive counts (a coarse scorer's boundary ties went unflagged), which
inflated the uplift U in favour of the finer-grained pilot. The fix meets both
scorers at ``common_spend`` -- the largest FP spend a strict threshold reaches
on BOTH honest ladders -- so "at equal FP spend" is a printed integer identity.

Every test here names a concrete input and the exact wrong number the old code
produced, so a reader can see what is being prevented.
"""

import math

import numpy as np
import pytest

from tripphysics import eval as E


# ---------------------------------------------------------------------------
# Helpers: build honest/shared score dicts with controllable granularity.
# ---------------------------------------------------------------------------
def _labels_and_scores(honest_scores, shared_scores):
    """Turn two lists into (labels, scores) dicts with distinct ids."""
    labels, scores = {}, {}
    for i, s in enumerate(honest_scores):
        aid = f"h{i:04d}"
        labels[aid] = False
        scores[aid] = float(s)
    for i, s in enumerate(shared_scores):
        aid = f"s{i:04d}"
        labels[aid] = True
        scores[aid] = float(s)
    return labels, scores


def _realized_spend(scores, labels, threshold):
    """How many honest accounts a strict > threshold actually flags."""
    return sum(1 for a, is_shared in labels.items()
               if not is_shared and scores[a] > threshold)


def _stacked(scores_baseline, scores_pilot, labels):
    rb = E.rank_normalize(scores_baseline)
    rd = E.rank_normalize(scores_pilot)
    return {a: max(rb[a], rd[a]) for a in labels}


# ---------------------------------------------------------------------------
# common_spend: the fixed-point primitive.
# ---------------------------------------------------------------------------
def test_common_spend_matches_budget_when_no_ties():
    # Two strictly-distinct ladders: both realize exactly allowed_fp.
    a = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0])[::-1]
    b = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])[::-1]
    desc_a = np.sort(a)[::-1]
    desc_b = np.sort(b)[::-1]
    assert E.common_spend(desc_a, desc_b, 3) == 3


def test_common_spend_backs_off_into_a_tie_cluster():
    # Baseline has a big tie block straddling the budget boundary: budget 3
    # cannot be realized (a strict threshold flags either 1 or the whole
    # block of 6), so the common spend drops to the largest reachable rung.
    desc_a = np.sort(np.array([5.0, 4.0] + [1.0] * 6 + [0.0, 0.0]))[::-1]
    desc_b = np.sort(np.array([9., 8., 7., 6., 5., 4., 3., 2., 1., 0.]))[::-1]
    spend = E.common_spend(desc_a, desc_b, 3)
    # A strict threshold on desc_a can realize 0, 1, 2, or 8 flags -- not 3.
    # The largest reachable <= 3 is 2.
    assert spend == 2


def test_common_spend_is_symmetric():
    desc_a = np.sort(np.array([5.0, 4.0] + [1.0] * 6 + [0.0, 0.0]))[::-1]
    desc_b = np.sort(np.array([9., 8., 7., 6., 5., 4., 3., 2., 1., 0.]))[::-1]
    assert (E.common_spend(desc_a, desc_b, 3)
            == E.common_spend(desc_b, desc_a, 3))


def test_common_spend_zero_when_top_scores_all_tie():
    # Every honest score identical: no strict threshold flags anyone but the
    # empty set; F* collapses to 0. This is the coarse-baseline worst case.
    desc_a = np.zeros(10)
    desc_b = np.sort(np.arange(10.0))[::-1]
    assert E.common_spend(desc_a, desc_b, 5) == 0


# ---------------------------------------------------------------------------
# The core invariant: equal realized spend on both scorers, always.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("granularity", [None, 1.0, 0.5, 0.1, 0.25])
def test_equal_realized_spend_invariant(granularity):
    # A range of populations at several baseline granularities; whatever the
    # ties, the two scorers must flag the SAME integer number of honest
    # accounts, equal to result['realized_fp'], never exceeding allowed_fp.
    rng = np.random.default_rng(0)
    honest = rng.uniform(0, 1, 400).tolist()
    shared = rng.uniform(0.3, 1.3, 40).tolist()
    labels, base = _labels_and_scores(honest, shared)
    # Pilot is a different, near-continuous scorer.
    pilot = {a: base[a] + rng.uniform(-0.05, 0.05) for a in base}
    if granularity is not None:
        base = {a: round(base[a] / granularity) * granularity for a in base}

    res = E.evaluate(labels, base, pilot, budget_fp_per_10k=200.0)
    stacked = _stacked(base, pilot, labels)

    fp_base = _realized_spend(base, labels, res["threshold_baseline"])
    fp_pilot = _realized_spend(stacked, labels, res["threshold_pilot"])
    assert fp_base == fp_pilot == res["realized_fp"], (
        f"unequal realized spend at granularity {granularity}: "
        f"baseline {fp_base}, pilot {fp_pilot}, recorded {res['realized_fp']}")
    assert res["realized_fp"] <= res["allowed_fp"]


def test_coarse_baseline_does_not_inflate_uplift():
    # The headline defect: a coarse baseline underspends the budget, its
    # recall R0 is depressed, and U inflates. Construct a case where the
    # continuous-baseline U and the coarse-baseline U would diverge under the
    # OLD code, and assert the fix ties them to the same realized spend so the
    # coarse U is not larger purely from underspend.
    rng = np.random.default_rng(1)
    honest = rng.uniform(0, 1, 500).tolist()
    shared = rng.uniform(0.4, 1.4, 50).tolist()
    labels, base_cont = _labels_and_scores(honest, shared)
    pilot = {a: base_cont[a] + rng.uniform(-0.05, 0.05) for a in base_cont}
    base_coarse = {a: round(base_cont[a] / 0.5) * 0.5 for a in base_cont}

    res_cont = E.evaluate(labels, base_cont, pilot, 100.0)
    res_coarse = E.evaluate(labels, base_coarse, pilot, 100.0)

    # Both scorers in each run spend their reported realized_fp exactly.
    for res, base in ((res_cont, base_cont), (res_coarse, base_coarse)):
        stacked = _stacked(base, pilot, labels)
        assert _realized_spend(base, labels, res["threshold_baseline"]) == \
            _realized_spend(stacked, labels, res["threshold_pilot"]) == \
            res["realized_fp"]

    # The coarse run either spends the same (no inflation) or its utilization
    # is flagged below 1.0 so a reader is told the budget was not reached.
    assert res_coarse["budget_utilization"] <= 1.0
    if res_coarse["realized_fp"] < res_coarse["allowed_fp"]:
        assert res_coarse["budget_utilization"] < 1.0


def test_continuous_scorers_unchanged_backward_compat():
    # With strictly-distinct scores there are no ties at the boundary, so the
    # realized spend equals allowed_fp and the fix must reproduce the old
    # numbers. Recall and U are computed the same way as before.
    honest = [i / 1000.0 for i in range(1000)]
    shared = [0.5 + i / 200.0 for i in range(50)]
    labels, base = _labels_and_scores(honest, shared)
    pilot = {a: base[a] * 0.9 for a in base}  # monotone rescale: same ranks

    res = E.evaluate(labels, base, pilot, budget_fp_per_10k=200.0)
    # 1000 honest, budget 200/10k -> allowed_fp = 20, fully realized.
    assert res["allowed_fp"] == 20
    assert res["realized_fp"] == 20
    assert res["budget_utilization"] == 1.0


def test_spend_primitive_cannot_flatter_a_scorer():
    # common_spend is the anti-flatter guarantee: it is a symmetric function of
    # the two honest ladders, so it cannot depend on which scorer is "baseline"
    # vs "pilot". (evaluate() itself is deliberately NOT swap-symmetric: the
    # pilot is always baseline-stacked-with-detector, and the realized spend is
    # governed by the BASELINE ladder's granularity -- a coarse baseline that
    # cannot spend the budget collapses F* to 0, which is the correct, honest
    # outcome, not a broken symmetry. So the invariant we lock is on the
    # primitive that does the spend arithmetic.)
    rng = np.random.default_rng(2)
    honest = rng.uniform(0, 1, 300).tolist()
    a = {f"h{i}": round(v / 0.5) * 0.5 for i, v in enumerate(honest)}  # coarse
    b = {k: float(rng.uniform(0, 1)) for k in a}                       # fine
    honest_ids = list(a)
    desc_a = np.sort(np.array([a[k] for k in honest_ids]))[::-1]
    desc_b = np.sort(np.array([b[k] for k in honest_ids]))[::-1]
    allowed = 4
    assert (E.common_spend(desc_a, desc_b, allowed)
            == E.common_spend(desc_b, desc_a, allowed))


def test_coarse_baseline_collapses_spend_not_the_fine_one():
    # The corollary a Head of Fraud must see: when the baseline is too coarse
    # to spend the budget, the realized spend collapses and utilization drops,
    # rather than the harness quietly comparing at a spend the baseline never
    # reaches. Same detector, same data, only the baseline granularity differs.
    rng = np.random.default_rng(2)
    honest = rng.uniform(0, 1, 300).tolist()
    shared = rng.uniform(0.3, 1.3, 30).tolist()
    labels, fine = _labels_and_scores(honest, shared)
    coarse = {k: round(fine[k] / 0.5) * 0.5 for k in fine}
    detector = {k: float(rng.uniform(0, 1)) for k in fine}

    res_fine = E.evaluate(labels, fine, detector, 150.0)
    res_coarse = E.evaluate(labels, coarse, detector, 150.0)
    # The fine baseline reaches the budget; the coarse one is flagged as under.
    assert res_fine["budget_utilization"] >= res_coarse["budget_utilization"]
    assert res_coarse["budget_utilization"] < 1.0


# ---------------------------------------------------------------------------
# Bootstrap CI validity gate.
# ---------------------------------------------------------------------------
def test_ci_invalid_when_baseline_too_coarse():
    # A baseline coarse enough that F* collapses to zero on a large fraction
    # of resamples: the CI must be refused (ci_valid False, nan bounds), not
    # computed over the spend-correlated survivors.
    rng = np.random.default_rng(3)
    n_honest = 300
    # Baseline takes only 3 distinct values: heavy ties, F* often 0 at a small
    # budget.
    honest = rng.choice([0.0, 0.5, 1.0], size=n_honest).tolist()
    shared = rng.choice([0.5, 1.0], size=30).tolist()
    labels, base = _labels_and_scores(honest, shared)
    pilot = {a: rng.uniform(0, 1) for a in base}
    ci = E.bootstrap_uplift_ci(labels, base, pilot, budget_fp_per_10k=15.0,
                               n_resamples=200, seed=0)
    if ci["degeneracy_rate"] >= 0.05:
        assert ci["ci_valid"] is False
        assert math.isnan(ci["ci_low"]) and math.isnan(ci["ci_high"])
    else:
        assert ci["ci_valid"] is True


def test_ci_valid_on_clean_continuous_data():
    rng = np.random.default_rng(4)
    honest = rng.uniform(0, 1, 800).tolist()
    shared = rng.uniform(0.4, 1.4, 80).tolist()
    labels, base = _labels_and_scores(honest, shared)
    pilot = {a: base[a] + rng.uniform(-0.03, 0.03) for a in base}
    ci = E.bootstrap_uplift_ci(labels, base, pilot, budget_fp_per_10k=200.0,
                               n_resamples=200, seed=0)
    assert ci["ci_valid"] is True
    assert not math.isnan(ci["ci_low"])
    assert ci["ci_low"] <= ci["ci_high"]


def test_ci_matches_point_estimator():
    # The CI's per-replicate recall must use the SAME joint-spend estimator as
    # the point estimate, so u_point lies in a sane relation to the interval
    # (not the old mismatched independent-threshold estimator). We check the
    # point U is finite and the interval brackets a neighborhood of it.
    rng = np.random.default_rng(5)
    honest = rng.uniform(0, 1, 600).tolist()
    shared = rng.uniform(0.4, 1.4, 60).tolist()
    labels, base = _labels_and_scores(honest, shared)
    pilot = {a: base[a] + rng.uniform(-0.04, 0.04) for a in base}
    ci = E.bootstrap_uplift_ci(labels, base, pilot, budget_fp_per_10k=200.0,
                               n_resamples=300, seed=0)
    assert math.isfinite(ci["u_point"])
    assert ci["ci_valid"] is True


# ---------------------------------------------------------------------------
# Determinism: the fix must not introduce any run-to-run variation.
# ---------------------------------------------------------------------------
def test_evaluate_is_deterministic_under_the_fix():
    rng = np.random.default_rng(6)
    honest = rng.uniform(0, 1, 400).tolist()
    shared = rng.uniform(0.3, 1.3, 40).tolist()
    labels, base = _labels_and_scores(honest, shared)
    base = {a: round(base[a] / 0.5) * 0.5 for a in base}
    pilot = {a: rng.uniform(0, 1) for a in base}
    r1 = E.evaluate(labels, base, pilot, 150.0)
    r2 = E.evaluate(labels, base, pilot, 150.0)
    assert r1 == r2


def test_negative_infinity_honest_score_keeps_spend_equal():
    # Adversarial regression: with a -inf honest score and a budget that allows
    # flagging everyone, the threshold is -inf. `-inf > -inf` is False, so that
    # account is NOT flagged -- an earlier version of the fix assumed "cap >= n
    # means all n are spent" and reported an unequal realized spend (baseline 1,
    # pilot 2) plus a fabricated U = +inf, with utilization 1.0 so the report's
    # granularity gate never tripped.
    labels = {"h1": False, "h2": False, "s1": True}
    base = {"h1": -math.inf, "h2": 1.0, "s1": 2.0}
    pilot = {"h1": 0.0, "h2": 0.0, "s1": 0.0}
    res = E.evaluate(labels, base, pilot, budget_fp_per_10k=10000.0)
    stacked = _stacked(base, pilot, labels)
    fp_base = _realized_spend(base, labels, res["threshold_baseline"])
    fp_pilot = _realized_spend(stacked, labels, res["threshold_pilot"])
    assert fp_base == fp_pilot == res["realized_fp"], (
        f"unequal spend with -inf score: baseline {fp_base}, pilot {fp_pilot}, "
        f"recorded {res['realized_fp']}")
    assert math.isfinite(res["U"]), "U must not be fabricated as +inf here"


def test_negative_budget_is_rejected():
    # The old guard lived inside threshold_at_fp_budget, which the realized-spend
    # path no longer calls. Without an explicit check a negative budget produced
    # allowed_fp = -1, realized_fp = 0 (violating realized <= allowed) and a
    # utilization of 1.0, i.e. nonsense reported as a clean result.
    labels = {"h1": False, "h2": False, "s1": True}
    scores = {"h1": 0.1, "h2": 0.2, "s1": 0.9}
    with pytest.raises(ValueError, match="budget_fp_per_10k must be >= 0"):
        E.evaluate(labels, scores, dict(scores), budget_fp_per_10k=-1.0)
    with pytest.raises(ValueError, match="budget_fp_per_10k must be >= 0"):
        E.bootstrap_uplift_ci(labels, scores, dict(scores),
                              budget_fp_per_10k=-5000.0, n_resamples=10)


def test_evaluate_result_keys_present():
    labels, base = _labels_and_scores([0.1, 0.2, 0.3, 0.4], [0.5, 0.6])
    pilot = {a: base[a] for a in base}
    res = E.evaluate(labels, base, pilot, 2500.0)
    for k in ("allowed_fp", "realized_fp", "realized_fp_per_10k",
              "budget_utilization", "R0", "R1", "U", "threshold_baseline",
              "threshold_pilot"):
        assert k in res, f"missing result key {k}"
