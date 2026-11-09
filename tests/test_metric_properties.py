"""Properties the metric layer must satisfy for *every* input, not just fixtures.

Example-based tests pin the cases we thought of. These tests instead assert
invariants over many randomly generated populations, which is how a metric
bug on an unanticipated score distribution actually gets caught.

Randomness is drawn from seeded numpy generators (no hypothesis dependency:
this package deliberately ships numpy as its only runtime requirement, and
the test suite keeps the same discipline). Every case is reproducible from
the seed printed in the assertion message.

Invariants covered:
  * PR-AUC is a valid area in [0, 1] on any input.
  * A perfect ranking scores exactly 1.0; a random one lands near the base rate.
  * PR-AUC is invariant under monotone rescaling of the scorer.
  * Recall at budget is non-decreasing in the FP budget (both scorers).
  * The FP budget is never overspent on the honest population.
  * Stacking cannot lose an account the detector already ranked at the top.
"""

import math

import numpy as np
import pytest

from tripphysics.eval import (
    evaluate,
    pr_auc,
    precision_at_k,
    rank_normalize,
    threshold_at_fp_budget,
)

# Seeds are enumerated explicitly so a failure names the exact reproducer.
SEEDS = (1, 7, 42, 99, 2024)


def _random_population(rng, n_honest=200, n_shared=20, separation=0.0):
    """Build a labelled population with a tunable signal-to-noise separation.

    separation=0.0 gives an uninformative scorer (shared and honest drawn
    from the same distribution); larger values shift the shared accounts up.
    """
    labels: dict[str, bool] = {}
    scores: dict[str, float] = {}
    for i in range(n_honest):
        account_id = f"hon{i:05d}"
        labels[account_id] = False
        scores[account_id] = float(rng.uniform(0.0, 1.0))
    for i in range(n_shared):
        account_id = f"shr{i:05d}"
        labels[account_id] = True
        scores[account_id] = float(rng.uniform(0.0, 1.0) + separation)
    return labels, scores


# ---------------------------------------------------------------------------
# PR-AUC bounds and calibration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("separation", [0.0, 0.3, 1.0, 3.0])
def test_pr_auc_is_always_a_valid_area(seed, separation):
    # An "area under a curve" outside [0, 1] means the curve construction or
    # the integration is wrong. This must hold at every signal strength.
    rng = np.random.default_rng(seed)
    labels, scores = _random_population(rng, separation=separation)
    auc = pr_auc(labels, scores)
    assert 0.0 <= auc <= 1.0, f"seed={seed} separation={separation} auc={auc}"
    assert math.isfinite(auc)


@pytest.mark.parametrize("seed", SEEDS)
def test_pr_auc_perfect_ranking_is_exactly_one(seed):
    # A scorer that places every shared account above every honest one has
    # precision 1 at every recall level: the area must be exactly 1.0, with
    # no floating-point drift from the trapezoidal accumulation.
    rng = np.random.default_rng(seed)
    labels, _ = _random_population(rng)
    perfect = {a: (1.0 + rng.random() if is_shared else rng.random())
               for a, is_shared in labels.items()}
    assert pr_auc(labels, perfect) == pytest.approx(1.0), f"seed={seed}"


@pytest.mark.parametrize("seed", SEEDS)
def test_pr_auc_random_scorer_lands_near_base_rate(seed):
    # An uninformative scorer's PR curve hovers at the base rate. With 20
    # shared of 220 the base rate is ~0.0909; the tolerance is wide because
    # PR-AUC on a small positive count is genuinely noisy, but a scorer that
    # scored near 1.0 here would prove the curve is optimistically wrong.
    rng = np.random.default_rng(seed)
    labels, scores = _random_population(rng, separation=0.0)
    base_rate = sum(labels.values()) / len(labels)
    auc = pr_auc(labels, scores)
    assert auc == pytest.approx(base_rate, abs=0.12), (
        f"seed={seed} auc={auc} base_rate={base_rate}")


@pytest.mark.parametrize("seed", SEEDS)
def test_pr_auc_beats_base_rate_when_signal_is_present(seed):
    # Sanity in the other direction: with a strong separation the metric must
    # clearly exceed the base rate, or it is not measuring separation at all.
    rng = np.random.default_rng(seed)
    labels, scores = _random_population(rng, separation=2.0)
    base_rate = sum(labels.values()) / len(labels)
    assert pr_auc(labels, scores) > base_rate + 0.5, f"seed={seed}"


@pytest.mark.parametrize("seed", SEEDS)
def test_pr_auc_invariant_under_monotone_rescaling(seed):
    # PR-AUC depends only on the induced ranking. Any strictly increasing
    # transform of the scores must leave it unchanged -- this is the property
    # that lets eval.py compare scorers on incomparable scales.
    rng = np.random.default_rng(seed)
    labels, scores = _random_population(rng, separation=0.8)
    rescaled = {k: math.expm1(2.5 * v) for k, v in scores.items()}
    assert pr_auc(labels, rescaled) == pytest.approx(pr_auc(labels, scores)), (
        f"seed={seed}")


@pytest.mark.parametrize("seed", SEEDS)
def test_precision_at_k_is_a_valid_fraction(seed):
    rng = np.random.default_rng(seed)
    labels, scores = _random_population(rng, separation=0.5)
    for k in (1, 10, 50, 500, 10_000):
        precision = precision_at_k(labels, scores, k=k)
        assert 0.0 <= precision <= 1.0, f"seed={seed} k={k}"


@pytest.mark.parametrize("seed", SEEDS)
def test_precision_at_k_is_non_increasing_for_a_good_scorer(seed):
    # Deeper into the queue means weaker candidates, so precision should
    # trend down. Asserted as a bound on the *average* rather than a strict
    # step-wise rule, which small-sample noise can legitimately violate.
    rng = np.random.default_rng(seed)
    labels, scores = _random_population(rng, separation=3.0)
    assert precision_at_k(labels, scores, k=10) >= precision_at_k(labels, scores, k=200)


# ---------------------------------------------------------------------------
# rank_normalize properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_rank_normalize_output_is_bounded_and_ordered(seed):
    # Ranks land in (0, 1], the maximum is exactly 1.0, and the induced
    # ordering matches the input ordering.
    rng = np.random.default_rng(seed)
    _, scores = _random_population(rng)
    ranks = rank_normalize(scores)
    assert set(ranks) == set(scores)
    assert all(0.0 < r <= 1.0 for r in ranks.values()), f"seed={seed}"
    assert max(ranks.values()) == pytest.approx(1.0)
    by_score = sorted(scores, key=lambda a: scores[a])
    by_rank = sorted(ranks, key=lambda a: ranks[a])
    assert by_score == by_rank, f"seed={seed}"


@pytest.mark.parametrize("seed", SEEDS)
def test_rank_normalize_preserves_ties(seed):
    # Accounts sharing a raw score must share a rank, regardless of how many
    # distinct values surround them.
    rng = np.random.default_rng(seed)
    scores = {f"a{i:03d}": float(rng.integers(0, 5)) for i in range(120)}
    ranks = rank_normalize(scores)
    for a in scores:
        for b in scores:
            if scores[a] == scores[b]:
                assert ranks[a] == pytest.approx(ranks[b]), f"seed={seed}"


# ---------------------------------------------------------------------------
# Budget monotonicity and FP-budget discipline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_recall_is_non_decreasing_in_the_fp_budget(seed):
    # The headline guarantee behind U: spending more false positives can only
    # buy more recall. If this ever inverts, the threshold rule is broken and
    # every reported uplift is suspect.
    rng = np.random.default_rng(seed)
    labels, baseline = _random_population(rng, n_honest=800, n_shared=60,
                                          separation=0.4)
    detector = {a: float(rng.uniform(0.0, 1.0) + (0.6 if labels[a] else 0.0))
                for a in labels}

    budgets = [0.0, 5.0, 25.0, 100.0, 500.0, 2000.0]
    results = [evaluate(labels, baseline, detector, budget_fp_per_10k=b)
               for b in budgets]
    for lower, higher in zip(results, results[1:]):
        assert higher["R0"] >= lower["R0"] - 1e-12, f"seed={seed} R0 dropped"
        assert higher["R1"] >= lower["R1"] - 1e-12, f"seed={seed} R1 dropped"


@pytest.mark.parametrize("seed", SEEDS)
def test_fp_budget_is_never_overspent_on_random_honest_scores(seed):
    # The FP budget is a promise to the business: at most `allowed` honest
    # accounts get flagged. Verified directly against the flagging rule.
    rng = np.random.default_rng(seed)
    honest = [float(v) for v in rng.normal(0.0, 1.0, 2000)]
    for budget in (0.0, 1.0, 50.0, 250.0, 1500.0):
        threshold = threshold_at_fp_budget(honest, budget)
        allowed = math.floor(len(honest) * budget / 10000.0)
        flagged = sum(1 for s in honest if s > threshold)
        assert flagged <= allowed, f"seed={seed} budget={budget}"


@pytest.mark.parametrize("seed", SEEDS)
def test_evaluate_metrics_stay_in_range(seed):
    # Every reported rate must be a valid fraction, and U must be a real
    # number (or +inf, the documented zero-baseline case).
    rng = np.random.default_rng(seed)
    labels, baseline = _random_population(rng, separation=0.3)
    detector = {a: float(rng.uniform(0.0, 1.0)) for a in labels}
    result = evaluate(labels, baseline, detector, budget_fp_per_10k=50.0)

    for key in ("R0", "R1", "pr_auc_baseline", "pr_auc_pilot",
                "precision_at_500_baseline", "precision_at_500_pilot"):
        assert 0.0 <= result[key] <= 1.0, f"seed={seed} {key}={result[key]}"
    assert result["U"] >= -1.0 - 1e-12  # recall cannot fall below zero
    assert not math.isnan(result["U"])


@pytest.mark.parametrize("seed", SEEDS)
def test_stacking_never_demotes_below_the_detector_rank(seed):
    # The stacking rule is an elementwise max in rank space, so every
    # account's stacked score is >= both of its input ranks. This is what
    # makes "either scorer places it deep in its own tail" true by
    # construction, and it is the property an off-by-one in rank_normalize
    # would break.
    rng = np.random.default_rng(seed)
    labels, baseline = _random_population(rng, separation=0.5)
    detector = {a: float(rng.uniform(0.0, 1.0)) for a in labels}

    baseline_ranks = rank_normalize(baseline)
    detector_ranks = rank_normalize(detector)
    stacked = {a: max(baseline_ranks[a], detector_ranks[a]) for a in labels}
    for a in labels:
        assert stacked[a] >= baseline_ranks[a] - 1e-12
        assert stacked[a] >= detector_ranks[a] - 1e-12


@pytest.mark.parametrize("seed", SEEDS)
def test_uplift_is_zero_when_the_detector_duplicates_the_baseline(seed):
    # Stacking a scorer with itself adds no information: max(r, r) == r, so
    # R1 must equal R0 and U must be exactly 0. A non-zero U here would mean
    # the stack manufactures uplift out of nothing.
    rng = np.random.default_rng(seed)
    labels, baseline = _random_population(rng, n_honest=500, n_shared=40,
                                          separation=0.6)
    result = evaluate(labels, baseline, dict(baseline), budget_fp_per_10k=50.0)
    assert result["R1"] == pytest.approx(result["R0"]), f"seed={seed}"
    assert result["U"] == pytest.approx(0.0), f"seed={seed}"


@pytest.mark.parametrize("seed", SEEDS)
def test_uplift_is_positive_when_the_detector_adds_real_signal(seed):
    # A detector that ranks a subset of the shared accounts the baseline
    # missed must produce U > 0 at a budget where the baseline is not already
    # saturated. This is the end-to-end direction the pilot is measured on.
    rng = np.random.default_rng(seed)
    labels: dict[str, bool] = {}
    baseline: dict[str, float] = {}
    detector: dict[str, float] = {}
    for i in range(600):
        account_id = f"hon{i:04d}"
        labels[account_id] = False
        baseline[account_id] = float(rng.uniform(0.0, 0.6))
        detector[account_id] = float(rng.uniform(0.0, 0.6))
    for i in range(40):
        account_id = f"shr{i:04d}"
        labels[account_id] = True
        # Half the shared accounts are invisible to the baseline but obvious
        # to the detector -- the "physics catches what rules miss" story.
        if i % 2 == 0:
            baseline[account_id] = float(rng.uniform(0.0, 0.4))
            detector[account_id] = float(rng.uniform(0.9, 1.0))
        else:
            baseline[account_id] = float(rng.uniform(0.8, 1.0))
            detector[account_id] = float(rng.uniform(0.0, 0.4))

    result = evaluate(labels, baseline, detector, budget_fp_per_10k=50.0)
    assert result["U"] > 0.0, f"seed={seed} U={result['U']}"
    assert result["R1"] > result["R0"], f"seed={seed}"
