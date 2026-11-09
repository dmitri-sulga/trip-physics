"""Invariants that must survive every future change to the pipeline.

The other suites pin *values*: golden metrics, hand-computed fixtures,
per-feature geometry. Values are expected to move when the detector is
recalibrated or the evaluation methodology is sharpened. The properties in
this module are different in kind -- they are the claims that make the
harness trustworthy at all, and none of them may ever be renegotiated by a
tuning pass:

* **No label leakage.** Feature extraction and both scorers never read
  ``is_shared``. Inverting every label in the corpus must leave features and
  scores bit-identical. If this test ever fails, every uplift number the
  harness has ever produced is void -- which is why it is asserted bitwise
  and not with a tolerance.
* **Determinism.** A seed fully determines a run: same seed, bit-identical
  arrays; no dependence on global numpy state or on dict insertion order.
* **Monotonicity.** Recall at the budgeted threshold never *decreases* when
  the false-positive budget grows. A non-monotone budget curve means the
  operating point could be gamed by widening the budget.
* **Budget discipline.** The threshold never lets through more honest
  accounts than the budget allows, on the population the threshold was set
  on. Over-spending here would make every reported R0/R1 pair a comparison
  at two different budgets -- the exact failure the harness exists to avoid.

Adversarial robustness lives here too: an account must not be able to lower
its own suspicion by emitting garbage (see the NaN-injection tests). That is
a security property, not a quality metric.
"""

import copy
import math

import numpy as np
import pytest

from tripphysics.baseline import NaiveBaseline
from tripphysics.detector import score_account
from tripphysics.eval import (
    evaluate,
    pr_auc,
    precision_at_k,
    rank_normalize,
    recall_at_threshold,
    threshold_at_fp_budget,
)
from tripphysics.features import FEATURE_NAMES, extract_features
from tripphysics.synth import cell_id_from_xy, generate_dataset


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _scored_corpus(n_accounts=120, shared_fraction=0.25, seed=5):
    """Generate a corpus and return (accounts, labels, baseline, detector)."""
    accounts = generate_dataset(n_accounts=n_accounts,
                                shared_fraction=shared_fraction,
                                seed=seed)["accounts"]
    labels = {a["account_id"]: a["is_shared"] for a in accounts}
    naive = NaiveBaseline()
    baseline_scores = {a["account_id"]: naive.score(a) for a in accounts}
    detector_scores = {a["account_id"]: score_account(extract_features(a))
                       for a in accounts}
    return accounts, labels, baseline_scores, detector_scores


def _feature_matrix(accounts) -> np.ndarray:
    """Every account's feature vector as one float64 array, stable order."""
    return np.array(
        [[extract_features(a)[k] for k in FEATURE_NAMES] for a in accounts],
        dtype=float,
    )


def _invert_labels(accounts):
    """Deep copy of the corpus with every is_shared flag flipped."""
    flipped = copy.deepcopy(accounts)
    for account in flipped:
        account["is_shared"] = not account["is_shared"]
    return flipped


# ---------------------------------------------------------------------------
# THE critical invariant: no label leakage
#
# extract_features receives the whole account dict, is_shared included. The
# only thing stopping a future edit from reading it is a test that fails
# loudly when it does.
# ---------------------------------------------------------------------------

def test_inverting_every_label_does_not_move_a_single_feature_bit():
    accounts, *_ = _scored_corpus(n_accounts=150, shared_fraction=0.3, seed=5)
    before = _feature_matrix(accounts)
    after = _feature_matrix(_invert_labels(accounts))
    assert before.shape == after.shape
    # np.array_equal, not approx: leakage that shifts a feature by 1e-12 is
    # still leakage, and a tolerance would hide it.
    assert np.array_equal(before, after), (
        "LABEL LEAKAGE: feature extraction changed when is_shared was "
        "inverted. Every uplift number produced by this harness is invalid "
        "until this is fixed.")


def test_inverting_every_label_does_not_move_a_single_detector_score_bit():
    accounts, *_ = _scored_corpus(n_accounts=150, shared_fraction=0.3, seed=5)
    before = np.array([score_account(extract_features(a)) for a in accounts])
    after = np.array([score_account(extract_features(a))
                      for a in _invert_labels(accounts)])
    assert np.array_equal(before, after), (
        "LABEL LEAKAGE: the detector scored differently after labels were "
        "inverted.")


def test_inverting_every_label_does_not_move_a_single_baseline_score_bit():
    # The baseline is the comparison arm; leakage there would inflate R0 and
    # understate uplift, which is the *flattering* direction to get wrong.
    accounts, *_ = _scored_corpus(n_accounts=150, shared_fraction=0.3, seed=5)
    naive = NaiveBaseline()
    before = np.array([naive.score(a) for a in accounts])
    after = np.array([naive.score(a) for a in _invert_labels(accounts)])
    assert np.array_equal(before, after), (
        "LABEL LEAKAGE: NaiveBaseline read the label.")


def test_features_are_identical_when_the_label_key_is_absent_entirely():
    # Strongest form of the claim: the key does not need to exist. A real
    # production export of unlabelled accounts must extract identically.
    accounts, *_ = _scored_corpus(n_accounts=40, shared_fraction=0.5, seed=6)
    stripped = copy.deepcopy(accounts)
    for account in stripped:
        account.pop("is_shared", None)
    assert np.array_equal(_feature_matrix(accounts), _feature_matrix(stripped))


def test_evaluate_uses_labels_only_through_the_labels_argument():
    # evaluate() must take ground truth exclusively from `labels`. Feeding it
    # accounts whose embedded is_shared flags are inverted must not change a
    # single metric, since the scores were computed label-blind.
    accounts, labels, baseline, detector = _scored_corpus(
        n_accounts=200, shared_fraction=0.2, seed=8)
    flipped_accounts = _invert_labels(accounts)
    naive = NaiveBaseline()
    baseline_after = {a["account_id"]: naive.score(a) for a in flipped_accounts}
    detector_after = {a["account_id"]: score_account(extract_features(a))
                      for a in flipped_accounts}

    before = evaluate(labels, baseline, detector)
    after = evaluate(labels, baseline_after, detector_after)
    assert before == after


# ---------------------------------------------------------------------------
# Determinism: the seed is the whole state
# ---------------------------------------------------------------------------

def test_two_runs_at_one_seed_give_bit_identical_feature_matrices():
    first = _feature_matrix(
        generate_dataset(n_accounts=60, shared_fraction=0.2, seed=21)["accounts"])
    second = _feature_matrix(
        generate_dataset(n_accounts=60, shared_fraction=0.2, seed=21)["accounts"])
    assert np.array_equal(first, second)


def test_scoring_is_independent_of_global_numpy_state():
    # Nothing in the scoring path may draw from np.random. If it did, an
    # unrelated caller's RNG use would silently change reported scores.
    accounts, *_ = _scored_corpus(n_accounts=40, shared_fraction=0.2, seed=12)
    np.random.seed(0)
    first = np.array([score_account(extract_features(a)) for a in accounts])
    np.random.seed(0)
    np.random.random(5000)  # perturb the legacy global state
    second = np.array([score_account(extract_features(a)) for a in accounts])
    assert np.array_equal(first, second)


def test_evaluate_is_independent_of_dict_insertion_order():
    # Dicts preserve insertion order in Python; every metric here sorts keys
    # explicitly so that an export written in a different order evaluates to
    # the same numbers. Reversing insertion order must be a no-op.
    _, labels, baseline, detector = _scored_corpus(
        n_accounts=200, shared_fraction=0.2, seed=14)
    reversed_labels = dict(reversed(list(labels.items())))
    reversed_baseline = dict(reversed(list(baseline.items())))
    reversed_detector = dict(reversed(list(detector.items())))

    assert evaluate(labels, baseline, detector) == evaluate(
        reversed_labels, reversed_baseline, reversed_detector)


def test_rank_normalize_is_independent_of_dict_insertion_order():
    scores = {"c": 0.3, "a": 0.9, "b": 0.3, "d": 0.1}
    shuffled = {"d": 0.1, "b": 0.3, "a": 0.9, "c": 0.3}
    assert rank_normalize(scores) == rank_normalize(shuffled)


def test_feature_extraction_does_not_mutate_its_input():
    # extract_features sorts the trip list; it must sort a copy. A caller
    # that re-reads the account afterwards has to see the original order.
    accounts, *_ = _scored_corpus(n_accounts=20, shared_fraction=0.3, seed=15)
    account = accounts[0]
    before = copy.deepcopy(account)
    extract_features(account)
    assert account == before


# ---------------------------------------------------------------------------
# Monotonicity of recall in the false-positive budget
# ---------------------------------------------------------------------------

BUDGET_LADDER = [0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0]


def test_threshold_never_rises_as_the_budget_grows():
    # The primitive behind the property: a bigger budget buys a lower bar.
    _, labels, baseline, _ = _scored_corpus(n_accounts=400, shared_fraction=0.1,
                                            seed=16)
    honest = [baseline[a] for a, shared in labels.items() if not shared]
    thresholds = [threshold_at_fp_budget(honest, b) for b in BUDGET_LADDER]
    for lower, higher in zip(thresholds, thresholds[1:]):
        assert higher <= lower, (
            f"threshold rose with the budget: {lower} -> {higher}")


def test_recall_is_non_decreasing_in_the_fp_budget():
    _, labels, baseline, detector = _scored_corpus(n_accounts=400,
                                                   shared_fraction=0.1, seed=16)
    r0_curve, r1_curve = [], []
    for budget in BUDGET_LADDER:
        result = evaluate(labels, baseline, detector, budget_fp_per_10k=budget)
        r0_curve.append(result["R0"])
        r1_curve.append(result["R1"])

    for name, curve in (("R0", r0_curve), ("R1", r1_curve)):
        for lower, higher in zip(curve, curve[1:]):
            assert higher >= lower - 1e-12, (
                f"{name} fell when the budget grew: {curve}")


def test_recall_at_threshold_is_non_increasing_in_the_threshold():
    labels = {f"s{i}": True for i in range(20)}
    labels.update({f"h{i}": False for i in range(80)})
    scores = {a: (i / 100.0) for i, a in enumerate(sorted(labels))}
    recalls = [recall_at_threshold(scores, labels, t)
               for t in np.linspace(-0.1, 1.1, 25)]
    for higher, lower in zip(recalls, recalls[1:]):
        assert lower <= higher + 1e-12


# ---------------------------------------------------------------------------
# Budget discipline: the promise the whole comparison rests on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("budget", [0.0, 1.0, 13.0, 50.0, 137.0, 500.0, 999.0])
def test_fp_budget_is_never_overspent_on_the_calibration_population(budget):
    # The threshold is set on the honest population; on that same population
    # the number of accounts flagged must never exceed floor(n * budget/10k).
    # Under-spending is allowed (ties at the threshold are not flagged).
    _, labels, baseline, detector = _scored_corpus(n_accounts=600,
                                                   shared_fraction=0.08, seed=18)
    honest_ids = [a for a, shared in labels.items() if not shared]
    allowed = math.floor(len(honest_ids) * budget / 10000.0)

    for name, scores in (("baseline", baseline), ("detector", detector)):
        threshold = threshold_at_fp_budget([scores[a] for a in honest_ids], budget)
        flagged = sum(1 for a in honest_ids if scores[a] > threshold)
        assert flagged <= allowed, (
            f"{name}: flagged {flagged} honest accounts at budget {budget}, "
            f"allowed {allowed}")


def test_stacked_pilot_also_respects_the_budget():
    # R1 is measured on the *stacked* scorer, so budget discipline has to
    # hold after stacking -- that is the arm the headline number comes from.
    _, labels, baseline, detector = _scored_corpus(n_accounts=600,
                                                   shared_fraction=0.08, seed=18)
    honest_ids = [a for a, shared in labels.items() if not shared]
    budget = 50.0
    stacked_ranks_b = rank_normalize(baseline)
    stacked_ranks_d = rank_normalize(detector)
    stacked = {a: max(stacked_ranks_b[a], stacked_ranks_d[a]) for a in labels}

    threshold = threshold_at_fp_budget([stacked[a] for a in honest_ids], budget)
    flagged = sum(1 for a in honest_ids if stacked[a] > threshold)
    assert flagged <= math.floor(len(honest_ids) * budget / 10000.0)


def test_zero_budget_flags_no_honest_account():
    # The strictest operating point: with no budget at all, not one honest
    # account may be flagged on the calibration population.
    _, labels, baseline, _ = _scored_corpus(n_accounts=300, shared_fraction=0.1,
                                            seed=19)
    honest_ids = [a for a, shared in labels.items() if not shared]
    threshold = threshold_at_fp_budget([baseline[a] for a in honest_ids], 0.0)
    assert sum(1 for a in honest_ids if baseline[a] > threshold) == 0


def test_both_arms_are_evaluated_at_the_same_budget():
    # The headline claim of the harness: R0 and R1 come from one budget. If
    # evaluate() ever thresholded the two arms differently, U would be a
    # comparison between two operating points and therefore meaningless.
    _, labels, baseline, detector = _scored_corpus(n_accounts=400,
                                                   shared_fraction=0.1, seed=20)
    for budget in (10.0, 50.0, 200.0):
        result = evaluate(labels, baseline, detector, budget_fp_per_10k=budget)
        assert result["budget_fp_per_10k"] == budget
        honest_ids = [a for a, shared in labels.items() if not shared]
        allowed = math.floor(len(honest_ids) * budget / 10000.0)
        flagged_baseline = sum(1 for a in honest_ids
                               if baseline[a] > result["threshold_baseline"])
        assert flagged_baseline <= allowed


# ---------------------------------------------------------------------------
# Adversarial robustness: garbage must never buy a lower score
#
# These are security tests. An evasion that works by degrading data quality
# is worse than a missed detection, because it is cheap and repeatable.
# ---------------------------------------------------------------------------

def _fix(t, x, y, speed=10.0, is_mock=False, cell_id=None):
    return {
        "t": float(t),
        "x_m": float(x),
        "y_m": float(y),
        "accuracy_m": 5.0,
        "speed_mps": float(speed),
        "is_mock": bool(is_mock),
        "cell_id": int(cell_id_from_xy(x, y)) if cell_id is None else int(cell_id),
    }


def _account(trips, account_id="adv"):
    return {"account_id": account_id, "is_shared": False, "trips": trips}


def _trip(fixes, trip_id="t0"):
    return {
        "trip_id": trip_id,
        "account_id": "adv",
        "t_start": fixes[0]["t"] if fixes else 0.0,
        "fixes": fixes,
    }


def _teleporting_trips(poison=None):
    """Two trips whose endpoints teleport 40 km apart in 60 s.

    With ``poison`` set, one extra fix carrying that non-finite value is
    appended to the first trip -- the "emit garbage" evasion.
    """
    first = [_fix(0.0, 0.0, 0.0), _fix(10.0, 100.0, 0.0), _fix(20.0, 200.0, 0.0)]
    if poison is not None:
        first.append(_fix(30.0, poison, 0.0))
    second = [_fix(80.0, 40_000.0, 0.0), _fix(90.0, 40_100.0, 0.0),
              _fix(100.0, 40_200.0, 0.0)]
    return [_trip(first, "t0"), _trip(second, "t1")]


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@pytest.mark.parametrize("poison", [float("inf"), float("-inf")])
def test_injecting_an_infinite_fix_never_lowers_the_handover_signal(poison):
    # The inter-trip teleport is measured between trip endpoints. Appending
    # one poisoned fix must not erase it: an adversary who can zero out a
    # feature by emitting garbage has a free evasion. +/-inf is safe here
    # because the excess distance saturates at HANDOVER_SAT_M; NaN is not,
    # which is DEFECT-4 below.
    clean = extract_features(_account(_teleporting_trips(None)))
    poisoned = extract_features(_account(_teleporting_trips(poison)))
    assert poisoned["handover_gap_score"] >= clean["handover_gap_score"] - 1e-12, (
        f"EVASION: a single {poison} fix lowered handover_gap_score from "
        f"{clean['handover_gap_score']} to {poisoned['handover_gap_score']}")


# DEFECT-4 (FIXED): a single NaN coordinate used to ERASE the *inter-trip*
# handover signal -- the poisoned fix became the trip endpoint, np.hypot
# returned NaN for the gap distance, np.mean propagated it across all gaps,
# and the "finite or 0.0" guard rewrote the aggregate to the cleanest
# possible value. Measured on a 40 km inter-trip teleport at the time:
# handover_gap_score 1.0 -> 0.0, score 0.2935 -> 0.1215, a 59% drop in
# suspicion bought with one garbage fix.
#
# Non-finite fixes are now discarded BEFORE any differencing, so they can no
# longer poison an aggregate, and the discard rate is surfaced as its own
# feature (nonfinite_rate) rather than passing silently.

@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_injecting_a_nan_fix_never_lowers_the_handover_signal():
    clean = extract_features(_account(_teleporting_trips(None)))
    poisoned = extract_features(_account(_teleporting_trips(float("nan"))))
    assert poisoned["handover_gap_score"] >= clean["handover_gap_score"] - 1e-12, (
        "EVASION: a single NaN fix lowered the inter-trip handover signal")


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_the_teleport_survives_nan_injection_intact():
    # The positive statement of the fix: the signal is not merely "not
    # lowered", it is preserved at full strength, because the bad fix is
    # dropped and the surviving endpoints still show the 40 km jump.
    clean = extract_features(_account(_teleporting_trips(None)))
    poisoned = extract_features(_account(_teleporting_trips(float("nan"))))
    assert clean["handover_gap_score"] == pytest.approx(1.0)
    assert poisoned["handover_gap_score"] == pytest.approx(1.0)


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_emitting_garbage_raises_suspicion_rather_than_lowering_it():
    # The design decision behind the fix: unparseable telemetry is itself an
    # integrity smell. An account that emits a NaN must end up scoring at
    # least as high as the same account without one -- otherwise the evasion
    # is merely made harder rather than removed.
    clean = extract_features(_account(_teleporting_trips(None)))
    poisoned = extract_features(_account(_teleporting_trips(float("nan"))))
    assert poisoned["nonfinite_rate"] > 0.0
    assert clean["nonfinite_rate"] == 0.0
    assert score_account(poisoned) >= score_account(clean)


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@pytest.mark.parametrize("poison", [float("nan"), float("inf"), float("-inf")])
def test_every_non_finite_flavour_is_counted_in_the_discard_rate(poison):
    # NaN, +inf and -inf must all be treated as unparseable. Counting only
    # NaN would leave the same evasion open through the infinities.
    #
    # The suppressed overflow warnings come from the fixture, not the code
    # under test: make_fix precomputes a cell_id via cell_id_from_xy, which
    # is deliberately permissive about non-finite coordinates (see its
    # docstring). extract_features discards the fix before any cell join, so
    # the warning is an artifact of building the poisoned input.
    features = extract_features(_account(_teleporting_trips(poison)))
    assert features["nonfinite_rate"] > 0.0, (
        f"{poison} was not counted as a discarded fix")


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@pytest.mark.parametrize("poison", [float("nan"), float("inf"), float("-inf")])
def test_features_stay_finite_and_complete_under_poisoned_input(poison):
    # Whatever the fix does to the *value*, the output contract holds: exact
    # key set, finite floats. A downstream consumer must never see a NaN.
    features = extract_features(_account(_teleporting_trips(poison)))
    assert set(features) == set(FEATURE_NAMES)
    for key, value in features.items():
        assert isinstance(value, float)
        assert math.isfinite(value), f"{key} is not finite under {poison}"


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_poisoned_account_still_receives_a_valid_score():
    score = score_account(extract_features(_account(_teleporting_trips(float("nan")))))
    assert 0.0 <= score <= 1.0
    assert math.isfinite(score)


def test_mock_flags_cannot_be_diluted_below_the_saturation_by_padding():
    # A rate feature can in principle be gamed by padding with clean fixes.
    # mock_rate saturates at 5% in the detector, so an account with a burst
    # of mock fixes must stay saturated under realistic padding. This pins
    # the intent: the evasion has to be expensive, not one extra trip.
    mock_fixes = [_fix(i * 10.0, i * 100.0, 0.0, is_mock=True) for i in range(20)]
    clean_fixes = [_fix(1000.0 + i * 10.0, i * 100.0, 0.0) for i in range(100)]
    burst_only = extract_features(_account([_trip(mock_fixes, "t0")]))
    padded = extract_features(
        _account([_trip(mock_fixes, "t0"), _trip(clean_fixes, "t1")]))

    assert burst_only["mock_rate"] == pytest.approx(1.0)
    # 20 mock of 120 total = 16.7%, still far above the 5% saturation point.
    assert padded["mock_rate"] > 0.05


# ---------------------------------------------------------------------------
# Metric well-formedness: results are always in range and JSON-safe
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [3, 11, 29])
def test_all_evaluate_outputs_are_in_their_declared_ranges(seed):
    _, labels, baseline, detector = _scored_corpus(n_accounts=250,
                                                   shared_fraction=0.12, seed=seed)
    result = evaluate(labels, baseline, detector)

    for key in ("R0", "R1", "pr_auc_baseline", "pr_auc_pilot",
                "precision_at_500_baseline", "precision_at_500_pilot"):
        assert 0.0 <= result[key] <= 1.0, f"{key} out of range: {result[key]}"
    assert result["n_accounts"] == 250
    assert 0 <= result["n_shared"] <= result["n_accounts"]
    assert math.isfinite(result["U"]) or result["U"] == math.inf


@pytest.mark.parametrize("seed", [3, 11, 29])
def test_evaluate_output_is_json_serialisable(seed):
    # The result dict is written straight into the shipped JSON artifact;
    # numpy scalars would break json.dump at report time, not at test time.
    import json

    _, labels, baseline, detector = _scored_corpus(n_accounts=120,
                                                   shared_fraction=0.15, seed=seed)
    result = evaluate(labels, baseline, detector)
    for key, value in result.items():
        assert type(value) in (float, int), f"{key} is {type(value)}, not a builtin"
    json.dumps(result)


def test_detector_score_is_bounded_for_every_account_in_a_corpus():
    accounts, *_ = _scored_corpus(n_accounts=200, shared_fraction=0.2, seed=31)
    for account in accounts:
        score = score_account(extract_features(account))
        assert 0.0 <= score <= 1.0


def test_pr_auc_and_precision_are_bounded_on_random_scorers():
    # A scorer with no relationship to the labels still has to produce a
    # valid area and a valid precision -- bounds are a contract, not a
    # consequence of the detector working.
    rng = np.random.default_rng(7)
    labels = {f"a{i}": bool(i % 7 == 0) for i in range(200)}
    scores = {a: float(rng.random()) for a in labels}
    assert 0.0 <= pr_auc(labels, scores) <= 1.0
    assert 0.0 <= precision_at_k(labels, scores, k=50) <= 1.0
