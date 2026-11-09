"""Calibration of the features against the population they claim to describe.

A feature can be arithmetically correct and still be *miscalibrated*: it
computes what its docstring says, but the number it hands the detector does
not mean what the detector assumes it means. That failure is invisible to
unit tests over hand-built fixtures, because every fixture is checked against
the same wrong intuition that produced the constant.

The guard against it is to measure each feature on the population it is meant
to separate, and to state, as an assertion, what the feature must look like
on honest accounts and on shared ones. Two properties are pinned here:

* **A null baseline that is actually null.** ``style_bimodality`` subtracts a
  constant so that "a single driver" reads ~0. If that constant is set below
  the score an honest unimodal sample really gets, every honest account
  arrives at the detector carrying standing evidence of a second driver.
* **Discriminative power proportional to weight.** A feature weighted as
  heavily as the physics signals has to separate the classes about as well.
  A near-coin-flip feature with a heavy weight is a noise amplifier.

Both are asserted as ranges over a generated corpus, not as pinned values:
the point is the property, and the tolerances are wide enough that an honest
recalibration passes while the miscalibration this file was written against
does not.
"""

import math

import numpy as np
import pytest

from tripphysics.detector import (
    W_CELL_MISMATCH,
    W_HANDOVER_GAP,
    W_MOCK_RATE,
    W_OVERLAP,
    W_STYLE_BIMODALITY,
)
from tripphysics.features import FEATURE_NAMES, _split_separation, extract_features
from tripphysics.synth import generate_dataset


# ---------------------------------------------------------------------------
# Corpora: an all-honest population and a labelled mixed one
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def honest_features():
    """Features for a purely honest corpus -- the null population."""
    accounts = generate_dataset(n_accounts=1200, shared_fraction=0.0,
                                seed=101)["accounts"]
    return [extract_features(a) for a in accounts]


@pytest.fixture(scope="module")
def labelled_features():
    """(honest, shared) feature lists from one balanced corpus."""
    accounts = generate_dataset(n_accounts=1600, shared_fraction=0.5,
                                seed=202)["accounts"]
    honest, shared = [], []
    for account in accounts:
        (shared if account["is_shared"] else honest).append(
            extract_features(account))
    return honest, shared


def _column(features, key) -> np.ndarray:
    return np.array([f[key] for f in features], dtype=float)


def _rank_auc(negative: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUC: P(positive ranks above negative), ties at 0.5.

    Hand-rolled (average ranks via argsort) so the suite keeps its numpy-only
    dependency footprint. 0.5 = no separation, 1.0 = perfect.
    """
    combined = np.concatenate([negative, positive])
    order = combined.argsort(kind="stable")
    ranks = np.empty(combined.size, dtype=float)
    sorted_values = combined[order]
    i = 0
    while i < combined.size:
        j = i
        while j + 1 < combined.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum_positive = ranks[negative.size:].sum()
    n_neg, n_pos = negative.size, positive.size
    return (rank_sum_positive - n_pos * (n_pos + 1) / 2.0) / (n_neg * n_pos)


# ---------------------------------------------------------------------------
# style_bimodality: the null baseline must actually be null
#
# The feature standardises the raw split statistic against the null a
# single-mode sample of the same size produces. If that null is set too low,
# honest accounts inherit standing evidence of a second driver.
#
# DEFECT-5 (FIXED): the null used to be a flat constant 0.55, well below the
# ~0.64 a genuinely unimodal sample scores, so honest single-driver accounts
# read ~0.23 bimodality instead of ~0. It is now standardised against a
# size-aware null (mean and sd both functions of n), which additionally puts
# small and large accounts on one scale -- a flat offset could not, because
# the null sd shrinks as 1/sqrt(n).
#
# Note on the statistic used below: the feature clips negative z to zero, so
# the null distribution is one-sided with roughly half its mass exactly at 0.
# The MEDIAN is therefore the right test of centring; the mean of a clipped
# one-sided variable is positive by construction and would fail even for a
# perfectly calibrated feature.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_trips", [6, 7, 10, 12, 20, 30, 50])
def test_unimodal_samples_are_centred_at_zero_for_every_sample_size(n_trips):
    # A sample drawn from ONE Gaussian has no second mode, so the typical
    # such sample must read exactly "no evidence". Checked per sample size:
    # a null that is right on average but wrong per size would let small
    # accounts alarm at a different rate than large ones. Sizes include both
    # parities, since an even n admits an exactly balanced split and an odd
    # one does not -- the null mean differs measurably between them.
    rng = np.random.default_rng(100 + n_trips)
    scores = np.array([_split_separation(rng.standard_normal(n_trips))
                       for _ in range(800)])
    median = float(np.median(scores))
    assert median < 0.10, (
        f"n={n_trips}: median unimodal bimodality is {median:.4f}, not ~0. "
        f"The null baseline is set below the score a single-mode sample "
        f"really achieves, so honest accounts carry standing evidence of a "
        f"second driver.")


@pytest.mark.parametrize("n_trips", [6, 10, 20, 40])
def test_the_unimodal_false_alarm_tail_is_bounded_for_every_sample_size(n_trips):
    # Centring alone is not calibration: the tail matters, because that is
    # where the detector's evidence transform lives. A single driver must
    # only rarely produce a high reading.
    rng = np.random.default_rng(200 + n_trips)
    scores = np.array([_split_separation(rng.standard_normal(n_trips))
                       for _ in range(1500)])
    assert float((scores > 0.5).mean()) < 0.15, (
        f"n={n_trips}: {(scores > 0.5).mean():.1%} of single-driver samples "
        f"exceed 0.5")
    assert float((scores >= 1.0).mean()) < 0.02, (
        f"n={n_trips}: single-driver samples saturate the feature too often")


def test_the_null_is_stable_across_sample_sizes():
    # The point of standardising rather than subtracting a constant: a
    # 6-trip account and a 40-trip account must have comparable false-alarm
    # rates. A flat offset cannot achieve this, since the null sd shrinks as
    # 1/sqrt(n) while the offset stays put.
    rates = []
    for n_trips in (6, 10, 20, 40):
        rng = np.random.default_rng(300 + n_trips)
        scores = np.array([_split_separation(rng.standard_normal(n_trips))
                           for _ in range(1500)])
        rates.append(float((scores > 0.5).mean()))
    assert max(rates) - min(rates) < 0.10, (
        f"false-alarm rate varies too much with trip count: {rates}")


def test_clearly_bimodal_samples_score_high():
    # The other side of the calibration: a genuine two-mode sample must land
    # near the top of the range, or the feature is not just unbiased but deaf.
    rng = np.random.default_rng(1)
    scores = []
    for n in (10, 20, 40):
        for _ in range(100):
            left = rng.standard_normal(n // 2) * 0.3 - 3.0
            right = rng.standard_normal(n - n // 2) * 0.3 + 3.0
            scores.append(_split_separation(np.concatenate([left, right])))
    mean_score = float(np.mean(scores))
    assert mean_score > 0.80, (
        f"clearly bimodal samples average only {mean_score:.4f}")


def test_bimodality_separates_one_mode_from_two_by_a_wide_margin():
    # Stated as a gap rather than two absolute levels, so the test survives a
    # change of scale but not a change of ordering.
    rng = np.random.default_rng(2)
    unimodal = [_split_separation(rng.standard_normal(20)) for _ in range(200)]
    bimodal = []
    for _ in range(200):
        left = rng.standard_normal(10) * 0.3 - 3.0
        right = rng.standard_normal(10) * 0.3 + 3.0
        bimodal.append(_split_separation(np.concatenate([left, right])))
    assert float(np.mean(bimodal)) - float(np.mean(unimodal)) > 0.60


def test_honest_population_bimodality_is_centred_near_zero(honest_features):
    # The corpus-level consequence: on a population containing no shared
    # accounts at all, the median honest account must read essentially "one
    # driver". A median far above zero is the miscalibration in situ.
    values = _column(honest_features, "style_bimodality")
    median = float(np.median(values))
    assert median < 0.10, (
        f"median honest style_bimodality is {median:.4f}; on an all-honest "
        f"corpus the typical account should read ~0")


def test_most_honest_accounts_are_below_the_detector_midpoint(honest_features):
    # The detector's soft threshold puts its 0.5 evidence point at
    # BIMODALITY_CENTER. The large majority of honest accounts must sit
    # below it, or the feature contributes evidence by default. The bar is
    # set at 10% rather than something tighter because this feature is
    # explicitly a weak, confounded signal carrying a small weight -- what
    # would be intolerable is the *typical* honest account clearing the
    # midpoint, which the median assertion above rules out.
    from tripphysics.detector import BIMODALITY_CENTER

    values = _column(honest_features, "style_bimodality")
    fraction_above = float((values > BIMODALITY_CENTER).mean())
    assert fraction_above < 0.10, (
        f"{fraction_above:.1%} of honest accounts exceed the detector's "
        f"bimodality midpoint {BIMODALITY_CENTER}")


def test_bimodality_stays_within_its_declared_range(honest_features):
    values = _column(honest_features, "style_bimodality")
    assert values.min() >= 0.0
    assert values.max() <= 1.0


def test_bimodality_is_zero_for_accounts_with_too_few_trips():
    # Below the minimum trip count a two-way split is pure noise; the feature
    # is defined to return exactly 0.0 rather than a small random number.
    from tripphysics.features import _MIN_TRIPS_BIMODALITY

    rng = np.random.default_rng(3)
    for n in range(0, _MIN_TRIPS_BIMODALITY):
        assert _split_separation(rng.standard_normal(n)) == 0.0


def test_bimodality_of_a_constant_sample_is_zero():
    # Zero variance: there is no split to find, and the guard must return 0.0
    # rather than dividing by a zero total variance.
    assert _split_separation(np.zeros(20)) == 0.0
    assert _split_separation(np.full(20, 7.5)) == 0.0


# ---------------------------------------------------------------------------
# Feature-level discriminative power
#
# Each feature is measured against the labels it is supposed to separate. The
# thresholds are deliberately loose: the claim is "this feature carries
# signal", not "this feature carries exactly this much signal".
# ---------------------------------------------------------------------------

def test_the_physics_features_separate_the_classes(labelled_features):
    # The signals the detector weights most heavily -- the ones a single
    # honest driver cannot produce -- must be the ones that actually
    # separate. If any of these degrades to a coin flip, the detector is
    # relying on a signal that no longer exists.
    honest, shared = labelled_features
    for key, floor in (("handover_gap_score", 0.60),
                       ("overlap_rate", 0.60),
                       ("max_jump_m", 0.65),
                       ("mock_rate", 0.65),
                       ("cell_mismatch_rate", 0.65)):
        auc = _rank_auc(_column(honest, key), _column(shared, key))
        assert auc > floor, f"{key} separates at AUC {auc:.4f}, below {floor}"


def test_no_feature_is_anti_predictive(labelled_features):
    # A feature whose AUC falls below 0.5 points the wrong way: shared
    # accounts score *lower* on it than honest ones. The detector treats
    # every feature as monotone-increasing evidence, so an anti-predictive
    # feature actively subtracts signal.
    #
    # nonfinite_rate is excluded by construction, not by convenience: the
    # synthetic generator emits no malformed telemetry, so the feature is
    # identically zero on both classes and its AUC is exactly 0.5. It exists
    # to close an evasion vector on real data (see the NaN-injection tests),
    # and a corpus that never exercises it cannot say anything about its
    # direction either way.
    honest, shared = labelled_features
    for key in FEATURE_NAMES:
        if key == "nonfinite_rate":
            continue
        auc = _rank_auc(_column(honest, key), _column(shared, key))
        assert auc >= 0.48, (
            f"{key} is anti-predictive (AUC {auc:.4f}): shared accounts "
            f"score lower on it than honest accounts do")


def test_the_unexercised_feature_is_genuinely_constant(labelled_features):
    # Justifies the exclusion above rather than assuming it: nonfinite_rate
    # must be zero everywhere on this corpus. If a future generator starts
    # emitting malformed fixes, this fails and the exclusion gets revisited
    # instead of quietly hiding a real signal.
    honest, shared = labelled_features
    for group in (honest, shared):
        assert float(_column(group, "nonfinite_rate").max()) == 0.0


# DEFECT-6 (FIXED): style_bimodality used to carry W_STYLE_BIMODALITY = 2.0,
# the third-heaviest weight in the detector, while separating the classes at
# AUC ~0.55 -- near a coin flip. mock_rate separated at 0.74 on weight 1.2 and
# cell_mismatch_rate at 0.75 on weight 1.0, i.e. far more evidence for far
# less weight. The weight is now 0.3, matching the evidence the feature
# actually carries. Note the AUC itself did not improve, and is not expected
# to: the cause is a construct limit (two drivers' cruise-speed separation is
# smaller than one driver's own trip-to-trip spread), not a coding bug. The
# feature is kept for interpretability and weighted accordingly.

def test_heavily_weighted_features_carry_proportionate_signal(labelled_features):
    # The calibration claim that matters operationally. A feature carrying a
    # weight at or above the mock-rate weight is being treated as a primary
    # signal; it must separate at least as well as the weakest feature the
    # detector considers primary. A near-coin-flip feature with a heavy
    # weight amplifies noise into the score of every account.
    honest, shared = labelled_features
    heavy = {
        "handover_gap_score": W_HANDOVER_GAP,
        "overlap_rate": W_OVERLAP,
        "style_bimodality": W_STYLE_BIMODALITY,
        "mock_rate": W_MOCK_RATE,
        "cell_mismatch_rate": W_CELL_MISMATCH,
    }
    weak = []
    for key, weight in heavy.items():
        auc = _rank_auc(_column(honest, key), _column(shared, key))
        if weight >= W_MOCK_RATE and auc < 0.60:
            weak.append(f"{key} (weight {weight}, AUC {auc:.4f})")
    assert not weak, (
        "features weighted as primary signals but separating no better than "
        "a coin flip: " + "; ".join(weak) + ". Either the feature needs "
        "recalibration or its weight needs to come down to match the "
        "evidence it actually carries.")


def test_every_feature_is_finite_across_a_whole_corpus(honest_features):
    for features in honest_features:
        assert set(features) == set(FEATURE_NAMES)
        for key, value in features.items():
            assert math.isfinite(value), f"{key} is not finite"


def test_rate_features_stay_within_zero_and_one(honest_features):
    for key in ("mock_rate", "cell_mismatch_rate", "overlap_rate",
                "night_share", "style_bimodality"):
        values = _column(honest_features, key)
        assert values.min() >= 0.0, f"{key} went negative"
        assert values.max() <= 1.0, f"{key} exceeded 1.0"


def test_non_negative_features_never_go_below_zero(honest_features):
    for key in ("max_jump_m", "max_speed_kmh", "accel_p99_ms2",
                "gps_jitter_m", "handover_gap_score"):
        assert _column(honest_features, key).min() >= 0.0, f"{key} went negative"


# ---------------------------------------------------------------------------
# The honest population must not look guilty by default
# ---------------------------------------------------------------------------

def test_the_typical_honest_account_scores_low(honest_features):
    # Sanity on the operating point: on an all-honest corpus the median
    # account must sit well below any plausible flagging threshold, or the
    # detector's bias term is doing the work instead of the evidence.
    from tripphysics.detector import score_account

    scores = np.array([score_account(f) for f in honest_features])
    assert float(np.median(scores)) < 0.35, (
        f"median honest score is {float(np.median(scores)):.4f}")
    assert float((scores > 0.5).mean()) < 0.10, (
        "more than 10% of an all-honest corpus scores above 0.5")
