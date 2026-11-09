"""Physics-first scoring of profile-sharing suspicion.

Turns the eleven per-account features from ``features.py`` into a single
suspicion score in [0, 1]. Design constraints, in priority order:

1. **Auditable.** Every step is a named constant plus a monotone transform.
   There is deliberately no fitted model here: nothing to overfit on
   synthetic data, nothing that cannot be reviewed line by line.
2. **Physics before heuristics.** The heaviest weights go to signals that a
   single honest driver cannot plausibly produce: two trips running at the
   same time, a phone that "teleports" between the end of one trip and the
   start of the next, two distinct motor-control signatures on one account.
3. **Monotone.** More evidence never lowers the score, so the threshold
   sweep in ``eval.py`` behaves predictably.

Score construction::

    evidence_i = monotone transform of feature_i into [0, 1]
    raw        = sum_i( WEIGHT_i * evidence_i )
    score      = 1 / (1 + exp(-(raw - SCORE_BIAS)))     # logistic squash

Only the *ordering* of scores matters downstream (eval.py thresholds by
quantile of the honest population); the logistic squash exists purely so a
human reading a score sees a stable 0..1 range.
"""

import math

# ---------------------------------------------------------------------------
# Evidence transforms: physical scales
#
# Each constant answers one question: "at what raw value does this feature
# stop looking like noise and start looking like account sharing?"
# ---------------------------------------------------------------------------

# End-of-trip -> start-of-next-trip displacement (largest observed, metres).
# A parked vehicle stays where it stopped; a handed-over account "teleports".
# 3 km with no plausible travel time is the midpoint of suspicion; beyond
# ~7-8 km the evidence saturates.
JUMP_CENTER_M = 3000.0
JUMP_WIDTH_M = 1500.0

# Maximum plausible speed (km/h). Urban ride-hailing rarely sustains >130;
# 150 is the suspicion midpoint (motorway burst vs. GPS glitch vs. spoofed
# feed), and by 220+ the trace is physically implausible for this fleet.
SPEED_CENTER_KMH = 150.0
SPEED_WIDTH_KMH = 25.0

# 99th percentile of longitudinal acceleration (m/s^2). Comfortable braking
# stays under ~3 m/s^2. A p99 above ~4.5 means a heavy harsh-event tail --
# which is exactly what a *mixture* of two driving styles produces, because
# the aggressive driver's tail dominates the percentile.
ACCEL_CENTER_MS2 = 4.5
ACCEL_WIDTH_MS2 = 1.0

# Median GPS scatter (metres). Phone GNSS under open sky sits at 3-10 m.
# Sustained scatter above ~25 m points at a degraded, replayed or synthetic
# position feed. Weak signal on its own (urban canyons exist) -- see weight.
JITTER_CENTER_M = 25.0
JITTER_WIDTH_M = 8.0

# Rates that saturate quickly: even a small fraction of affected fixes is
# close to conclusive, so evidence ramps linearly and caps at these values.
MOCK_RATE_SATURATION = 0.05       # 5% of fixes flagged mock => full evidence
CELL_MISMATCH_SATURATION = 0.10   # GPS in one place, radio in another
OVERLAP_RATE_SATURATION = 0.05    # one pair of simultaneous trips is damning

# Handover gap score (from features.py): dimensionless, >= 0, scaled so that
# ~1.0 corresponds to one clearly implausible trip-to-trip handover (required
# speed between trips far above what any vehicle could do). Diminishing
# returns above that: x / (x + half-saturation).
HANDOVER_HALF_SATURATION = 1.0

# Bimodality of the driving-style distribution. `features.style_bimodality`
# is already standardised against its own single-driver null and rescaled to
# [0, 1] (1.0 = 3 null sigmas), so the midpoint here is a point on THAT
# scale, not a raw variance ratio: 0.33 is one null sigma of excess
# structure. Note this feature is weak by construction -- see its weight.
BIMODALITY_CENTER = 0.33
BIMODALITY_WIDTH = 0.15

# Fraction of fixes discarded as non-finite. Valid telemetry is exactly 0.0
# here, so any sustained nonzero rate means a client emitting unparseable
# data. Saturates fast, but carries a small weight: broken SDKs and genuine
# upload corruption exist, so this is an integrity smell, not proof of
# sharing. It matters mostly that it points the RIGHT way -- garbage input
# must raise suspicion, never lower it.
NONFINITE_RATE_SATURATION = 0.02

# Share of trips at night. One honest driver clusters around their own
# schedule; a shift-shared account covers the clock. Weakest signal here --
# plenty of honest night drivers exist -- hence the small weight below.
NIGHT_SHARE_CENTER = 0.45
NIGHT_SHARE_WIDTH = 0.10

# ---------------------------------------------------------------------------
# Weights: how much each unit of evidence contributes to the raw score.
#
# Ordered by how hard the signal is for a single honest driver to produce.
# ---------------------------------------------------------------------------

W_OVERLAP = 2.5           # two trips at once: physically impossible for one person
W_HANDOVER_GAP = 2.2      # teleporting between trips: the core physics tell
W_MAX_JUMP = 1.5          # raw displacement discontinuity (backs up handover)
W_MOCK_RATE = 1.2         # OS-level mock-location flag: direct but evadable
W_CELL_MISMATCH = 1.0     # GPS position disagrees with serving cell
W_MAX_SPEED = 0.8         # implausible speed: shared *or* glitchy hardware
W_ACCEL_P99 = 0.6         # harsh-event tail: style-adjacent, weaker alone
W_NONFINITE_RATE = 0.5    # unparseable telemetry: integrity smell, and must
                          # push suspicion UP so garbage is never an escape
W_GPS_JITTER = 0.5        # scatter anomaly: spoof tools get this wrong
W_NIGHT_SHARE = 0.3       # clock coverage: deliberately tiny, easy to confound
# Two motor-control signatures on one account. The construct is sound and
# the statistic is now correctly calibrated against its own null, but it is
# EMPIRICALLY WEAK: ROC-AUC 0.51 over 7 seeds at n=2000, barely above a
# coin. The cause is a signal-to-noise limit, not a bug -- the tier-level
# cruise-speed separation between two drivers (1.2-3.0 m/s) is smaller than
# the spread of one account's own per-trip median speeds (~2.9 m/s), and on
# a median of 18 trips no two-way split can resolve a mixture at SNR < 1.
# Even a ground-truth simulation with an exact 50/50 driver mix only reaches
# AUC 0.52-0.59 across the tier separations, so this is a ceiling of the
# measurement, not of the implementation.
#
# It is therefore demoted from 2.0 to the "weak and confounded" tier
# alongside gps_jitter_m, rather than deleted: at 0.3 it can corroborate a
# case an analyst is already looking at without materially moving rankings.
# Carrying it at 2.0 was the real defect -- it made a coin flip the third
# heaviest voice in the score. Revisit the weight only with labelled
# production data, where the true style separation may be far wider than
# the synthetic corpus assumes.
W_STYLE_BIMODALITY = 0.3

# Logistic squash midpoint. Chosen so an all-zero feature vector lands near
# 0.09 (every transform here is smooth, so zero features still leave a small
# positive residue -- an account with NO trips is mapped to a hard 0.0
# separately, see `_is_evidence_free`) and full evidence on the top three
# signals alone reaches ~0.976. Shifts every score by the same monotone map,
# so downstream thresholding (which is quantile-based) is unaffected by this
# choice.
SCORE_BIAS = 2.5

REQUIRED_FEATURES = frozenset({
    "max_jump_m",
    "max_speed_kmh",
    "accel_p99_ms2",
    "gps_jitter_m",
    "mock_rate",
    "cell_mismatch_rate",
    "handover_gap_score",
    "style_bimodality",
    "overlap_rate",
    "night_share",
    "nonfinite_rate",
})


def _logistic(x: float) -> float:
    """Standard logistic 1 / (1 + e^-x), numerically safe for large |x|."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    # For very negative x, exp(-x) overflows; compute via exp(x) instead.
    e = math.exp(x)
    return e / (1.0 + e)


def _soft_threshold(x: float, center: float, width: float) -> float:
    """Monotone map onto (0, 1): 0.5 at `center`, 0.95 at center + 3*width.

    A smooth version of "is x above the physically plausible limit?" --
    smooth so that measurement noise near the limit does not flip evidence
    on and off. It never reaches exactly 1: the evidence keeps growing with
    x, so a more extreme observation always outranks a merely extreme one.
    """
    return _logistic((x - center) / width)


def _saturating_rate(rate: float, saturation: float) -> float:
    """Linear ramp for a rate in [0, 1]: full evidence at `saturation`.

    Used where a small fraction of affected observations is already close
    to conclusive (mock fixes, cell mismatches, overlapping trips).
    """
    return min(max(rate, 0.0) / saturation, 1.0)


def _diminishing(x: float, half_saturation: float) -> float:
    """Monotone map [0, inf) -> [0, 1): x / (x + h), evidence 0.5 at x = h.

    Used for open-ended scores where the second and third implausible event
    add less than the first one did.
    """
    x = max(x, 0.0)
    return x / (x + half_saturation)


def _evidence(features: dict[str, float]) -> dict[str, float]:
    """Map each raw feature to an evidence value in [0, 1]. Monotone per key."""
    return {
        "max_jump_m": _soft_threshold(
            features["max_jump_m"], JUMP_CENTER_M, JUMP_WIDTH_M),
        "max_speed_kmh": _soft_threshold(
            features["max_speed_kmh"], SPEED_CENTER_KMH, SPEED_WIDTH_KMH),
        "accel_p99_ms2": _soft_threshold(
            features["accel_p99_ms2"], ACCEL_CENTER_MS2, ACCEL_WIDTH_MS2),
        "gps_jitter_m": _soft_threshold(
            features["gps_jitter_m"], JITTER_CENTER_M, JITTER_WIDTH_M),
        "mock_rate": _saturating_rate(
            features["mock_rate"], MOCK_RATE_SATURATION),
        "cell_mismatch_rate": _saturating_rate(
            features["cell_mismatch_rate"], CELL_MISMATCH_SATURATION),
        "handover_gap_score": _diminishing(
            features["handover_gap_score"], HANDOVER_HALF_SATURATION),
        "style_bimodality": _soft_threshold(
            features["style_bimodality"], BIMODALITY_CENTER, BIMODALITY_WIDTH),
        "overlap_rate": _saturating_rate(
            features["overlap_rate"], OVERLAP_RATE_SATURATION),
        "night_share": _soft_threshold(
            features["night_share"], NIGHT_SHARE_CENTER, NIGHT_SHARE_WIDTH),
        "nonfinite_rate": _saturating_rate(
            features["nonfinite_rate"], NONFINITE_RATE_SATURATION),
    }


_WEIGHTS = {
    "max_jump_m": W_MAX_JUMP,
    "max_speed_kmh": W_MAX_SPEED,
    "accel_p99_ms2": W_ACCEL_P99,
    "gps_jitter_m": W_GPS_JITTER,
    "mock_rate": W_MOCK_RATE,
    "cell_mismatch_rate": W_CELL_MISMATCH,
    "handover_gap_score": W_HANDOVER_GAP,
    "style_bimodality": W_STYLE_BIMODALITY,
    "overlap_rate": W_OVERLAP,
    "night_share": W_NIGHT_SHARE,
    "nonfinite_rate": W_NONFINITE_RATE,
}


def _is_evidence_free(features: dict[str, float]) -> bool:
    """True when the feature vector carries no observation at all.

    An account with no trips (or only unusable ones) produces exactly the
    all-zero feature vector. That is *absence of evidence*, which is not the
    same thing as evidence of innocence, and the two must not land on the
    same score. Because every transform in `_evidence` is a smooth logistic,
    a zero feature still yields a small positive evidence value, so an empty
    account would otherwise accumulate the same ~0.09 floor as a genuinely
    clean one -- and would sort ahead of nothing, yet below nothing either.
    We map it to a hard 0.0 instead: "nothing to say", explicitly.
    """
    return all(features[k] == 0.0 for k in sorted(REQUIRED_FEATURES))


def score_account(features: dict[str, float]) -> float:
    """Suspicion score in [0, 1] for one account's feature dict.

    Fails loudly (ValueError) on missing feature keys rather than silently
    imputing zeros: in a fraud pipeline a silently-degraded feature is worse
    than a crash.

    An all-zero feature vector (an account with no usable trips) scores a
    hard 0.0 -- see :func:`_is_evidence_free`. Every other input goes
    through the weighted-evidence sum, which is monotone in each feature.
    """
    missing = REQUIRED_FEATURES - features.keys()
    if missing:
        raise ValueError(f"missing feature keys: {sorted(missing)}")
    if _is_evidence_free(features):
        return 0.0
    evidence = _evidence(features)
    raw = sum(_WEIGHTS[k] * evidence[k] for k in sorted(evidence))
    return _logistic(raw - SCORE_BIAS)


def explain(features: dict[str, float]) -> dict:
    """Per-feature breakdown of one account's score, for case review.

    Returns ``{"evidence": {...}, "contribution": {...}, "raw": float,
    "score": float}`` where ``contribution[k] = WEIGHT_k * evidence[k]``.
    Useful when an analyst asks *why* an account was flagged.

    The ``score`` reported here always equals :func:`score_account` on the
    same input, including the evidence-free 0.0 case: a breakdown that
    disagreed with the score that flagged the account would be worse than
    no breakdown at all.
    """
    missing = REQUIRED_FEATURES - features.keys()
    if missing:
        raise ValueError(f"missing feature keys: {sorted(missing)}")
    if _is_evidence_free(features):
        zeros = {k: 0.0 for k in sorted(REQUIRED_FEATURES)}
        return {"evidence": zeros, "contribution": dict(zeros),
                "raw": 0.0, "score": 0.0}
    evidence = _evidence(features)
    contribution = {k: _WEIGHTS[k] * evidence[k] for k in evidence}
    raw = sum(contribution[k] for k in sorted(contribution))
    return {
        "evidence": evidence,
        "contribution": contribution,
        "raw": raw,
        "score": _logistic(raw - SCORE_BIAS),
    }
