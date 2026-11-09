"""Account-level physics features for profile-sharing detection.

Every feature is a plain, auditable statistic over one account's trip log --
no fitted parameters, no opaque transforms. The detector downstream combines
them; this module only measures.

Feature reference (exact keys returned by `extract_features`)
-------------------------------------------------------------
max_jump_m          Largest in-trip displacement between consecutive fixes
                    IN EXCESS of what a 250 km/h hard cap allows for the
                    observed time gap, metres. Honest logs sit at 0 with a
                    thin tail from GPS cold starts; spoof re-anchoring
                    produces km-scale values.
max_speed_kmh       Maximum reported speed across all fixes, km/h.
accel_p99_ms2       99th percentile of |d(speed)/dt| between consecutive
                    fixes, m/s^2. At ~10 s telemetry cadence this is a
                    speed-volatility proxy (typical 0.3-1.5), not true
                    instantaneous acceleration.
gps_jitter_m        Median deviation of each interior fix from the midpoint
                    of its neighbours, metres. Proxy for positional noise:
                    honest receivers show a device-typical floor; spoofed
                    tracks are suspiciously glassy (near zero).
mock_rate           Fraction of fixes with the OS mock-location flag set.
cell_mismatch_rate  Fraction of fixes whose reported serving cell disagrees
                    with the cell implied by the reported position. NOTE:
                    here the expected cell comes from the synthetic world's
                    grid (`synth.cell_id_from_xy`); on live data replace
                    this join with the operator's cell-location database.
handover_gap_score  Mean implausibility of inter-trip gaps: for consecutive
                    trips, distance from trip N's end to trip N+1's start,
                    less what a generous 54 km/h door-to-door average covers
                    in the gap time, saturated at 5 km. Nonzero means the
                    account "teleports" between trips -- the classic
                    phone-handover signature.
style_bimodality    Evidence that trips cluster into two distinct driving
                    signatures, in [0, 1]. Balance-weighted best two-way
                    split (Otsu-style between/total variance ratio) of a
                    per-trip style score (winsorised z of median speed plus
                    winsorised z of speed volatility), STANDARDISED against
                    the null distribution a single-mode sample of the same
                    size produces, then capped at 3 null sigmas.
                    WEAK SIGNAL, kept for interpretability, not for lift:
                    measured ROC-AUC is 0.51 (7 seeds, n=2000), i.e. close
                    to a coin. Cause is a construct limit, not a coding bug
                    -- the two drivers' cruise-speed separation (1.2-3.0
                    m/s by tier) is smaller than the spread of one account's
                    own per-trip median speeds (~2.9 m/s, mostly driven by
                    the 10% of trips that run at highway cruise), so on a
                    median of 18 trips no two-way split can recover the
                    mixture. Even a ground-truth simulation with an exact
                    50/50 driver mix only reaches AUC 0.52-0.59 across the
                    tier separations, and widening the separation far beyond
                    what the tiers use still caps out near 0.65. Weighted
                    accordingly in `detector.py` (0.3, the "confounded"
                    tier).
overlap_rate        Fraction of trip pairs that overlap in time beyond a
                    30 s tolerance -- one account driving two trips at
                    once. Counted over ALL pairs via an interval sweep, so
                    a trip nested inside a longer one is not missed.
night_share         Fraction of trips starting in the 00:00-06:00 local
                    window. Weak alone (honest night-shift drivers exist);
                    useful only in combination.
nonfinite_rate      Fraction of raw fixes discarded as non-finite (NaN/inf
                    in t, x, y or speed). Such fixes are dropped BEFORE any
                    differencing so they cannot poison an aggregate, and
                    the discard rate is reported here as evidence in its
                    own right: a client emitting unparseable telemetry is
                    itself suspicious. Honest logs sit at exactly 0.0.
                    NOTE: `synth.py` never emits a non-finite fix for any
                    account, so on the synthetic corpus this feature is
                    identically 0.0 and its standalone AUC is exactly 0.50.
                    It exists as an anti-evasion guard for real data, not
                    as a source of synthetic lift -- see the adversarial
                    note below.

All values are finite Python floats; degenerate accounts (few trips, short
trips) fall back to 0.0 rather than NaN. Extraction is deterministic and
side-effect free.

Adversarial note
----------------
Feature code in a fraud pipeline is attack surface: the account being
measured controls the input. Two rules follow, and both are enforced above.
(1) Malformed input must never make an account look CLEANER than valid
input would -- otherwise emitting garbage becomes a cheap evasion. Hence
non-finite fixes are excised at read time and counted, never allowed to
NaN-poison an aggregate that a downstream guard would then rewrite to the
minimum-suspicion value. (2) A degraded feature must be visible, so the
discard shows up as `nonfinite_rate` rather than as silence.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# The synthetic world's cell grid doubles as the "cell-location database"
# for the mismatch feature. Swap this import for a real tower join on
# production data; the feature semantics stay identical.
from .synth import cell_id_from_xy

__all__ = ["extract_features", "FEATURE_NAMES"]

FEATURE_NAMES = (
    "max_jump_m", "max_speed_kmh", "accel_p99_ms2", "gps_jitter_m",
    "mock_rate", "cell_mismatch_rate", "handover_gap_score",
    "style_bimodality", "overlap_rate", "night_share",
    "nonfinite_rate",
)

# --- physical plausibility constants (documented, deliberately generous) ---
V_TRIP_MAX_MPS = 69.4     # 250 km/h: nothing on wheels sustains more between fixes
V_GAP_MAX_MPS = 15.0      # 54 km/h door-to-door average between trips, incl. parking
GAP_FLOOR_S = 60.0        # floor on gap time so overlaps score as maximal handovers
HANDOVER_SAT_M = 5_000.0  # excess distance at which a handover saturates to 1.0
OVERLAP_TOL_S = 30.0      # clock skew / upload rounding tolerance
NIGHT_START_H, NIGHT_END_H = 0.0, 6.0
DAY_S = 86_400.0
MIN_DT_S = 1.0            # guards divide-by-zero on duplicate timestamps
_MIN_TRIPS_BIMODALITY = 6  # below this a two-way split is pure noise
_Z_WINSOR = 2.5           # caps per-dimension z so one outlier trip cannot
                          # masquerade as a second driving mode

# --- null model for the balanced-Otsu split statistic ---
#
# A UNIMODAL sample is not centred at zero on this statistic: the best
# balanced two-way split of a single Gaussian already captures a large share
# of the variance purely by chance. Measured on 20k simulated unimodal
# samples per n (two independent standard-normal per-trip dimensions pushed
# through the same winsorised-z pipeline used below), the raw statistic's
# null distribution is:
#
#     n     mean     sd          n     mean     sd
#     6     0.666    0.129       20    0.640    0.072
#     7     0.599    0.101       30    0.640    0.059
#     12    0.643    0.092       50    0.642    0.046
#     15    0.613    0.079       100   0.641    0.033
#
# Two structural facts drive the model below:
#
# 1. The mean depends on the PARITY of n, not just its size. An even n can
#    be split into two exactly equal halves, which maximises the balance
#    weight 2*min(k,n-k)/n; an odd n cannot, so it scores systematically
#    lower. Both branches converge to ~0.634 with a 1/n correction.
# 2. The sd shrinks as ~1/sqrt(n). A fixed offset therefore cannot put small
#    and large accounts on one scale: subtracting a constant leaves a
#    6-trip account and a 30-trip account with wildly different false-alarm
#    rates for the same nominal value.
#
# So we standardise rather than subtract: the feature reports how many null
# standard deviations the observed split sits above what one driver produces
# by chance. Coefficients are least-squares fits to the table above
# (max abs error 0.009 on the mean, 0.013 on the sd) and validated on a
# held-out simulation stream (null z-scores come out mean ~0, sd ~1).
_OTSU_NULL_MEAN_EVEN = (0.6368, 0.1243)   # (intercept, 1/n coefficient)
_OTSU_NULL_MEAN_ODD = (0.6327, -0.2632)
_OTSU_NULL_SD = (0.0051, 0.2885)          # (intercept, 1/sqrt(n) coefficient)
# Null-sigma excess at which the feature saturates to 1.0. The simulated
# null tail reaches z=2.21 at p99 and z=2.81 at p99.9, so a single driver
# essentially never reaches 3 sigma by chance.
_OTSU_Z_FULL = 3.0


def _otsu_null_mean(n: int) -> float:
    """Expected balanced-Otsu score for a UNIMODAL sample of size n."""
    a, b = _OTSU_NULL_MEAN_EVEN if n % 2 == 0 else _OTSU_NULL_MEAN_ODD
    return a + b / n


def _otsu_null_sd(n: int) -> float:
    """Standard deviation of that null, which shrinks as ~1/sqrt(n)."""
    a, b = _OTSU_NULL_SD
    return a + b / math.sqrt(n)


def _split_separation(scalar: np.ndarray) -> float:
    """Two-mode evidence for a 1-D sample, in [0, 1].

    Sorts the sample, tries every split with >=2 points on each side and
    takes max over splits of (between-class variance / total variance)
    weighted by 2*min(n1,n2)/n. The balance weight kills the degenerate
    "one outlier vs the rest" splits that plague raw Otsu on small samples.
    Hand-rolled on purpose: a reviewer can verify it by eye.

    The raw maximum is then STANDARDISED against the null distribution a
    single-mode sample of the same size produces (see `_otsu_null_mean` /
    `_otsu_null_sd`): the returned value is the null-sigma excess, squashed
    into [0, 1] by dividing by `_OTSU_Z_FULL`. 0 therefore means "no more
    structure than one driver's ordinary trip-to-trip variation", not
    "no variance". Standardising rather than subtracting a constant is what
    puts a 6-trip and a 30-trip account on the same scale.

    Honest limitation: on realistic per-trip signatures this statistic is
    weak (measured ROC-AUC ~0.53). See the `style_bimodality` note in the
    module docstring and the small weight it carries in `detector.py`.
    """
    n = int(scalar.size)
    if n < _MIN_TRIPS_BIMODALITY:
        return 0.0
    s = np.sort(scalar)
    total_var = float(s.var())
    if total_var <= 1e-12:
        return 0.0
    best = 0.0
    for k in range(2, n - 1):          # k = size of the left class
        m1 = float(s[:k].mean())
        m2 = float(s[k:].mean())
        between = k * (n - k) / (n * n) * (m1 - m2) ** 2
        balance = 2.0 * min(k, n - k) / n
        best = max(best, between / total_var * balance)
    z = (best - _otsu_null_mean(n)) / _otsu_null_sd(n)
    return min(max(z, 0.0) / _OTSU_Z_FULL, 1.0)


def _zscore(a: np.ndarray) -> np.ndarray:
    sd = float(a.std())
    if sd <= 1e-9:
        return np.zeros_like(a)
    return (a - a.mean()) / sd


def extract_features(account: dict[str, Any]) -> dict[str, float]:
    """Compute the eleven account-level features from an Account dict.

    Input shape (see `synth.generate_dataset` / schema/input_schema.md):
      {"account_id": str, "is_shared": bool, "trips": [Trip, ...]}
    The `is_shared` label is never read here -- extraction is label-blind.
    """
    trips = sorted(account.get("trips") or [],
                   key=lambda tr: float(tr["t_start"]))

    max_jump = 0.0
    max_speed_mps = 0.0
    accel_chunks: list[np.ndarray] = []
    resid_chunks: list[np.ndarray] = []
    n_fix = 0
    n_fix_raw = 0
    n_nonfinite = 0
    n_mock = 0
    n_mismatch = 0
    night_trips = 0
    trip_med_speed: list[float] = []
    trip_accel_p90: list[float] = []
    # Per-trip endpoints for the inter-trip pass: (t_start, t_end, start/end xy).
    bounds: list[tuple[float, float, float, float, float, float]] = []

    for tr in trips:
        fx = tr["fixes"]
        if not fx:
            continue
        t = np.array([f["t"] for f in fx], dtype=float)
        x = np.array([f["x_m"] for f in fx], dtype=float)
        y = np.array([f["y_m"] for f in fx], dtype=float)
        s = np.array([f["speed_mps"] for f in fx], dtype=float)
        mock = np.array([f["is_mock"] for f in fx], dtype=bool)
        cell = np.array([f["cell_id"] for f in fx], dtype=np.int64)

        # Drop non-finite fixes BEFORE any differencing. NaN/inf in a single
        # coordinate otherwise propagates through np.diff/np.hypot into every
        # running aggregate for the whole account, and a downstream
        # "finite or 0.0" guard would then rewrite a poisoned max_jump_m to
        # 0.0 -- the cleanest possible value. That hands an attacker a way to
        # LOWER their own suspicion by emitting garbage, so the discard must
        # happen here and be surfaced as evidence (`nonfinite_rate`) rather
        # than silently zeroing an aggregate.
        n_raw = t.size
        finite = (np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
                  & np.isfinite(s))
        n_bad = int(n_raw - np.count_nonzero(finite))
        n_fix_raw += n_raw
        n_nonfinite += n_bad
        if n_bad:
            t, x, y, s = t[finite], x[finite], y[finite], s[finite]
            mock, cell = mock[finite], cell[finite]
        n = t.size
        if n == 0:
            # Every fix in this trip was garbage: it contributes only to
            # nonfinite_rate, and cannot anchor an inter-trip handover.
            continue

        n_fix += n
        n_mock += int(np.count_nonzero(mock))
        n_mismatch += int(np.count_nonzero(cell != cell_id_from_xy(x, y)))
        max_speed_mps = max(max_speed_mps, float(s.max()))

        start_hour = (float(t[0]) % DAY_S) / 3600.0
        if NIGHT_START_H <= start_hour < NIGHT_END_H:
            night_trips += 1
        bounds.append((float(t[0]), float(t[-1]),
                       float(x[0]), float(y[0]), float(x[-1]), float(y[-1])))

        if n >= 2:
            dt = np.maximum(np.diff(t), MIN_DT_S)
            dist = np.hypot(np.diff(x), np.diff(y))
            # In-trip teleport: distance beyond the hard physical cap.
            max_jump = max(max_jump, float((dist - dt * V_TRIP_MAX_MPS).max()))
            accel = np.abs(np.diff(s)) / dt
            accel_chunks.append(accel)
            if n >= 5:
                trip_med_speed.append(float(np.median(s)))
                trip_accel_p90.append(float(np.percentile(accel, 90)))
        if n >= 3:
            # Deviation from the local straight-line path: noise proxy.
            rx = x[1:-1] - 0.5 * (x[:-2] + x[2:])
            ry = y[1:-1] - 0.5 * (y[:-2] + y[2:])
            resid_chunks.append(np.hypot(rx, ry))

    # ---- inter-trip pass: handovers and overlaps ----
    handover_score = 0.0
    overlap_rate = 0.0
    if len(bounds) >= 2:
        gap_scores = []
        for prev, nxt in zip(bounds, bounds[1:]):
            gap_t = nxt[0] - prev[1]                      # start(N+1) - end(N)
            gap_d = float(np.hypot(nxt[2] - prev[4], nxt[3] - prev[5]))
            # Distance the account covered "for free": whatever a generous
            # door-to-door average cannot explain in the observed gap.
            excess = gap_d - max(gap_t, GAP_FLOOR_S) * V_GAP_MAX_MPS
            gap_scores.append(min(max(excess, 0.0) / HANDOVER_SAT_M, 1.0))
        handover_score = float(np.mean(gap_scores))

        # Overlaps: count every trip that runs concurrently with ANY other
        # trip, not just with its immediate successor. Comparing only
        # consecutive pairs misses nested trips -- one long trip spanning
        # several short ones scores as a single overlap however many trips
        # it swallows, which understates exactly the pattern that matters
        # most (a driver whose account is live continuously while a second
        # driver runs rides inside that window).
        #
        # Sweep, O(n log n): trips are already sorted by start time, so a
        # trip overlaps something earlier iff it starts more than the
        # tolerance before the MAXIMUM end time seen so far. Tracking the
        # running max (rather than the previous end) is what makes nesting
        # visible. Each side of an overlapping pair is marked, so a trip
        # buried inside a longer one counts, as does the trip that hides it.
        overlapped = [False] * len(bounds)
        # (end_time, index) of the trip with the largest end seen so far.
        max_end, max_end_idx = bounds[0][1], 0
        for i in range(1, len(bounds)):
            if bounds[i][0] < max_end - OVERLAP_TOL_S:
                overlapped[i] = True
                overlapped[max_end_idx] = True
            if bounds[i][1] > max_end:
                max_end, max_end_idx = bounds[i][1], i
        # Denominator kept at len(bounds) - 1 so the feature's scale (and
        # the tuned OVERLAP_RATE_SATURATION in detector.py) is unchanged:
        # one overlapping pair on a two-trip account still reads 1.0.
        overlap_rate = min(sum(overlapped) / (len(bounds) - 1), 1.0)

    # ---- style bimodality over per-trip signatures ----
    bimodality = 0.0
    if len(trip_med_speed) >= _MIN_TRIPS_BIMODALITY:
        # Winsorise each dimension: a couple of honest highway trips are
        # outliers, not a second driver.
        z_speed = np.clip(_zscore(np.array(trip_med_speed)), -_Z_WINSOR, _Z_WINSOR)
        z_accel = np.clip(_zscore(np.array(trip_accel_p90)), -_Z_WINSOR, _Z_WINSOR)
        bimodality = _split_separation(z_speed + z_accel)

    accel_all = np.concatenate(accel_chunks) if accel_chunks else np.array([0.0])
    resid_all = np.concatenate(resid_chunks) if resid_chunks else np.array([0.0])

    out = {
        "max_jump_m": max(max_jump, 0.0),
        "max_speed_kmh": max_speed_mps * 3.6,
        "accel_p99_ms2": float(np.percentile(accel_all, 99)),
        "gps_jitter_m": float(np.median(resid_all)),
        "mock_rate": (n_mock / n_fix) if n_fix else 0.0,
        "cell_mismatch_rate": (n_mismatch / n_fix) if n_fix else 0.0,
        "handover_gap_score": handover_score,
        "style_bimodality": bimodality,
        "overlap_rate": overlap_rate,
        "night_share": (night_trips / len(trips)) if trips else 0.0,
        "nonfinite_rate": (n_nonfinite / n_fix_raw) if n_fix_raw else 0.0,
    }
    # Hard guarantee: finite floats only, exact key set, stable order.
    #
    # Reaching this point with a non-finite value would be a bug, not input
    # data: every non-finite fix is filtered above. Crucially we do NOT fall
    # back to 0.0 here -- for max_jump_m and friends 0.0 is the *least*
    # suspicious value, so a silent coercion would turn an internal error
    # into an evasion channel. Fail loudly instead.
    bad = [k for k in FEATURE_NAMES if not np.isfinite(out[k])]
    if bad:
        raise ValueError(
            f"non-finite feature values for account "
            f"{account.get('account_id')!r}: {sorted(bad)}")
    return {k: float(out[k]) for k in FEATURE_NAMES}
