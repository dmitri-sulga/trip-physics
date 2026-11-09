"""Geometry and physics edge cases in feature extraction.

The harness works in a local planar frame (x_m / y_m in metres) rather than
lat/lon, so the classic spherical pathologies -- antimeridian wrap, polar
convergence -- cannot arise by construction. That is a design property worth
*testing*, not assuming: test_planar_frame_has_no_antimeridian_discontinuity
pins it, so a future switch to lat/lon cannot happen silently without a
failing test demanding a proper great-circle distance.

What can and does go wrong on a planar frame is covered here: zero
displacement, duplicate timestamps (dt = 0), single-fix trips, trips with no
fixes, unsorted trips, and non-finite coordinates fed by an adversary.

DEFECT-3 is covered below: a single non-finite coordinate used to erase the
in-trip evidence and *lower* an account's suspicion. It is fixed, and the
tests here pin the fixed behaviour so it cannot regress silently.
"""

import math

import pytest

from tripphysics.detector import score_account
from tripphysics.features import (
    FEATURE_NAMES,
    GAP_FLOOR_S,
    HANDOVER_SAT_M,
    V_GAP_MAX_MPS,
    V_TRIP_MAX_MPS,
    extract_features,
)
from tripphysics.synth import cell_id_from_xy

# ---------------------------------------------------------------------------
# Fixture builders: minimal, hand-shaped accounts (no synthetic generator, so
# each assertion traces to explicit coordinates).
# ---------------------------------------------------------------------------


def make_fix(t, x, y, speed=10.0, is_mock=False, cell_id=None, accuracy=5.0):
    """One GPS fix; cell_id defaults to the grid cell implied by (x, y)."""
    return {
        "t": float(t),
        "x_m": float(x),
        "y_m": float(y),
        "accuracy_m": float(accuracy),
        "speed_mps": float(speed),
        "is_mock": bool(is_mock),
        "cell_id": int(cell_id_from_xy(x, y)) if cell_id is None else int(cell_id),
    }


def make_trip(fixes, trip_id="t000", account_id="acc"):
    return {
        "trip_id": trip_id,
        "account_id": account_id,
        "t_start": fixes[0]["t"] if fixes else 0.0,
        "fixes": fixes,
    }


def make_account(trips, account_id="acc", is_shared=False):
    return {"account_id": account_id, "is_shared": is_shared, "trips": trips}


def straight_line_trip(t0, x0, y0, n=10, dt=10.0, step_m=100.0, **fix_kw):
    """A trip moving east at a constant, physically plausible rate."""
    return make_trip(
        [make_fix(t0 + i * dt, x0 + i * step_m, y0, **fix_kw) for i in range(n)],
        trip_id=f"t{int(t0):06d}",
    )


# ---------------------------------------------------------------------------
# Contract: shape and finiteness on degenerate accounts
# ---------------------------------------------------------------------------

def test_account_with_no_trips_returns_all_zero_features():
    features = extract_features(make_account([]))
    assert set(features) == set(FEATURE_NAMES)
    assert all(value == 0.0 for value in features.values())


def test_account_with_missing_trips_key_is_tolerated():
    # extract_features uses .get("trips") -- an account record with the key
    # absent must not raise, since real exports drop empty lists.
    features = extract_features({"account_id": "acc", "is_shared": False})
    assert all(value == 0.0 for value in features.values())


def test_trip_with_no_fixes_is_skipped_not_crashed():
    features = extract_features(make_account([make_trip([])]))
    assert set(features) == set(FEATURE_NAMES)
    assert all(math.isfinite(v) for v in features.values())


def test_single_fix_trip_yields_no_kinematic_signal():
    # One fix admits no differences, so every difference-based feature must
    # stay at zero rather than reading an artifact off a length-1 array.
    features = extract_features(make_account([make_trip([make_fix(0, 0, 0, speed=10.0)])]))
    assert features["max_jump_m"] == 0.0
    assert features["accel_p99_ms2"] == 0.0
    assert features["gps_jitter_m"] == 0.0
    assert features["max_speed_kmh"] == pytest.approx(36.0)  # 10 m/s


def test_all_feature_values_are_plain_finite_floats():
    account = make_account([straight_line_trip(0, 0, 0), straight_line_trip(3600, 0, 0)])
    features = extract_features(account)
    for key, value in features.items():
        assert type(value) is float, f"{key} is {type(value)}"
        assert math.isfinite(value), f"{key} is not finite"


def test_extraction_is_label_blind():
    # The docstring promises extraction never reads is_shared. Flipping the
    # label must not move a single feature.
    trips = [straight_line_trip(0, 0, 0), straight_line_trip(3600, 0, 0)]
    honest = extract_features(make_account(trips, is_shared=False))
    shared = extract_features(make_account(trips, is_shared=True))
    assert honest == shared


def test_extraction_is_pure_and_repeatable():
    account = make_account([straight_line_trip(0, 0, 0), straight_line_trip(7200, 5000, 0)])
    first = extract_features(account)
    second = extract_features(account)
    assert first == second


# ---------------------------------------------------------------------------
# Zero displacement and dt = 0
# ---------------------------------------------------------------------------

def test_stationary_vehicle_produces_no_jump_and_no_jitter():
    # A parked car reporting identical positions: no displacement, no noise.
    fixes = [make_fix(i * 10.0, 0.0, 0.0, speed=0.0) for i in range(10)]
    features = extract_features(make_account([make_trip(fixes)]))
    assert features["max_jump_m"] == 0.0
    assert features["gps_jitter_m"] == 0.0
    assert features["accel_p99_ms2"] == 0.0
    assert features["max_speed_kmh"] == 0.0


def test_duplicate_timestamps_do_not_divide_by_zero():
    # dt = 0 between fixes (duplicate upload / clock stall). MIN_DT_S floors
    # the divisor, so acceleration must stay finite instead of blowing up.
    fixes = [
        make_fix(100.0, 0.0, 0.0, speed=0.0),
        make_fix(100.0, 0.0, 0.0, speed=30.0),   # same timestamp, speed jump
        make_fix(100.0, 50.0, 0.0, speed=30.0),  # same timestamp, moved
    ]
    features = extract_features(make_account([make_trip(fixes)]))
    assert all(math.isfinite(v) for v in features.values())
    # dt floors to MIN_DT_S = 1 s, so |d speed| / dt = 30 m/s^2 is reported.
    assert features["accel_p99_ms2"] == pytest.approx(30.0, rel=0.05)


def test_backwards_timestamps_do_not_produce_negative_jump():
    # A fix arriving out of order gives a negative dt; the floor keeps the
    # excess-distance computation sane and max_jump_m must stay >= 0.
    fixes = [
        make_fix(200.0, 0.0, 0.0),
        make_fix(100.0, 10.0, 0.0),  # earlier than its predecessor
        make_fix(300.0, 20.0, 0.0),
    ]
    features = extract_features(make_account([make_trip(fixes)]))
    assert features["max_jump_m"] >= 0.0
    assert all(math.isfinite(v) for v in features.values())


def test_max_jump_is_zero_for_physically_plausible_motion():
    # 100 m per 10 s = 10 m/s, far below the 250 km/h cap: no excess distance.
    features = extract_features(make_account([straight_line_trip(0, 0, 0)]))
    assert features["max_jump_m"] == 0.0


def test_max_jump_measures_distance_in_excess_of_the_hard_cap():
    # A 50 km hop in 10 s. The cap allows V_TRIP_MAX_MPS * 10 s; the feature
    # reports exactly the remainder, so the value is checkable by hand.
    dt = 10.0
    hop_m = 50_000.0
    fixes = [make_fix(0.0, 0.0, 0.0), make_fix(dt, hop_m, 0.0)]
    features = extract_features(make_account([make_trip(fixes)]))
    assert features["max_jump_m"] == pytest.approx(hop_m - dt * V_TRIP_MAX_MPS)


# ---------------------------------------------------------------------------
# Planar frame: the antimeridian / pole pathologies must be absent
# ---------------------------------------------------------------------------

def test_planar_frame_has_no_antimeridian_discontinuity():
    # Guard test. In lat/lon, a step from +179 to -179 degrees is a 2-degree
    # move that naive subtraction reads as 358 degrees. This harness uses a
    # local planar frame in metres, so no such wrap exists: two trips with a
    # large negative x-coordinate delta behave exactly like the mirrored pair
    # with a large positive delta. If someone migrates the schema to lat/lon
    # without adding great-circle distance, this test fails and says why.
    eastward = make_account([
        straight_line_trip(0, -1_000.0, 0.0, n=6),
        straight_line_trip(600.0, 1_000.0, 0.0, n=6),
    ])
    westward = make_account([
        straight_line_trip(0, 1_000.0, 0.0, n=6, step_m=-100.0),
        straight_line_trip(600.0, -1_000.0, 0.0, n=6, step_m=-100.0),
    ])
    east_features = extract_features(eastward)
    west_features = extract_features(westward)
    # Displacement magnitudes are sign-symmetric, so the geometry features
    # must match exactly under the mirror.
    assert east_features["max_jump_m"] == pytest.approx(west_features["max_jump_m"])
    assert east_features["handover_gap_score"] == pytest.approx(
        west_features["handover_gap_score"])


def test_features_are_translation_invariant():
    # A planar frame has no privileged origin (no "pole"): shifting an entire
    # account by a constant vector must not change any physics feature. This
    # is the property that would break under a spherical parameterisation.
    trips = [straight_line_trip(0, 0.0, 0.0), straight_line_trip(3600.0, 2_000.0, 0.0)]
    shifted = [straight_line_trip(0, 30_000.0, -25_000.0),
               straight_line_trip(3600.0, 32_000.0, -25_000.0)]
    origin_features = extract_features(make_account(trips))
    shifted_features = extract_features(make_account(shifted))
    for key in ("max_jump_m", "max_speed_kmh", "accel_p99_ms2", "gps_jitter_m",
                "handover_gap_score", "overlap_rate", "night_share"):
        assert origin_features[key] == pytest.approx(shifted_features[key]), key


def test_large_coordinates_do_not_overflow():
    # Coordinates far outside the synthetic world's +-45 km plane must still
    # produce finite features (a real export may use a different projection
    # origin with much larger values).
    fixes = [make_fix(i * 10.0, 1e7 + i * 100.0, -1e7, speed=10.0) for i in range(8)]
    features = extract_features(make_account([make_trip(fixes)]))
    assert all(math.isfinite(v) for v in features.values())


# ---------------------------------------------------------------------------
# Inter-trip geometry: handovers and overlaps
# ---------------------------------------------------------------------------

def test_single_trip_has_no_handover_or_overlap_signal():
    # Both features need at least two trips to be defined.
    features = extract_features(make_account([straight_line_trip(0, 0, 0)]))
    assert features["handover_gap_score"] == 0.0
    assert features["overlap_rate"] == 0.0


def test_colocated_consecutive_trips_score_no_handover():
    # Trip 2 starts where trip 1 ended, an hour later: nothing implausible.
    first = straight_line_trip(0.0, 0.0, 0.0, n=6)          # ends at x = 500
    second = straight_line_trip(3600.0, 500.0, 0.0, n=6)
    features = extract_features(make_account([first, second]))
    assert features["handover_gap_score"] == 0.0
    assert features["overlap_rate"] == 0.0


def test_teleport_between_trips_saturates_the_handover_score():
    # Trip 2 starts 40 km away 60 s after trip 1 ended: far beyond what the
    # generous 54 km/h door-to-door allowance covers, so the score saturates.
    first = straight_line_trip(0.0, 0.0, 0.0, n=6)          # ends at t=50, x=500
    second = straight_line_trip(110.0, 40_000.0, 0.0, n=6)
    features = extract_features(make_account([first, second]))
    assert features["handover_gap_score"] == pytest.approx(1.0)


def test_handover_score_matches_its_hand_computed_definition():
    # excess = gap_distance - max(gap_time, GAP_FLOOR_S) * V_GAP_MAX_MPS,
    # then divided by HANDOVER_SAT_M and clipped to 1. Worked by hand here so
    # a change to the constants surfaces as a failing expectation.
    first = straight_line_trip(0.0, 0.0, 0.0, n=6)   # ends t = 50 s, x = 500 m
    gap_t = 100.0
    gap_d = 3_000.0
    second = straight_line_trip(50.0 + gap_t, 500.0 + gap_d, 0.0, n=6)
    expected_excess = gap_d - max(gap_t, GAP_FLOOR_S) * V_GAP_MAX_MPS
    expected = min(max(expected_excess, 0.0) / HANDOVER_SAT_M, 1.0)
    features = extract_features(make_account([first, second]))
    assert features["handover_gap_score"] == pytest.approx(expected)


def test_overlapping_trips_are_detected():
    # Trip 2 starts before trip 1 ends by more than the tolerance: one
    # account driving two trips at once.
    first = straight_line_trip(0.0, 0.0, 0.0, n=10)     # spans t = 0..90
    second = straight_line_trip(30.0, 0.0, 0.0, n=10)   # starts mid-trip-1
    features = extract_features(make_account([first, second]))
    assert features["overlap_rate"] == pytest.approx(1.0)


def test_small_overlap_within_tolerance_is_not_flagged():
    # A 10 s overlap is inside OVERLAP_TOL_S (30 s): clock skew and upload
    # rounding must not manufacture a fraud signal.
    first = straight_line_trip(0.0, 0.0, 0.0, n=10)      # ends at t = 90
    second = straight_line_trip(80.0, 900.0, 0.0, n=10)  # 10 s overlap
    features = extract_features(make_account([first, second]))
    assert features["overlap_rate"] == 0.0


def test_trips_are_sorted_by_start_time_before_the_inter_trip_pass():
    # Feature extraction sorts trips by t_start, so a shuffled export must
    # give the same answer as the ordered one.
    first = straight_line_trip(0.0, 0.0, 0.0, n=6)
    second = straight_line_trip(3600.0, 500.0, 0.0, n=6)
    ordered = extract_features(make_account([first, second]))
    shuffled = extract_features(make_account([second, first]))
    assert ordered == shuffled


# ---------------------------------------------------------------------------
# Rate features
# ---------------------------------------------------------------------------

def test_mock_rate_is_the_fraction_of_flagged_fixes():
    fixes = [make_fix(i * 10.0, i * 100.0, 0.0, is_mock=(i < 3)) for i in range(10)]
    features = extract_features(make_account([make_trip(fixes)]))
    assert features["mock_rate"] == pytest.approx(0.3)


def test_cell_mismatch_rate_is_zero_when_cells_match_position():
    # make_fix derives cell_id from (x, y) by default, so a clean trip has no
    # mismatch at all.
    features = extract_features(make_account([straight_line_trip(0, 0, 0)]))
    assert features["cell_mismatch_rate"] == 0.0


def test_cell_mismatch_rate_detects_radio_position_disagreement():
    # GPS says one place, the serving cell says another: the spoofing tell.
    fixes = [make_fix(i * 10.0, i * 100.0, 0.0, cell_id=999_999) for i in range(10)]
    features = extract_features(make_account([make_trip(fixes)]))
    assert features["cell_mismatch_rate"] == pytest.approx(1.0)


def test_night_share_counts_trips_starting_before_06_00():
    # t is seconds from midnight of day 0; 02:00 is night, 12:00 is not.
    night = straight_line_trip(2 * 3600.0, 0.0, 0.0)
    day = straight_line_trip(12 * 3600.0, 0.0, 0.0)
    assert extract_features(make_account([night]))["night_share"] == pytest.approx(1.0)
    assert extract_features(make_account([day]))["night_share"] == 0.0
    both = extract_features(make_account([night, day]))
    assert both["night_share"] == pytest.approx(0.5)


def test_night_share_uses_time_of_day_not_absolute_time():
    # A trip on day 5 at 02:00 is still a night trip.
    late = straight_line_trip(5 * 86_400.0 + 2 * 3600.0, 0.0, 0.0)
    assert extract_features(make_account([late]))["night_share"] == pytest.approx(1.0)


def test_style_bimodality_is_zero_below_the_minimum_trip_count():
    # Fewer than the minimum number of scorable trips: a two-way split would
    # be pure noise, so the feature must abstain rather than guess.
    trips = [straight_line_trip(i * 3600.0, 0.0, 0.0, n=10) for i in range(3)]
    assert extract_features(make_account(trips))["style_bimodality"] == 0.0


def test_style_bimodality_is_zero_for_a_single_consistent_driver():
    # Ten identical trips have no variance to split, so the null-baseline
    # subtraction must drive the feature to exactly zero.
    trips = [straight_line_trip(i * 3600.0, 0.0, 0.0, n=12) for i in range(10)]
    assert extract_features(make_account(trips))["style_bimodality"] == 0.0


# ---------------------------------------------------------------------------
# Non-finite input: DEFECT-3
# ---------------------------------------------------------------------------

def _teleporting_account(bad_value=None):
    """Account whose trip contains a 500 km jump, optionally poisoned.

    With bad_value=None this is an obvious in-trip teleport. With
    bad_value=nan/inf the same teleport also carries one non-finite fix.
    """
    fixes = [
        make_fix(0.0, 0.0, 0.0, speed=10.0),
        make_fix(10.0, 500_000.0 if bad_value is None else bad_value, 0.0, speed=10.0),
        make_fix(20.0, 500_000.0, 0.0, speed=10.0),
    ]
    return make_account([make_trip(fixes)])


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_non_finite_coordinates_still_yield_finite_features():
    # The output contract holds even on poisoned input: every value is a
    # finite float. (Whether the *value* is right is DEFECT-3, below.)
    #
    # The suppressed RuntimeWarnings are part of the finding: cell_id_from_xy
    # casts non-finite coordinates straight to int64, which numpy flags as an
    # invalid cast and an overflow. They are silenced here so the suite stays
    # readable; the underlying behaviour is DEFECT-3's territory.
    for bad in (float("nan"), float("inf"), float("-inf")):
        features = extract_features(_teleporting_account(bad))
        assert set(features) == set(FEATURE_NAMES)
        for key, value in features.items():
            assert math.isfinite(value), f"{key} not finite for {bad}"


def test_non_finite_speed_still_yields_finite_features():
    fixes = [
        make_fix(0.0, 0.0, 0.0, speed=float("nan")),
        make_fix(10.0, 100.0, 0.0, speed=10.0),
        make_fix(20.0, 200.0, 0.0, speed=10.0),
    ]
    features = extract_features(make_account([make_trip(fixes)]))
    assert all(math.isfinite(v) for v in features.values())


def test_clean_teleport_is_detected_as_a_large_jump():
    # Control for the NaN cases below: without the NaN, the teleport is caught.
    features = extract_features(_teleporting_account(None))
    assert features["max_jump_m"] > 100_000.0


# DEFECT-3 (FIXED): a single non-finite coordinate used to ERASE the in-trip
# teleport signal for the whole account instead of isolating the bad fix.
# np.hypot propagated NaN into the running max and the "finite or 0.0" guard
# rewrote the poisoned max_jump_m to 0.0 -- the cleanest possible value.
# Measured at the time: a 500 km in-trip jump scored max_jump_m = 499306 m
# (score 0.69); injecting one NaN fix into the same trip gave max_jump_m =
# 0.0 (score 0.27), so an adversary lowered their suspicion by emitting
# garbage. Non-finite fixes are now dropped before differencing and the
# discard rate is reported as nonfinite_rate.

@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_nan_injection_does_not_lower_suspicion():
    clean_score = score_account(extract_features(_teleporting_account(None)))
    poisoned_score = score_account(extract_features(_teleporting_account(float("nan"))))
    assert poisoned_score >= clean_score, (
        f"NaN injection lowered the score: {clean_score} -> {poisoned_score}")


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@pytest.mark.parametrize("poison", [float("nan"), float("inf"), float("-inf")])
def test_no_non_finite_flavour_lowers_suspicion(poison):
    # The evasion must be closed for every flavour of garbage, not just NaN.
    clean_score = score_account(extract_features(_teleporting_account(None)))
    poisoned_score = score_account(extract_features(_teleporting_account(poison)))
    assert poisoned_score >= clean_score, (
        f"{poison} injection lowered the score: {clean_score} -> {poisoned_score}")


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_the_surviving_fixes_still_show_the_in_trip_teleport():
    # Positive statement of the fix: dropping the bad fix leaves the two
    # clean fixes 500 km apart, so the jump is still measured -- the signal
    # is preserved, not merely prevented from inverting.
    poisoned = extract_features(_teleporting_account(float("nan")))
    assert poisoned["max_jump_m"] > 100_000.0
    assert poisoned["nonfinite_rate"] > 0.0
