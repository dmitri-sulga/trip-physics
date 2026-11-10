"""Unit checks for the manipulations in tripphysics.attack_measure.

Each manipulation is checked on a small hand-built account for the one
property its registry entry describes. The population-level counts (for
example 0/300 for silence-no-trips) are re-measured by
`scripts/check-claimed-numbers.sh --slow`, not here: they need the 3,000- and
300-account populations and take about a minute.
"""

import copy

import pytest

from tripphysics.attack_measure import (
    MANIPULATIONS,
    SPEED_CAP_MPS,
    clear_mock,
    cap_speed,
    deoverlap_restamp,
    drop_endpoint_fixes,
    forge_cell,
    full_client_evasion,
    kill_in_trip_teleports,
    shift_night_trips,
    silence_empty_fixes,
    silence_no_trips,
    split_trips,
)
from tripphysics.detector import score_account
from tripphysics.features import extract_features
from tripphysics.synth import cell_id_from_xy


def _trip(trip_id, t0, x0, n=14, step_m=100.0, dt=10.0, mock=False, cell=1):
    fixes = [{"t": t0 + i * dt, "x_m": x0 + i * step_m, "y_m": 0.0,
              "accuracy_m": 5.0, "speed_mps": step_m / dt + (30.0 if i == 3 else 0.0),
              "is_mock": mock, "cell_id": cell} for i in range(n)]
    return {"trip_id": trip_id, "account_id": "a", "t_start": t0, "fixes": fixes}


def _account():
    # Two trips that overlap in time (the second starts before the first
    # ends), mock flags and wrong cell ids, one trip at night (01:00), and an
    # in-trip teleport of 50 km between fixes 5 and 6 of the second trip.
    first = _trip("t1", 3600.0, 0.0, mock=True)
    second = _trip("t2", 3650.0, 20_000.0)
    for fix in second["fixes"][6:]:
        fix["x_m"] += 50_000.0
    return {"account_id": "a", "is_shared": True, "trips": [first, second]}


@pytest.mark.parametrize("name", sorted(MANIPULATIONS))
def test_manipulation_is_a_function_of_its_copy(name):
    account = _account()
    before = copy.deepcopy(account)
    MANIPULATIONS[name](copy.deepcopy(account))
    assert account == before


def test_silence_scores_exactly_zero():
    assert score_account(extract_features(silence_no_trips(_account()))) == 0.0
    assert score_account(extract_features(silence_empty_fixes(_account()))) == 0.0


def test_client_field_forgeries():
    a = forge_cell(clear_mock(cap_speed(_account())))
    fixes = [f for t in a["trips"] for f in t["fixes"]]
    assert not any(f["is_mock"] for f in fixes)
    assert all(f["cell_id"] == int(cell_id_from_xy(f["x_m"], f["y_m"])) for f in fixes)
    assert max(f["speed_mps"] for f in fixes) <= SPEED_CAP_MPS


def test_drop_endpoint_fixes_removes_three_from_each_end():
    a = drop_endpoint_fixes(_account())
    assert [len(t["fixes"]) for t in a["trips"]] == [8, 8]


def test_split_trips_cuts_at_the_midpoint_fix():
    a = split_trips(_account())
    assert [len(t["fixes"]) for t in a["trips"]] == [7, 7, 7, 7]
    assert a["trips"][1]["t_start"] == a["trips"][1]["fixes"][0]["t"]


def test_shift_night_trips_moves_the_night_trip_by_ten_hours():
    a = shift_night_trips(_account())
    assert a["trips"][0]["t_start"] == 3600.0 + 36_000.0
    assert a["trips"][0]["fixes"][0]["t"] == 3600.0 + 36_000.0


def test_kill_in_trip_teleports_removes_the_jump():
    assert extract_features(_account())["max_jump_m"] > 40_000.0
    assert extract_features(kill_in_trip_teleports(_account()))["max_jump_m"] == 0.0


def test_deoverlap_restamp_removes_the_overlap():
    assert extract_features(_account())["overlap_rate"] > 0.0
    a = deoverlap_restamp(_account())
    assert extract_features(a)["overlap_rate"] == 0.0
    assert a["trips"][1]["fixes"][0]["t"] == a["trips"][0]["fixes"][-1]["t"] + 120.0


def test_full_client_evasion_lowers_the_score():
    before = score_account(extract_features(_account()))
    after = score_account(extract_features(full_client_evasion(_account())))
    assert after < before
