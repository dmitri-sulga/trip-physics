"""Re-run part of the attack registry from this repository.

The verdicts in ``attacks/registry.json`` were first measured with one-off
scripts that are not part of this repository. This module re-implements the
manipulations that can be written down unambiguously from the registry's
``params`` text and the public API, and re-measures them under the
registry's protocol:

* honest population: 3,000 accounts, ``shared_fraction=0.0``, seed 4242;
  threshold = ``threshold_at_fp_budget`` at 50 FP per 10k honest accounts;
* attack population: 300 accounts, ``shared_fraction=1.0``, seed 777;
* "caught" = score strictly above that threshold, out of 300.

Every manipulation listed in :data:`MANIPULATIONS` reproduces the count in
the registry exactly; those entries carry a ``reproduce`` field and
``ATTACKS.md`` says so. Entries without that field are a record of what was
run and cannot be re-run from here: some need fabricated trips or random
draws whose exact form was not recorded, and the simulator-setting entries
(``tier-*``, ``mode-*`` and similar) need generator parameters that
``generate_dataset`` does not expose.

Usage::

    python -m tripphysics.attack_measure              # every reproducible entry
    python -m tripphysics.attack_measure split-trips  # one or more ids

About 40 s for the populations, then a few seconds per entry.
"""

from __future__ import annotations

import copy
import math
import statistics
import sys
from typing import Any, Callable

from .detector import score_account
from .eval import threshold_at_fp_budget
from .features import NIGHT_END_H, NIGHT_START_H, V_TRIP_MAX_MPS, extract_features
from .synth import DAY_S, cell_id_from_xy, generate_dataset

__all__ = ["MANIPULATIONS", "PROTOCOL", "measure", "reference"]

Account = dict[str, Any]

PROTOCOL = {
    "honest_accounts": 3_000, "honest_seed": 4242,
    "attack_accounts": 300, "attack_seed": 777,
    "budget_fp_per_10k": 50.0,
}

SPEED_CAP_MPS = 140.0 / 3.6      # 140 km/h
NIGHT_SHIFT_S = 10 * 3600.0      # shift-night-trips: forward by 10 hours
DEOVERLAP_GAP_S = 120.0          # deoverlap-restamp: start 120 s after the max end


def _fixes(account: Account):
    for trip in account["trips"]:
        yield from trip["fixes"]


def silence_no_trips(account: Account) -> Account:
    account["trips"] = []
    return account


def silence_empty_fixes(account: Account) -> Account:
    for trip in account["trips"]:
        trip["fixes"] = []
    return account


def clear_mock(account: Account) -> Account:
    for fix in _fixes(account):
        fix["is_mock"] = False
    return account


def forge_cell(account: Account) -> Account:
    for fix in _fixes(account):
        fix["cell_id"] = int(cell_id_from_xy(fix["x_m"], fix["y_m"]))
    return account


def cap_speed(account: Account) -> Account:
    for fix in _fixes(account):
        fix["speed_mps"] = min(fix["speed_mps"], SPEED_CAP_MPS)
    return account


def drop_endpoint_fixes(account: Account) -> Account:
    for trip in account["trips"]:
        if len(trip["fixes"]) >= 10:
            trip["fixes"] = trip["fixes"][3:-3]
    return account


def split_trips(account: Account) -> Account:
    """Cut every trip of 12+ fixes in two at the midpoint fix."""
    out = []
    for trip in account["trips"]:
        fixes = trip["fixes"]
        if len(fixes) >= 12:
            mid = len(fixes) // 2
            out.append({**trip, "fixes": fixes[:mid]})
            out.append({**trip, "trip_id": trip["trip_id"] + "_b",
                        "t_start": fixes[mid]["t"], "fixes": fixes[mid:]})
        else:
            out.append(trip)
    account["trips"] = out
    return account


def shift_night_trips(account: Account) -> Account:
    """Move every trip that starts in the night window 10 hours later."""
    for trip in account["trips"]:
        hour = (float(trip["t_start"]) % DAY_S) / 3600.0
        if NIGHT_START_H <= hour < NIGHT_END_H:
            trip["t_start"] += NIGHT_SHIFT_S
            for fix in trip["fixes"]:
                fix["t"] += NIGHT_SHIFT_S
    return account


def kill_in_trip_teleports(account: Account) -> Account:
    """Rescale any fix-to-fix step above 250 km/h to exactly the cap.

    Every later fix in the trip moves by the same vector, so the rest of the
    track stays continuous.
    """
    for trip in account["trips"]:
        fixes = trip["fixes"]
        for i in range(1, len(fixes)):
            dt = max(fixes[i]["t"] - fixes[i - 1]["t"], 1.0)
            dx = fixes[i]["x_m"] - fixes[i - 1]["x_m"]
            dy = fixes[i]["y_m"] - fixes[i - 1]["y_m"]
            dist = math.hypot(dx, dy)
            cap = dt * V_TRIP_MAX_MPS
            if dist > cap:
                k = cap / dist
                shift_x, shift_y = dx * k - dx, dy * k - dy
                for fix in fixes[i:]:
                    fix["x_m"] += shift_x
                    fix["y_m"] += shift_y
    return account


def deoverlap_restamp(account: Account) -> Account:
    """Restamp any trip that starts before the running maximum end time."""
    trips = sorted(account["trips"], key=lambda t: t["t_start"])
    max_end = -math.inf
    for trip in trips:
        if not trip["fixes"]:
            continue
        start = trip["fixes"][0]["t"]
        if start < max_end:
            shift = max_end + DEOVERLAP_GAP_S - start
            trip["t_start"] += shift
            for fix in trip["fixes"]:
                fix["t"] += shift
        max_end = max(max_end, trip["fixes"][-1]["t"])
    account["trips"] = trips
    return account


def full_client_evasion(account: Account) -> Account:
    """De-overlap, clear mock flags, forge cells, then kill teleports and cap speed.

    The order is the registry's: cell ids are forged from positions before
    teleports are rescaled, so fixes moved by the rescale keep their old cell.
    """
    for step in (deoverlap_restamp, clear_mock, forge_cell,
                 kill_in_trip_teleports, cap_speed):
        account = step(account)
    return account


MANIPULATIONS: dict[str, Callable[[Account], Account]] = {
    "silence-no-trips": silence_no_trips,
    "silence-empty-fixes": silence_empty_fixes,
    "client-mock-clear": clear_mock,
    "client-cell-forge": forge_cell,
    "speed-cap-under-center": cap_speed,
    "drop-endpoint-fixes": drop_endpoint_fixes,
    "split-trips": split_trips,
    "shift-night-trips": shift_night_trips,
    "kill-in-trip-teleports": kill_in_trip_teleports,
    "deoverlap-restamp": deoverlap_restamp,
    "full-client-evasion": full_client_evasion,
}


# full-client-evasion's registry evidence decomposes the result step by step;
# these are the partial chains it names, in the same order.
DECOMPOSITION: tuple[tuple[str, tuple[Callable[[Account], Account], ...]], ...] = (
    ("de-overlap alone", (deoverlap_restamp,)),
    ("mock-clear plus cell-forge", (clear_mock, forge_cell)),
    ("the three together", (deoverlap_restamp, clear_mock, forge_cell)),
    ("adding teleport re-anchoring",
     (deoverlap_restamp, clear_mock, forge_cell, kill_in_trip_teleports)),
)


def _chain(steps):
    def apply(account: Account) -> Account:
        for step in steps:
            account = step(account)
        return account
    return apply


def _score(accounts: list[Account]) -> list[float]:
    return [score_account(extract_features(a)) for a in accounts]


def reference() -> dict[str, Any]:
    """Threshold from the honest population and the unattacked attack population."""
    honest = generate_dataset(n_accounts=PROTOCOL["honest_accounts"],
                              shared_fraction=0.0,
                              seed=PROTOCOL["honest_seed"])["accounts"]
    honest_scores = _score(honest)
    threshold = threshold_at_fp_budget(honest_scores, PROTOCOL["budget_fp_per_10k"])
    shared = generate_dataset(n_accounts=PROTOCOL["attack_accounts"],
                              shared_fraction=1.0,
                              seed=PROTOCOL["attack_seed"])["accounts"]
    scores = _score(shared)
    return {"threshold": threshold,
            "honest_median": statistics.median(honest_scores),
            "accounts": shared,
            "caught": sum(s > threshold for s in scores),
            "median": statistics.median(scores)}


def measure(attack_id: str, ref: dict[str, Any],
            manipulate: Callable[[Account], Account] | None = None) -> dict[str, Any]:
    """Apply one manipulation to a copy of every attack account and count catches."""
    manipulate = manipulate or MANIPULATIONS[attack_id]
    attacked = [manipulate(copy.deepcopy(a)) for a in ref["accounts"]]
    scores = _score(attacked)
    return {"id": attack_id,
            "caught": sum(s > ref["threshold"] for s in scores),
            "of": len(scores),
            "median": statistics.median(scores)}


def main(argv: list[str] | None = None) -> None:
    ids = (argv if argv is not None else sys.argv[1:]) or list(MANIPULATIONS)
    unknown = [i for i in ids if i not in MANIPULATIONS]
    if unknown:
        raise SystemExit(f"not reproducible from this repository: {unknown}; "
                         f"reproducible ids: {list(MANIPULATIONS)}")
    ref = reference()
    print(f"threshold {ref['threshold']:.6f} (honest median "
          f"{ref['honest_median']:.2f}); unattacked "
          f"{ref['caught']}/{len(ref['accounts'])} caught, median score "
          f"{ref['median']:.2f}")
    for attack_id in ids:
        r = measure(attack_id, ref)
        print(f"{attack_id:24s} {r['caught']}/{r['of']} caught, "
              f"median score {r['median']:.2f}", flush=True)
        if attack_id == "full-client-evasion":
            for label, steps in DECOMPOSITION:
                d = measure(attack_id, ref, _chain(steps))
                print(f"  {label:30s} {d['caught']}/{d['of']} caught", flush=True)


if __name__ == "__main__":
    main()
