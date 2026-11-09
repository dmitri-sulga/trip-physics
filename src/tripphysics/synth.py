"""Synthetic trip-log corpus with known profile-sharing ground truth.

This module fabricates account-level GPS trip logs so that the rest of the
harness (features -> detector -> evaluation -> report) can be exercised
end-to-end without touching any production data.

Honesty note
------------
The synthetic corpus proves that the *harness* works: the plumbing, the
metric math, the determinism, the report. It does NOT prove that the
detector works on real traffic -- that is exactly what the 30-day shadow
window is for. Distributions below are tuned to be *plausible*, not to
match any operator's fleet.

What a "shared" account looks like here
---------------------------------------
A shared (rented / multi-driver) account mixes two latent driver profiles:

* handover discontinuities -- trip N ends in one part of town, trip N+1
  starts kilometres away after a gap too short to have driven it;
* bimodal driving style -- two cruise-speed / volatility signatures;
* occasional mock-location trips -- flagged fixes, suspiciously smooth
  tracks, serving-cell ids that disagree with the reported position,
  mid-trip position re-anchoring (teleports);
* occasional overlapping trips -- two drivers active at the same time.

Shared accounts come in three hardness tiers ("blatant", "moderate",
"subtle") so the detector cannot win by memorising one caricature. Honest
accounts are noisy on purpose: GPS multipath spikes, cold-start fixes that
are kilometres off, night-shift drivers, rare duplicated-upload overlaps.
Class distributions overlap; separation is earned, not given.

Determinism
-----------
Every random draw flows from `numpy.random.default_rng` streams derived
from the caller's seed (one independent child stream per account, keyed by
`[seed, 1, account_index]`). Same seed => byte-identical dataset. No
wall-clock anywhere; all timestamps are synthetic seconds from a synthetic
epoch (t=0 is midnight of day 0, local time).

All emitted structures are plain dicts/lists of Python scalars and are
JSON-serialisable as-is.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["generate_dataset", "cell_id_from_xy", "CELL_SIZE_M"]

# ---------------------------------------------------------------------------
# World constants
# ---------------------------------------------------------------------------
DAY_S = 86_400.0                # seconds per synthetic day
SIM_START_SPREAD_DAYS = 10.0    # accounts begin activity within the first days
FIX_INTERVAL_S = 10.0           # nominal telemetry cadence (jittered +-2 s)
WORLD_HALF_M = 45_000.0         # simulation plane is [-45 km, +45 km]^2
CELL_SIZE_M = 2_000.0           # serving-cell grid pitch (metres)
_CELL_OFFSET = 500              # keeps packed grid indices positive

# Trip length: lognormal around ~4.7 min, clipped to [2, 10] min. Short city
# hops dominate; the cap keeps the default corpus around ~1M fixes so the
# demo stays fast and memory-friendly.
_TRIP_MEDIAN_S = 280.0
_TRIP_SIGMA = 0.45
_TRIP_MIN_S, _TRIP_MAX_S = 120.0, 600.0

_V_HARD_CAP_MPS = 36.0          # ~130 km/h: reported speed ceiling
_P_HIGHWAY_TRIP = 0.10          # some trips run at highway cruise
_P_MULTIPATH_FIX = 0.003        # honest per-fix chance of a noise blow-up
_P_COLD_START_TRIP = 0.004      # honest per-trip chance of a km-scale first fix
_P_FALSE_MOCK_FIX = 0.0008      # OEM quirk: honest fix wrongly flagged mock
_P_NEIGHBOR_CELL_FIX = 0.02     # handover lag: cell id one grid step behind
_P_HONEST_OVERLAP_GAP = 0.0015  # duplicated-upload artifact in honest logs
_P_HONEST_NIGHT_DRIVER = 0.12   # honest night-shift subpopulation
_P_HONEST_RELOCATION = 0.04     # per long break: driver moves across town

# ---------------------------------------------------------------------------
# Shared-account hardness tiers.
#
# p_switch        chance, at each inter-trip gap, that the phone changes hands
# switch_modes    given a switch, probabilities of (overlap, remote, meet);
#                 the remainder is a "shift change":
#                   overlap -- second driver starts before the first ends
#                   remote  -- next trip starts across town after a gap far
#                              too short to have driven it (damning geometry)
#                   meet    -- drivers meet and pass the phone: co-located,
#                              geometrically invisible, only style/night help
#                   shift change -- incoming driver waits for their own shift:
#                              hours of gap, plausible geometry, but the
#                              night/style split shows up in aggregates
# zone_sep_km     distance between the two drivers' operating zones
# handover_gap_min  minutes between trips on a remote handover
# style_sep_mps   cruise-speed separation between the two driver profiles
# p_mock_trip     chance a second-driver trip runs through a location spoofer
# p_night_split   chance the second driver works nights (day+night pattern)
# ---------------------------------------------------------------------------
_TIERS: dict[str, dict[str, Any]] = {
    "blatant": {
        "p_switch": 0.62, "switch_modes": (0.30, 0.55, 0.08),
        "zone_sep_km": (8.0, 18.0), "handover_gap_min": (2.0, 8.0),
        "style_sep_mps": 3.0, "p_mock_trip": 0.30, "p_night_split": 0.70,
    },
    "moderate": {
        "p_switch": 0.42, "switch_modes": (0.16, 0.52, 0.12),
        "zone_sep_km": (5.0, 12.0), "handover_gap_min": (4.0, 14.0),
        "style_sep_mps": 2.0, "p_mock_trip": 0.10, "p_night_split": 0.55,
    },
    "subtle": {
        "p_switch": 0.28, "switch_modes": (0.06, 0.40, 0.16),
        "zone_sep_km": (3.0, 8.0), "handover_gap_min": (8.0, 25.0),
        "style_sep_mps": 1.2, "p_mock_trip": 0.03, "p_night_split": 0.35,
    },
}
_TIER_NAMES = ("blatant", "moderate", "subtle")
_TIER_PROBS = (0.25, 0.45, 0.30)


# ---------------------------------------------------------------------------
# Serving-cell grid
# ---------------------------------------------------------------------------
def cell_id_from_xy(x_m, y_m):
    """Map a position to its serving-cell id in the synthetic world.

    The synthetic network is a square grid of `CELL_SIZE_M` cells; the id
    packs the two grid indices into one integer. Accepts scalars or numpy
    arrays (vectorised). On live data this mapping must be replaced by the
    operator's own cell-location join -- see `features.extract_features`.

    Caveat for live use: non-finite coordinates are NOT rejected here.
    Casting NaN/inf to int64 is undefined, and numpy quietly returns a
    plausible-looking id (NaN and -inf both yield 50000500, inf yields
    49900500) rather than failing. This helper is deliberately left
    permissive because test fixtures and callers build poisoned fixes with
    it on purpose; the guarantee that no such coordinate ever reaches a
    cell comparison is enforced one level up, where `extract_features`
    discards non-finite fixes before the mismatch join. If you swap this
    for a real operator cell-location join, keep that filter in place.
    """
    ix = np.floor(np.asarray(x_m) / CELL_SIZE_M).astype(np.int64) + _CELL_OFFSET
    iy = np.floor(np.asarray(y_m) / CELL_SIZE_M).astype(np.int64) + _CELL_OFFSET
    packed = ix * 100_000 + iy
    if np.ndim(packed) == 0:
        return int(packed)
    return packed


# ---------------------------------------------------------------------------
# Driver styles and shift patterns
# ---------------------------------------------------------------------------
def _draw_base_style(rng: np.random.Generator) -> dict[str, float]:
    """One driver's latent signature: cruise speed, volatility, GPS quality."""
    jitter = float(np.clip(rng.lognormal(np.log(6.0), 0.35), 2.5, 18.0))
    return {
        "cruise_mps": float(np.clip(rng.normal(11.5, 1.7), 7.0, 17.0)),
        "sigma_v": float(np.clip(rng.normal(2.0, 0.45), 1.1, 3.2)),  # speed volatility
        "p_stop": float(rng.uniform(0.04, 0.09)),                    # traffic stops
        "jitter_m": jitter,                                          # GPS noise (1-sigma, per axis)
        "acc_base_m": jitter * 1.3,                                  # reported accuracy scale
    }


def _derive_second_style(rng: np.random.Generator, base: dict[str, float],
                         sep_mps: float) -> dict[str, float]:
    """Second driver on the same account: shifted cruise, different volatility.

    Same handset, so GPS jitter stays close to the first driver's -- the
    style split must come from *driving*, not from device artifacts.
    """
    sign = 1.0 if rng.random() < 0.5 else -1.0
    return {
        "cruise_mps": float(np.clip(
            base["cruise_mps"] + sign * sep_mps * rng.uniform(0.8, 1.2), 6.0, 18.0)),
        "sigma_v": float(np.clip(
            base["sigma_v"] * (1.0 + 0.22 * sep_mps * rng.uniform(0.6, 1.2)), 1.1, 4.5)),
        "p_stop": float(np.clip(base["p_stop"] * rng.uniform(0.8, 1.6), 0.03, 0.15)),
        "jitter_m": float(np.clip(base["jitter_m"] * rng.uniform(0.9, 1.15), 2.5, 20.0)),
        "acc_base_m": base["acc_base_m"],
    }


def _draw_shift(rng: np.random.Generator, night: bool) -> dict[str, float]:
    """Working-hours pattern: centre hour and spread of trip start times."""
    if night:
        return {"center_h": float(rng.normal(0.8, 1.5) % 24.0), "sd_h": 1.9}
    return {"center_h": float(rng.uniform(7.0, 17.0)), "sd_h": 2.5}


def _next_shift_start(rng: np.random.Generator, t_after: float,
                      shift: dict[str, float]) -> float:
    """First plausible shift start strictly after `t_after` (plus >=1 h rest)."""
    hour = rng.normal(shift["center_h"], shift["sd_h"]) % 24.0
    cand = math.floor(t_after / DAY_S) * DAY_S + hour * 3600.0
    while cand < t_after + 3600.0:
        cand += DAY_S
    return cand + float(rng.uniform(0.0, 900.0))


# ---------------------------------------------------------------------------
# Single-trip kinematics
# ---------------------------------------------------------------------------
def _simulate_trip(rng: np.random.Generator, trip_cruise: float,
                   style: dict[str, float], t0: float, start_xy: np.ndarray,
                   mock: bool, cell_anchor: np.ndarray | None):
    """Simulate one trip; returns (fixes, true_end_xy, t_end).

    Kinematics: speed follows a mean-reverting walk around `trip_cruise`
    with per-style volatility and random full stops (traffic lights);
    heading is a slow random walk; position integrates speed. Reported
    position adds per-style GPS noise. Mock trips are the spoofer
    caricature: glassy-smooth path, near-zero jitter, flagged fixes,
    mismatched serving cells, occasional km-scale re-anchoring teleports
    and absurd single-fix speeds.
    """
    duration = float(np.clip(rng.lognormal(np.log(_TRIP_MEDIAN_S), _TRIP_SIGMA),
                             _TRIP_MIN_S, _TRIP_MAX_S))
    n = max(8, int(round(duration / FIX_INTERVAL_S)))
    steps = rng.uniform(FIX_INTERVAL_S - 2.0, FIX_INTERVAL_S + 2.0, n - 1)
    t = t0 + np.concatenate(([0.0], np.cumsum(steps)))

    # --- speed: mean-reverting walk with stops (pre-drawn noise, tiny loop) ---
    sigma_v = style["sigma_v"] * (0.35 if mock else 1.0)   # spoofed tracks glide
    p_stop = 0.0 if mock else style["p_stop"]
    eps = rng.normal(0.0, sigma_v, n)
    stop_draw = rng.random(n)
    v = np.empty(n)
    v[0] = max(0.0, float(rng.normal(0.55 * trip_cruise, 2.0)))
    for i in range(1, n):
        vi = v[i - 1] + 0.25 * (trip_cruise - v[i - 1]) + eps[i]
        if stop_draw[i] < p_stop:
            vi = 0.0
        v[i] = _V_HARD_CAP_MPS if vi > _V_HARD_CAP_MPS else (0.0 if vi < 0.0 else vi)

    # --- position: integrate speed along a slowly wandering heading ---
    heading = rng.uniform(0.0, 2.0 * math.pi) + np.concatenate(
        ([0.0], np.cumsum(rng.normal(0.0, 0.22, n - 1))))
    seg = v[1:] * steps
    x_true = start_xy[0] + np.concatenate(([0.0], np.cumsum(seg * np.cos(heading[1:]))))
    y_true = start_xy[1] + np.concatenate(([0.0], np.cumsum(seg * np.sin(heading[1:]))))

    if mock:
        # Spoofer re-anchors the fake route mid-trip: everything downstream
        # of the teleport index shifts by a km-scale vector.
        candidates = np.arange(2, n - 1)
        n_tp = min(int(rng.integers(1, 4)), candidates.size)
        for idx in np.sort(rng.choice(candidates, size=n_tp, replace=False)):
            ang = rng.uniform(0.0, 2.0 * math.pi)
            hop = rng.uniform(2_000.0, 8_000.0)
            x_true[idx:] += hop * math.cos(ang)
            y_true[idx:] += hop * math.sin(ang)

    # --- reported position: GPS noise, multipath spikes, cold starts ---
    noise = rng.normal(0.0, 0.3 if mock else style["jitter_m"], (n, 2))
    mp = np.zeros(n, dtype=bool)
    if not mock:
        mp = rng.random(n) < _P_MULTIPATH_FIX
        noise[mp] *= rng.uniform(8.0, 25.0, (int(mp.sum()), 1))
    x = x_true + noise[:, 0]
    y = y_true + noise[:, 1]
    if not mock and rng.random() < _P_COLD_START_TRIP:
        # Receiver reports a stale cached position for the first fix or two,
        # then snaps back -- the classic honest source of km-scale jumps. This
        # is the single honest source of km-scale positional jumps in the
        # corpus; a separate per-fix re-lock spike was specified but never
        # added a distinct behavior, so it is not carried as a dead constant.
        k = int(rng.integers(1, 3))
        ang = rng.uniform(0.0, 2.0 * math.pi)
        off = rng.uniform(1_500.0, 6_000.0)
        x[:k] += off * math.cos(ang)
        y[:k] += off * math.sin(ang)
    np.clip(x, -WORLD_HALF_M, WORLD_HALF_M, out=x)
    np.clip(y, -WORLD_HALF_M, WORLD_HALF_M, out=y)

    # --- reported speed ---
    s_rep = np.clip(v + rng.normal(0.0, 0.5, n), 0.0, None)
    if mock:
        spike = rng.random(n) < 0.02   # spoof tools emit garbage speeds sometimes
        s_rep[spike] = rng.uniform(45.0, 75.0, int(spike.sum()))

    # --- mock flag and reported accuracy ---
    if mock:
        is_mock = rng.random(n) < 0.9              # flag flickers, never clean 100%
        accuracy = np.clip(rng.normal(3.0, 0.3, n), 1.5, None)  # suspiciously perfect
    else:
        is_mock = rng.random(n) < _P_FALSE_MOCK_FIX
        accuracy = rng.lognormal(np.log(style["acc_base_m"]), 0.3, n)
        accuracy[mp] *= rng.uniform(3.0, 8.0, int(mp.sum()))

    # --- serving cell ---
    if mock:
        # GPS says "route", the radio still talks to towers near the real
        # handset location: cells anchor kilometres away from the track.
        anchor = cell_anchor + rng.normal(0.0, 60.0, (n, 2))
        ix = np.floor(anchor[:, 0] / CELL_SIZE_M).astype(np.int64) + _CELL_OFFSET
        iy = np.floor(anchor[:, 1] / CELL_SIZE_M).astype(np.int64) + _CELL_OFFSET
    else:
        ix = np.floor(x / CELL_SIZE_M).astype(np.int64) + _CELL_OFFSET
        iy = np.floor(y / CELL_SIZE_M).astype(np.int64) + _CELL_OFFSET
        # Handover lag: id occasionally one grid step off the reported fix.
        flip = rng.random(n) < _P_NEIGHBOR_CELL_FIX
        delta = rng.choice(np.array([-1, 1]), size=n)
        axis_x = rng.random(n) < 0.5
        ix = np.where(flip & axis_x, ix + delta, ix)
        iy = np.where(flip & ~axis_x, iy + delta, iy)
    cells = ix * 100_000 + iy

    fixes = [
        {"t": ti, "x_m": xi, "y_m": yi, "accuracy_m": acc,
         "speed_mps": si, "is_mock": mi, "cell_id": ci}
        for ti, xi, yi, acc, si, mi, ci in zip(
            t.tolist(), x.tolist(), y.tolist(), accuracy.tolist(),
            s_rep.tolist(), is_mock.tolist(), cells.tolist())
    ]
    return fixes, np.array([x_true[-1], y_true[-1]]), float(t[-1])


# ---------------------------------------------------------------------------
# Account assembly
# ---------------------------------------------------------------------------
def _build_account(rng: np.random.Generator, account_id: str, is_shared: bool,
                   trips_range: tuple) -> dict[str, Any]:
    """Assemble one account: schedule, positions, drivers, trips."""
    n_trips = int(rng.integers(int(trips_range[0]), int(trips_range[1]) + 1))
    styles = [_draw_base_style(rng)]
    zones = [rng.uniform(-15_000.0, 15_000.0, 2)]   # driver's operating area

    if is_shared:
        tier = _TIERS[_TIER_NAMES[int(rng.choice(len(_TIER_NAMES), p=_TIER_PROBS))]]
        styles.append(_derive_second_style(rng, styles[0], tier["style_sep_mps"]))
        ang = rng.uniform(0.0, 2.0 * math.pi)
        sep = rng.uniform(*tier["zone_sep_km"]) * 1_000.0
        zone_b = zones[0] + sep * np.array([math.cos(ang), math.sin(ang)])
        zones.append(np.clip(zone_b, -WORLD_HALF_M + 5_000.0, WORLD_HALF_M - 5_000.0))
        shifts = [_draw_shift(rng, night=False),
                  _draw_shift(rng, night=rng.random() < tier["p_night_split"])]
    else:
        tier = None
        shifts = [_draw_shift(rng, night=rng.random() < _P_HONEST_NIGHT_DRIVER)]

    driver = 0
    t_next = _next_shift_start(rng, rng.uniform(0.0, SIM_START_SPREAD_DAYS) * DAY_S,
                               shifts[0])
    pos = zones[0] + rng.normal(0.0, 2_000.0, 2)
    t_end_max = t_next   # forward anchor so overlaps do not cascade backwards

    trips: list[dict[str, Any]] = []
    for k in range(n_trips):
        style = styles[driver]
        trip_cruise = float(np.clip(style["cruise_mps"] + rng.normal(0.0, 0.55), 5.0, 17.0))
        if rng.random() < _P_HIGHWAY_TRIP:
            trip_cruise = min(trip_cruise * 1.7, 32.0)

        mock = bool(is_shared and driver == 1 and rng.random() < tier["p_mock_trip"])
        cell_anchor = None
        if mock:
            # Real handset sits kilometres from the spoofed route.
            ang = rng.uniform(0.0, 2.0 * math.pi)
            far = rng.uniform(5_000.0, 20_000.0)
            cell_anchor = pos + far * np.array([math.cos(ang), math.sin(ang)])

        fixes, end_xy, t_end = _simulate_trip(
            rng, trip_cruise, style, t_next, pos, mock, cell_anchor)
        trips.append({"trip_id": f"{account_id}_t{k:03d}",
                      "account_id": account_id,
                      "t_start": fixes[0]["t"],
                      "fixes": fixes})
        if k == n_trips - 1:
            break

        t_start_this = fixes[0]["t"]
        t_anchor = max(t_end, t_end_max)
        t_end_max = t_anchor

        # ---- plan the next trip: who drives, when, from where ----
        if is_shared and rng.random() < tier["p_switch"]:
            driver = 1 - driver
            p_ov, p_rem, p_meet = tier["switch_modes"]
            u = rng.random()
            if u < p_ov:
                # Both drivers active at once: physically impossible for
                # one person, and the second start is across town.
                t_next = max(t_end - rng.uniform(60.0, 420.0), t_start_this + 30.0)
                pos = zones[driver] + rng.normal(0.0, 1_500.0, 2)
            elif u < p_ov + p_rem:
                # Remote handover: next trip starts across town after a gap
                # far too short to have driven the distance.
                t_next = t_anchor + rng.uniform(*tier["handover_gap_min"]) * 60.0
                pos = zones[driver] + rng.normal(0.0, 1_500.0, 2)
            elif u < p_ov + p_rem + p_meet:
                # Co-located handover: drivers meet and pass the phone.
                # Geometrically invisible -- only style/night can catch it.
                t_next = t_anchor + rng.uniform(600.0, 3_000.0)
                pos = end_xy + rng.normal(0.0, 300.0, 2)
            else:
                # Shift change: incoming driver starts on their own working
                # hours, from their own area. Hours of gap keep the geometry
                # plausible; the tell is the day/night and style split.
                t_next = _next_shift_start(rng, t_anchor, shifts[driver])
                pos = zones[driver] + rng.normal(0.0, 2_000.0, 2)
        else:
            # Same driver keeps going (this is the only branch honest
            # accounts ever take).
            r = rng.random()
            if r < _P_HONEST_OVERLAP_GAP:
                # Duplicated-upload artifact: co-located, tiny overlap.
                t_next = max(t_end - rng.uniform(30.0, 120.0), t_start_this + 30.0)
                pos = end_xy + rng.normal(0.0, 150.0, 2)
            elif r < _P_HONEST_OVERLAP_GAP + 0.55:
                # Short break between rides; next pickup is nearby.
                t_next = t_anchor + rng.uniform(300.0, 2_400.0)
                pos = end_xy + rng.normal(0.0, 400.0, 2)
            else:
                # Long break: rest until the next shift, restart near base.
                if not is_shared and rng.random() < _P_HONEST_RELOCATION:
                    # Honest cross-town move -- large distance but with
                    # hours of gap, so the geometry stays plausible.
                    ang = rng.uniform(0.0, 2.0 * math.pi)
                    hop = rng.uniform(8_000.0, 30_000.0)
                    zones[0] = np.clip(
                        zones[0] + hop * np.array([math.cos(ang), math.sin(ang)]),
                        -30_000.0, 30_000.0)
                t_next = _next_shift_start(rng, t_anchor, shifts[driver])
                pos = zones[driver] + rng.normal(0.0, 2_000.0, 2)
        t_end_max = max(t_end_max, t_end)

    return {"account_id": account_id, "is_shared": bool(is_shared), "trips": trips}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate_dataset(n_accounts: int = 2000, shared_fraction: float = 0.03,
                     trips_per_account: tuple = (6, 30),
                     seed: int = 42) -> dict[str, Any]:
    """Build a labelled synthetic corpus of driver accounts.

    Parameters
    ----------
    n_accounts : total accounts to fabricate.
    shared_fraction : fraction of accounts with profile-sharing behaviour
        (ground truth label `is_shared`). Rounded to the nearest count.
    trips_per_account : inclusive (min, max) trips per account, uniform.
    seed : master seed; the same seed reproduces the dataset exactly.

    Returns
    -------
    dict with:
      "accounts": list of Account dicts
          {"account_id", "is_shared", "trips": [Trip, ...]},
          each Trip = {"trip_id", "account_id", "t_start", "fixes": [Fix, ...]},
          each Fix  = {"t", "x_m", "y_m", "accuracy_m", "speed_mps",
                       "is_mock", "cell_id"};
      "meta": {"seed", "n_accounts", "shared_fraction"}.

    Everything is JSON-serialisable Python scalars; no numpy types leak out.
    """
    if n_accounts < 1:
        raise ValueError("n_accounts must be >= 1")
    if not 0.0 <= shared_fraction <= 1.0:
        raise ValueError("shared_fraction must be in [0, 1]")
    lo, hi = int(trips_per_account[0]), int(trips_per_account[1])
    if lo < 2 or hi < lo:
        raise ValueError("trips_per_account must satisfy 2 <= min <= max")

    # Master stream picks which accounts are shared; each account then gets
    # its own child stream so per-account draws never interleave.
    master = np.random.default_rng([int(seed), 0])
    n_shared = int(round(n_accounts * shared_fraction))
    shared_ids = (set(map(int, master.choice(n_accounts, size=n_shared, replace=False)))
                  if n_shared else set())

    accounts = []
    for i in range(n_accounts):
        rng = np.random.default_rng([int(seed), 1, i])
        accounts.append(_build_account(rng, f"acc_{i:06d}", i in shared_ids,
                                       (lo, hi)))
    return {
        "accounts": accounts,
        "meta": {"seed": int(seed), "n_accounts": int(n_accounts),
                 "shared_fraction": float(shared_fraction)},
    }
