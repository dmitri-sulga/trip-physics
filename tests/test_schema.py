"""Schema contract: exact field sets, JSON round-trips, lossless JSONL flattening.

The JSONL flattening test mirrors schema/input_schema.md exactly: if it
passes, the documented 10-field export format provably carries everything the
in-memory model holds.
"""

import json
import math

from tripphysics.features import extract_features
from tripphysics.synth import generate_dataset

FIX_KEYS = {"t", "x_m", "y_m", "accuracy_m", "speed_mps", "is_mock", "cell_id"}
TRIP_KEYS = {"trip_id", "account_id", "t_start", "fixes"}
ACCOUNT_KEYS = {"account_id", "is_shared", "trips"}
FEATURE_KEYS = {
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
    # Added when non-finite fixes stopped being zero-filled into the
    # aggregates and started being discarded and counted instead.
    "nonfinite_rate",
}


def _small_dataset():
    return generate_dataset(n_accounts=40, shared_fraction=0.1, seed=5)


def test_dataset_json_round_trip():
    # Plain dicts, JSON-serializable, and float-exact through dump/load.
    dataset = _small_dataset()
    assert json.loads(json.dumps(dataset)) == dataset


def test_exact_field_sets_and_types():
    dataset = _small_dataset()
    meta = dataset["meta"]
    assert {"seed", "n_accounts", "shared_fraction"} <= set(meta)
    assert meta["seed"] == 5
    assert meta["n_accounts"] == 40
    for account in dataset["accounts"]:
        assert set(account) == ACCOUNT_KEYS
        assert isinstance(account["account_id"], str)
        assert isinstance(account["is_shared"], bool)
        for trip in account["trips"]:
            assert set(trip) == TRIP_KEYS
            assert trip["account_id"] == account["account_id"]
            assert isinstance(trip["trip_id"], str)
            assert isinstance(trip["t_start"], float)
            assert trip["fixes"], "trip without fixes"
            for fix in trip["fixes"]:
                assert set(fix) == FIX_KEYS
                for key in ("t", "x_m", "y_m", "accuracy_m", "speed_mps"):
                    assert isinstance(fix[key], float)
                assert isinstance(fix["is_mock"], bool)
                assert isinstance(fix["cell_id"], int) and not isinstance(fix["cell_id"], bool)
                assert fix["accuracy_m"] > 0
                assert fix["speed_mps"] >= 0


def test_jsonl_flatten_round_trip(tmp_path):
    # Flatten to the documented 10-field JSONL + labels sidecar, read it back,
    # rebuild the account model, and require exact equality with the original.
    dataset = _small_dataset()
    jsonl_path = tmp_path / "shadow_log.jsonl"
    labels_path = tmp_path / "labels.jsonl"

    with open(jsonl_path, "w") as jsonl_file, open(labels_path, "w") as labels_file:
        for account in dataset["accounts"]:
            labels_file.write(json.dumps(
                {"account_id": account["account_id"], "is_shared": account["is_shared"]}
            ) + "\n")
            for trip in account["trips"]:
                for fix in trip["fixes"]:
                    row = {
                        "account_id": account["account_id"],
                        "trip_id": trip["trip_id"],
                        "t_start": trip["t_start"],
                        **fix,
                    }
                    assert len(row) == 10  # the schema is exactly ten fields
                    jsonl_file.write(json.dumps(row) + "\n")

    labels = {}
    with open(labels_path) as labels_file:
        for line in labels_file:
            record = json.loads(line)
            labels[record["account_id"]] = record["is_shared"]

    # Rebuild: group fixes by (account_id, trip_id), preserving encounter order.
    trips_by_key: dict[tuple[str, str], dict] = {}
    account_order: dict[str, list[tuple[str, str]]] = {}
    with open(jsonl_path) as jsonl_file:
        for line in jsonl_file:
            row = json.loads(line)
            key = (row["account_id"], row["trip_id"])
            if key not in trips_by_key:
                trips_by_key[key] = {
                    "trip_id": row["trip_id"],
                    "account_id": row["account_id"],
                    "t_start": row["t_start"],
                    "fixes": [],
                }
                account_order.setdefault(row["account_id"], []).append(key)
            fix = {k: v for k, v in row.items() if k in FIX_KEYS}
            trips_by_key[key]["fixes"].append(fix)

    rebuilt = [
        {
            "account_id": account_id,
            "is_shared": labels[account_id],
            "trips": [trips_by_key[key] for key in keys],
        }
        for account_id, keys in account_order.items()
    ]

    assert rebuilt == dataset["accounts"]


def test_feature_vector_keys_and_finiteness():
    dataset = _small_dataset()
    for account in dataset["accounts"]:
        features = extract_features(account)
        assert set(features) == FEATURE_KEYS
        for key, value in features.items():
            assert isinstance(value, float), f"{key} is {type(value)}"
            assert math.isfinite(value), f"{key} is not finite"
