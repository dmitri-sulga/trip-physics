# Input schema — shadow-log trip fixes

This is the complete input contract for running the harness on a real
shadow log. Ten fields per GPS fix, one JSON object per line (JSONL). Nothing else
is read.

## Fix record (one JSONL line per GPS fix)

| #   | Field        | Type   | Unit / range | Description                                                                                                           |
| --- | ------------ | ------ | ------------ | --------------------------------------------------------------------------------------------------------------------- |
| 1   | `account_id` | string | opaque       | Stable pseudonymous driver-account key. Hash it on your side; the harness never needs the real ID.                    |
| 2   | `trip_id`    | string | opaque       | Stable key for the trip this fix belongs to.                                                                          |
| 3   | `t_start`    | float  | seconds      | Trip start time. Epoch offset or any consistent zero — only differences matter. Identical for all fixes of a trip.    |
| 4   | `t`          | float  | seconds      | Fix timestamp, same clock as `t_start`.                                                                               |
| 5   | `x_m`        | float  | metres       | Easting in a local planar frame (see projection note).                                                                |
| 6   | `y_m`        | float  | metres       | Northing in the same frame.                                                                                           |
| 7   | `accuracy_m` | float  | metres, > 0  | Reported horizontal accuracy of the fix.                                                                              |
| 8   | `speed_mps`  | float  | m/s, >= 0    | Speed as reported by the positioning stack.                                                                           |
| 9   | `is_mock`    | bool   |              | OS-level mock-location flag (e.g. Android `isFromMockProvider`).                                                      |
| 10  | `cell_id`    | int    | opaque       | Serving-cell identifier (hashed to an int is fine). Used only for GPS-vs-cell consistency; never resolved to a place. |

### Projection note (`x_m`, `y_m`)

The physics features are thresholded in metres and m/s, so coordinates must be
planar metres, not degrees. Project lat/lon to a local frame before export —
UTM zone of the city, or a simple equirectangular projection around the city
centre is fine at city scale. The frame origin is irrelevant; only distances
and speeds within one account are used.

## Example JSONL

```jsonl
{"account_id": "acc_9f3a", "trip_id": "trip_0451", "t_start": 1000.0, "t": 1000.0, "x_m": 1250.4, "y_m": -830.2, "accuracy_m": 8.0, "speed_mps": 0.0, "is_mock": false, "cell_id": 30412}
{"account_id": "acc_9f3a", "trip_id": "trip_0451", "t_start": 1000.0, "t": 1002.0, "x_m": 1252.1, "y_m": -828.9, "accuracy_m": 7.5, "speed_mps": 1.4, "is_mock": false, "cell_id": 30412}
{"account_id": "acc_9f3a", "trip_id": "trip_0452", "t_start": 4600.0, "t": 4600.0, "x_m": 9801.7, "y_m": 4470.3, "accuracy_m": 12.0, "speed_mps": 0.0, "is_mock": false, "cell_id": 18777}
```

Grouping fixes by (`account_id`, `trip_id`) and sorting by `t` reconstructs the
in-memory model used throughout the package:

```
Fix     = {"t", "x_m", "y_m", "accuracy_m", "speed_mps", "is_mock", "cell_id"}
Trip    = {"trip_id", "account_id", "t_start", "fixes": [Fix, ...]}
Account = {"account_id", "is_shared", "trips": [Trip, ...]}
```

`tests/test_schema.py::test_jsonl_flatten_round_trip` proves this flattening is
lossless — the JSONL above carries everything the detector uses.

## Labels sidecar (evaluation only)

Scoring runs without labels. Computing R0/R1/U at the end of the shadow window
requires ground truth for the window's accounts — one CSV or JSONL with:

| Field        | Type   | Description                                                                 |
| ------------ | ------ | --------------------------------------------------------------------------- |
| `account_id` | string | Same key as above.                                                          |
| `is_shared`  | bool   | Confirmed profile-sharing during the window (manual review / case outcome). |

Label quality bounds metric quality: unreviewed accounts silently counted as
honest deflate both R0 and R1.

## Baseline scores sidecar

To compare against the current stack, export its per-account scores as a CSV
(`account_id,score`, one header line) and load with
`tripphysics.baseline.load_scores_csv`. No production code changes needed.

## What the harness deliberately does not read

- **No PII.** No names, phone numbers, emails, documents, photos.
- **No rider data.** Driver-side fixes only; nothing about who was in the car.
- **No payment data.** No earnings, payouts, cards, invoices.
- **No addresses or POI resolution.** Coordinates stay in an anonymous local
  frame; `cell_id` may be salted-hashed before export.
- **No device fingerprints** beyond the single OS mock-location bit.
- **No live access.** Batch JSONL export, processed locally, in a container
  with egress blocked (see Dockerfile).

If a field in this table looks like more than the detection task needs, that is
a design bug in the contract.
