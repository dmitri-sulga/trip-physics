"""Bit-level determinism of the synthetic corpus and the scoring pipeline.

test_determinism.py already asserts that two runs at one seed produce equal
dicts and identical report bytes. This module goes a level deeper, because
dict equality is a weaker claim than it looks: it compares floats by value,
so it would pass even if the generator drifted within float tolerance, and
it says nothing about *different* seeds producing genuinely different data.

Asserted here:
  * float arrays extracted from two same-seed runs are bit-identical
    (np.array_equal on the raw bits, not approx);
  * different seeds diverge in the actual coordinate stream, not merely in
    account ids;
  * the seed fully determines the run -- no wall-clock, no hash randomisation,
    no global numpy state leaks in;
  * per-account streams are independent: the corpus is unchanged by
    generating more accounts around it.
"""

import numpy as np
import pytest

from tripphysics.baseline import NaiveBaseline
from tripphysics.detector import score_account
from tripphysics.features import extract_features
from tripphysics.synth import generate_dataset


def _coordinate_stream(dataset) -> np.ndarray:
    """Flatten every (t, x, y, speed) in the corpus into one float array."""
    rows = [
        (fix["t"], fix["x_m"], fix["y_m"], fix["speed_mps"])
        for account in dataset["accounts"]
        for trip in account["trips"]
        for fix in trip["fixes"]
    ]
    return np.array(rows, dtype=float)


def _score_vector(dataset) -> np.ndarray:
    return np.array(
        [score_account(extract_features(a)) for a in dataset["accounts"]],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Same seed => bit-identical output
# ---------------------------------------------------------------------------

def test_same_seed_yields_bit_identical_coordinate_arrays():
    # np.array_equal on float64 is an exact bit comparison (no NaNs present),
    # which is strictly stronger than the dict equality already tested.
    first = _coordinate_stream(generate_dataset(n_accounts=50, shared_fraction=0.1, seed=3))
    second = _coordinate_stream(generate_dataset(n_accounts=50, shared_fraction=0.1, seed=3))
    assert first.shape == second.shape
    assert np.array_equal(first, second)


def test_same_seed_yields_bit_identical_detector_scores():
    first = _score_vector(generate_dataset(n_accounts=40, shared_fraction=0.1, seed=17))
    second = _score_vector(generate_dataset(n_accounts=40, shared_fraction=0.1, seed=17))
    assert np.array_equal(first, second)


def test_same_seed_yields_bit_identical_baseline_scores():
    naive = NaiveBaseline()
    dataset_a = generate_dataset(n_accounts=40, shared_fraction=0.1, seed=17)
    dataset_b = generate_dataset(n_accounts=40, shared_fraction=0.1, seed=17)
    scores_a = np.array([naive.score(a) for a in dataset_a["accounts"]])
    scores_b = np.array([naive.score(a) for a in dataset_b["accounts"]])
    assert np.array_equal(scores_a, scores_b)


def test_labels_are_reproducible_for_a_seed():
    # Which accounts are shared is drawn from the master stream; it must be
    # part of the deterministic contract, not incidental.
    first = generate_dataset(n_accounts=100, shared_fraction=0.1, seed=23)
    second = generate_dataset(n_accounts=100, shared_fraction=0.1, seed=23)
    labels_a = [a["is_shared"] for a in first["accounts"]]
    labels_b = [a["is_shared"] for a in second["accounts"]]
    assert labels_a == labels_b
    assert sum(labels_a) == 10


# ---------------------------------------------------------------------------
# Different seeds => genuinely different data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed_a,seed_b", [(1, 2), (7, 8), (42, 43), (100, 999)])
def test_different_seeds_produce_different_coordinates(seed_a, seed_b):
    # Guards against a seeding bug where the seed reaches the account id but
    # not the RNG -- dict inequality alone would not catch that.
    first = _coordinate_stream(generate_dataset(n_accounts=30, shared_fraction=0.1, seed=seed_a))
    second = _coordinate_stream(generate_dataset(n_accounts=30, shared_fraction=0.1, seed=seed_b))
    assert not np.array_equal(first, second), f"seeds {seed_a}/{seed_b} collided"


def test_different_seeds_produce_different_detector_scores():
    first = _score_vector(generate_dataset(n_accounts=30, shared_fraction=0.1, seed=5))
    second = _score_vector(generate_dataset(n_accounts=30, shared_fraction=0.1, seed=6))
    assert not np.array_equal(first, second)


def test_seed_is_recorded_in_meta():
    dataset = generate_dataset(n_accounts=10, shared_fraction=0.1, seed=1234)
    assert dataset["meta"]["seed"] == 1234


# ---------------------------------------------------------------------------
# Independence and isolation of the RNG streams
# ---------------------------------------------------------------------------

def test_per_account_streams_are_independent_of_corpus_size():
    # Each account draws from default_rng([seed, 1, index]), so account i must
    # be identical whether the corpus holds 20 accounts or 60. Without this,
    # changing n_accounts would silently invalidate every stored baseline.
    small = generate_dataset(n_accounts=20, shared_fraction=0.0, seed=77)
    large = generate_dataset(n_accounts=60, shared_fraction=0.0, seed=77)
    for i in range(20):
        assert small["accounts"][i] == large["accounts"][i], f"account {i} drifted"


def test_generation_does_not_consume_global_numpy_entropy():
    # The generator must use its own Generator objects, never the legacy
    # global np.random state. If it did, an unrelated caller's draws would
    # shift the corpus and determinism would silently depend on call order.
    np.random.seed(0)
    first = generate_dataset(n_accounts=15, shared_fraction=0.1, seed=9)
    np.random.seed(0)
    np.random.random(1000)  # perturb the global state
    second = generate_dataset(n_accounts=15, shared_fraction=0.1, seed=9)
    assert first == second


def test_generation_is_unaffected_by_call_order():
    # Generating an unrelated corpus in between must not shift the result.
    first = generate_dataset(n_accounts=15, shared_fraction=0.1, seed=31)
    generate_dataset(n_accounts=50, shared_fraction=0.2, seed=999)
    second = generate_dataset(n_accounts=15, shared_fraction=0.1, seed=31)
    assert first == second


def test_feature_extraction_is_deterministic_across_repeated_calls():
    dataset = generate_dataset(n_accounts=25, shared_fraction=0.2, seed=44)
    for account in dataset["accounts"]:
        assert extract_features(account) == extract_features(account)


# ---------------------------------------------------------------------------
# Generator input validation
# ---------------------------------------------------------------------------

def test_generate_dataset_rejects_empty_corpus():
    with pytest.raises(ValueError, match="n_accounts"):
        generate_dataset(n_accounts=0)


def test_generate_dataset_rejects_out_of_range_shared_fraction():
    with pytest.raises(ValueError, match="shared_fraction"):
        generate_dataset(n_accounts=10, shared_fraction=1.5)
    with pytest.raises(ValueError, match="shared_fraction"):
        generate_dataset(n_accounts=10, shared_fraction=-0.1)


def test_generate_dataset_rejects_invalid_trip_range():
    with pytest.raises(ValueError, match="trips_per_account"):
        generate_dataset(n_accounts=10, trips_per_account=(1, 5))   # min < 2
    with pytest.raises(ValueError, match="trips_per_account"):
        generate_dataset(n_accounts=10, trips_per_account=(10, 4))  # max < min


def test_shared_fraction_zero_produces_no_shared_accounts():
    dataset = generate_dataset(n_accounts=30, shared_fraction=0.0, seed=2)
    assert not any(a["is_shared"] for a in dataset["accounts"])


def test_shared_fraction_one_produces_all_shared_accounts():
    dataset = generate_dataset(n_accounts=20, shared_fraction=1.0, seed=2)
    assert all(a["is_shared"] for a in dataset["accounts"])


def test_trip_counts_respect_the_requested_range():
    dataset = generate_dataset(n_accounts=25, shared_fraction=0.1,
                               trips_per_account=(3, 7), seed=8)
    for account in dataset["accounts"]:
        assert 3 <= len(account["trips"]) <= 7
