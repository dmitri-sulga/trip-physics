"""Golden regression anchor: the headline numbers must not move silently.

The anchor configuration (2000 accounts, 3% shared, seed 42, 50 FP/10k) is a
fixed single-seed regression run, small enough to execute in the test suite.
It is NOT the README headline, which is the median of a 20-seed sweep at
20,000 accounts (``examples/run_demo.py --full``). These tests pin its
metrics so that any change to feature extraction, detector weights, the
stacking rule, or the threshold math shows up as a failing assertion naming
the exact number that moved -- rather than as a quietly different result.

Tolerances are deliberately tight but not exact. Floating-point accumulation
order can shift the last bits across numpy builds and platforms, so the
anchors carry an absolute tolerance around 1e-9 for derived rates and a
slightly wider band for U. A real regression moves these numbers by
percentage points, not by 1e-9.

Regenerating the anchors after an INTENTIONAL change: evaluate the
configuration above (the ``demo_result`` fixture below does exactly that),
and copy the R0 / R1 / U / PR-AUC values here, in the same commit as the
change that moved them, with a note saying why.
"""

import math

import pytest

from tripphysics.baseline import NaiveBaseline
from tripphysics.detector import score_account
from tripphysics.eval import evaluate
from tripphysics.features import extract_features
from tripphysics.report import U_TARGET
from tripphysics.synth import generate_dataset

# --- the demo configuration, as run by examples/run_demo.py -----------------
DEMO_N_ACCOUNTS = 2000
DEMO_SHARED_FRACTION = 0.03
DEMO_SEED = 42
DEMO_BUDGET = 50.0

# --- golden values, measured on the configuration above ---------------------
GOLDEN = {
    "R0": 0.5166666666666667,
    "R1": 0.65,
    "U": 0.2580645161290322,
    "pr_auc_baseline": 0.5677044709450273,
    # Moved 0.7809983 -> 0.7849702 when style_bimodality was recalibrated
    # against a size-aware null and its weight cut from 2.0 to 0.3 (the
    # feature separates at AUC ~0.54, so it was over-weighted). Verified
    # before re-anchoring: R0/R1/U are unchanged to the last bit, the
    # baseline arm is untouched, inverting every label leaves the detector
    # scores bit-identical, and nonfinite_rate is identically zero on this
    # corpus -- so the gain is a genuine ranking improvement in the tail,
    # not leakage and not the new feature smuggling in signal.
    "pr_auc_pilot": 0.7849702305101559,
    "precision_at_500_baseline": 0.08,
    "precision_at_500_pilot": 0.106,
    "n_accounts": 2000,
    "n_shared": 60,
    "threshold_baseline": 0.3446114369501466,
    "threshold_pilot": 0.9805,
}

TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def demo_result():
    """Run the full demo pipeline once and share it across the module."""
    dataset = generate_dataset(n_accounts=DEMO_N_ACCOUNTS,
                               shared_fraction=DEMO_SHARED_FRACTION,
                               seed=DEMO_SEED)
    accounts = dataset["accounts"]
    labels = {a["account_id"]: a["is_shared"] for a in accounts}
    naive = NaiveBaseline()
    baseline_scores = {a["account_id"]: naive.score(a) for a in accounts}
    detector_scores = {a["account_id"]: score_account(extract_features(a))
                       for a in accounts}
    return evaluate(labels, baseline_scores, detector_scores,
                    budget_fp_per_10k=DEMO_BUDGET)


@pytest.mark.parametrize("metric", sorted(GOLDEN))
def test_golden_metric_is_unchanged(demo_result, metric):
    expected = GOLDEN[metric]
    actual = demo_result[metric]
    if isinstance(expected, int):
        assert actual == expected, f"{metric}: {actual} != {expected}"
    else:
        assert actual == pytest.approx(expected, abs=TOLERANCE), (
            f"{metric} moved: {actual} vs golden {expected}. If this change "
            f"was intentional, regenerate the anchors (see module docstring) "
            f"in the same commit and say why.")


def test_anchor_seed_uplift_sits_above_the_pilot_bar(demo_result):
    # A readable restatement of the golden U anchor (0.258), not a claim about
    # the detector: this one 2,000-account seed happens to land above the
    # pilot bar U_TARGET. At this population size a single seed ranges from
    # about +2% to +42% (README, "Sample size"); the 20-seed median at 20,000
    # accounts sits BELOW the bar, and no pass/fail applies to synthetic data.
    assert demo_result["U"] >= U_TARGET, (
        f"U = {demo_result['U']:.4f} fell below U_TARGET {U_TARGET}")


def test_stacking_improves_pr_auc_on_the_demo_run(demo_result):
    # PR-AUC is budget-free: it shows the separation is real before any
    # threshold is chosen. The README leans on this claim explicitly.
    assert demo_result["pr_auc_pilot"] > demo_result["pr_auc_baseline"]


def test_pilot_recall_beats_baseline_recall_on_the_demo_run(demo_result):
    assert demo_result["R1"] > demo_result["R0"]


def test_demo_population_shape(demo_result):
    # 3% of 2000 = 60 shared accounts; a drift here means the corpus changed.
    assert demo_result["n_accounts"] == DEMO_N_ACCOUNTS
    assert demo_result["n_shared"] == 60
    assert demo_result["budget_fp_per_10k"] == DEMO_BUDGET


def test_demo_metrics_are_all_finite(demo_result):
    for key, value in demo_result.items():
        assert math.isfinite(float(value)), f"{key} is not finite"


# ---------------------------------------------------------------------------
# The uplift must not be a single lucky seed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [1, 7, 42, 99])
def test_detector_adds_signal_across_independent_seeds(seed):
    # A golden anchor on one seed proves reproducibility, not robustness. If
    # uplift only appeared at seed 42, the headline would be cherry-picked.
    # These are smaller corpora so the test stays fast; the claim is only
    # directional (PR-AUC improves, U > 0), never a specific value.
    dataset = generate_dataset(n_accounts=300, shared_fraction=0.05, seed=seed)
    accounts = dataset["accounts"]
    labels = {a["account_id"]: a["is_shared"] for a in accounts}
    naive = NaiveBaseline()
    baseline_scores = {a["account_id"]: naive.score(a) for a in accounts}
    detector_scores = {a["account_id"]: score_account(extract_features(a))
                       for a in accounts}
    result = evaluate(labels, baseline_scores, detector_scores,
                      budget_fp_per_10k=DEMO_BUDGET)

    assert result["pr_auc_pilot"] > result["pr_auc_baseline"], (
        f"seed={seed}: stacking did not improve PR-AUC")
    assert result["U"] > 0.0, f"seed={seed}: U = {result['U']}"


def test_detector_separates_shared_from_honest_accounts():
    # The most basic claim the detector makes: on a labelled corpus, shared
    # accounts score higher on average than honest ones. Stated as a mean
    # comparison with a wide margin, not a per-account guarantee.
    dataset = generate_dataset(n_accounts=400, shared_fraction=0.1, seed=13)
    shared, honest = [], []
    for account in dataset["accounts"]:
        score = score_account(extract_features(account))
        (shared if account["is_shared"] else honest).append(score)

    assert shared and honest
    mean_shared = sum(shared) / len(shared)
    mean_honest = sum(honest) / len(honest)
    assert mean_shared > mean_honest, (
        f"shared mean {mean_shared:.4f} <= honest mean {mean_honest:.4f}")
