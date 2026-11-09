"""Eval math on a tiny hand-computable fixture.

This test doubles as documentation of the U metric. Every number below is
worked out by hand in the comments; the assertions are exact.

Semantics under test (see src/tripphysics/eval.py):
- FP allowance = floor(n_honest * budget / 10000); flagging is strictly
  score > threshold; threshold = (allowance+1)-th highest honest score.
- evaluate() stacks internally: R1 is computed on
  max(rank_normalize(baseline), rank_normalize(detector)), where
  rank_normalize maps a score to its 1-based ascending rank / n.
"""

import math

import numpy as np
import pytest

from tripphysics.eval import evaluate

# ---------------------------------------------------------------------------
# Fixture: 5 honest accounts (h1..h5) + 3 shared accounts (s1..s3).
# All raw scores are distinct, so rank normalization is trivially rank/8.
#
# Budget: 15 FP per 10k honest. With 5 honest accounts the allowance is
# floor(5 * 15 / 10000) = floor(0.0075) = 0 honest accounts flagged.
#
# R0 — raw baseline scores, descending:
#   s3 .90, s2 .70, h5 .60, h4 .55, s1 .30, h3 .15, h2 .10, h1 .05
#   Zero-FP threshold = highest honest score = .60 (h5); flagged = strictly
#   above .60 -> {s2, s3} of 3 shared            => R0 = 2/3
#
# R1 — evaluate() stacks ranks internally. Ascending ranks (rank/8):
#   baseline: h1 1/8, h2 2/8, h3 3/8, s1 4/8, h4 5/8, h5 6/8, s2 7/8, s3 8/8
#   detector: h1 1/8, h2 2/8, h3 3/8, h4 4/8, h5 5/8, s2 6/8, s1 7/8, s3 8/8
#   stacked = per-account max:
#     h1 1/8, h2 2/8, h3 3/8, h4 5/8, h5 6/8 | s1 7/8, s2 7/8, s3 8/8
#   Zero-FP threshold = highest honest stacked = 6/8 (h5); flagged =
#   strictly above 6/8 -> {s1, s2, s3}           => R1 = 3/3 = 1.0
#
# U = (R1 - R0) / R0 = (1 - 2/3) / (2/3) = 0.5   => +50% uplift.
#
# Robustness of the fixture: no shared score ties an honest score in either
# raw or stacked space, and even a convention that rounded the fractional
# 0.0075 allowance UP to one honest FP would move both thresholds down one
# honest account (baseline .55 / stacked 5/8) without changing which shared
# accounts clear them — the asserted R0/R1/U are convention-independent.
# ---------------------------------------------------------------------------

LABELS = {
    "h1": False, "h2": False, "h3": False, "h4": False, "h5": False,
    "s1": True, "s2": True, "s3": True,
}

BASELINE = {
    "h1": 0.05, "h2": 0.10, "h3": 0.15, "h4": 0.55, "h5": 0.60,
    "s1": 0.30, "s2": 0.70, "s3": 0.90,
}

# Detector story: strong physics evidence on s1 (which the baseline missed),
# moderate on s2, strong on s3; honest accounts stay at the bottom.
DETECTOR = {
    "h1": 0.05, "h2": 0.10, "h3": 0.15, "h4": 0.20, "h5": 0.25,
    "s1": 0.70, "s2": 0.60, "s3": 0.95,
}

REQUIRED_KEYS = {
    "R0", "R1", "U", "budget_fp_per_10k",
    "pr_auc_baseline", "pr_auc_pilot",
    "precision_at_500_baseline", "precision_at_500_pilot",
    "n_accounts", "n_shared",
}


def test_hand_computed_r0_r1_u():
    result = evaluate(LABELS, BASELINE, DETECTOR, budget_fp_per_10k=15.0)
    assert result["R0"] == pytest.approx(2 / 3)  # {s2, s3} of 3 shared
    assert result["R1"] == pytest.approx(1.0)    # {s1, s2, s3} of 3 shared
    assert result["U"] == pytest.approx(0.5)     # (1 - 2/3) / (2/3)


def test_hand_computed_thresholds():
    # Zero honest FPs allowed -> threshold pins to the highest honest score:
    # 0.60 in raw baseline space, 6/8 = 0.75 in stacked rank space.
    result = evaluate(LABELS, BASELINE, DETECTOR, budget_fp_per_10k=15.0)
    assert result["threshold_baseline"] == pytest.approx(0.60)
    assert result["threshold_pilot"] == pytest.approx(6 / 8)


def test_hand_computed_pr_auc_and_precision():
    # Baseline PR curve (anchor (R=0, P=1), trapezoid over recall):
    #   s3: R=1/3 P=1      -> + (1/3)*1        = 1/3
    #   s2: R=2/3 P=1      -> + (1/3)*1        = 1/3
    #   h5, h4: dR=0                            (precision drops, no area)
    #   s1: R=1   P=3/5    -> + (1/3)*(1/2 + 3/5)/2 = 11/60
    #   h3, h2, h1: dR=0
    #   PR-AUC(baseline) = 1/3 + 1/3 + 11/60 = 51/60 = 0.85
    # Stacked pilot ranks separate perfectly (all shared above all honest):
    #   PR-AUC(pilot) = 1.0
    # precision@500 with only 8 accounts truncates to top-8 = everyone:
    #   3 shared / 8 accounts = 0.375 for both scorers.
    result = evaluate(LABELS, BASELINE, DETECTOR, budget_fp_per_10k=15.0)
    assert result["pr_auc_baseline"] == pytest.approx(0.85)
    assert result["pr_auc_pilot"] == pytest.approx(1.0)
    assert result["precision_at_500_baseline"] == pytest.approx(3 / 8)
    assert result["precision_at_500_pilot"] == pytest.approx(3 / 8)


def test_counts_and_required_keys():
    result = evaluate(LABELS, BASELINE, DETECTOR, budget_fp_per_10k=15.0)
    assert REQUIRED_KEYS <= set(result)
    assert result["n_accounts"] == 8
    assert result["n_shared"] == 3
    assert result["budget_fp_per_10k"] == 15.0
    for value in result.values():
        assert isinstance(value, (int, float))
        assert not isinstance(value, bool)
        assert math.isfinite(float(value))


def test_budget_monotonicity():
    # A larger FP budget can only lower the threshold, so recall at a higher
    # budget must be >= recall at a lower budget, for both scorers.
    rng = np.random.default_rng(123)
    labels: dict[str, bool] = {}
    baseline: dict[str, float] = {}
    detector: dict[str, float] = {}
    for i in range(1200):  # honest: scores bounded away from the shared top
        account_id = f"hon{i:04d}"
        labels[account_id] = False
        baseline[account_id] = float(rng.uniform(0.0, 0.7))
        detector[account_id] = float(rng.uniform(0.0, 0.7))
    for i in range(60):  # shared: overlapping but right-shifted scores
        account_id = f"shr{i:04d}"
        labels[account_id] = True
        baseline[account_id] = float(rng.uniform(0.3, 1.0))
        detector[account_id] = float(rng.uniform(0.5, 1.0))

    budgets = [5.0, 15.0, 50.0, 200.0]
    results = [evaluate(labels, baseline, detector, budget_fp_per_10k=b) for b in budgets]
    r0 = [r["R0"] for r in results]
    r1 = [r["R1"] for r in results]
    for lower, higher in zip(r0, r0[1:]):
        assert higher >= lower
    for lower, higher in zip(r1, r1[1:]):
        assert higher >= lower
