"""Regression anchors for the uncertainty layer of the demo run.

test_golden_metrics.py pins the point estimates of one seed. This module
pins the numbers that qualify them: the spread across seeds, the bootstrap
interval, and the out-of-sample false-positive check. Those are the figures
that decide whether the headline is a result or a coincidence, so they need
the same protection against silent drift.

Two scales appear here deliberately:

* **n=2000, the demo corpus.** Cheap enough to assert on directly. Its role
  is to document that at this size the uplift is dominated by noise -- the
  across-seed sd is roughly 8 percentage points and the 95% interval spans
  zero. This is asserted, not merely commented, so that nobody can quote a
  single 2000-account seed as a result without a test objecting.
* **n=20000, the quotable corpus.** Where the interval actually tightens.
  Marked ``slow`` because scoring it takes minutes; deselect with
  ``-m "not slow"``.

Tolerances are wide (a few percentage points) on purpose. These are
statistics with real sampling variance; pinning them to 1e-9 like the point
estimates would produce a suite that fails whenever anything legitimate
changes. What they must catch is a *structural* regression: an interval that
stops covering, a spread that collapses to zero because the seeds stopped
varying, an out-of-sample FP rate that quietly runs away from its budget.

Regenerating after an INTENTIONAL change: rerun the measurement in the
docstring of each anchor block and update the numbers in the same commit as
the change that moved them, with a note saying why.
"""

import pytest

from tripphysics.baseline import NaiveBaseline
from tripphysics.detector import score_account
from tripphysics.eval import (
    bootstrap_uplift_ci,
    evaluate,
    holdout_fp_check,
    summarize_across_seeds,
)
from tripphysics.features import extract_features
from tripphysics.synth import generate_dataset

DEMO_SHARED_FRACTION = 0.03
DEMO_BUDGET = 50.0
DEMO_N_ACCOUNTS = 2000
QUOTABLE_N_ACCOUNTS = 20000
SWEEP_SEEDS = (1, 2, 3)


def _score_population(n_accounts, shared_fraction, seed):
    accounts = generate_dataset(n_accounts=n_accounts,
                                shared_fraction=shared_fraction,
                                seed=seed)["accounts"]
    labels = {a["account_id"]: a["is_shared"] for a in accounts}
    naive = NaiveBaseline()
    baseline = {a["account_id"]: naive.score(a) for a in accounts}
    detector = {a["account_id"]: score_account(extract_features(a))
                for a in accounts}
    return labels, baseline, detector


# ---------------------------------------------------------------------------
# The demo corpus: small, noisy, and documented as such
#
# Measured at n=2000, shared_fraction=0.03, seeds 1-3, budget 50/10k:
#   U per seed: 0.1176, 0.1875, 0.0250
#   median 0.1176   mean 0.1100   sd 0.0815   min 0.0250   max 0.1875
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def demo_sweep():
    uplifts = []
    for seed in SWEEP_SEEDS:
        labels, baseline, detector = _score_population(
            DEMO_N_ACCOUNTS, DEMO_SHARED_FRACTION, seed)
        uplifts.append(evaluate(labels, baseline, detector,
                                budget_fp_per_10k=DEMO_BUDGET)["U"])
    return uplifts


def test_demo_sweep_uplifts_are_unchanged(demo_sweep):
    # Per-seed anchors. These are exact reruns of a deterministic pipeline,
    # so they carry a tight tolerance -- unlike the derived statistics below.
    expected = [0.11764705882352938, 0.18749999999999997, 0.025000000000000078]
    assert demo_sweep == pytest.approx(expected, abs=1e-9)


def test_demo_sweep_summary_is_unchanged(demo_sweep):
    summary = summarize_across_seeds(demo_sweep, target=0.20)
    assert summary["median"] == pytest.approx(0.1176, abs=5e-3)
    assert summary["mean"] == pytest.approx(0.1100, abs=5e-3)
    assert summary["sd"] == pytest.approx(0.0815, abs=5e-3)
    assert summary["min"] == pytest.approx(0.0250, abs=5e-3)
    assert summary["max"] == pytest.approx(0.1875, abs=5e-3)
    assert summary["n_seeds"] == len(SWEEP_SEEDS)


def test_the_demo_corpus_is_too_small_to_quote_a_single_seed(demo_sweep):
    # The finding that motivated moving the quotable run to 20k accounts, and
    # the reason the harness reports a median rather than a seed. Asserted so
    # that a future change which makes single-seed numbers look stable has to
    # justify itself here rather than slipping through.
    summary = summarize_across_seeds(demo_sweep)
    assert summary["sd"] > 0.03, (
        f"across-seed sd collapsed to {summary['sd']:.4f}; if the seeds "
        f"genuinely stopped varying, check that the seed still reaches the "
        f"generator before re-anchoring")
    assert summary["max"] - summary["min"] > 0.10


def test_the_demo_confidence_interval_spans_zero():
    # At 60 positives the interval cannot exclude "no effect". This is the
    # honest reading of the small corpus and it must stay visible: an
    # interval that suddenly excluded zero at this sample size would mean
    # the resampling stopped accounting for a real source of variance.
    labels, baseline, detector = _score_population(
        DEMO_N_ACCOUNTS, DEMO_SHARED_FRACTION, 1)
    ci = bootstrap_uplift_ci(labels, baseline, detector,
                             budget_fp_per_10k=DEMO_BUDGET,
                             n_resamples=300, seed=0)
    assert ci["ci_low"] == pytest.approx(-0.0256, abs=0.03)
    assert ci["ci_high"] == pytest.approx(0.2430, abs=0.03)
    assert ci["ci_low"] < 0.0 < ci["ci_high"], (
        "the 2000-account interval no longer spans zero; verify this is a "
        "real improvement and not a narrowed resampling procedure")
    assert ci["n_degenerate"] == 0


# ---------------------------------------------------------------------------
# The quotable corpus: where the interval actually tightens
#
# Measured at n=20000, shared_fraction=0.03, seed 1, budget 50/10k:
#   R0 0.5600   R1 0.6400   U 0.142857   n_shared 600
#   PR-AUC baseline 0.6075   pilot 0.7307
#   bootstrap 95% CI (300 resamples, seed 0): [0.0925, 0.1925]
#   holdout FP per 10k (50/50 split, seed 0): baseline 46.4, pilot 70.1
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def quotable_run():
    labels, baseline, detector = _score_population(
        QUOTABLE_N_ACCOUNTS, DEMO_SHARED_FRACTION, 1)
    return labels, baseline, detector


@pytest.mark.slow
def test_quotable_run_point_estimates_are_unchanged(quotable_run):
    labels, baseline, detector = quotable_run
    result = evaluate(labels, baseline, detector, budget_fp_per_10k=DEMO_BUDGET)
    assert result["n_accounts"] == QUOTABLE_N_ACCOUNTS
    assert result["n_shared"] == 600
    assert result["R0"] == pytest.approx(0.56, abs=1e-9)
    assert result["R1"] == pytest.approx(0.64, abs=1e-9)
    assert result["U"] == pytest.approx(0.142857142857, abs=1e-9)
    assert result["pr_auc_baseline"] == pytest.approx(0.607508, abs=1e-5)
    assert result["pr_auc_pilot"] == pytest.approx(0.730676, abs=1e-5)


@pytest.mark.slow
def test_the_quotable_interval_is_tighter_than_the_demo_one(quotable_run):
    # The whole reason the sample size went up: 600 positives instead of 60
    # roughly halves the interval width and lifts it clear of zero.
    labels, baseline, detector = quotable_run
    ci = bootstrap_uplift_ci(labels, baseline, detector,
                             budget_fp_per_10k=DEMO_BUDGET,
                             n_resamples=300, seed=0)
    assert ci["ci_low"] == pytest.approx(0.0925, abs=0.03)
    assert ci["ci_high"] == pytest.approx(0.1925, abs=0.03)
    assert ci["ci_high"] - ci["ci_low"] < 0.12, (
        "the 20k interval widened; at 600 positives it should be well under "
        "half the demo corpus's width")
    assert ci["ci_low"] > 0.0, "the quotable interval no longer excludes zero"


@pytest.mark.slow
def test_the_out_of_sample_fp_rate_is_measured_and_bounded(quotable_run):
    # The budget is a promise about NEXT week's honest accounts, not the ones
    # the threshold was fitted to. This anchor keeps the size of that
    # optimism visible: the pilot arm overspends its 50/10k budget on held-out
    # honest accounts, which is a real and reportable limitation, not a bug to
    # be hidden. What must not happen silently is that overshoot growing.
    labels, baseline, detector = quotable_run
    check = holdout_fp_check(labels, baseline, detector,
                             budget_fp_per_10k=DEMO_BUDGET, seed=0)
    assert check["realized_fp_per_10k_baseline"] == pytest.approx(46.4, abs=8.0)
    assert check["realized_fp_per_10k_pilot"] == pytest.approx(70.1, abs=12.0)
    # Sized, not ignored: a 2x budget overshoot out of sample would mean the
    # in-sample threshold is not usable as a deployment promise at all.
    assert check["realized_fp_per_10k_pilot"] < 2.0 * DEMO_BUDGET, (
        "the pilot arm now overspends its FP budget by more than 2x on "
        "held-out honest accounts")
    assert check["n_calibration"] + check["n_measurement"] == sum(
        1 for is_shared in labels.values() if not is_shared)
