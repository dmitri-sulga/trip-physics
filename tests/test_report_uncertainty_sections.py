"""Rendering of the uncertainty sections, and the defensive branches around them.

The report grew three sections that exist to stop a reader over-reading a
single number: the seed sweep, the bootstrap interval, and the held-out
false-positive check. Each is optional -- driven by a key in ``meta`` -- so
each has a present branch and an absent branch, and a report that silently
dropped one would still render perfectly well.

These tests pin the content that carries the caveat, not the formatting. What
must survive a refactor is that the sweep reports a MEDIAN and every seed, that
the interval is shown with its bounds, and that the held-out FP rate is printed
next to the budget it is supposed to respect. Those three are the difference
between a report and a brochure.

Also covered here are the last defensive branches in the pipeline: trips whose
fixes are all unparseable, the saturated-budget shortcut inside the bootstrap,
and the input validation on the held-out check.
"""

import json
import math

import pytest

from tripphysics.eval import bootstrap_uplift_ci, holdout_fp_check
from tripphysics.features import FEATURE_NAMES, extract_features
from tripphysics.report import render_report

BASE_RESULT = {
    "R0": 0.56, "R1": 0.64, "U": 0.142857,
    "budget_fp_per_10k": 50.0,
    "pr_auc_baseline": 0.6075, "pr_auc_pilot": 0.7307,
    "precision_at_500_baseline": 0.676, "precision_at_500_pilot": 0.782,
    "n_accounts": 20000, "n_shared": 600,
    "threshold_baseline": 0.3493, "threshold_pilot": 0.9804,
}
BASE_META = {"seed": 1, "n_accounts": 20000, "shared_fraction": 0.03}

SWEEP = {
    "median": 0.1428, "mean": 0.1402, "sd": 0.0212,
    "min": 0.1010, "max": 0.1809, "n_seeds": 5,
    "target": 0.20, "n_at_or_above_target": 0,
    "fraction_at_or_above_target": 0.0,
}
SWEEP_VALUES = [{"seed": s, "U": u} for s, u in
                zip(range(1, 6), (0.1428, 0.1010, 0.1809, 0.1355, 0.1408))]
CI = {"u_point": 0.1428, "ci_low": 0.0925, "ci_high": 0.1925,
      "n_resamples": 300, "confidence": 0.95, "n_degenerate": 0}
FP_CHECK = {"budget_fp_per_10k": 50.0, "calibration_fraction": 0.5,
            "n_calibration": 9700, "n_measurement": 9700,
            "realized_fp_per_10k_baseline": 46.4,
            "realized_fp_per_10k_pilot": 70.1}


def _render(tmp_path, meta, name="report", result=None):
    md_path = tmp_path / f"{name}.md"
    render_report(result or BASE_RESULT, meta,
                  str(tmp_path / f"{name}.json"), str(md_path))
    return md_path.read_text()


# ---------------------------------------------------------------------------
# Seed sweep section
# ---------------------------------------------------------------------------

def test_sweep_section_is_absent_without_sweep_data(tmp_path):
    markdown = _render(tmp_path, BASE_META, "nosweep")
    assert "U across independent seeds" not in markdown


def test_sweep_section_reports_the_median_as_the_headline(tmp_path):
    # The anti-cherry-picking claim: the headline is the median across seeds,
    # not the displayed seed and certainly not the best one.
    markdown = _render(tmp_path, dict(BASE_META, seed_sweep=SWEEP), "sweep")
    assert "U across independent seeds" in markdown
    assert "median" in markdown
    assert "HEADLINE" in markdown
    assert "not the best draw" in markdown


def test_sweep_section_reports_spread_and_range(tmp_path):
    markdown = _render(tmp_path, dict(BASE_META, seed_sweep=SWEEP), "spread")
    assert "sd" in markdown
    assert "2.1pp" in markdown          # sd rendered in percentage points
    assert "| seeds | 5 |" in markdown


def test_sweep_never_scores_the_simulator_against_the_pilot_bar(tmp_path):
    # Stronger than disclaiming the count: the report must not compute it at
    # all. "N of M seeds at or above the target" reads as a pass rate however
    # it is captioned, and the report tells the reader two paragraphs earlier
    # that the synthetic number and the pilot bar are not comparable.
    markdown = _render(tmp_path, dict(BASE_META, seed_sweep=SWEEP), "rate")

    assert "at or above" not in markdown
    assert "pass rate" not in markdown
    # The count itself, in the shapes it would take for this fixture.
    assert f"{SWEEP['n_at_or_above_target']} of {SWEEP['n_seeds']} seeds" not in markdown

    # What must be there instead: the spread on its own terms, and the reason
    # the bar sits above it.
    assert "synthetic spread" in markdown
    assert "on purpose" in markdown


def test_per_seed_table_lists_every_seed_with_none_dropped(tmp_path):
    # A sweep that reported only its summary could hide a seed. Every value
    # must appear, and the report must say that none were dropped.
    markdown = _render(
        tmp_path,
        dict(BASE_META, seed_sweep=SWEEP, seed_sweep_values=SWEEP_VALUES),
        "perseed")
    assert "none dropped" in markdown
    for row in SWEEP_VALUES:
        assert f"| {row['seed']} |" in markdown


def test_per_seed_table_is_absent_when_only_the_summary_is_given(tmp_path):
    markdown = _render(tmp_path, dict(BASE_META, seed_sweep=SWEEP), "summaryonly")
    assert "none dropped" not in markdown


# ---------------------------------------------------------------------------
# Bootstrap interval section
# ---------------------------------------------------------------------------

def test_ci_section_is_absent_without_ci_data(tmp_path):
    markdown = _render(tmp_path, BASE_META, "noci")
    assert "Bootstrap confidence interval" not in markdown


def test_ci_section_reports_both_bounds(tmp_path):
    markdown = _render(tmp_path, dict(BASE_META, bootstrap_ci=CI), "ci")
    assert "Bootstrap confidence interval" in markdown
    assert "95% CI" in markdown
    # Bounds are rendered as percentages, truncated to one decimal.
    assert "[+9.2%, +19.2%]" in markdown


def test_ci_section_states_that_threshold_variance_is_included(tmp_path):
    # The methodological point of re-deriving the threshold per replicate. If
    # this sentence goes, a reader cannot tell whether the interval is honest.
    markdown = _render(tmp_path, dict(BASE_META, bootstrap_ci=CI), "civar")
    assert "threshold is re-derived on every replicate" in markdown


def test_ci_section_discloses_excluded_degenerate_replicates(tmp_path):
    # Replicates where U was undefined are dropped; the count must be stated,
    # or an interval computed from a fraction of the resamples looks clean.
    degenerate = dict(CI, n_degenerate=12)
    markdown = _render(tmp_path, dict(BASE_META, bootstrap_ci=degenerate), "cideg")
    assert "12 replicate(s)" in markdown
    assert "excluded" in markdown


def test_ci_section_is_skipped_when_the_interval_is_undefined(tmp_path):
    # NaN bounds mean every replicate was degenerate. Rendering "[nan, nan]"
    # as a confidence interval would be worse than saying nothing, so the
    # whole section is suppressed. (The raw values still appear in the meta
    # dump at the foot of the report, which is a faithful record of the run
    # rather than a claim about it -- hence the narrower assertion below.)
    undefined = dict(CI, ci_low=float("nan"), ci_high=float("nan"),
                     n_degenerate=300)
    markdown = _render(tmp_path, dict(BASE_META, bootstrap_ci=undefined), "cinan")
    assert "Bootstrap confidence interval" not in markdown
    assert "95% CI" not in markdown


# ---------------------------------------------------------------------------
# Held-out false-positive check
# ---------------------------------------------------------------------------

def test_fp_check_section_is_absent_without_data(tmp_path):
    # Matched on the heading, not on the phrase "out-of-sample": that phrase
    # also appears in the standing honest-limits prose, which is present in
    # every report by design.
    markdown = _render(tmp_path, BASE_META, "nofp")
    assert "Is the FP budget promise kept out-of-sample?" not in markdown
    assert "held-out honest accounts" not in markdown


def test_fp_check_section_prints_realized_rate_against_the_budget(tmp_path):
    markdown = _render(tmp_path, dict(BASE_META, holdout_fp_check=FP_CHECK), "fp")
    assert "Is the FP budget promise kept out-of-sample?" in markdown
    assert "46.4 / 10k" in markdown
    assert "70.1 / 10k" in markdown
    assert "50 / 10k" in markdown


def test_fp_check_section_explains_why_in_sample_is_a_tautology(tmp_path):
    # Without this sentence the section reads as a redundant restatement of
    # the budget rather than as the measurement of its optimism.
    markdown = _render(tmp_path, dict(BASE_META, holdout_fp_check=FP_CHECK), "fpwhy")
    assert "tautology" in markdown
    assert "honest error bar" in markdown


def test_fp_check_does_not_hide_an_overspent_budget(tmp_path):
    # The pilot arm overspends its budget out of sample. That number has to
    # be printed as-is; a report that clipped it to the budget would be
    # asserting the very thing the check exists to test.
    markdown = _render(tmp_path, dict(BASE_META, holdout_fp_check=FP_CHECK), "fpover")
    assert FP_CHECK["realized_fp_per_10k_pilot"] > FP_CHECK["budget_fp_per_10k"]
    assert "70.1" in markdown


def test_all_uncertainty_sections_render_together(tmp_path):
    # The shipped configuration: every section present at once must produce
    # one valid document, and the JSON artifact must stay strictly parseable.
    meta = dict(BASE_META, seed_sweep=SWEEP, seed_sweep_values=SWEEP_VALUES,
                bootstrap_ci=CI, holdout_fp_check=FP_CHECK)
    json_path = tmp_path / "full.json"
    md_path = tmp_path / "full.md"
    render_report(BASE_RESULT, meta, str(json_path), str(md_path))

    markdown = md_path.read_text()
    for heading in ("U across independent seeds",
                    "Bootstrap confidence interval",
                    "Is the FP budget promise kept out-of-sample?",
                    "How to read U"):
        assert heading in markdown
    payload = json.loads(json_path.read_text())
    assert payload["meta"]["seed_sweep"]["n_seeds"] == 5


# ---------------------------------------------------------------------------
# Remaining defensive branches in the pipeline
# ---------------------------------------------------------------------------

def _fix(t, x, y, speed=10.0):
    return {"t": float(t), "x_m": float(x), "y_m": float(y),
            "accuracy_m": 5.0, "speed_mps": float(speed),
            "is_mock": False, "cell_id": 0}


def _trip(fixes, trip_id="t0"):
    return {"trip_id": trip_id, "account_id": "a",
            "t_start": fixes[0]["t"] if fixes else 0.0, "fixes": fixes}


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_a_trip_whose_fixes_are_all_unparseable_is_skipped():
    # Such a trip contributes to nonfinite_rate and nothing else: it cannot
    # anchor a handover, because it has no surviving endpoint to anchor with.
    nan = float("nan")
    garbage = _trip([_fix(0, nan, 0), _fix(10, nan, 0)], "bad")
    good = _trip([_fix(100, 0, 0), _fix(110, 100, 0), _fix(120, 200, 0)], "good")
    features = extract_features(
        {"account_id": "a", "is_shared": False, "trips": [garbage, good]})

    assert set(features) == set(FEATURE_NAMES)
    # 2 of 5 raw fixes were discarded.
    assert features["nonfinite_rate"] == pytest.approx(0.4)
    # The surviving trip is clean and alone, so no handover is inferred.
    assert features["handover_gap_score"] == 0.0
    assert all(math.isfinite(v) for v in features.values())


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_an_account_whose_every_fix_is_unparseable_reports_a_full_discard_rate():
    nan = float("nan")
    features = extract_features({
        "account_id": "a", "is_shared": False,
        "trips": [_trip([_fix(0, nan, 0), _fix(10, nan, 0)], "bad")],
    })
    assert features["nonfinite_rate"] == pytest.approx(1.0)
    assert all(math.isfinite(v) for v in features.values())


def _account_with_enough_trips_for_bimodality(account_id="acct_42"):
    from tripphysics.features import _MIN_TRIPS_BIMODALITY

    trips = [
        {"trip_id": f"t{k}", "account_id": account_id, "t_start": k * 1000.0,
         "fixes": [_fix(k * 1000 + i * 10, i * 100, 0, speed=10.0 + k)
                   for i in range(6)]}
        for k in range(_MIN_TRIPS_BIMODALITY + 1)
    ]
    return {"account_id": account_id, "is_shared": False, "trips": trips}


def test_a_non_finite_aggregate_raises_instead_of_being_coerced(monkeypatch):
    # The last line of defence. Non-finite fixes are filtered upstream, so
    # reaching the end of extract_features with a NaN aggregate would be an
    # internal bug -- and coercing it to 0.0 would convert that bug into an
    # evasion channel, because 0.0 is the LEAST suspicious value for
    # max_jump_m and friends. Simulated here by forcing an internal to
    # misbehave, since no input can reach this state.
    import tripphysics.features as features_module

    account = _account_with_enough_trips_for_bimodality()
    # Sanity: the normal path produces a finite value for this account.
    assert math.isfinite(extract_features(account)["style_bimodality"])

    monkeypatch.setattr(features_module, "_split_separation",
                        lambda scalar: float("nan"))
    with pytest.raises(ValueError, match="non-finite feature values"):
        extract_features(account)


def test_the_non_finite_guard_names_the_account_and_the_feature(monkeypatch):
    # An internal error in a batch of 100k accounts is only actionable if it
    # says which account and which feature went wrong.
    import tripphysics.features as features_module

    monkeypatch.setattr(features_module, "_split_separation",
                        lambda scalar: float("nan"))
    with pytest.raises(ValueError, match=r"'acct_42'.*style_bimodality"):
        extract_features(_account_with_enough_trips_for_bimodality("acct_42"))


def test_bootstrap_handles_a_budget_that_flags_everyone():
    # When the budget allows flagging every honest account the threshold is
    # -inf and recall is 1.0 for both arms, so U is exactly 0. This is the
    # saturated shortcut inside the resampling loop.
    labels = {f"s{i}": True for i in range(5)}
    labels.update({f"h{i}": False for i in range(20)})
    scores = {a: (0.9 if a.startswith("s") else 0.1) for a in labels}
    result = bootstrap_uplift_ci(labels, scores, scores,
                                 budget_fp_per_10k=100000.0,
                                 n_resamples=20, seed=0)
    assert result["ci_low"] == pytest.approx(0.0)
    assert result["ci_high"] == pytest.approx(0.0)
    assert result["n_degenerate"] == 0


@pytest.mark.parametrize("fraction", [0.0, 1.0, 1.5, -0.2])
def test_holdout_check_rejects_an_out_of_range_split(fraction):
    labels = {"h1": False, "h2": False, "s1": True}
    scores = {"h1": 0.1, "h2": 0.2, "s1": 0.9}
    with pytest.raises(ValueError, match="calibration_fraction"):
        holdout_fp_check(labels, scores, scores, calibration_fraction=fraction)


def test_holdout_check_needs_at_least_two_honest_accounts():
    # With one honest account there is no split to make: calibration and
    # measurement cannot both be non-empty.
    labels = {"h1": False, "s1": True}
    scores = {"h1": 0.1, "s1": 0.9}
    with pytest.raises(ValueError, match="at least 2 honest"):
        holdout_fp_check(labels, scores, scores)
