"""Baseline adapters, detector scoring, and report rendering.

These three modules carry the harness's contact points with the outside
world -- the CSV a client exports their production scores into, the score an
analyst reads off a flagged account, and the two report artifacts that leave
the machine. `load_scores_csv` in particular was the least-covered code in
the package while being the single function a pilot user is most likely to
hit first, with their own malformed file.
"""

import json
import math

import pytest

from tripphysics.baseline import NaiveBaseline, load_scores_csv
from tripphysics.detector import (
    REQUIRED_FEATURES,
    SCORE_BIAS,
    explain,
    score_account,
)
from tripphysics.report import U_TARGET, render_report

ZERO_FEATURES = {name: 0.0 for name in REQUIRED_FEATURES}


def write_csv(tmp_path, content, name="scores.csv"):
    path = tmp_path / name
    path.write_text(content)
    return str(path)


# ---------------------------------------------------------------------------
# load_scores_csv: accepted formats
# ---------------------------------------------------------------------------

def test_load_scores_csv_with_header(tmp_path):
    path = write_csv(tmp_path, "account_id,score\nacc_1,0.5\nacc_2,0.25\n")
    assert load_scores_csv(path) == {"acc_1": 0.5, "acc_2": 0.25}


def test_load_scores_csv_without_header(tmp_path):
    path = write_csv(tmp_path, "acc_1,0.5\nacc_2,0.25\n")
    assert load_scores_csv(path) == {"acc_1": 0.5, "acc_2": 0.25}


def test_load_scores_csv_rejects_decimal_commas(tmp_path):
    # A European Excel export writes 0,73 rather than 0.73. The csv module then
    # splits "acc_1,0,73" into three fields, the score column reads "0", and
    # float("0") parses cleanly: every score silently collapses to zero while
    # the account count still reconciles. The pilot would be scored against a
    # column of zeros and nothing would look wrong until the debrief.
    path = write_csv(tmp_path, "account_id,score\nacc_1,0,73\nacc_2,0,42\n")
    with pytest.raises(ValueError) as excinfo:
        load_scores_csv(path)
    message = str(excinfo.value)
    assert "expected 2 columns, got 3" in message
    assert ":2:" in message          # the offending line is named


def test_load_scores_csv_strips_excel_byte_order_mark(tmp_path):
    # Excel's "CSV UTF-8" prepends a BOM. Without utf-8-sig the first header
    # cell reads "﻿account_id", the header is not recognised, and the row
    # is treated as data: one account is silently renamed and never joins
    # against the trip data.
    path = write_csv(tmp_path, "﻿account_id,score\nacc_1,0.5\n")
    scores = load_scores_csv(path)
    assert scores == {"acc_1": 0.5}
    assert not any(key.startswith("﻿") for key in scores)


def test_load_scores_csv_header_is_case_insensitive(tmp_path):
    path = write_csv(tmp_path, "Account_ID,SCORE\nacc_1,0.75\n")
    assert load_scores_csv(path) == {"acc_1": 0.75}


def test_load_scores_csv_tolerates_column_order_and_extra_columns(tmp_path):
    # A production export rarely has exactly two columns in the right order.
    path = write_csv(tmp_path, "score,model_version,account_id\n0.5,v3,acc_1\n")
    assert load_scores_csv(path) == {"acc_1": 0.5}


def test_load_scores_csv_skips_blank_lines(tmp_path):
    path = write_csv(tmp_path, "account_id,score\n\nacc_1,0.5\n\nacc_2,0.6\n\n")
    assert load_scores_csv(path) == {"acc_1": 0.5, "acc_2": 0.6}


def test_load_scores_csv_strips_whitespace_around_ids(tmp_path):
    path = write_csv(tmp_path, "account_id,score\n  acc_1  ,0.5\n")
    assert load_scores_csv(path) == {"acc_1": 0.5}


def test_load_scores_csv_accepts_scientific_and_negative_scores(tmp_path):
    # Scores are only required to be ordered, so log-odds and negative values
    # are legitimate production exports.
    path = write_csv(tmp_path, "account_id,score\nacc_1,1e-3\nacc_2,-2.5\n")
    assert load_scores_csv(path) == {"acc_1": 0.001, "acc_2": -2.5}


# ---------------------------------------------------------------------------
# load_scores_csv: failure modes (these must be loud)
# ---------------------------------------------------------------------------

def test_load_scores_csv_rejects_empty_file(tmp_path):
    with pytest.raises(ValueError, match="empty CSV"):
        load_scores_csv(write_csv(tmp_path, ""))


def test_load_scores_csv_rejects_whitespace_only_file(tmp_path):
    with pytest.raises(ValueError, match="empty CSV"):
        load_scores_csv(write_csv(tmp_path, "\n\n   \n"))


def test_load_scores_csv_rejects_duplicate_account_ids(tmp_path):
    # Silent deduplication is how an offline evaluation quietly goes wrong:
    # the second row would overwrite the first and nobody would know.
    path = write_csv(tmp_path, "account_id,score\nacc_1,0.5\nacc_1,0.9\n")
    with pytest.raises(ValueError, match="duplicate account_id"):
        load_scores_csv(path)


def test_load_scores_csv_rejects_non_numeric_score(tmp_path):
    path = write_csv(tmp_path, "account_id,score\nacc_1,not_a_number\n")
    with pytest.raises(ValueError, match="is not a number"):
        load_scores_csv(path)


def test_load_scores_csv_rejects_empty_account_id(tmp_path):
    path = write_csv(tmp_path, "account_id,score\n,0.5\n")
    with pytest.raises(ValueError, match="empty account_id"):
        load_scores_csv(path)


def test_load_scores_csv_rejects_headerless_file_with_wrong_column_count(tmp_path):
    path = write_csv(tmp_path, "acc_1,0.5,extra\n")
    with pytest.raises(ValueError, match="two-column"):
        load_scores_csv(path)


def test_load_scores_csv_rejects_short_row(tmp_path):
    path = write_csv(tmp_path, "account_id,score\nacc_1\n")
    with pytest.raises(ValueError, match="expected at least"):
        load_scores_csv(path)


def test_load_scores_csv_error_messages_carry_the_line_number(tmp_path):
    # An operator debugging a 100k-row export needs the row, not just "bad file".
    path = write_csv(tmp_path, "account_id,score\nacc_1,0.5\nacc_2,oops\n")
    with pytest.raises(ValueError, match=r":3:"):
        load_scores_csv(path)


# ---------------------------------------------------------------------------
# load_scores_csv: non-finite scores
#
# float() happily accepts "nan", "inf" and "-inf", and every one of them
# corrupts the evaluation silently rather than failing: NaN sorts to the front
# of a descending sort, so the budgeted threshold becomes NaN and
# `score > threshold` is False for every account, collapsing recall to zero.
# Since this loader is the documented integration point for a production score
# export -- where NULL scores are entirely normal -- these must be rejected
# loudly, naming the offending account.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("literal", ["nan", "NaN", "inf", "-inf", "Infinity"])
def test_load_scores_csv_rejects_non_finite_scores(tmp_path, literal):
    path = write_csv(tmp_path, f"account_id,score\nacc_1,{literal}\n")
    with pytest.raises(ValueError, match="not finite"):
        load_scores_csv(path)


def test_non_finite_score_error_names_the_account(tmp_path):
    # The account id is the actionable part of the message: an operator has
    # to go and fix (or drop) that specific row in a 100k-row export.
    path = write_csv(tmp_path,
                     "account_id,score\ngood_1,0.5\nbroken_acct,nan\n")
    with pytest.raises(ValueError, match=r"broken_acct"):
        load_scores_csv(path)


def test_non_finite_score_error_carries_file_and_line(tmp_path):
    path = write_csv(tmp_path,
                     "account_id,score\ngood_1,0.5\ngood_2,0.6\nbad_3,inf\n")
    with pytest.raises(ValueError, match=r"scores\.csv:4:"):
        load_scores_csv(path)


def test_non_finite_score_error_explains_the_consequence(tmp_path):
    # The message has to say why this is fatal rather than merely unusual,
    # or the natural reaction is to work around it by coercing NaN to zero --
    # which reintroduces the silent corruption under a different name.
    path = write_csv(tmp_path, "account_id,score\nacc_1,nan\n")
    with pytest.raises(ValueError, match="thresholding and ranking"):
        load_scores_csv(path)


def test_load_scores_csv_still_accepts_finite_extremes(tmp_path):
    # The check must reject non-finite values only. Very large, very small
    # and negative scores are legitimate production outputs (logits, log-odds)
    # and must keep loading.
    path = write_csv(tmp_path,
                     "account_id,score\nacc_1,1e308\nacc_2,-1e308\nacc_3,0.0\n")
    scores = load_scores_csv(path)
    assert scores == {"acc_1": 1e308, "acc_2": -1e308, "acc_3": 0.0}
    assert all(math.isfinite(v) for v in scores.values())


def test_load_scores_csv_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scores_csv(str(tmp_path / "does_not_exist.csv"))


# ---------------------------------------------------------------------------
# NaiveBaseline
# ---------------------------------------------------------------------------

def _fix(is_mock=False, cell_id=1):
    return {"t": 0.0, "x_m": 0.0, "y_m": 0.0, "accuracy_m": 5.0,
            "speed_mps": 0.0, "is_mock": is_mock, "cell_id": cell_id}


def test_naive_baseline_scores_empty_account_as_zero():
    assert NaiveBaseline().score({"account_id": "a", "trips": []}) == 0.0


def test_naive_baseline_tolerates_missing_trips_key():
    assert NaiveBaseline().score({"account_id": "a"}) == 0.0


def test_naive_baseline_tolerates_trips_without_fixes():
    assert NaiveBaseline().score({"trips": [{"fixes": []}]}) == 0.0


def test_naive_baseline_saturates_on_mock_fixes():
    # Fully mock-flagged account: the mock component maxes out at W_MOCK.
    account = {"trips": [{"fixes": [_fix(is_mock=True)] * 20}]}
    assert NaiveBaseline().score(account) == pytest.approx(NaiveBaseline.W_MOCK)


def test_naive_baseline_scores_clean_single_cell_account_as_zero():
    account = {"trips": [{"fixes": [_fix()] * 20}]}
    assert NaiveBaseline().score(account) == 0.0


def test_naive_baseline_rewards_anchor_cell_diversity():
    # Every trip anchors to a different cell: maximum diversity component.
    account = {"trips": [{"fixes": [_fix(cell_id=i)] * 3} for i in range(5)]}
    assert NaiveBaseline().score(account) == pytest.approx(
        NaiveBaseline.W_ANCHOR_DIVERSITY)


def test_naive_baseline_score_is_bounded():
    account = {"trips": [{"fixes": [_fix(is_mock=True, cell_id=i)] * 3}
                         for i in range(5)]}
    score = NaiveBaseline().score(account)
    assert 0.0 <= score <= 1.0


def test_naive_baseline_breaks_modal_cell_ties_deterministically():
    # Two cells appear equally often in a trip; the documented rule picks the
    # smallest cell_id, so the score must not depend on fix ordering.
    forward = {"trips": [{"fixes": [_fix(cell_id=1), _fix(cell_id=2)]},
                         {"fixes": [_fix(cell_id=1), _fix(cell_id=2)]}]}
    backward = {"trips": [{"fixes": [_fix(cell_id=2), _fix(cell_id=1)]},
                          {"fixes": [_fix(cell_id=2), _fix(cell_id=1)]}]}
    assert NaiveBaseline().score(forward) == NaiveBaseline().score(backward)


# ---------------------------------------------------------------------------
# detector.score_account
# ---------------------------------------------------------------------------

# The saturated vector is built from REQUIRED_FEATURES so that adding a
# feature to the detector cannot leave this fixture silently one key short.
SATURATED_FEATURES = {name: 1e6 for name in REQUIRED_FEATURES}


def test_score_is_bounded_for_zero_and_saturated_evidence():
    zero = score_account(ZERO_FEATURES)
    saturated = score_account(SATURATED_FEATURES)
    assert 0.0 <= zero < saturated < 1.0


def test_an_account_with_no_observations_scores_a_hard_zero():
    # An all-zero feature vector means "no usable trips", i.e. ABSENCE of
    # evidence -- which is not the same claim as evidence of innocence. It is
    # mapped to an explicit 0.0 so an empty account cannot share a score with
    # a genuinely clean one that was actually observed.
    assert score_account(ZERO_FEATURES) == 0.0


def test_an_observed_but_clean_account_sits_above_the_empty_one():
    # The distinction the hard zero exists to preserve. A feature vector that
    # is *almost* all-zero has been observed, so it takes the ordinary
    # logistic floor -- strictly above the "nothing to say" score.
    observed_clean = dict(ZERO_FEATURES, night_share=1e-9)
    floor = score_account(observed_clean)
    bias_only = 1.0 / (1.0 + math.exp(SCORE_BIAS))
    assert score_account(ZERO_FEATURES) < floor
    # All-zero *features* are not zero *evidence*: the soft-threshold
    # transforms are logistics, so a feature sitting at 0 well below its
    # centre still contributes a small positive tail. The floor is therefore
    # slightly above logistic(-SCORE_BIAS) and must stay in that region.
    assert bias_only < floor < 0.12, floor
    assert floor == pytest.approx(0.0946, abs=5e-4)


def test_score_rejects_missing_feature_keys():
    # A silently-degraded feature is worse than a crash in a fraud pipeline.
    with pytest.raises(ValueError, match="missing feature keys"):
        score_account({"mock_rate": 0.0})


def test_score_ignores_unknown_extra_features():
    # Extra keys are tolerated so a caller can pass a richer dict.
    enriched = dict(ZERO_FEATURES, some_future_feature=99.0)
    assert score_account(enriched) == score_account(ZERO_FEATURES)


@pytest.mark.parametrize("feature", sorted(REQUIRED_FEATURES))
def test_score_is_monotone_in_every_feature(feature):
    # The stated design constraint: more evidence never lowers the score.
    # A sign error in any single evidence transform would break this.
    low = dict(ZERO_FEATURES)
    high = dict(ZERO_FEATURES, **{feature: 1.0 if feature in
                                  ("mock_rate", "cell_mismatch_rate",
                                   "overlap_rate", "style_bimodality",
                                   "night_share") else 1e6})
    assert score_account(high) >= score_account(low), feature


def test_score_handles_negative_feature_values_without_crashing():
    # Garbage-in from a broken upstream join must not raise or leave [0, 1].
    negative = {name: -1.0 for name in REQUIRED_FEATURES}
    score = score_account(negative)
    assert 0.0 <= score <= 1.0


def test_explain_matches_score_account():
    features = dict(ZERO_FEATURES, overlap_rate=0.04, handover_gap_score=1.5)
    breakdown = explain(features)
    assert breakdown["score"] == pytest.approx(score_account(features))


def test_explain_contributions_sum_to_raw():
    features = dict(ZERO_FEATURES, mock_rate=0.03, max_jump_m=4000.0)
    breakdown = explain(features)
    assert breakdown["raw"] == pytest.approx(sum(breakdown["contribution"].values()))


def test_explain_evidence_is_bounded_per_feature():
    features = dict(ZERO_FEATURES, max_jump_m=1e9, overlap_rate=5.0)
    breakdown = explain(features)
    for name, value in breakdown["evidence"].items():
        assert 0.0 <= value <= 1.0, name


def test_explain_covers_every_required_feature():
    breakdown = explain(ZERO_FEATURES)
    assert set(breakdown["evidence"]) == set(REQUIRED_FEATURES)
    assert set(breakdown["contribution"]) == set(REQUIRED_FEATURES)


def test_explain_rejects_missing_features():
    with pytest.raises(ValueError, match="missing feature keys"):
        explain({"mock_rate": 0.0})


def test_overlap_outweighs_the_weakest_signals():
    # Weight ordering is a documented design claim: a physically impossible
    # signal (two trips at once) must dominate a confoundable one (night work).
    overlap = dict(ZERO_FEATURES, overlap_rate=0.05)
    night = dict(ZERO_FEATURES, night_share=1.0)
    assert score_account(overlap) > score_account(night)


# ---------------------------------------------------------------------------
# report.render_report
# ---------------------------------------------------------------------------

MINIMAL_RESULT = {
    "R0": 0.5, "R1": 0.65, "U": 0.30, "budget_fp_per_10k": 50.0,
    "pr_auc_baseline": 0.55, "pr_auc_pilot": 0.75,
    "precision_at_500_baseline": 0.08, "precision_at_500_pilot": 0.11,
    "n_accounts": 2000, "n_shared": 60,
    "threshold_baseline": 0.34, "threshold_pilot": 0.98,
}
MINIMAL_META = {"seed": 42, "n_accounts": 2000, "shared_fraction": 0.03}


def test_render_report_writes_both_artifacts(tmp_path):
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    render_report(MINIMAL_RESULT, MINIMAL_META, str(json_path), str(md_path))
    assert json_path.exists() and md_path.exists()

    payload = json.loads(json_path.read_text())
    assert payload["metrics"]["U"] == pytest.approx(0.30)
    assert payload["meta"]["seed"] == 42


def test_render_report_creates_missing_parent_directories(tmp_path):
    json_path = tmp_path / "nested" / "deeper" / "report.json"
    md_path = tmp_path / "nested" / "deeper" / "report.md"
    render_report(MINIMAL_RESULT, MINIMAL_META, str(json_path), str(md_path))
    assert json_path.exists() and md_path.exists()


def test_render_report_is_byte_identical_across_renders(tmp_path):
    # No timestamps, no hostnames: the artifacts are pure functions of input.
    first_json, first_md = tmp_path / "a.json", tmp_path / "a.md"
    second_json, second_md = tmp_path / "b.json", tmp_path / "b.md"
    render_report(MINIMAL_RESULT, MINIMAL_META, str(first_json), str(first_md))
    render_report(MINIMAL_RESULT, MINIMAL_META, str(second_json), str(second_md))
    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_md.read_bytes() == second_md.read_bytes()


def test_render_report_emits_strictly_parseable_json_for_infinite_uplift(tmp_path):
    # U = +inf is a documented outcome. json.dumps would emit bare `Infinity`,
    # which is not valid JSON; _json_safe must map it to null so jq and
    # strict parsers downstream do not choke.
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    result = dict(MINIMAL_RESULT, U=math.inf, R0=0.0)
    render_report(result, MINIMAL_META, str(json_path), str(md_path))

    raw = json_path.read_text()
    assert "Infinity" not in raw
    payload = json.loads(raw)  # strict parse must succeed
    assert payload["metrics"]["U"] is None


def test_markdown_reports_infinite_uplift_in_words(tmp_path):
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    render_report(dict(MINIMAL_RESULT, U=math.inf, R0=0.0), MINIMAL_META,
                  str(json_path), str(md_path))
    markdown = md_path.read_text()
    assert "+inf" in markdown
    assert "baseline recall is zero" in markdown


def _render_markdown_text(tmp_path, name, result, meta):
    md_path = tmp_path / f"{name}.md"
    render_report(result, meta, str(tmp_path / f"{name}.json"), str(md_path))
    return md_path.read_text()


def test_synthetic_run_never_renders_a_pass_fail_verdict(tmp_path):
    # THE claim this section of the report exists to protect. On synthetic
    # data the generator and the detector were written by the same hand, so
    # the detector is correct by construction and a "MEETS the target" box
    # would be a manufactured result. Whatever U comes out, a synthetic run
    # must say only that the pipeline ran.
    for label, u_value in (("above", U_TARGET + 0.05),
                           ("below", U_TARGET - 0.05),
                           ("huge", U_TARGET + 5.0)):
        markdown = _render_markdown_text(
            tmp_path, label, dict(MINIMAL_RESULT, U=u_value), MINIMAL_META)
        assert "SYNTHETIC DRY-RUN" in markdown
        assert "MEETS the target" not in markdown, (
            f"synthetic run with U={u_value} claimed to meet the target")
        assert "BELOW the target" not in markdown


def test_synthetic_is_the_default_when_meta_is_silent(tmp_path):
    # Fail safe: a meta dict that forgets to say what it is must be treated
    # as synthetic. The dangerous default is the one that renders a verdict.
    meta_without_flag = {k: v for k, v in MINIMAL_META.items()
                         if k != "synthetic"}
    markdown = _render_markdown_text(
        tmp_path, "default", dict(MINIMAL_RESULT, U=U_TARGET + 0.05),
        meta_without_flag)
    assert "SYNTHETIC DRY-RUN" in markdown
    assert "MEETS the target" not in markdown


def test_real_data_verdict_reflects_the_pilot_target(tmp_path):
    # Only a run against real labels is compared to U_TARGET, and there the
    # verdict must track the bar in both directions.
    passing = _render_markdown_text(
        tmp_path, "pass", dict(MINIMAL_RESULT, U=U_TARGET + 0.05),
        dict(MINIMAL_META, synthetic=False))
    failing = _render_markdown_text(
        tmp_path, "fail", dict(MINIMAL_RESULT, U=U_TARGET - 0.05),
        dict(MINIMAL_META, synthetic=False))
    assert "MEETS the target" in passing
    assert "BELOW the target" in failing


def test_real_data_verdict_at_exactly_the_target_is_a_pass(tmp_path):
    # The bar is ">= U_TARGET"; the boundary must not fall through to BELOW.
    markdown = _render_markdown_text(
        tmp_path, "boundary", dict(MINIMAL_RESULT, U=U_TARGET),
        dict(MINIMAL_META, synthetic=False))
    assert "MEETS the target" in markdown


def test_infinite_uplift_on_real_data_is_not_reported_as_a_pass(tmp_path):
    # U = +inf means the baseline caught nothing, which makes the ratio
    # undefined rather than infinitely good. Rendering it as a pass would
    # turn the worst-case baseline into the best-case headline.
    markdown = _render_markdown_text(
        tmp_path, "inf_real", dict(MINIMAL_RESULT, U=math.inf, R0=0.0),
        dict(MINIMAL_META, synthetic=False))
    assert "MEETS the target" not in markdown
    assert "undefined" in markdown


def test_markdown_renders_the_budget_curve_when_present(tmp_path):
    # meta["budget_curve"] drives the "U across FP budgets" section, which is
    # the harness's defence against a cherry-picked operating point.
    meta = dict(MINIMAL_META, budget_curve=[
        {"budget_fp_per_10k": 10.0, "R0": 0.3, "R1": 0.4, "U": 0.33, "default": False},
        {"budget_fp_per_10k": 50.0, "R0": 0.5, "R1": 0.65, "U": 0.30, "default": True},
    ])
    md_path = tmp_path / "report.md"
    render_report(MINIMAL_RESULT, meta, str(tmp_path / "report.json"), str(md_path))
    markdown = md_path.read_text()
    assert "U across FP budgets" in markdown
    assert "← default" in markdown


def test_markdown_budget_curve_handles_infinite_uplift_rows(tmp_path):
    meta = dict(MINIMAL_META, budget_curve=[
        {"budget_fp_per_10k": 1.0, "R0": 0.0, "R1": 0.1, "U": math.inf, "default": False},
    ])
    md_path = tmp_path / "report.md"
    render_report(MINIMAL_RESULT, meta, str(tmp_path / "report.json"), str(md_path))
    assert "+inf" in md_path.read_text()


def test_markdown_carries_the_honest_limits_section(tmp_path):
    # The synthetic-data caveat is a load-bearing part of this artifact: it
    # must never be dropped from a report that leaves the machine.
    md_path = tmp_path / "report.md"
    render_report(MINIMAL_RESULT, MINIMAL_META, str(tmp_path / "report.json"),
                  str(md_path))
    markdown = md_path.read_text()
    assert "Honest limits" in markdown
    assert "proves the harness" in markdown
    assert "NaiveBaseline" in markdown


def test_json_report_is_sorted_and_stable(tmp_path):
    # sort_keys=True keeps diffs between runs meaningful.
    json_path = tmp_path / "report.json"
    render_report(MINIMAL_RESULT, MINIMAL_META, str(json_path),
                  str(tmp_path / "report.md"))
    raw = json_path.read_text()
    assert raw.endswith("\n")
    assert json.loads(raw) == json.loads(json.dumps(json.loads(raw)))
