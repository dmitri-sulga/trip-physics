"""Bring-Your-Own-Scores CLI: the input traps and the honest refusals.

Two kinds of assertion here, and the second kind is the point.

The ordinary kind checks that a malformed file is rejected with exit code 2
and a message naming the file. The important kind checks that inputs which
*parse cleanly and then produce a meaningless number* are refused too: a
constant score column and an all-honest label column both make
``eval.evaluate`` return 0.0 without raising, and a zero in a debrief is
harder to catch than a crash. Those two are pinned as errors, and the
absence of any metric in their output is asserted, not assumed.

The third property tested is the U refusal: with one score column U is
undefined, and the run must say so and name the flag that fixes it rather
than printing a zero or comparing the column against itself.
"""

import json

import pytest

from tripphysics.byos import main

# ---------------------------------------------------------------------------
# Fixture helpers. Populations are kept small but above the thresholds that
# would trip the coarse-score and zero-allowance notes, except where a test
# is specifically about those.
# ---------------------------------------------------------------------------


def write(path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def scores_csv(tmp_path, n: int = 400, with_labels: bool = True,
               name: str = "scores.csv") -> str:
    """A separable population: every 10th account is shared and scores high.

    Deliberately separable so any failure in a test is about the input
    handling under test, not about a marginal ranking.
    """
    header = "account_id,score,label\n" if with_labels else "account_id,score\n"
    lines = [header]
    for i in range(n):
        shared = i % 10 == 0
        score = 0.90 + i / (n * 100.0) if shared else 0.10 + i / (n * 100.0)
        lines.append(f"acc_{i:04d},{score:.6f}"
                     + (f",{1 if shared else 0}\n" if with_labels else "\n"))
    return write(tmp_path / name, "".join(lines))


def labels_csv(tmp_path, n: int = 400, name: str = "labels.csv") -> str:
    lines = ["account_id,label\n"]
    for i in range(n):
        lines.append(f"acc_{i:04d},{1 if i % 10 == 0 else 0}\n")
    return write(tmp_path / name, "".join(lines))


def baseline_csv(tmp_path, n: int = 400, name: str = "baseline.csv") -> str:
    """A weaker scorer: shared accounts score high, but less separably."""
    lines = ["account_id,score\n"]
    for i in range(n):
        shared = i % 10 == 0
        score = 0.60 + i / (n * 50.0) if shared else 0.20 + i / (n * 50.0)
        lines.append(f"acc_{i:04d},{score:.6f}\n")
    return write(tmp_path / name, "".join(lines))


# ---------------------------------------------------------------------------
# Happy paths.
# ---------------------------------------------------------------------------


def test_two_scorers_reports_u_and_writes_json(tmp_path, capsys):
    out = str(tmp_path / "byos")
    code = main(["--scores", scores_csv(tmp_path),
                 "--baseline-scores", baseline_csv(tmp_path),
                 "--out", out, "--resamples", "50"])
    assert code == 0
    captured = capsys.readouterr().out
    assert "U (relative uplift)" in captured
    assert "U 95% CI" in captured

    payload = json.loads((tmp_path / "byos.json").read_text())
    assert payload["mode"] == "labels_two_scorers"
    assert payload["headline"]["n_shared"] == 40
    assert payload["bootstrap_ci"]["n_resamples"] == 50
    # The whole budget sweep is present, so the operating point cannot have
    # been picked after seeing the result.
    assert [row["budget_fp_per_10k"] for row in payload["budget_curve"]] == [
        15.0, 30.0, 50.0, 100.0, 200.0]


def test_report_json_carries_no_account_ids_or_scores(tmp_path):
    """The artifact that may leave the machine must be aggregate-only."""
    out = str(tmp_path / "byos")
    main(["--scores", scores_csv(tmp_path),
          "--baseline-scores", baseline_csv(tmp_path),
          "--out", out, "--resamples", "20"])
    text = (tmp_path / "byos.json").read_text()
    assert "acc_0000" not in text
    assert "acc_" not in text


def test_labels_from_separate_file_match_inline_labels(tmp_path, capsys):
    """A combined CSV and a split pair describe the same population."""
    out_a = str(tmp_path / "a")
    main(["--scores", scores_csv(tmp_path), "--out", out_a])
    capsys.readouterr()

    out_b = str(tmp_path / "b")
    main(["--scores", scores_csv(tmp_path, with_labels=False,
                                 name="scores_only.csv"),
          "--labels", labels_csv(tmp_path), "--out", out_b])
    capsys.readouterr()

    a = json.loads((tmp_path / "a.json").read_text())
    b = json.loads((tmp_path / "b.json").read_text())
    assert a["pr_auc"] == b["pr_auc"]
    assert a["budget_curve"] == b["budget_curve"]


# ---------------------------------------------------------------------------
# The U refusal: the single most important honesty property of this module.
# ---------------------------------------------------------------------------


def test_single_scorer_refuses_to_print_u(tmp_path, capsys):
    out = str(tmp_path / "byos")
    code = main(["--scores", scores_csv(tmp_path), "--out", out])
    assert code == 0
    captured = capsys.readouterr().out

    assert "U is NOT reported" in captured
    assert "--baseline-scores" in captured
    # No fabricated value anywhere: not a zero, not a percentage.
    assert "U (relative uplift)" not in captured
    assert "U = 0" not in captured

    payload = json.loads((tmp_path / "byos.json").read_text())
    assert payload["U"] is None
    assert "two score columns" in payload["U_reason"]


def test_single_scorer_still_reports_what_is_defined(tmp_path, capsys):
    """Refusing U must not suppress the metrics that ARE defined for one scorer."""
    out = str(tmp_path / "byos")
    main(["--scores", scores_csv(tmp_path), "--out", out])
    captured = capsys.readouterr().out
    assert "PR-AUC" in captured
    # The label carries the k actually used: on a fixture smaller than 500
    # accounts, precision_at_k truncates, and printing "precision@500" would
    # overstate the depth of the queue the number describes.
    assert "precision@" in captured
    assert "asked for 500" in captured or "precision@500" in captured

    payload = json.loads((tmp_path / "byos.json").read_text())
    assert payload["pr_auc"] > 0.5
    assert all(row["recall"] >= 0.0 for row in payload["budget_curve"])


# ---------------------------------------------------------------------------
# The two silent failures. Both parse cleanly; both must be refused BEFORE
# any metric is printed, since the number they produce is a plausible zero.
# ---------------------------------------------------------------------------


def test_constant_score_column_is_refused_not_scored(tmp_path, capsys):
    path = write(tmp_path / "const.csv",
                 "account_id,score,label\n"
                 + "".join(f"acc_{i},0.5,{i % 2}\n" for i in range(200)))
    code = main(["--scores", path, "--out", str(tmp_path / "byos")])
    assert code == 2

    captured = capsys.readouterr()
    assert "every one of the 200 scores is the same value" in captured.err
    # Nothing resembling a result may have been printed, and no artifact
    # written: a zero from this input is the failure mode being prevented.
    assert "PR-AUC" not in captured.out
    assert "U" not in captured.out
    assert not (tmp_path / "byos.json").exists()


def test_all_honest_labels_are_refused_not_scored(tmp_path, capsys):
    path = write(tmp_path / "honest.csv",
                 "account_id,score,label\n"
                 + "".join(f"acc_{i},0.{i:03d},0\n" for i in range(200)))
    code = main(["--scores", path, "--out", str(tmp_path / "byos")])
    assert code == 2

    captured = capsys.readouterr()
    assert "no confirmed-shared accounts" in captured.err
    assert "nothing was measured" in captured.err
    assert "PR-AUC" not in captured.out
    assert not (tmp_path / "byos.json").exists()


def test_all_positive_labels_are_refused_with_an_operator_message(tmp_path,
                                                                  capsys):
    """evaluate() raises here, but its text is developer-facing; re-explain."""
    path = write(tmp_path / "pos.csv",
                 "account_id,score,label\n"
                 + "".join(f"acc_{i},0.{i:03d},1\n" for i in range(200)))
    code = main(["--scores", path, "--out", str(tmp_path / "byos")])
    assert code == 2
    err = capsys.readouterr().err
    assert "no honest accounts" in err
    assert "per 10,000 honest accounts" in err
    assert "Traceback" not in err


def test_rejection_prints_no_report_banner(tmp_path, capsys):
    """An error followed by a report header reads as 'failed, then ran anyway'."""
    path = write(tmp_path / "honest.csv",
                 "account_id,score,label\n"
                 + "".join(f"acc_{i},0.{i:03d},0\n" for i in range(200)))
    assert main(["--scores", path, "--out", str(tmp_path / "byos")]) == 2
    assert "measuring your scores" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Malformed input. The score column is parsed by baseline.load_scores_csv, so
# these assert that its errors reach the user formatted rather than as a
# traceback -- not that the parse itself is reimplemented here.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,body,expected", [
    ("empty.csv", "", "empty CSV"),
    ("dup.csv", "account_id,score\nacc_1,0.5\nacc_1,0.7\n", "duplicate account_id"),
    ("nan.csv", "account_id,score\nacc_1,0.5\nacc_2,nan\n", "not finite"),
    ("comma.csv", "account_id,score\nacc_1,0,73\nacc_2,0,44\n", "decimal commas"),
    ("text.csv", "account_id,score\nacc_1,high\n", "is not a number"),
])
def test_malformed_score_file_exits_2_without_a_traceback(tmp_path, capsys,
                                                          name, body, expected):
    path = write(tmp_path / name, body)
    code = main(["--scores", path, "--out", str(tmp_path / "byos")])
    assert code == 2
    err = capsys.readouterr().err
    assert expected in err
    assert name in err  # the message names the file
    assert "Traceback" not in err
    assert not (tmp_path / "byos.json").exists()


def test_missing_score_file_names_the_flag(tmp_path, capsys):
    code = main(["--scores", str(tmp_path / "absent.csv"),
                 "--out", str(tmp_path / "byos")])
    assert code == 2
    err = capsys.readouterr().err
    assert "no such file" in err
    assert "--scores" in err


@pytest.mark.parametrize("body,expected", [
    ("account_id,label\nacc_0000,maybe\n", "is not recognised"),
    ("account_id,label\nacc_0000,\n", "empty label"),
    ("account_id,label\nacc_0000,1\nacc_0000,0\n", "duplicate account_id"),
    ("account_id,label\n", "no labelled accounts"),
])
def test_malformed_label_file_exits_2(tmp_path, capsys, body, expected):
    path = write(tmp_path / "labels.csv", body)
    code = main(["--scores", scores_csv(tmp_path, with_labels=False),
                 "--labels", path, "--out", str(tmp_path / "byos")])
    assert code == 2
    err = capsys.readouterr().err
    assert expected in err
    assert "Traceback" not in err


def test_label_tokens_are_not_parsed_as_python_truthiness(tmp_path, capsys):
    """bool('0') and bool('false') are True; reading labels that way inverts them."""
    path = write(tmp_path / "labels.csv",
                 "account_id,label\n"
                 + "".join(f"acc_{i:04d},{'false' if i % 10 else 'true'}\n"
                           for i in range(400)))
    code = main(["--scores", scores_csv(tmp_path, with_labels=False),
                 "--labels", path, "--out", str(tmp_path / "byos")])
    assert code == 0
    capsys.readouterr()
    payload = json.loads((tmp_path / "byos.json").read_text())
    assert payload["n_shared"] == 40  # not 400


def test_mismatched_account_ids_name_the_overlap_and_the_missing_ids(tmp_path,
                                                                     capsys):
    path = write(tmp_path / "labels.csv",
                 "account_id,label\nother_1,1\nother_2,0\nacc_0000,1\n")
    code = main(["--scores", scores_csv(tmp_path, with_labels=False),
                 "--labels", path, "--out", str(tmp_path / "byos")])
    assert code == 2
    err = capsys.readouterr().err
    assert "2 of 3 accounts" in err       # the overlap counts
    assert "other_1" in err               # the first few missing ids
    assert "Traceback" not in err         # evaluate()'s message never surfaces raw


def test_partially_labelled_population_reports_what_was_dropped(tmp_path,
                                                                capsys):
    """Scored-but-unlabelled accounts are excluded, and the count is stated.

    An unconfirmed account is not a confirmed-honest one, so it cannot enter
    the honest population that the FP budget is spent over. Dropping it
    silently would shrink that denominator without saying so.
    """
    partial = write(tmp_path / "partial.csv",
                    "account_id,label\n"
                    + "".join(f"acc_{i:04d},{1 if i % 10 == 0 else 0}\n"
                              for i in range(200)))
    code = main(["--scores", scores_csv(tmp_path, with_labels=False),
                 "--labels", partial, "--out", str(tmp_path / "byos")])
    assert code == 0
    captured = capsys.readouterr().out
    assert "200 of 400 scored accounts have no label" in captured

    payload = json.loads((tmp_path / "byos.json").read_text())
    assert payload["n_accounts"] == 200
    assert payload["n_shared"] == 20


def test_two_sources_of_labels_are_refused(tmp_path, capsys):
    """A label column AND --labels: picking one would silently mislabel."""
    code = main(["--scores", scores_csv(tmp_path),
                 "--labels", labels_csv(tmp_path),
                 "--out", str(tmp_path / "byos")])
    assert code == 2
    assert "Two sources of ground truth" in capsys.readouterr().err


def test_baseline_missing_labelled_accounts_is_refused(tmp_path, capsys):
    short = write(tmp_path / "short_baseline.csv",
                  "account_id,score\nacc_0000,0.5\nacc_0001,0.4\n")
    code = main(["--scores", scores_csv(tmp_path), "--baseline-scores", short,
                 "--out", str(tmp_path / "byos")])
    assert code == 2
    err = capsys.readouterr().err
    assert "U compares two scorers over one population" in err


# ---------------------------------------------------------------------------
# No labels: say what is measurable, never a placeholder for what is not.
# ---------------------------------------------------------------------------


def test_without_labels_names_what_cannot_be_computed(tmp_path, capsys):
    out = str(tmp_path / "byos")
    code = main(["--scores", scores_csv(tmp_path, with_labels=False),
                 "--out", out])
    assert code == 0
    captured = capsys.readouterr().out

    assert "NOT computable without labels" in captured
    assert "PR-AUC" in captured  # named as unavailable...
    # ...and never accompanied by a value. No metric line may carry a number.
    assert "PR-AUC:" not in captured
    assert "recall" not in captured.split("Accounts scored")[1]

    payload = json.loads((tmp_path / "byos.json").read_text())
    assert payload["mode"] == "no_labels"
    assert set(payload["not_computable"]) == {
        "recall_at_budget", "pr_auc", "precision_at_k", "U"}
    assert "pr_auc" not in payload or payload.get("pr_auc") is None


def test_without_labels_still_reports_thresholds_and_flag_counts(tmp_path,
                                                                 capsys):
    out = str(tmp_path / "byos")
    main(["--scores", scores_csv(tmp_path, with_labels=False), "--out", out])
    capsys.readouterr()
    payload = json.loads((tmp_path / "byos.json").read_text())
    curve = payload["budget_curve"]
    assert [row["budget_fp_per_10k"] for row in curve] == [15.0, 30.0, 50.0,
                                                           100.0, 200.0]
    # A larger budget never raises the threshold, so flag counts are monotone.
    counts = [row["n_flagged"] for row in curve]
    assert counts == sorted(counts)


# ---------------------------------------------------------------------------
# Quantization: a budget too small to buy one alert must say so rather than
# printing a threshold that looks like an operating point.
# ---------------------------------------------------------------------------


def test_tiny_population_says_the_budget_buys_no_alerts(tmp_path, capsys):
    path = write(tmp_path / "tiny.csv",
                 "account_id,score,label\n"
                 "acc_1,0.10,0\nacc_2,0.20,0\nacc_3,0.90,1\nacc_4,0.40,0\n")
    code = main(["--scores", path, "--out", str(tmp_path / "byos")])
    assert code == 0
    captured = capsys.readouterr().out
    assert "budget buys 0 honest alerts" in captured
    assert "none of them is an operating point" in captured

    payload = json.loads((tmp_path / "byos.json").read_text())
    assert all(row["budget_quantizes_to_zero"] for row in payload["budget_curve"])


def test_thin_positive_count_is_qualified_before_the_numbers(tmp_path, capsys):
    """The 150-positive pilot minimum, printed above the table it qualifies."""
    path = write(tmp_path / "thin.csv",
                 "account_id,score,label\n"
                 + "".join(f"acc_{i:04d},0.{i:03d},{1 if i % 100 == 0 else 0}\n"
                           for i in range(400)))
    assert main(["--scores", path, "--out", str(tmp_path / "byos")]) == 0
    captured = capsys.readouterr().out
    assert "pre-registers a minimum of 150 positives" in captured
    assert captured.index("plumbing check") < captured.index("PR-AUC")


# ---------------------------------------------------------------------------
# Determinism: same input, same seed, byte-identical report -- the property
# the rest of this repo is built on.
# ---------------------------------------------------------------------------


def test_two_runs_produce_identical_reports(tmp_path, capsys):
    args = ["--scores", scores_csv(tmp_path),
            "--baseline-scores", baseline_csv(tmp_path), "--resamples", "50"]
    main(args + ["--out", str(tmp_path / "one")])
    main(args + ["--out", str(tmp_path / "two")])
    capsys.readouterr()
    assert (tmp_path / "one.json").read_bytes() == (tmp_path / "two.json").read_bytes()


def test_runs_with_every_socket_constructor_blocked(tmp_path, capsys,
                                                    monkeypatch):
    """The no-egress claim, enforced rather than asserted in prose.

    The guide tells a fraud engineer their data never leaves the machine and
    points at `docker run --network=none` as the enforcement. That only holds
    if no code path here opens a socket, so the property is pinned here too:
    with every socket constructor replaced by a function that raises, a full
    two-scorer run must still complete and exit 0.
    """
    import socket

    def deny(*args, **kwargs):
        raise AssertionError("byos attempted a network call")

    for name in ("socket", "create_connection", "socketpair", "getaddrinfo"):
        monkeypatch.setattr(socket, name, deny)

    code = main(["--scores", scores_csv(tmp_path),
                 "--baseline-scores", baseline_csv(tmp_path),
                 "--out", str(tmp_path / "byos"), "--resamples", "20"])
    capsys.readouterr()
    assert code == 0
    assert (tmp_path / "byos.json").exists()


def test_negative_budget_is_refused(tmp_path, capsys):
    code = main(["--scores", scores_csv(tmp_path), "--budget", "-1",
                 "--out", str(tmp_path / "byos")])
    assert code == 2
    assert "--budget must be >= 0" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The held-out FP budget block: promised vs realized with a bootstrap CI, the
# single-scorer deliverable. Degenerate inputs must refuse, not fabricate.
# ---------------------------------------------------------------------------


def test_single_scorer_reports_promised_vs_realized_with_ci(tmp_path, capsys):
    out = str(tmp_path / "byos")
    code = main(["--scores", scores_csv(tmp_path, n=1000), "--out", out,
                 "--resamples", "50"])
    assert code == 0
    captured = capsys.readouterr().out
    assert "promised 50/10k -> realized" in captured
    assert "95% CI" in captured

    payload = json.loads((tmp_path / "byos.json").read_text())
    hb = payload["holdout_fp_budget"]
    assert hb["computable"] is True
    assert hb["n_resamples"] == 50
    assert hb["realized_fp_per_10k"] >= 0.0
    assert hb["ci_low"] <= hb["ci_high"]
    assert payload["headline_budget_fp_per_10k"] == 50.0


def test_holdout_budget_respects_the_budget_flag(tmp_path, capsys):
    out = str(tmp_path / "byos")
    code = main(["--scores", scores_csv(tmp_path, n=1000), "--out", out,
                 "--budget", "100", "--resamples", "50"])
    assert code == 0
    assert "promised 100/10k -> realized" in capsys.readouterr().out
    payload = json.loads((tmp_path / "byos.json").read_text())
    assert payload["holdout_fp_budget"]["budget_fp_per_10k"] == 100.0


def test_holdout_budget_refuses_when_budget_floors_to_zero(tmp_path, capsys):
    """360 honest -> 180-account calibration half -> floor(180*50/10k) = 0."""
    out = str(tmp_path / "byos")
    code = main(["--scores", scores_csv(tmp_path), "--out", out])
    assert code == 0
    captured = capsys.readouterr().out
    assert "not computable" in captured
    assert "calibration half" in captured
    # The refusal must not be accompanied by a fabricated realized number.
    assert "promised 50/10k -> realized" not in captured

    payload = json.loads((tmp_path / "byos.json").read_text())
    hb = payload["holdout_fp_budget"]
    assert hb["computable"] is False
    assert hb["realized_fp_per_10k"] is None
    assert "calibration half" in hb["reason"]


def test_holdout_budget_refuses_with_one_honest_account(tmp_path, capsys):
    path = write(tmp_path / "tiny.csv",
                 "account_id,score,label\n"
                 "acc_1,0.10,0\nacc_2,0.90,1\nacc_3,0.80,1\n")
    code = main(["--scores", path, "--out", str(tmp_path / "byos")])
    assert code == 0
    captured = capsys.readouterr().out
    assert "not computable" in captured

    payload = json.loads((tmp_path / "byos.json").read_text())
    hb = payload["holdout_fp_budget"]
    assert hb["computable"] is False
    assert "honest account" in hb["reason"]


def test_holdout_budget_block_is_deterministic(tmp_path, capsys):
    args = ["--scores", scores_csv(tmp_path, n=1000), "--resamples", "50"]
    main(args + ["--out", str(tmp_path / "one")])
    main(args + ["--out", str(tmp_path / "two")])
    capsys.readouterr()
    assert (tmp_path / "one.json").read_bytes() == \
        (tmp_path / "two.json").read_bytes()
    assert (tmp_path / "one.md").read_bytes() == \
        (tmp_path / "two.md").read_bytes()


def test_holdout_budget_ci_point_matches_two_scorer_check():
    """Same seed, same split: the point estimate must equal holdout_fp_check's
    baseline number when both scorer arguments are the same column."""
    from tripphysics.eval import holdout_fp_budget_ci, holdout_fp_check

    labels, scores = {}, {}
    for i in range(1000):
        account = f"acc_{i:04d}"
        shared = i % 10 == 0
        labels[account] = shared
        scores[account] = (0.9 if shared else 0.1) + i / 100000.0

    ci = holdout_fp_budget_ci(labels, scores, 50.0, n_resamples=10, seed=3)
    check = holdout_fp_check(labels, scores, scores, 50.0, seed=3)
    assert ci["computable"] is True
    assert ci["realized_fp_per_10k"] == check["realized_fp_per_10k_baseline"]
    assert ci["n_calibration"] == check["n_calibration"]
    assert ci["n_measurement"] == check["n_measurement"]


# ---------------------------------------------------------------------------
# The budget the two-scorer headline was measured at must be printed next to
# it, and an unspent budget must be explained, not left as a suspicious zero.
# ---------------------------------------------------------------------------


def test_two_scorers_print_realized_spend_and_utilization(tmp_path, capsys):
    code = main(["--scores", scores_csv(tmp_path),
                 "--baseline-scores", baseline_csv(tmp_path),
                 "--out", str(tmp_path / "byos"), "--resamples", "20"])
    assert code == 0
    captured = capsys.readouterr().out
    assert "realized FP spend" in captured
    assert "utilization" in captured


def coarse_baseline_csv(tmp_path, n: int = 400,
                        name: str = "coarse_baseline.csv") -> str:
    """Every honest account ties at 0.5: no strict threshold can spend >0."""
    lines = ["account_id,score\n"]
    for i in range(n):
        shared = i % 10 == 0
        score = 0.90 + i / (n * 100.0) if shared else 0.50
        lines.append(f"acc_{i:04d},{score:.6f}\n")
    return write(tmp_path / name, "".join(lines))


def test_mega_tie_at_the_top_is_named_as_the_cause(tmp_path, capsys):
    """A tie block bigger than the budget makes realized spend 0; the run must
    say which ladder is responsible and that R0/R1 sit at zero spend."""
    code = main(["--scores", scores_csv(tmp_path),
                 "--baseline-scores", coarse_baseline_csv(tmp_path),
                 "--out", str(tmp_path / "byos"), "--resamples", "20"])
    assert code == 0
    captured = capsys.readouterr().out
    assert "realized spend is 0" in captured
    assert "baseline" in captured
    assert "tie" in captured


def test_identical_sweep_rows_are_explained(tmp_path, capsys):
    """Five identical rows read as 'the tool is broken'; the run must say the
    budgets collapsed to one realized spend instead."""
    code = main(["--scores", scores_csv(tmp_path),
                 "--baseline-scores", coarse_baseline_csv(tmp_path),
                 "--out", str(tmp_path / "byos"), "--resamples", "20"])
    assert code == 0
    captured = capsys.readouterr().out
    assert "every row of this sweep is identical" in captured
    assert "not broken" in captured


def test_low_utilization_prints_the_granularity_warning(tmp_path, capsys):
    """Baseline spends some budget but under half of it: U must be qualified.

    At a 200/10k budget over 360 honest accounts the budget allows 7 honest
    flags; the baseline below can realize only 3 (three distinct honest
    scores above a tie block), so utilization is 3/7 and the warning must
    print. Both scorers rank the same three honest accounts on top, so the
    stacked ladder stays fine-grained and cannot collapse the spend to 0.
    """
    cand_lines = ["account_id,score,label\n"]
    base_lines = ["account_id,score\n"]
    for i in range(400):
        shared = i % 10 == 0
        if shared:
            cand, base = 0.90 + i / 40000.0, 0.90 + i / 40000.0
        elif i in (1, 2, 3):
            # Top honest accounts in BOTH scorers, in the same order.
            cand, base = 0.20 - i / 100.0, 0.81 - i / 100.0
        elif i % 10 < 9:
            cand, base = 0.10 + i / 40000.0, 0.50   # baseline tie block
        else:
            cand, base = 0.10 + i / 40000.0, 0.10 + i / 40000.0
        cand_lines.append(f"acc_{i:04d},{cand:.6f},{1 if shared else 0}\n")
        base_lines.append(f"acc_{i:04d},{base:.6f}\n")
    scores = write(tmp_path / "cand.csv", "".join(cand_lines))
    coarse = write(tmp_path / "coarse.csv", "".join(base_lines))

    code = main(["--scores", scores,
                 "--baseline-scores", coarse, "--budget", "200",
                 "--out", str(tmp_path / "byos"), "--resamples", "20"])
    assert code == 0
    captured = capsys.readouterr().out
    assert "cannot spend the budget" in captured
    assert "utilization" in captured


def test_coarse_baseline_scores_get_the_tie_note_too(tmp_path, capsys):
    """The distinct-value note fired only for --scores; the baseline column is
    where the coarse export usually is."""
    lines = ["account_id,score\n"]
    for i in range(400):
        lines.append(f"acc_{i:04d},{(i % 5) / 10.0:.1f}\n")
    coarse = write(tmp_path / "five_values.csv", "".join(lines))
    code = main(["--scores", scores_csv(tmp_path),
                 "--baseline-scores", coarse,
                 "--out", str(tmp_path / "byos"), "--resamples", "20"])
    assert code == 0
    captured = capsys.readouterr().out
    assert "five_values.csv has only 5 distinct score values" in captured


# ---------------------------------------------------------------------------
# Two former tracebacks: a non-UTF-8 label file and an unwritable --out. Both
# must fail as one operator sentence, exit 2, before any metric is printed.
# ---------------------------------------------------------------------------


def test_non_utf8_label_file_exits_2_without_a_traceback(tmp_path, capsys):
    bad = tmp_path / "labels.csv"
    bad.write_bytes("account_id,label\nacc_0000,1\n".encode("utf-16"))
    code = main(["--scores", scores_csv(tmp_path, with_labels=False),
                 "--labels", str(bad), "--out", str(tmp_path / "byos")])
    assert code == 2
    err = capsys.readouterr().err
    assert "labels.csv" in err
    assert "UTF-8" in err
    assert "Traceback" not in err


def test_unwritable_out_fails_before_any_metric(tmp_path, capsys):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    out = str(blocker / "sub" / "byos")
    code = main(["--scores", scores_csv(tmp_path), "--out", out])
    assert code == 2
    captured = capsys.readouterr()
    assert "cannot be written" in captured.err
    assert "Traceback" not in captured.err
    # Failed BEFORE the banner and the numbers, not after them.
    assert "measuring your scores" not in captured.out
    assert "PR-AUC" not in captured.out


def test_out_probe_does_not_truncate_an_existing_report(tmp_path, capsys):
    """The writability probe must never destroy the previous run's report."""
    out = str(tmp_path / "byos")
    main(["--scores", scores_csv(tmp_path), "--out", out])
    capsys.readouterr()
    first = (tmp_path / "byos.json").read_bytes()
    assert first  # non-empty: the probe on the SECOND run must not blank it
    code = main(["--scores", scores_csv(tmp_path), "--out", out])
    capsys.readouterr()
    assert code == 0
    assert (tmp_path / "byos.json").read_bytes() == first


# ---------------------------------------------------------------------------
# The markdown report: the caveats must reach the file the numbers reach.
# ---------------------------------------------------------------------------


def test_markdown_report_is_written_and_carries_the_caveats(tmp_path, capsys):
    out = str(tmp_path / "byos")
    code = main(["--scores", scores_csv(tmp_path), "--out", out])
    assert code == 0
    assert "byos.md" in capsys.readouterr().out

    text = (tmp_path / "byos.md").read_text()
    # The stdout qualifications, verbatim, in the persistent artifact: the
    # thin-label note and the held-out budget refusal both fire on this input.
    assert "pre-registers a minimum of 150 positives" in text
    assert "not computable" in text
    assert "U is NOT reported" in text
    # Aggregate-only, like the JSON.
    assert "acc_" not in text


def test_markdown_report_carries_the_two_scorer_headline(tmp_path, capsys):
    out = str(tmp_path / "byos")
    code = main(["--scores", scores_csv(tmp_path),
                 "--baseline-scores", baseline_csv(tmp_path),
                 "--out", out, "--resamples", "20"])
    assert code == 0
    capsys.readouterr()
    text = (tmp_path / "byos.md").read_text()
    assert "U (relative uplift)" in text
    assert "realized FP spend" in text
    assert "acc_" not in text
