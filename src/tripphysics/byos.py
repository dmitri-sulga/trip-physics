"""Bring Your Own Scores: run *your* risk scores through this harness.

The rest of this repo measures a detector that has to be integrated first.
This module measures scores you already have. Export one or two CSVs, run one
command locally, and get the same budgeted-recall arithmetic that ``eval.py``
applies to the demo -- with no data leaving the machine and none of this code
running against the database the scores came from.

    python -m tripphysics.byos --scores scores.csv --labels labels.csv --out out/byos

The score column is parsed by :func:`baseline.load_scores_csv`, unchanged, so
the traps that loader already catches -- decimal commas, duplicate ids,
NaN, BOM, ragged rows -- are caught identically here. What this module adds
is the *evaluation-shaped* failure: an input that parses perfectly and then
produces a number that is arithmetically correct and analytically worthless.
Two of those are live in ``eval.evaluate`` today:

* every score identical -> U = 0.0, no error. A constant column (a broken
  export, a model that collapsed) is indistinguishable from "the detector
  added nothing" if you only read the headline.
* every label honest -> R0 = R1 = U = 0.0, no error. Nothing has been
  measured at all; the zero is the absence of positives, not a result.
  (All-positive labels do raise, in ``bootstrap_uplift_ci``; all-honest
  does not.)

Both are refused here, before any metric is printed, because a plausible
zero costs more than a crash: a crash gets fixed in ten minutes, a plausible
zero gets quoted in a debrief.

The other decision this module makes, and the one worth reading twice:
**U is an uplift of one scorer over another.** With a single score column
there is exactly one scorer, so U is not defined -- not zero, not "no
uplift", undefined. This module refuses to print it and says which second
file would make it computable. That refusal is the whole point of running
someone else's measurement code rather than your own.
"""

import argparse
import contextlib
import csv
import io
import json
import math
import os
import sys

from .baseline import load_scores_csv
from .eval import (
    DEFAULT_BUDGET_FP_PER_10K,
    bootstrap_uplift_ci,
    evaluate,
    holdout_fp_budget_ci,
    holdout_fp_check,
    pr_auc,
    precision_at_k,
    rank_normalize,
    threshold_at_fp_budget,
)

# Budgets swept in every report, matching the demo's sweep so a BYOS run and
# a demo run can be read side by side. The operating point is never a single
# number chosen after seeing the result.
BUDGET_SWEEP = (15.0, 30.0, 50.0, 100.0, 200.0)

# Truthy/falsy spellings accepted in a label column. Deliberately closed: a
# label file is small and hand-made often enough that guessing at an
# unrecognised token ("Y", "shared?", "2") would be guessing at ground truth.
_TRUE_TOKENS = frozenset({"1", "true", "t", "yes", "y", "shared", "positive"})
_FALSE_TOKENS = frozenset({"0", "false", "f", "no", "n", "honest", "negative"})


class InputError(Exception):
    """An input problem the user can fix, phrased for the user.

    Everything raised with this type is printed as a single sentence naming
    the file, the line where one is known, and the fix -- then the process
    exits 2. No traceback reaches the operator: a stack trace tells them
    about our call graph, which is not the thing that is broken.
    """


def load_labels_csv(path: str) -> dict[str, bool]:
    """Load ``{account_id: is_shared}`` from a two-column CSV.

    Accepted like :func:`baseline.load_scores_csv`: header ``account_id`` and
    ``label`` (case-insensitive, any order, extra columns ignored), or a
    headerless two-column file. ``utf-8-sig`` for the same Excel-BOM reason.

    Labels parse from the closed vocabularies above rather than via
    ``bool(value)``, which would read "0" and "false" as True and silently
    turn an entire honest population into positives. An unrecognised token
    is an error naming the line and listing what is accepted, because the
    alternative -- picking an interpretation -- fabricates ground truth.
    """
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = [row for row in csv.reader(fh)
                    if row and any(cell.strip() for cell in row)]
    except FileNotFoundError:
        raise InputError(f"{path}: no such file. Pass the path to your label "
                         f"CSV with --labels, or drop --labels and put a "
                         f"'label' column in the score file.") from None
    except UnicodeDecodeError as exc:
        raise InputError(
            f"{path}: not valid UTF-8 (byte {exc.object[exc.start]:#04x} at "
            f"position {exc.start}). Re-export the file as UTF-8 (or plain "
            f"ASCII) CSV; other encodings are refused rather than guessed at, "
            f"since a wrong guess silently corrupts account ids.") from None
    except OSError as exc:
        raise InputError(f"{path}: cannot be read ({exc.strerror}).") from None

    if not rows:
        raise InputError(f"{path}: empty CSV. Expected 'account_id,label' with "
                         f"one row per labelled account.")

    header = [cell.strip().lower() for cell in rows[0]]
    if "account_id" in header and "label" in header:
        id_col, label_col = header.index("account_id"), header.index("label")
        data_rows, first_line, n_expected = rows[1:], 2, len(rows[0])
    else:
        if len(rows[0]) != 2:
            raise InputError(
                f"{path}: no 'account_id'/'label' header and not a plain "
                f"two-column file (first row has {len(rows[0])} columns). Add a "
                f"header row 'account_id,label', or reduce the file to two "
                f"comma-separated columns.")
        id_col, label_col = 0, 1
        data_rows, first_line, n_expected = rows, 1, 2

    labels: dict[str, bool] = {}
    for offset, row in enumerate(data_rows):
        line = first_line + offset
        # Same exact-field-count rule as the score loader, for the same
        # reason: a longer row means the delimiter or the quoting is wrong,
        # and reading the nominal column of a misaligned row returns a value
        # that parses. Here that would assign the wrong label to an account.
        if len(row) != n_expected:
            raise InputError(
                f"{path}:{line}: expected {n_expected} columns, got {len(row)}: "
                f"{row!r}. A row that does not match the header means the "
                f"delimiter or the quoting is wrong; fix it before loading, "
                f"since a misaligned row still parses and mislabels an account.")
        account_id = row[id_col].strip()
        if not account_id:
            raise InputError(f"{path}:{line}: empty account_id.")
        if account_id in labels:
            raise InputError(
                f"{path}:{line}: duplicate account_id {account_id!r}. Two rows "
                f"for one account means one of them is wrong and we cannot "
                f"tell which; de-duplicate the export.")
        token = row[label_col].strip().lower()
        if not token:
            raise InputError(
                f"{path}:{line}: empty label for account {account_id!r}. An "
                f"unlabelled account is not an honest account -- drop the row "
                f"if the outcome is unknown, rather than defaulting it to 0.")
        if token in _TRUE_TOKENS:
            labels[account_id] = True
        elif token in _FALSE_TOKENS:
            labels[account_id] = False
        else:
            raise InputError(
                f"{path}:{line}: label {row[label_col]!r} for account "
                f"{account_id!r} is not recognised. Use 1/0, true/false, or "
                f"yes/no. Guessing at an unknown token would fabricate ground "
                f"truth, so it is refused.")
    if not labels:
        raise InputError(f"{path}: header row only, no labelled accounts.")
    return labels


def load_scores_with_optional_labels(path: str) -> tuple[dict[str, float],
                                                         dict[str, bool] | None]:
    """Read the score file, and its ``label`` column when it has one.

    A single combined CSV is what most people actually export -- one query,
    one file -- so it is supported directly. The scores go through
    :func:`baseline.load_scores_csv` verbatim rather than being re-parsed
    here: that loader is where the decimal-comma and non-finite guards live,
    and a second implementation of the same parse would drift away from them.
    The label column is then read on a separate pass over the same file.

    Returns ``(scores, labels_or_None)``.
    """
    try:
        scores = load_scores_csv(path)
    except FileNotFoundError:
        raise InputError(f"{path}: no such file. Pass the path to your score "
                         f"CSV with --scores.") from None
    except OSError as exc:
        raise InputError(f"{path}: cannot be read ({exc.strerror}).") from None
    except ValueError as exc:
        # load_scores_csv already names file, line and fix; it is written for
        # this reader. Re-raise as InputError so main() formats it like every
        # other input problem instead of letting it surface as a traceback.
        raise InputError(str(exc)) from None

    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [row for row in csv.reader(fh)
                if row and any(cell.strip() for cell in row)]
    header = [cell.strip().lower() for cell in rows[0]]
    if "account_id" not in header or "label" not in header:
        return scores, None

    id_col, label_col = header.index("account_id"), header.index("label")
    labels: dict[str, bool] = {}
    for offset, row in enumerate(rows[1:]):
        line = 2 + offset
        account_id = row[id_col].strip()
        token = row[label_col].strip().lower()
        if not token:
            raise InputError(
                f"{path}:{line}: empty label for account {account_id!r}. The "
                f"file has a 'label' column, so every row must carry an "
                f"outcome; drop rows whose outcome is unknown rather than "
                f"leaving them blank, which would count them as honest.")
        if token in _TRUE_TOKENS:
            labels[account_id] = True
        elif token in _FALSE_TOKENS:
            labels[account_id] = False
        else:
            raise InputError(
                f"{path}:{line}: label {row[label_col]!r} for account "
                f"{account_id!r} is not recognised. Use 1/0, true/false, or "
                f"yes/no.")
    return scores, labels


def check_scores_usable(scores: dict[str, float], path: str,
                        role: str = "score") -> None:
    """Refuse a score column that cannot rank anything.

    The failure this exists for: with every score equal, every threshold is
    that same value, ``score > threshold`` is False for every account, and
    ``evaluate`` returns R0 = R1 = U = 0.0 without raising. Zero is a number
    a reader accepts. It looks like "the detector added nothing" and it is
    actually "the column is constant" -- a collapsed model, a failed join, a
    default value written for every row. Checked before any metric is
    computed, because the point is that no metric is printable from this
    input.
    """
    if not scores:
        raise InputError(f"{path}: no accounts loaded.")
    distinct = len(set(scores.values()))
    if distinct == 1:
        value = next(iter(scores.values()))
        raise InputError(
            f"{path}: every one of the {len(scores)} {role}s is the same value "
            f"({value!r}). A constant column cannot rank accounts, so every "
            f"threshold flags nobody and every metric would come out as a "
            f"plausible-looking zero rather than an error. Check the export "
            f"query -- a collapsed model, a failed join, or a default value "
            f"written to every row all look like this.")


def warn_on_coarse_scores(scores: dict[str, float], path: str) -> None:
    """Qualify, but do not refuse, a score column with heavy ties.

    A rule-count scorer legitimately takes a handful of values, so this is
    not an error. It does change how the budget reads: thresholds land on a
    large block of equal scores, and since flagging is strictly
    greater-than, no strict threshold can flag part of a tie block. The
    evaluation handles this by measuring BOTH scorers at the largest FP spend
    a strict threshold reaches on both -- the ``budget_utilization`` in the
    report shows how much of the nominal budget that realized spend covers, so
    a coarse column that cannot spend the budget is visible rather than
    silently compared at an unequal spend. Printed above the numbers it
    qualifies, not below them.
    """
    distinct = len(set(scores.values()))
    if distinct < 10 and len(scores) > 100:
        print(f"note: {path} has only {distinct} distinct score values across "
              f"{len(scores)} accounts. Thresholds land on large ties. Because "
              f"an account is flagged only when its score is strictly above "
              f"the threshold, the budget may not be fully spendable on this "
              f"column; the report measures both scorers at their common "
              f"realized spend and prints the budget utilization, so any "
              f"shortfall is shown rather than hidden.")
        print()


def check_labels_usable_hard(labels: dict[str, bool], path: str) -> None:
    """Refuse a label column with nothing on one side of the boundary.

    All-honest is the silent one: ``evaluate`` computes a threshold, finds no
    shared accounts to recall, and returns R0 = R1 = U = 0.0. Nothing was
    measured; the zero is the absence of positives. All-positive raises
    inside ``threshold_at_fp_budget`` instead, but its message is written for
    a caller, not an operator, so it is pre-empted here with the same shape
    of explanation.
    """
    n_shared = sum(1 for is_shared in labels.values() if is_shared)
    n_honest = len(labels) - n_shared
    if n_shared == 0:
        raise InputError(
            f"{path}: all {len(labels)} labels are honest (0/false); there are "
            f"no confirmed-shared accounts. Recall is a fraction of the shared "
            f"population, so with none of them every metric would return zero "
            f"without failing -- a zero that reads as 'the scorer caught "
            f"nothing' when it means 'nothing was measured'. Export the "
            f"confirmed cases for this window, or widen the window.")
    if n_honest == 0:
        raise InputError(
            f"{path}: all {len(labels)} labels are positive (1/true); there are "
            f"no honest accounts. The false-positive budget is defined per "
            f"10,000 honest accounts, so with none of them no threshold can be "
            f"set at all. Export the full scored population, not only the "
            f"confirmed cases.")


def warn_on_thin_labels(labels: dict[str, bool], path: str) -> None:
    """Say when the positive count is too small for the numbers to be quotable.

    The minimum fixed in advance for a quotable run is 150 confirmed
    positives (docs/method.md). Below it
    the output is still correct arithmetic, and someone testing the plumbing
    on one week of data should still get it -- but recall moves in visible
    steps of 1/n_shared, so the headline is a plumbing check rather than a
    result. Printed before the numbers, since a caveat underneath a table is
    a caveat nobody reads.
    """
    n_shared = sum(1 for is_shared in labels.values() if is_shared)
    if n_shared < 150:
        print(f"note: only {n_shared} confirmed-shared account(s) in {path}. "
              f"Recall moves in steps of 1/{n_shared} "
              f"({100.0 / n_shared:.2f}pp), and this repo "
              f"pre-registers a minimum of 150 positives before a number is "
              f"quotable. Below that, treat what follows as a plumbing check, "
              f"not a measurement.")
        print()


def warn_on_dropped_accounts(n_scored: int, labels: dict[str, bool],
                             path: str) -> None:
    """Report scored accounts excluded for want of a label."""
    dropped = n_scored - len(labels)
    if dropped > 0:
        print(f"note: {dropped} of {n_scored} scored accounts have no label in "
              f"{path} and are excluded. An unconfirmed account is not a "
              f"confirmed-honest one; the false-positive budget below is spent "
              f"over the {len(labels)} labelled accounts only.")
        print()


def require_matching_populations(scores: dict[str, float],
                                 labels: dict[str, bool],
                                 scores_path: str, labels_path: str) -> None:
    """Refuse a label set that reaches accounts the score file does not.

    ``evaluate`` requires identical key sets and raises a developer-facing
    message when they differ. Two files exported from two systems almost
    never match exactly, and the interesting case is not "they differ" but
    "which side is short": labels covering a fraction of the scored
    population is normal -- you only confirmed some cases, and the rest are
    dropped with a count. Labels reaching accounts that have no score is not
    normal: the score export lost rows, and every recall number would then
    be computed over a smaller population than the one that was labelled.
    """
    missing = sorted(labels.keys() - scores.keys())
    if missing:
        raise InputError(
            f"{len(missing)} of {len(labels)} accounts in {labels_path} have no "
            f"row in {scores_path} (first few: {missing[:5]}). Every labelled "
            f"account needs a score, otherwise recall is computed against a "
            f"population smaller than the one you labelled and comes out "
            f"biased. Export both files from the same account list and the "
            f"same window.")


# Below this fraction of the allowed FP budget actually spent, the headline U
# is qualified in place: a comparison at a spend far under the budget is a
# different measurement from the one the budget names. Same floor as the
# synthetic report path uses; duplicated here because the BYOS console is its
# own artifact and must carry its own qualifications.
UTIL_FLOOR = 0.50


def _max_realizable_spend(values: list[float], cap: int) -> int:
    """Largest honest-FP count a strict threshold realizes on ONE ladder <= cap.

    The single-scorer mirror of ``eval.common_spend``: the fixed point of
    "how many scores sit strictly above the score at position ``spend``".
    Used only to name which column is responsible when the joint realized
    spend falls short of the budget.
    """
    desc = sorted(values, reverse=True)
    n = len(desc)
    spend = min(int(cap), n)
    while spend > 0:
        if spend >= n:
            realized = sum(1 for v in desc if v > -math.inf)
        else:
            realized = sum(1 for v in desc if v > desc[spend])
        if realized == spend:
            return spend
        spend = realized
    return 0


def warn_on_unspent_budget(headline: dict, labels: dict[str, bool],
                           baseline: dict[str, float],
                           candidate: dict[str, float],
                           budget: float) -> None:
    """Say, next to the headline, when the budget was not actually spent.

    The headline U is measured at the realized spend, which is correct -- but
    correct arithmetic quoted against the wrong budget is how the number gets
    misread. Three cases, each printed as its own cause rather than left as a
    suspicious zero: the budget quantizes to nothing at this population size;
    a tie block at the top of a ladder is bigger than the whole budget, so no
    strict threshold can spend anything; or coarse scores spend some of the
    budget but less than half of it.
    """
    allowed = headline["allowed_fp"]
    realized = headline["realized_fp"]
    utilization = headline["budget_utilization"]
    n_honest = headline["n_accounts"] - headline["n_shared"]

    if allowed == 0:
        need = math.ceil(10000.0 / budget) if budget > 0 else 0
        print(f"    note: the {budget:g}/10k budget floors to 0 permitted "
              f"honest flags over {n_honest} honest accounts, so R0, R1 and U "
              f"above are measured at zero spend -- the threshold is the "
              f"highest honest score, not an operating point. This budget "
              f"needs at least {need} honest accounts to buy one alert.")
        print()
        return

    honest_ids = [a for a, is_shared in labels.items() if not is_shared]
    if realized == 0:
        # Which ladder is to blame: a top tie block larger than the budget
        # means the smallest nonzero spend that column can realize already
        # exceeds it. The distinct-value heuristic in warn_on_coarse_scores
        # does not catch this (one mega-tie among thousands of distinct
        # values), so it is diagnosed from the ladders directly.
        b_ranks = rank_normalize(baseline)
        d_ranks = rank_normalize(candidate)
        ladders = (
            ("baseline", [baseline[a] for a in honest_ids]),
            ("stacked", [max(b_ranks[a], d_ranks[a]) for a in honest_ids]),
        )
        culprits = [name for name, ladder in ladders
                    if _max_realizable_spend(ladder, allowed) == 0]
        which = " and ".join(culprits) if culprits else "one of the score columns"
        print(f"    note: the budget allows {allowed} honest flag(s) but the "
              f"realized spend is 0. The top of the {which} honest-score "
              f"ladder is a block of tied scores larger than the budget, and "
              f"an account is flagged only when strictly above the threshold, "
              f"so a tie block can never be partially flagged. R0/R1 above "
              f"are measured at zero spend and say nothing about recall at "
              f"the {budget:g}/10k operating point; a finer-grained score "
              f"column is the fix.")
        print()
    elif utilization < UTIL_FLOOR:
        b_spend = _max_realizable_spend([baseline[a] for a in honest_ids],
                                        allowed)
        which = ("baseline" if b_spend < allowed else "candidate")
        print(f"    note: score granularity cannot spend the budget: a strict "
              f"threshold reaches only {realized} of the {allowed} honest "
              f"flags the budget allows (utilization "
              f"{utilization * 100:.0f}%). U above is measured at that "
              f"realized spend, not at the nominal budget -- comparing at a "
              f"spend the coarser column never reaches would overstate the "
              f"finer one. A finer-grained {which} export (calibrated "
              f"probabilities rather than rounded buckets) would spend more "
              f"of it.")
        print()


def _fmt_threshold(value: float) -> str:
    """Render a threshold, spelling out the flag-everything sentinel."""
    if value == -math.inf:
        return "-inf (budget allows flagging every honest account)"
    return f"{value:.6g}"


def describe_without_labels(scores: dict[str, float], scores_path: str,
                            budgets: tuple[float, ...]) -> dict:
    """Everything computable from a score column alone -- and what is not.

    Deliberately short. Without labels there is no recall, no PR-AUC and no
    U, and the temptation is to fill the gap with distribution statistics
    dressed up as an evaluation. What follows is the honest list: where the
    threshold falls for each budget, how many accounts that flags, and the
    shape of the tail. None of it says whether the flagged accounts are
    fraudsters. That question needs outcomes.
    """
    values = sorted(scores.values())
    n = len(values)

    print("Labels were not supplied, so this run measures the score column "
          "only.")
    print()
    print("  NOT computable without labels:")
    print("    recall at budget (R0/R1), PR-AUC, precision@k, and U.")
    print("    Each of those is a statement about whether the accounts you "
          "flag")
    print("    are actually shared, which the scores alone cannot answer.")
    print("    Add a label column (account_id,label with 1/0) and rerun.")
    print()
    print(f"  Accounts scored: {n}")
    print(f"  Distinct values: {len(set(values))} "
          f"(ties block a threshold from separating accounts)")
    print(f"  Range:           {values[0]:.6g} … {values[-1]:.6g}")
    print(f"  Median:          {values[n // 2]:.6g}")
    print()
    print("  Treating the whole population as honest, the alert budget buys:")
    print()
    print("    budget/10k   threshold        accounts flagged")

    rows = []
    for budget in budgets:
        threshold = threshold_at_fp_budget(values, budget)
        flagged = sum(1 for v in values if v > threshold)
        allowed = math.floor(n * budget / 10000.0)
        note = ""
        if allowed == 0:
            # A real case at pilot scale: 15/10k of 3,000 accounts is 0.045
            # accounts, floored to zero. The threshold is then the maximum
            # score, nothing is flagged, and quoting it as an operating point
            # would be quoting an artifact of the population size.
            note = "  <- budget rounds to 0 alerts at this population size"
        print(f"    {budget:>7.0f}    {_fmt_threshold(threshold):<16} "
              f"{flagged:>6}{note}")
        rows.append({"budget_fp_per_10k": float(budget),
                     "threshold": float(threshold),
                     "n_flagged": int(flagged),
                     "budget_quantizes_to_zero": allowed == 0})
    print()
    if all(r["budget_quantizes_to_zero"] for r in rows):
        print(f"  Every budget rounds to zero alerts over {n} accounts: the "
              f"budget is")
        print(f"  expressed per 10,000 honest accounts, and this population is "
              f"too")
        print(f"  small for any of them to buy a single alert. Score more "
              f"accounts.")
        print()

    return {
        "mode": "no_labels",
        "scores_file": scores_path,
        "n_accounts": n,
        "n_distinct_scores": len(set(values)),
        "score_min": float(values[0]),
        "score_max": float(values[-1]),
        "score_median": float(values[n // 2]),
        "budget_curve": rows,
        "not_computable": ["recall_at_budget", "pr_auc", "precision_at_k", "U"],
        "not_computable_reason": (
            "no labels supplied; every one of these metrics is defined against "
            "confirmed outcomes"),
    }


def evaluate_single_scorer(scores: dict[str, float], labels: dict[str, bool],
                           scores_path: str,
                           budgets: tuple[float, ...],
                           budget: float = DEFAULT_BUDGET_FP_PER_10K,
                           n_resamples: int = 1000,
                           seed: int = 0) -> dict:
    """Metrics for one scorer against labels -- and no U, stated as such.

    This is the honest half of the module. Everything here is well-defined
    for a single scorer: PR-AUC ranks the column against outcomes,
    precision@k answers "if analysts work the top k, how many are real", and
    the budget curve says where the threshold falls and what recall it buys.

    U does not belong in that list. U = (R1 - R0) / R0 compares two scorers;
    with one score column there is no R0 to divide by. Reporting U = 0 here,
    or quietly comparing the column against itself, would be reporting a
    number that means nothing -- and it would be *the* number the reader
    remembers. So it is named as missing, with the exact file that would
    supply it.
    """
    n_shared = sum(1 for is_shared in labels.values() if is_shared)
    honest_ids = [a for a, is_shared in labels.items() if not is_shared]
    n_honest = len(honest_ids)

    print(f"Scored population: {len(labels)} accounts "
          f"({n_shared} confirmed shared, {n_honest} honest)")
    print(f"Source: {scores_path}")
    print()
    # Label precision@k with the k actually used. precision_at_k truncates to
    # the population size, so on a 200-account export "precision@500" would be
    # precision@200 under a name that overstates the depth of the review queue
    # it describes.
    k100, k500 = min(100, len(labels)), min(500, len(labels))
    print(f"  PR-AUC:          {pr_auc(labels, scores):.4f}")
    print(f"  precision@{k100}:   {precision_at_k(labels, scores, 100):.4f}"
          + ("" if k100 == 100 else
             f"   (asked for 100; only {len(labels)} accounts scored)"))
    print(f"  precision@{k500}:   {precision_at_k(labels, scores, 500):.4f}"
          + ("" if k500 == 500 else
             f"   (asked for 500; only {len(labels)} accounts scored)"))
    print()
    print("  budget/10k   threshold        flagged   TP    recall")

    honest_scores = [scores[a] for a in honest_ids]
    curve = []
    for b in budgets:  # NOT `budget`: that name is the headline budget below
        threshold = threshold_at_fp_budget(honest_scores, b)
        flagged = sum(1 for a in labels if scores[a] > threshold)
        caught = sum(1 for a, is_shared in labels.items()
                     if is_shared and scores[a] > threshold)
        recall = caught / n_shared
        allowed = math.floor(n_honest * b / 10000.0)
        # When the allowance floors to zero the budget has bought nothing:
        # the threshold is simply the highest honest score, so the row
        # measures "shared accounts scoring above every honest account"
        # rather than an operating point at this budget. Distinct budgets
        # then collapse onto identical rows, which is the visible symptom.
        note = "  <- budget buys 0 honest alerts here" if allowed == 0 else ""
        print(f"    {b:>7.0f}    {_fmt_threshold(threshold):<16} "
              f"{flagged:>6}  {caught:>4}   {recall:>6.3f}{note}")
        curve.append({"budget_fp_per_10k": float(b),
                      "threshold": float(threshold),
                      "n_flagged": int(flagged),
                      "n_true_positive": int(caught),
                      "recall": float(recall),
                      "budget_quantizes_to_zero": allowed == 0})
    print()
    if all(row["budget_quantizes_to_zero"] for row in curve):
        print(f"  Every budget above rounds to zero permitted honest alerts "
              f"over {n_honest}")
        print(f"  honest accounts: the budget is expressed per 10,000 of them, "
              f"and the")
        print(f"  smallest one here ({min(budgets):.0f}/10k) would need "
              f"{math.ceil(10000.0 / min(budgets))} honest accounts to buy a")
        print(f"  single alert. Each row is therefore the same threshold — the "
              f"highest")
        print(f"  honest score — and none of them is an operating point you "
              f"can run at.")
        print()

    holdout = holdout_fp_budget_ci(labels, scores, budget,
                                   n_resamples=n_resamples, seed=seed)
    if holdout["computable"]:
        print(f"  Held-out FP spend against the promised budget (threshold "
              f"fitted on")
        print(f"  {holdout['n_calibration']} honest accounts, spend measured "
              f"on the {holdout['n_measurement']} it never saw):")
        print()
        print(f"    promised {budget:g}/10k -> realized "
              f"{holdout['realized_fp_per_10k']:.1f}/10k  "
              f"[{holdout['confidence'] * 100:.0f}% CI "
              f"{holdout['ci_low']:.1f}, {holdout['ci_high']:.1f}]")
        print()
        print(f"    The gap between the two numbers is the optimism of "
              f"fitting and")
        print(f"    measuring a threshold on the same accounts: next week's "
              f"honest")
        print(f"    accounts are the measurement half, not the calibration "
              f"half. The")
        print(f"    interval is a percentile bootstrap over {n_resamples} "
              f"resamples, each half")
        print(f"    resampled by account with the threshold re-fitted every "
              f"time.")
    else:
        print(f"  Held-out FP spend against the promised budget: not "
              f"computable --")
        print(f"  {holdout['reason']}.")
    print()

    print("  U is NOT reported: it is an uplift of one scorer over another,")
    print("  U = (R1 - R0) / R0, and this run has exactly one score column.")
    print("  There is no R0 to divide by, so U is undefined here -- not zero.")
    print("  To get it, export your CURRENT production scores for the same")
    print("  accounts as a second CSV and add:")
    print("      --baseline-scores production_scores.csv")
    print("  The column you passed to --scores is then measured as the")
    print("  candidate, stacked on that baseline, at the same budget.")
    print()

    return {
        "mode": "labels_single_scorer",
        "scores_file": scores_path,
        "n_accounts": len(labels),
        "n_shared": int(n_shared),
        "n_honest": int(n_honest),
        "pr_auc": float(pr_auc(labels, scores)),
        "precision_at_100": float(precision_at_k(labels, scores, 100)),
        "precision_at_500": float(precision_at_k(labels, scores, 500)),
        "budget_curve": curve,
        "headline_budget_fp_per_10k": float(budget),
        "holdout_fp_budget": holdout,
        "U": None,
        "U_reason": (
            "U is an uplift of one scorer over another and requires two score "
            "columns; this run supplied one. Pass --baseline-scores to compute "
            "it."),
    }


def evaluate_two_scorers(candidate: dict[str, float],
                         baseline: dict[str, float],
                         labels: dict[str, bool],
                         candidate_path: str, baseline_path: str,
                         budgets: tuple[float, ...],
                         budget: float,
                         n_resamples: int, seed: int) -> dict:
    """Full uplift comparison: the same call the synthetic demo makes.

    ``evaluate`` stacks the candidate onto the baseline in rank space and
    measures both at one budget; ``bootstrap_uplift_ci`` puts an interval
    around U; ``holdout_fp_check`` says how much of the budget the stack
    actually spends on honest accounts it was not calibrated on. Nothing is
    reimplemented here -- the arithmetic that produced the numbers in
    ``out/report.json`` is the arithmetic that runs on your data.
    """
    print(f"Baseline (current stack): {baseline_path}")
    print(f"Candidate (your new scores): {candidate_path}")
    n_shared = sum(1 for is_shared in labels.values() if is_shared)
    print(f"Population: {len(labels)} accounts "
          f"({n_shared} confirmed shared, {len(labels) - n_shared} honest)")
    print()

    try:
        headline = evaluate(labels, baseline, candidate, budget)
    except ValueError as exc:
        # evaluate() validates key-set equality itself; align_populations
        # should have caught any mismatch already, so reaching here means an
        # assumption above is wrong. Re-explain rather than showing the
        # developer-facing text raw.
        raise InputError(
            f"the evaluation refused this population: {exc}. Both score files "
            f"and the label file must cover exactly the same account ids.") from None

    print(f"  At budget {budget:.0f} false positives per 10,000 honest "
          f"accounts:")
    print(f"    R0 (baseline recall):   {headline['R0']:.4f}")
    print(f"    R1 (stacked recall):    {headline['R1']:.4f}")
    if math.isinf(headline["U"]):
        print("    U:                      undefined (baseline recall is 0 at "
              "this budget, so the ratio has no denominator)")
    else:
        print(f"    U (relative uplift):    {headline['U'] * 100:+.1f}%")
    print(f"    thresholds:             baseline "
          f"{_fmt_threshold(headline['threshold_baseline'])}, stacked "
          f"{_fmt_threshold(headline['threshold_pilot'])}")
    print(f"    realized FP spend:      {headline['realized_fp']} of the "
          f"{headline['allowed_fp']} honest flags the budget allows "
          f"({headline['realized_fp_per_10k']:.1f}/10k, utilization "
          f"{headline['budget_utilization'] * 100:.0f}%)")
    print()
    warn_on_unspent_budget(headline, labels, baseline, candidate, budget)
    print(f"    PR-AUC:        {headline['pr_auc_baseline']:.4f} -> "
          f"{headline['pr_auc_pilot']:.4f}")
    print(f"    precision@500: {headline['precision_at_500_baseline']:.4f} -> "
          f"{headline['precision_at_500_pilot']:.4f}")
    print()

    ci = bootstrap_uplift_ci(labels, baseline, candidate, budget,
                             n_resamples=n_resamples, seed=seed)
    if math.isnan(ci["ci_low"]):
        print(f"    bootstrap CI: not computable -- baseline recall was zero in "
              f"all {ci['n_degenerate']} of {n_resamples} resamples, so U had "
              f"no denominator in any of them.")
    else:
        print(f"    U 95% CI: [{ci['ci_low'] * 100:+.1f}%, "
              f"{ci['ci_high'] * 100:+.1f}%] over {n_resamples} resamples"
              + (f", {ci['n_degenerate']} excluded as degenerate"
                 if ci["n_degenerate"] else ""))
    print()

    holdout = None
    if len(labels) - n_shared >= 2:
        holdout = holdout_fp_check(labels, baseline, candidate, budget,
                                   seed=seed)
        print(f"    Held-out FP check (threshold fitted on half the honest "
              f"accounts,")
        print(f"    measured on the other half): budget {budget:.0f}/10k -> "
              f"baseline {holdout['realized_fp_per_10k_baseline']:.0f}/10k, "
              f"stacked {holdout['realized_fp_per_10k_pilot']:.0f}/10k.")
        print(f"    The gap is how much of the uplift is bought with budget the")
        print(f"    in-sample threshold does not admit to spending.")
        print()

    print("  U across budgets (the operating point is not chosen after the "
          "fact):")
    print()
    print("    budget/10k        R0        R1         U")
    curve = []
    for b in budgets:
        row = evaluate(labels, baseline, candidate, b)
        u_text = ("undefined" if math.isinf(row["U"])
                  else f"{row['U'] * 100:+.1f}%")
        print(f"    {b:>7.0f}     {row['R0']:>7.4f}   {row['R1']:>7.4f}   "
              f"{u_text:>9}")
        curve.append({"budget_fp_per_10k": float(b), "R0": float(row["R0"]),
                      "R1": float(row["R1"]), "U": float(row["U"]),
                      "realized_fp": int(row["realized_fp"])})
    print()
    if len(curve) > 1 and len({(r["R0"], r["R1"], r["U"], r["realized_fp"])
                               for r in curve}) == 1:
        spend = curve[0]["realized_fp"]
        print(f"    note: every row of this sweep is identical. All "
              f"{len(curve)} budgets realize the")
        print(f"    same honest-FP spend ({spend} flag(s)), so they share one "
              f"threshold. The")
        print(f"    sweep is not broken -- score granularity or population "
              f"size makes these")
        print(f"    budgets indistinguishable here; the notes above the "
              f"headline say which.")
        print()

    return {
        "mode": "labels_two_scorers",
        "candidate_scores_file": candidate_path,
        "baseline_scores_file": baseline_path,
        "headline_budget_fp_per_10k": float(budget),
        "headline": headline,
        "bootstrap_ci": ci,
        "holdout_fp_check": holdout,
        "budget_curve": curve,
    }


def _json_safe(value):
    """Replace non-finite floats with None so the JSON is strictly parseable.

    U is +inf whenever baseline recall is zero at the budget (eval.py), and it
    flows straight into the two-scorer payload. ``json.dump`` would happily
    write the literal ``Infinity``, which is not JSON: jq, spreadsheets and
    every strict parser reject the file. report.py already guards its own
    output this way; the BYOS report -- the one artifact meant to leave the
    operator's machine -- must not be the exception.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class _Tee:
    """Write-through stream: everything printed also lands in a buffer.

    The console output IS the report -- every qualification is printed above
    the number it applies to. Persisting it verbatim is how those
    qualifications reach the file the numbers reach, instead of living only
    in a scrollback buffer that closes with the terminal.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, text: str) -> None:
        for stream in self._streams:
            stream.write(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def check_out_writable(out_prefix: str) -> None:
    """Refuse an unwritable --out BEFORE any metric is computed or printed.

    Discovering the broken path after the numbers means the operator has seen
    a full report and then an error, and cannot tell whether the run counted.
    The probe appends nothing and removes the file again if it created it, so
    an existing report is never truncated by a run that later fails.
    """
    directory = os.path.dirname(out_prefix)
    path = f"{out_prefix}.json"
    try:
        if directory:
            os.makedirs(directory, exist_ok=True)
        existed = os.path.exists(path)
        with open(path, "a", encoding="utf-8"):
            pass
        if not existed:
            os.remove(path)
    except OSError as exc:
        raise InputError(
            f"{path}: cannot be written ({exc.strerror or exc}). --out must "
            f"point somewhere writable; nothing has been measured yet, so fix "
            f"the path and rerun.") from None


def render_markdown(payload: dict, transcript: str) -> str:
    """The human-readable report: the run's own console output, kept verbatim.

    Deliberately a transcript rather than a re-rendering. Every caveat this
    module prints -- thin labels, coarse scores, dropped accounts, unspent
    budget, the held-out budget block -- is already placed above the number
    it qualifies, in reading order. A second renderer would be a second list
    of caveats to keep in sync, and the one that drifts is always the one in
    the file that gets forwarded.
    """
    lines = [
        "# Score measurement — run report",
        "",
        f"Mode: `{payload.get('mode', 'unknown')}`.",
        "",
        "The transcript below is the run's console output, verbatim. Every",
        "qualification (thin labels, coarse scores, dropped accounts, unspent",
        "budget) appears above the numbers it applies to, in the order an",
        "operator saw them. The machine-readable version of the same run is",
        "the `.json` written alongside this file. Both are aggregate-only:",
        "no account id and no score appears in either.",
        "",
        "```text",
        transcript.rstrip("\n"),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_report(payload: dict, out_prefix: str,
                 transcript: str) -> tuple[str, str]:
    """Write the run as JSON plus a markdown transcript next to it.

    Aggregate only: counts, rates, thresholds. No account id and no score
    ever enters these files, so they are the artifacts that can safely leave
    the machine the data cannot. Non-finite values (a +inf U) are written as
    null and ``allow_nan=False`` makes any that slip through a loud error
    rather than a silently invalid file.
    """
    directory = os.path.dirname(out_prefix)
    json_path = f"{out_prefix}.json"
    md_path = f"{out_prefix}.md"
    try:
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(_json_safe(payload), fh, indent=2, sort_keys=True,
                      allow_nan=False)
            fh.write("\n")
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(render_markdown(payload, transcript))
    except OSError as exc:
        raise InputError(
            f"{out_prefix}: cannot write the report ({exc.strerror or exc}). "
            f"Point --out at a writable path and rerun.") from None
    return json_path, md_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tripphysics.byos",
        description="Measure your own risk scores with this harness's "
                    "budgeted-recall arithmetic. Reads CSV, writes JSON, "
                    "makes zero network calls.")
    parser.add_argument("--scores", required=True,
                        help="CSV with account_id,score (and optionally label)")
    parser.add_argument("--labels", default=None,
                        help="CSV with account_id,label (1/0). Omit if the "
                             "score file already carries a label column.")
    parser.add_argument("--baseline-scores", default=None,
                        help="CSV of your CURRENT production scores for the "
                             "same accounts. Required for U -- without it U "
                             "is undefined and is not reported.")
    parser.add_argument("--out", default="out/byos",
                        help="output prefix; writes <prefix>.json "
                             "(default: out/byos)")
    parser.add_argument("--budget", type=float,
                        default=DEFAULT_BUDGET_FP_PER_10K,
                        help=f"headline false-positive budget per 10,000 "
                             f"honest accounts (default: "
                             f"{DEFAULT_BUDGET_FP_PER_10K:.0f})")
    parser.add_argument("--resamples", type=int, default=1000,
                        help="bootstrap resamples for the U interval "
                             "(default: 1000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the bootstrap and the held-out "
                             "split, so a rerun reproduces exactly (default: 0)")
    return parser


def run(argv: list[str] | None = None) -> int:
    """Parse arguments, evaluate, print. Returns the process exit code.

    Split from :func:`main` so tests can call it without the process exiting
    underneath them.
    """
    args = build_parser().parse_args(argv)

    # `float("nan")` and `float("inf")` both parse happily from the command
    # line. A NaN budget compares False against every bound, so it slipped past
    # a bare `< 0` check and then produced a misdiagnosed error further down.
    if not math.isfinite(args.budget):
        raise InputError(f"--budget must be a finite number, got "
                         f"{args.budget!r}: it is a count of false positives "
                         f"allowed per 10,000 honest accounts.")
    if args.budget < 0:
        raise InputError("--budget must be >= 0: it is a count of false "
                         "positives allowed per 10,000 honest accounts.")
    if args.resamples < 1:
        raise InputError("--resamples must be >= 1.")
    # The output path is part of the input: an unwritable --out discovered
    # after the metrics have scrolled by leaves the operator with a report
    # they cannot tell the status of. Probed here, with every other refusal.
    check_out_writable(args.out)

    # -- Load and validate everything BEFORE a single line of report is
    # printed. An error followed by a report banner reads as "it failed and
    # then ran anyway", and on a terminal where stderr and stdout interleave
    # the operator cannot tell which of the two to believe. Nothing below
    # this block can reject an input.
    scores, inline_labels = load_scores_with_optional_labels(args.scores)
    check_scores_usable(scores, args.scores)

    labels = inline_labels
    labels_path = args.scores
    if args.labels:
        if inline_labels is not None:
            # Two sources of ground truth is a question we must not answer by
            # picking one: the operator believes both are in play, and the
            # wrong choice mislabels the evaluation without any visible sign.
            raise InputError(
                f"{args.scores} already has a 'label' column and --labels "
                f"{args.labels} was also given. Two sources of ground truth "
                f"would silently disagree; drop one.")
        labels = load_labels_csv(args.labels)
        labels_path = args.labels

    baseline = None
    n_scored = len(scores)  # before narrowing to the labelled subset below
    if labels is not None:
        check_labels_usable_hard(labels, labels_path)
        require_matching_populations(scores, labels, args.scores, labels_path)
        scores = {a: scores[a] for a in labels}
        # Re-check AFTER narrowing to the labelled subset. A file can vary
        # across all scored accounts and still be constant within the labelled
        # ones -- the constant-column refusal has to run on the population the
        # metrics are actually computed over, or a flat zero result slips
        # through looking like "the detector added nothing".
        check_scores_usable(scores, args.scores)

        if args.baseline_scores:
            baseline, baseline_labels = load_scores_with_optional_labels(
                args.baseline_scores)
            check_scores_usable(baseline, args.baseline_scores,
                                role="baseline score")
            # An inline label column in the baseline file used to be discarded
            # in silence, even when it contradicted the labels being used. Two
            # disagreeing sources of ground truth must not be resolved by
            # quietly picking one.
            if baseline_labels is not None:
                conflicts = sorted(
                    a for a, is_shared in baseline_labels.items()
                    if a in labels and labels[a] != is_shared)
                if conflicts:
                    raise InputError(
                        f"{args.baseline_scores} carries its own 'label' column "
                        f"and it disagrees with {labels_path} on "
                        f"{len(conflicts)} account(s) (first few: "
                        f"{conflicts[:5]}). Two sources of ground truth would "
                        f"silently mislabel the evaluation; drop the label "
                        f"column from the baseline file or reconcile the two.")
            missing = sorted(labels.keys() - baseline.keys())
            if missing:
                raise InputError(
                    f"{len(missing)} of {len(labels)} labelled accounts have no "
                    f"row in {args.baseline_scores} (first few: {missing[:5]}). "
                    f"U compares two scorers over one population, so both score "
                    f"files must cover every labelled account.")
            baseline = {a: baseline[a] for a in labels}
            check_scores_usable(baseline, args.baseline_scores,
                                role="baseline score")

    # -- Past this point the run produces numbers. Banner, then the
    # qualifications that shape how those numbers must be read, then them.
    # Everything printed is also captured verbatim into <prefix>.md, so the
    # caveats travel with the numbers instead of dying in the scrollback.
    transcript = io.StringIO()
    with contextlib.redirect_stdout(_Tee(sys.stdout, transcript)):
        print()
        print("=" * 72)
        print("  trip-physics — measuring your scores, on your machine")
        print("=" * 72)
        print()

        if labels is None:
            warn_on_coarse_scores(scores, args.scores)
            payload = describe_without_labels(scores, args.scores,
                                              BUDGET_SWEEP)
        else:
            warn_on_coarse_scores(scores, args.scores)
            if baseline is not None:
                # The baseline column deserves the same granularity note as
                # the candidate: a coarse *baseline* is the column that
                # usually cannot spend the budget.
                warn_on_coarse_scores(baseline, args.baseline_scores)
            warn_on_thin_labels(labels, labels_path)
            warn_on_dropped_accounts(n_scored, labels, labels_path)
            if baseline is not None:
                payload = evaluate_two_scorers(
                    scores, baseline, labels, args.scores,
                    args.baseline_scores, BUDGET_SWEEP, args.budget,
                    args.resamples, args.seed)
            else:
                payload = evaluate_single_scorer(
                    scores, labels, args.scores, BUDGET_SWEEP, args.budget,
                    args.resamples, args.seed)

    payload["labels_file"] = labels_path if labels is not None else None
    json_path, md_path = write_report(payload, args.out,
                                      transcript.getvalue())
    print(f"Written: {json_path} and {md_path} (aggregate metrics only — no "
          f"account id and no score is in either)")
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point: turn every input problem into one sentence and exit 2.

    A traceback tells the operator about our call graph. What they need is
    the file, the line, and the fix -- so InputError is formatted and
    everything else is allowed through, since an unexpected exception here is
    our bug and hiding it would waste their time reporting a symptom.
    """
    try:
        return run(argv)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
