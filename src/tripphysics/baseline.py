"""Baseline scorers: the "current stack" stand-in and the CSV adapter.

Two ways to provide a baseline for the uplift comparison in ``eval.py``:

* :class:`NaiveBaseline` -- a deliberately simple placeholder used by the
  synthetic demo so the harness runs end to end out of the box.
* :func:`load_scores_csv` -- the adapter you actually want in a shadow
  pilot: export your production risk scores per account and feed them in
  without touching any code in this package.
"""

import csv
import math
from collections import Counter
from typing import Protocol

# Accounts are plain JSON-style dicts; see schema/input_schema.md for the
# exact field list. No classes on purpose: the data model must survive a
# round trip through JSONL untouched.
Account = dict


class Baseline(Protocol):
    """Anything that maps an account to a risk score (higher = more suspect)."""

    def score(self, account: Account) -> float:
        ...


class NaiveBaseline:
    """Placeholder stand-in for the incumbent scoring stack.

    **REPLACE THIS.** This class exists only so the synthetic demo has a
    baseline to compare against. Any uplift measured against it says
    nothing about uplift over your real stack. For the shadow pilot,
    export your production scores per account and load them with
    :func:`load_scores_csv` -- the evaluation code takes a plain
    ``{account_id: score}`` dict and never needs to know where the numbers
    came from.

    What it does (roughly what a first-pass device-integrity rule does):

    * *mock-location rate*: fraction of GPS fixes the OS flagged as coming
      from a mock provider, saturating at ``MOCK_SATURATION``;
    * *anchor-cell diversity*: each trip is anchored to its most frequent
      serving cell; an account whose trips anchor to many distinct cells
      relative to its trip count looks like several devices/people. This is
      a crude proxy for the classic accounts-per-device signal, computable
      from trip fixes alone.
    """

    # Fraction of mock-flagged fixes at which the mock component maxes out.
    MOCK_SATURATION = 0.05
    # Component weights; sum to 1.0 so the score stays in [0, 1].
    W_MOCK = 0.7
    W_ANCHOR_DIVERSITY = 0.3

    def score(self, account: Account) -> float:
        trips = account.get("trips", [])
        all_fixes = [fix for trip in trips for fix in trip.get("fixes", [])]
        if not all_fixes:
            return 0.0

        # Component 1: mock-location rate, saturating early -- even a small
        # fraction of mock fixes is a strong device-integrity signal.
        mock_rate = sum(1 for f in all_fixes if f["is_mock"]) / len(all_fixes)
        mock_component = min(mock_rate / self.MOCK_SATURATION, 1.0)

        # Component 2: anchor-cell diversity. Modal cell per trip; ties are
        # broken toward the smallest cell_id so the score is deterministic.
        anchors = []
        for trip in trips:
            fixes = trip.get("fixes", [])
            if not fixes:
                continue
            counts = Counter(f["cell_id"] for f in fixes)
            modal_cell = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            anchors.append(modal_cell)
        if len(anchors) <= 1:
            diversity = 0.0
        else:
            # 0.0 when every trip anchors to one cell, 1.0 when every trip
            # anchors somewhere new.
            diversity = (len(set(anchors)) - 1) / (len(anchors) - 1)

        return self.W_MOCK * mock_component + self.W_ANCHOR_DIVERSITY * diversity


def load_scores_csv(path: str) -> dict[str, float]:
    """Load per-account baseline scores from a CSV file.

    This is the injection point for *your* production scores: the shadow
    pilot compares against whatever numbers you put here, with no code
    changes anywhere else.

    Accepted formats:

    * with header: columns ``account_id`` and ``score`` (case-insensitive,
      any column order, extra columns ignored);
    * without header: exactly two columns, ``account_id,score``.

    Fails loudly on malformed score values, on non-finite scores
    (NaN/inf -- see below), on duplicate account ids -- silent deduplication
    is how offline evaluations quietly go wrong -- and on rows whose field
    count disagrees with the header (the decimal-comma trap, below).

    The file is read as ``utf-8-sig`` so a UTF-8 BOM -- which every Excel
    "CSV UTF-8" export writes -- is stripped rather than glued onto the
    first header name (making ``account_id`` unrecognisable) or onto the
    first account id (silently renaming one account so it never joins to
    the trip data).

    Raises:
        ValueError: on an empty file, an unusable header, a row whose field
            count does not match the header, an empty or duplicate
            ``account_id``, a score that does not parse as a number, or a
            score that parses but is not finite. The message carries the
            file, the 1-based line and the account id.
    """
    scores: dict[str, float] = {}
    # utf-8-sig strips a leading BOM if present and behaves as plain utf-8
    # otherwise. newline="" is required by the csv module so that newlines
    # embedded in quoted fields survive.
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        rows = [row for row in reader if row and any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError(f"{path}: empty CSV")

    header = [cell.strip().lower() for cell in rows[0]]
    if "account_id" in header and "score" in header:
        id_col = header.index("account_id")
        score_col = header.index("score")
        data_rows = rows[1:]
        first_line = 2  # 1-based line number of the first data row
        n_expected = len(rows[0])
    else:
        if len(rows[0]) != 2:
            raise ValueError(
                f"{path}: no 'account_id'/'score' header and not a plain "
                f"two-column file (first row has {len(rows[0])} columns). "
                f"If this file is semicolon- or tab-separated, convert it to "
                f"commas first; this loader reads comma-separated files only.")
        id_col, score_col = 0, 1
        data_rows = rows
        first_line = 1
        n_expected = 2

    for offset, row in enumerate(data_rows):
        line = first_line + offset
        # Exact field count, not "at least enough columns". A row with MORE
        # fields than the header is the decimal-comma trap: a European export
        # writes `acc_1,0,73` for a score of 0.73, the csv module splits it
        # into three fields, and reading column 1 yields the string "0" --
        # which float() parses happily. Every score would collapse to its
        # integer part (on a 0-1 scale: to 0.0) with no error at all, and the
        # row count would still look right. That is the same class of silent
        # corruption the non-finite guard below exists to prevent, so it is
        # rejected here rather than trusted.
        if len(row) < n_expected:
            raise ValueError(
                f"{path}:{line}: expected at least {n_expected} columns, got "
                f"{len(row)}: {row!r}. A short row means a ragged export.")
        if len(row) > n_expected:
            raise ValueError(
                f"{path}:{line}: expected {n_expected} columns, got "
                f"{len(row)}: {row!r}. Extra fields usually mean decimal "
                f"commas (e.g. '0,73' for 0.73), which would silently "
                f"truncate every score to its integer part. Fix the "
                f"delimiter or quote the offending fields.")
        account_id = row[id_col].strip()
        if not account_id:
            raise ValueError(f"{path}:{line}: empty account_id")
        if account_id in scores:
            raise ValueError(f"{path}:{line}: duplicate account_id {account_id!r}")
        try:
            score = float(row[score_col])
        except ValueError:
            raise ValueError(
                f"{path}:{line}: score {row[score_col]!r} is not a number") from None
        # `float()` happily accepts "nan", "inf" and "-inf". Every one of
        # those silently destroys the evaluation downstream rather than
        # failing: NaN sorts to the front of a descending sort, so the
        # budgeted threshold becomes NaN and `score > threshold` is False
        # for every account, collapsing recall to zero; NaN also takes the
        # top rank in argsort, and tie-breaking with `(-score, id)` on NaN
        # yields an order that depends on dict insertion order, which
        # breaks determinism outright. A single NaN among 1000 honest
        # accounts moved PR-AUC from 0.779 to 0.310 with no warning.
        #
        # Since this loader is the documented integration point for a
        # production score export -- and NULL scores are entirely normal in
        # production -- the value must be rejected here, loudly, with the
        # offending account and line, exactly as duplicate ids are.
        if not math.isfinite(score):
            raise ValueError(
                f"{path}:{line}: score {row[score_col]!r} for account "
                f"{account_id!r} is not finite; NaN/inf silently corrupts "
                f"thresholding and ranking. Drop the account or substitute "
                f"an explicit numeric score before loading.")
        scores[account_id] = score
    return scores
