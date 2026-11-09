"""Shadow-evaluation math: threshold-at-budget, uplift, PR-AUC, precision@k.

This module is the part a fraud DS should read first, so everything is
implemented by hand (numpy for sorting only, no sklearn) and every step is
commented. The contract:

* Both scorers are evaluated at the **same false-positive budget** on the
  honest population -- recall numbers at mismatched budgets are how offline
  evaluations lie.
* The pilot scorer is the baseline **stacked** with the detector:
  ``pilot = max(rank_normalize(baseline), rank_normalize(detector))``.
  See :func:`rank_normalize` and :func:`evaluate` for why.
* The headline number is the uplift ``U = (R1 - R0) / R0`` where R0/R1 are
  baseline/pilot recall at the budgeted threshold.

Conventions used throughout: ``labels[account_id] is True`` means the
account is shared ("positive"); higher score = more suspicious; an account
is flagged when its score is **strictly greater** than the threshold.
"""

import math

import numpy as np


def rank_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Map scores to their normalized ranks in (0, 1]; ties get the average rank.

    Why ranks: the baseline and the detector produce scores on incomparable
    scales (a production stack might emit calibrated probabilities, logits,
    or rule counts). Rank normalization is invariant under any monotone
    rescaling of either scorer, so stacking with ``max`` afterwards compares
    "how deep into its own top tail does each scorer place this account"
    rather than raw numbers.

    Definition: sort ascending, assign 1-based ranks, average the ranks of
    tied values (so ties remain ties and the result is independent of dict
    ordering), divide by n. Highest score -> 1.0, lowest -> ~1/n.
    """
    if not scores:
        return {}
    ids = sorted(scores)  # fixed iteration order: determinism does not
    #                       depend on dict insertion order
    values = np.array([scores[i] for i in ids], dtype=float)
    n = len(ids)

    order = np.argsort(values, kind="stable")  # ascending
    sorted_values = values[order]

    # Average 1-based rank within each run of tied values.
    avg_ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        # positions i..j (0-based) hold ranks i+1..j+1; their average:
        avg_ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1

    ranks = np.empty(n, dtype=float)
    ranks[order] = avg_ranks
    return {account_id: float(r) / n for account_id, r in zip(ids, ranks)}


def threshold_at_fp_budget(honest_scores: list[float],
                           budget_fp_per_10k: float) -> float:
    """Score threshold that spends at most the FP budget on honest accounts.

    SINGLE-SCORER DIAGNOSTIC. This sets a budgeted threshold for ONE scorer
    and is used by the BYOS single-scorer view and the budget curves. It must
    NOT be used to compare two scorers: two scorers of different granularity
    thresholded here independently realize *different* false-positive counts
    at the same nominal budget (a coarse scorer's boundary ties go unflagged),
    which overstates uplift in favour of the finer-grained scorer. The
    two-scorer comparison in :func:`evaluate` meets both at :func:`common_spend`
    instead, so recall is compared at an equal *realized* spend.

    ``budget_fp_per_10k`` is the number of honest accounts we may falsely
    flag per 10,000 honest accounts (the weekly-active analog in a real
    deployment). Steps:

    1. ``allowed_fp = floor(n_honest * budget / 10000)`` -- the integer
       number of honest accounts the budget lets us flag.
    2. Sort honest scores descending.
    3. The threshold is the ``(allowed_fp + 1)``-th highest honest score.
       With the strict flagging rule ``score > threshold``, the honest
       accounts flagged are exactly those strictly above it: at most
       ``allowed_fp`` of them (exactly ``allowed_fp`` when scores are
       distinct; ties at the threshold value are *not* flagged, i.e. we
       err on the side of under-spending the budget).

    Returns ``-inf`` when the budget allows flagging every honest account.
    Monotone by construction: a larger budget never raises the threshold,
    so recall at budget is non-decreasing in the budget.
    """
    n_honest = len(honest_scores)
    if n_honest == 0:
        raise ValueError("no honest accounts: cannot set an FP-budget threshold")
    if budget_fp_per_10k < 0:
        raise ValueError("budget_fp_per_10k must be >= 0")

    allowed_fp = math.floor(n_honest * budget_fp_per_10k / 10000.0)
    if allowed_fp >= n_honest:
        return -math.inf

    descending = np.sort(np.array(honest_scores, dtype=float))[::-1]
    return float(descending[allowed_fp])


def _realized_spend_at(honest_desc: np.ndarray, cap: int) -> int:
    """Largest honest-FP count a strict ``score > t`` threshold can realize <= cap.

    ``honest_desc`` is the honest scores sorted DESCENDING. With the strict
    flagging rule, a threshold placed at ``honest_desc[cap]`` flags exactly
    the honest accounts scoring strictly above that value. When the score at
    position ``cap`` is tied with scores above it, that count is *less* than
    ``cap`` -- the ties at the boundary are not flagged. This returns that
    realized count, i.e. the true number of honest accounts spent at the
    tightest threshold that does not exceed ``cap``.

    ``cap >= n`` means the budget allows flagging every honest account: the
    threshold is -inf and all ``n`` are spent.
    """
    n = int(honest_desc.size)
    if cap < 0:
        return 0
    if cap >= n:
        # The budget allows flagging everyone, i.e. the threshold is -inf. That
        # is NOT the same as "all n are spent": strict `>` never flags a score
        # equal to the threshold, and a -inf score is not > -inf. Counting the
        # scores actually above -inf keeps the realized spend equal to what the
        # threshold really flags -- assuming n here made a -inf-scored honest
        # account count as spent on one scorer and not the other, which is the
        # exact unequal-spend bug this module exists to prevent.
        return int(np.count_nonzero(honest_desc > -math.inf))
    return int(np.count_nonzero(honest_desc > honest_desc[cap]))


def common_spend(desc_a: np.ndarray, desc_b: np.ndarray,
                 allowed_fp: int) -> int:
    """Largest honest-FP spend a strict threshold realizes on BOTH scorers <= budget.

    The tie-underspend problem: two scorers of different granularity, held to
    the same nominal FP budget, spend *different* realized numbers of false
    positives, because a coarse scorer's ties at the budget boundary are not
    flagged. Comparing recall at a mismatched realized spend is exactly how an
    offline evaluation lies -- and it lies in favour of whichever scorer
    happens to be finer-grained (here, the rank-space pilot).

    The fix is to meet at the largest spend both scorers can *actually*
    realize with a strict threshold, no greater than ``allowed_fp``::

        F* = max { s in [0, allowed_fp] :
                   realized_spend_a(s) == s  and  realized_spend_b(s) == s }

    Computed as a fixed point: start at ``allowed_fp``, take the smaller of
    the two scorers' realized spends at that cap, repeat. The sequence is
    non-increasing and bounded below by 0 (which every scorer can realize:
    threshold above the top score flags nobody), so it terminates, and the
    value it lands on is realizable by both by construction. F* is a symmetric
    function of the two anonymous score ladders -- swapping the arguments
    cannot change it -- which is why this cannot flatter one scorer over the
    other.
    """
    spend = int(allowed_fp)
    while True:
        s = min(_realized_spend_at(desc_a, spend),
                _realized_spend_at(desc_b, spend))
        if s == spend:
            return spend
        spend = s


def threshold_for_spend(honest_desc: np.ndarray, spend: int) -> float:
    """Strict-``>`` threshold that flags exactly ``spend`` honest accounts.

    ``honest_desc`` is the honest scores sorted descending. ``spend`` must be
    a realizable spend (a value :func:`common_spend` can return); the
    threshold is then the score at position ``spend``, so the ``spend``
    strictly-higher accounts are flagged and no tie at the boundary leaks in.
    ``spend >= n`` returns -inf (flag everyone).
    """
    n = int(honest_desc.size)
    if spend >= n:
        return -math.inf
    return float(honest_desc[spend])


def recall_at_threshold(scores: dict[str, float],
                        labels: dict[str, bool],
                        threshold: float) -> float:
    """Fraction of shared accounts with score strictly above the threshold."""
    shared = [a for a, is_shared in labels.items() if is_shared]
    if not shared:
        return 0.0
    caught = sum(1 for a in shared if scores[a] > threshold)
    return caught / len(shared)


def pr_auc(labels: dict[str, bool], scores: dict[str, float]) -> float:
    """Area under the precision-recall curve, trapezoidal rule, by hand.

    Construction:

    1. Sort accounts by score descending (ties broken by account_id only
       for a stable sort order; tied scores are then *grouped*, because a
       threshold can never separate equal scores -- ungrouped ties are a
       classic source of optimistic PR curves).
    2. After each group of tied scores, record one curve point::

           recall_i    = TP_i / n_positives
           precision_i = TP_i / (TP_i + FP_i)

    3. Anchor the curve at (recall=0, precision=1) -- the usual
       precision_recall_curve convention -- and integrate with the
       trapezoidal rule::

           AUC = sum_i (R_i - R_{i-1}) * (P_i + P_{i-1}) / 2

    Note: trapezoidal PR-AUC linearly interpolates precision between
    points, which is known to be slightly optimistic versus step-wise
    average precision. It is applied identically to both scorers here, so
    the *comparison* is fair; do not compare these values against AP
    numbers from other tooling.

    What the anchor does when the top group is mixed (DEFECT-1, open). The
    first trapezoid runs from the anchor (0, 1) to the first group's point.
    If the highest-scoring group of tied scores contains honest accounts,
    that segment credits precision the scorer never achieves at any
    threshold. In the extreme case -- every score tied -- the whole curve is
    the single segment from (0, 1) to (1, base rate), and the function
    returns ``(1 + base_rate) / 2`` (0.55 at a 0.10 base rate) where a
    non-informative scorer should get the base rate. The strict xfail in
    ``tests/test_metric_edge_cases.py`` pins the correct value, and a
    characterisation test pins today's. It is left open rather than patched
    for the constant case alone because the real fix (start the curve at the
    first group's own precision, or switch to average precision) changes
    every value this function returns. It does not touch the figures in
    the committed report: on that population the top group is 21 shared
    accounts for the baseline and 1 shared account for the stacked scorer,
    so the anchor and the first point both sit at precision 1. The
    bring-your-own-scores command refuses a constant score column before
    computing anything.

    Returns 0.0 when there are no positive labels.
    """
    n_pos = sum(1 for is_shared in labels.values() if is_shared)
    if n_pos == 0:
        return 0.0

    # Stable, deterministic order: score descending, then account_id.
    ordered = sorted(labels, key=lambda a: (-scores[a], a))

    recall_points = [0.0]
    precision_points = [1.0]  # anchor at (R=0, P=1)
    tp = 0
    fp = 0
    i = 0
    n = len(ordered)
    while i < n:
        # Consume one group of tied scores in full before emitting a point.
        j = i
        while j + 1 < n and scores[ordered[j + 1]] == scores[ordered[i]]:
            j += 1
        for k in range(i, j + 1):
            if labels[ordered[k]]:
                tp += 1
            else:
                fp += 1
        recall_points.append(tp / n_pos)
        precision_points.append(tp / (tp + fp))
        i = j + 1

    # Trapezoidal integration over recall.
    auc = 0.0
    for i in range(1, len(recall_points)):
        dr = recall_points[i] - recall_points[i - 1]
        auc += dr * (precision_points[i] + precision_points[i - 1]) / 2.0
    return auc


def precision_at_k(labels: dict[str, bool],
                   scores: dict[str, float],
                   k: int = 500) -> float:
    """Precision among the top-k accounts by score.

    Operational reading: "if analysts review the k highest-scored accounts,
    what fraction are actually shared?" Ties are broken by account_id so
    the metric is deterministic; when fewer than k accounts exist, all of
    them are taken and the divisor shrinks accordingly.
    """
    if k < 1:
        # k=0 would divide by zero; k<0 would silently take a suffix of the
        # ranking via Python slice semantics and return a plausible-looking
        # number for a nonsensical question. Both are caller bugs.
        raise ValueError(f"k must be >= 1, got {k}")
    if not labels:
        return 0.0
    ordered = sorted(labels, key=lambda a: (-scores[a], a))
    k_eff = min(k, len(ordered))
    hits = sum(1 for a in ordered[:k_eff] if labels[a])
    return hits / k_eff


# Default false-positive budget, in falsely-flagged honest accounts per 10k.
# 50/10k = 0.5% is a realistic driver-action review-queue tolerance for
# ride-hailing (strict enough that a false deactivation is costly, loose
# enough that a human review team can clear the queue). The pilot fixes THIS
# number against the operator's own stated tolerance before Day 1 — it is not a knob
# to be tuned for a flattering result. run_demo reports U across a range of
# budgets precisely so the operating point is never cherry-picked.
DEFAULT_BUDGET_FP_PER_10K = 50.0


def evaluate(labels: dict[str, bool],
             baseline_scores: dict[str, float],
             pilot_scores: dict[str, float],
             budget_fp_per_10k: float = DEFAULT_BUDGET_FP_PER_10K) -> dict:
    """Compare baseline vs. baseline-stacked-with-detector at one FP budget.

    Args:
        labels: ``{account_id: is_shared}`` ground truth.
        baseline_scores: the incumbent scorer (production export via
            ``baseline.load_scores_csv``, or ``NaiveBaseline`` in the demo).
        pilot_scores: the candidate detector's scores
            (``detector.score_account`` per account). Inside this function
            they are **stacked** with the baseline; R1 measures the stack,
            not the detector alone.
        budget_fp_per_10k: the MAXIMUM allowed false positives per 10k honest
            accounts. Both scorers are then measured at the same *realized*
            spend -- the largest FP count a strict threshold reaches on both
            (:func:`common_spend`) -- so different score granularities are
            compared fairly rather than at a mismatched realized spend.

    The stacking rule::

        pilot_stacked = max(rank_normalize(baseline), rank_normalize(detector))

    i.e. an account is high-risk if *either* scorer places it deep in its
    own top tail. Rank space makes the two tails comparable regardless of
    each scorer's raw scale (see :func:`rank_normalize`). One honest note:
    stacking does not *guarantee* R1 >= R0 -- the detector's false alarms
    spend part of the shared FP budget, and if it adds noise instead of
    signal, R1 can come out below R0. That is precisely what U measures,
    and why it can be negative.

    Steps (each delegated to a documented function above):

    1. Validate that all three dicts cover the same account ids.
    2. Build the stacked pilot scores.
    3. Find the common realized FP spend F* (:func:`common_spend`) and set
       BOTH thresholds to realize exactly F* honest flags. R0/R1 are recall
       on shared accounts at those thresholds, compared at equal spend.
    4. U = (R1 - R0) / R0. If R0 == 0 the ratio is undefined: U is
       reported as +inf when R1 > 0 (any catch is infinite relative
       improvement over zero) and 0.0 when both are zero.
    5. PR-AUC and precision@500 for both scorers, as budget-free views of
       the same comparison.

    Returns a dict with keys: "R0", "R1", "U", "budget_fp_per_10k",
    "allowed_fp" (integer FP the budget permits), "realized_fp" (the common
    spend F* both scorers are measured at), "realized_fp_per_10k",
    "budget_utilization" (realized_fp / allowed_fp; below 1.0 means a coarse
    scorer could not spend the full budget), "pr_auc_baseline",
    "pr_auc_pilot", "precision_at_500_baseline", "precision_at_500_pilot",
    "n_accounts", "n_shared", "threshold_baseline", "threshold_pilot". All
    values are plain Python floats/ints (JSON-serializable).
    """
    # -- Step 1: the three dicts must describe the same population. A partial
    # overlap silently evaluated would bias every number below.
    if not (labels.keys() == baseline_scores.keys() == pilot_scores.keys()):
        missing_b = sorted(labels.keys() - baseline_scores.keys())[:5]
        missing_p = sorted(labels.keys() - pilot_scores.keys())[:5]
        extra_b = sorted(baseline_scores.keys() - labels.keys())[:5]
        extra_p = sorted(pilot_scores.keys() - labels.keys())[:5]
        raise ValueError(
            "labels / baseline_scores / pilot_scores must cover identical "
            f"account ids; e.g. missing from baseline: {missing_b}, missing "
            f"from pilot: {missing_p}, unlabeled in baseline: {extra_b}, "
            f"unlabeled in pilot: {extra_p}")
    if not labels:
        raise ValueError("empty evaluation set")

    honest_ids = sorted(a for a, is_shared in labels.items() if not is_shared)
    n_accounts = len(labels)
    n_shared = n_accounts - len(honest_ids)

    # The FP budget is a promise about how many *honest* accounts get flagged,
    # so it is undefined without any. (The old code raised this from inside
    # threshold_at_fp_budget; the realized-spend path never calls it, so the
    # guard is stated here explicitly rather than relied on as a side effect.)
    if not honest_ids:
        raise ValueError("no honest accounts: cannot set an FP-budget threshold")
    # Same story for the budget itself: threshold_at_fp_budget rejected a
    # negative budget, and the realized-spend path never calls it. Without this
    # guard a negative budget yields allowed_fp < 0, a realized_fp that exceeds
    # it, and a utilization of 1.0 from the `allowed_fp > 0 else 1.0` branch --
    # a nonsense result reported as if it were fine.
    if budget_fp_per_10k < 0:
        raise ValueError("budget_fp_per_10k must be >= 0")

    # -- Step 2: build the stacked pilot first, because R0 and R1 are now
    # measured at a SHARED realized FP spend (see below), which needs both
    # honest-score ladders in hand before either threshold is set.
    baseline_ranks = rank_normalize(baseline_scores)
    detector_ranks = rank_normalize(pilot_scores)
    stacked = {a: max(baseline_ranks[a], detector_ranks[a]) for a in labels}

    # -- Step 3: the tie-aware symmetric spend. The FP budget names a MAXIMUM
    # count of honest accounts we may flag; but two scorers of different
    # granularity, thresholded independently at that budget, spend *different*
    # realized numbers of false positives, because a coarse scorer's ties at
    # the boundary go unflagged. Comparing R0 and R1 at mismatched realized
    # spend is precisely how an offline evaluation overstates uplift, and it
    # does so in favour of the finer-grained scorer -- here the rank-space
    # pilot. So both scorers are held to F*, the largest spend a strict
    # threshold realizes on BOTH ladders without exceeding the budget. Then
    # "at equal FP spend" is a printed integer identity, not a claim.
    n_honest = len(honest_ids)
    allowed_fp = math.floor(n_honest * budget_fp_per_10k / 10000.0)
    desc_baseline = np.sort(
        np.array([baseline_scores[a] for a in honest_ids], dtype=float))[::-1]
    desc_stacked = np.sort(
        np.array([stacked[a] for a in honest_ids], dtype=float))[::-1]
    realized_fp = common_spend(desc_baseline, desc_stacked,
                               min(allowed_fp, n_honest))

    threshold_baseline = threshold_for_spend(desc_baseline, realized_fp)
    threshold_pilot = threshold_for_spend(desc_stacked, realized_fp)
    r0 = recall_at_threshold(baseline_scores, labels, threshold_baseline)
    r1 = recall_at_threshold(stacked, labels, threshold_pilot)

    # -- Step 4: relative uplift.
    if r0 > 0.0:
        uplift = (r1 - r0) / r0
    elif r1 > 0.0:
        uplift = math.inf  # baseline caught nothing; any catch is infinite uplift
    else:
        uplift = 0.0

    # -- Step 5: budget-free views of the same comparison. budget_utilization
    # exposes when a coarse baseline could not spend the budget at all
    # (realized_fp << allowed_fp): report.py refuses a headline U below a floor
    # and prints the granularity warning instead of a number bought at an
    # unequal spend.
    return {
        "R0": float(r0),
        "R1": float(r1),
        "U": float(uplift),
        "budget_fp_per_10k": float(budget_fp_per_10k),
        "allowed_fp": int(allowed_fp),
        "realized_fp": int(realized_fp),
        "realized_fp_per_10k": (float(realized_fp) / n_honest * 10000.0
                                if n_honest else 0.0),
        "budget_utilization": (float(realized_fp) / allowed_fp
                               if allowed_fp > 0 else 1.0),
        "pr_auc_baseline": float(pr_auc(labels, baseline_scores)),
        "pr_auc_pilot": float(pr_auc(labels, stacked)),
        "precision_at_500_baseline": float(precision_at_k(labels, baseline_scores, 500)),
        "precision_at_500_pilot": float(precision_at_k(labels, stacked, 500)),
        "n_accounts": int(n_accounts),
        "n_shared": int(n_shared),
        "threshold_baseline": float(threshold_baseline),
        "threshold_pilot": float(threshold_pilot),
    }


def bootstrap_uplift_ci(labels: dict[str, bool],
                        baseline_scores: dict[str, float],
                        pilot_scores: dict[str, float],
                        budget_fp_per_10k: float = DEFAULT_BUDGET_FP_PER_10K,
                        n_resamples: int = 1000,
                        confidence: float = 0.95,
                        seed: int = 0) -> dict:
    """Percentile bootstrap confidence interval for U on a single population.

    Why this exists: a point estimate of U carries no information about how
    much of it is sampling noise. With 600 confirmed positives, R1 - R0 is a
    difference of two counts out of 600; the interval says how wide that is.
    The pilot pass bar (README) requires U "reported with a bootstrap 95% CI",
    so the code that produces it ships with the harness rather than being
    improvised against real data later.

    Method: resample accounts **with replacement, stratified** by label, so
    every resample keeps exactly the observed number of shared and honest
    accounts. Unstratified resampling would let the positive count wander
    (occasionally to zero at low base rates), which inflates the interval for
    a reason that has nothing to do with the detector. The threshold is
    re-derived on each resample's honest population -- recomputing the whole
    pipeline is the point, since threshold selection is itself a source of
    variance that a naive CI on fixed thresholds would hide.

    Resampling accounts is the right unit here because accounts are the
    independent sampling unit and the metric is defined per account.

    Args:
        labels / baseline_scores / pilot_scores / budget_fp_per_10k: as in
            :func:`evaluate`. Scores are computed once by the caller and only
            re-indexed here, so this is cheap relative to feature extraction.
        n_resamples: bootstrap replicates. 1000 is the usual floor for a
            stable 95% percentile interval.
        confidence: two-sided coverage, e.g. 0.95 -> 2.5th/97.5th percentiles.
        seed: RNG seed, so the interval is reproducible like everything else.

    Returns a dict with "u_point" (U on the observed sample), "ci_low",
    "ci_high", "n_resamples", "confidence", and "n_degenerate" (replicates
    where baseline recall was zero, making U undefined; these are excluded
    from the percentiles and reported so the exclusion is never silent).
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    shared_ids = sorted(a for a, is_shared in labels.items() if is_shared)
    honest_ids = sorted(a for a, is_shared in labels.items() if not is_shared)
    if not shared_ids:
        raise ValueError("no shared accounts: U is undefined")
    if not honest_ids:
        raise ValueError("no honest accounts: cannot set an FP-budget threshold")
    if budget_fp_per_10k < 0:
        raise ValueError("budget_fp_per_10k must be >= 0")

    point = evaluate(labels, baseline_scores, pilot_scores, budget_fp_per_10k)

    # Implementation note: this is the array-level equivalent of calling
    # evaluate() on each resample, not a different method. Rebuilding three
    # 20k-entry dicts per replicate (and re-sorting inside evaluate) costs
    # ~140ms x 1000 replicates; the vectorized form below is the same
    # arithmetic on numpy arrays. The one subtlety it preserves: the stacked
    # score depends on ranks over the WHOLE resampled population, so ranks are
    # recomputed per replicate rather than reused from the point estimate.
    rng = np.random.default_rng(seed)
    n_shared, n_honest = len(shared_ids), len(honest_ids)

    # Scores laid out once as [shared..., honest...]; resamples index into these.
    shared_base = np.array([baseline_scores[a] for a in shared_ids], dtype=float)
    honest_base = np.array([baseline_scores[a] for a in honest_ids], dtype=float)
    shared_det = np.array([pilot_scores[a] for a in shared_ids], dtype=float)
    honest_det = np.array([pilot_scores[a] for a in honest_ids], dtype=float)

    allowed_fp = math.floor(n_honest * budget_fp_per_10k / 10000.0)

    def _ranks(values: np.ndarray) -> np.ndarray:
        """Average-tied ranks / n — the array form of rank_normalize."""
        order = np.argsort(values, kind="stable")
        ordered = values[order]
        n = values.size
        out = np.empty(n, dtype=float)
        # Boundaries of runs of equal values.
        start = 0
        for end in range(1, n + 1):
            if end == n or ordered[end] != ordered[start]:
                out[start:end] = (start + end - 1) / 2.0 + 1.0
                start = end
        ranks = np.empty(n, dtype=float)
        ranks[order] = out
        return ranks / n

    def _joint_recalls(shared_base_v: np.ndarray, honest_base_v: np.ndarray,
                       shared_stk_v: np.ndarray,
                       honest_stk_v: np.ndarray) -> tuple[float, float]:
        """R0 and R1 at the SHARED realized spend F*, the array form of evaluate().

        This mirrors the point estimate exactly: both scorers meet at the
        largest FP spend a strict threshold realizes on both honest ladders,
        so each replicate's R0/R1 are compared at equal realized spend rather
        than each thresholded independently (the old, mismatched estimator
        that made the CI a different quantity than the point it wrapped).
        """
        desc_b = np.sort(honest_base_v)[::-1]
        desc_s = np.sort(honest_stk_v)[::-1]
        spend = common_spend(desc_b, desc_s, min(allowed_fp, n_honest))
        t_b = threshold_for_spend(desc_b, spend)
        t_s = threshold_for_spend(desc_s, spend)
        r0 = float(np.count_nonzero(shared_base_v > t_b)) / shared_base_v.size
        r1 = float(np.count_nonzero(shared_stk_v > t_s)) / shared_stk_v.size
        return r0, r1

    uplifts: list[float] = []
    n_degenerate = 0
    for _ in range(n_resamples):
        idx_shared = rng.integers(0, n_shared, n_shared)
        idx_honest = rng.integers(0, n_honest, n_honest)

        rs_shared_base = shared_base[idx_shared]
        rs_honest_base = honest_base[idx_honest]
        rs_shared_det = shared_det[idx_shared]
        rs_honest_det = honest_det[idx_honest]

        # Stack in rank space over the full resampled population, then measure
        # both scorers at the joint realized spend.
        base_all = np.concatenate((rs_shared_base, rs_honest_base))
        det_all = np.concatenate((rs_shared_det, rs_honest_det))
        stacked_all = np.maximum(_ranks(base_all), _ranks(det_all))
        r0, r1 = _joint_recalls(
            rs_shared_base, rs_honest_base,
            stacked_all[:n_shared], stacked_all[n_shared:])

        if r0 > 0.0:
            uplifts.append((r1 - r0) / r0)
        else:
            # Baseline recall zero at the joint spend: U undefined. Excluded
            # from the percentiles and counted, never silently folded in.
            n_degenerate += 1

    # CI validity gate. Dropping the degenerate replicates is correct only
    # when they are rare; when a coarse baseline drives F* to zero on a large
    # fraction of replicates (the typical rules-based-production case: F*
    # collapses on many resamples), the surviving subset is a spend-correlated, non-random
    # slice, and a percentile over it is NOT a 95% interval for U. Above a 5%
    # degeneracy rate the interval is refused rather than reported over a
    # biased subsample -- report.py must not print it when ci_valid is False.
    degeneracy_rate = n_degenerate / n_resamples if n_resamples else 1.0
    ci_valid = bool(uplifts) and degeneracy_rate < 0.05

    if not ci_valid:
        return {"u_point": float(point["U"]), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_resamples": int(n_resamples),
                "confidence": float(confidence),
                "n_degenerate": int(n_degenerate),
                "degeneracy_rate": float(degeneracy_rate),
                "ci_valid": False}

    tail = (1.0 - confidence) / 2.0 * 100.0
    return {
        "u_point": float(point["U"]),
        "ci_low": float(np.percentile(uplifts, tail)),
        "ci_high": float(np.percentile(uplifts, 100.0 - tail)),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
        "n_degenerate": int(n_degenerate),
        "degeneracy_rate": float(degeneracy_rate),
        "ci_valid": True,
    }


def holdout_fp_check(labels: dict[str, bool],
                     baseline_scores: dict[str, float],
                     pilot_scores: dict[str, float],
                     budget_fp_per_10k: float = DEFAULT_BUDGET_FP_PER_10K,
                     calibration_fraction: float = 0.5,
                     seed: int = 0) -> dict:
    """Measure the FP budget's honesty: calibrate on one half, spend on the other.

    The problem this quantifies: :func:`evaluate` picks the threshold on the
    same honest accounts it then measures false positives on. In-sample, the
    realized FP rate therefore hits the budget almost exactly by construction
    -- the number is a tautology, not a measurement. But the budget is a
    *promise* about how many honest people get flagged next week, and next
    week's honest accounts are not the ones the threshold was fitted to.

    This function splits the honest population at random, derives the
    threshold on the calibration half only, and reports the FP rate actually
    realized on the untouched measurement half. The gap between the budget and
    that number is the honest error bar on the promise.

    Note this is a *diagnostic*, not the headline path: U itself is reported
    in-sample, because splitting halves the calibration set and adds variance
    to the threshold without materially moving U at demo scale. The point is
    to make the optimism visible and sized, rather than to hide it behind a
    procedure.

    Returns "budget_fp_per_10k", "calibration_fraction", "n_calibration",
    "n_measurement", and, for each scorer, the realized FP per 10k on the
    held-out half: "realized_fp_per_10k_baseline" / "..._pilot".
    """
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")

    honest_ids = sorted(a for a, is_shared in labels.items() if not is_shared)
    if len(honest_ids) < 2:
        raise ValueError("need at least 2 honest accounts to split")

    rng = np.random.default_rng(seed)
    shuffled = list(honest_ids)
    rng.shuffle(shuffled)
    cut = max(1, min(len(shuffled) - 1,
                     int(round(len(shuffled) * calibration_fraction))))
    calibration, measurement = shuffled[:cut], shuffled[cut:]

    baseline_ranks = rank_normalize(baseline_scores)
    detector_ranks = rank_normalize(pilot_scores)
    stacked = {a: max(baseline_ranks[a], detector_ranks[a]) for a in labels}

    realized: dict[str, float] = {}
    for name, scores in (("baseline", baseline_scores), ("pilot", stacked)):
        threshold = threshold_at_fp_budget(
            [scores[a] for a in calibration], budget_fp_per_10k)
        flagged = sum(1 for a in measurement if scores[a] > threshold)
        realized[name] = flagged / len(measurement) * 10000.0

    return {
        "budget_fp_per_10k": float(budget_fp_per_10k),
        "calibration_fraction": float(calibration_fraction),
        "n_calibration": len(calibration),
        "n_measurement": len(measurement),
        "realized_fp_per_10k_baseline": float(realized["baseline"]),
        "realized_fp_per_10k_pilot": float(realized["pilot"]),
    }


def holdout_fp_budget_ci(labels: dict[str, bool],
                         scores: dict[str, float],
                         budget_fp_per_10k: float = DEFAULT_BUDGET_FP_PER_10K,
                         calibration_fraction: float = 0.5,
                         n_resamples: int = 1000,
                         confidence: float = 0.95,
                         seed: int = 0) -> dict:
    """Held-out FP spend of ONE scorer against its promised budget, with a CI.

    The single-scorer counterpart of :func:`holdout_fp_check`, and the number
    a budget promise is actually accountable to: the threshold is fitted on a
    calibration half of the honest population and the spend is measured on
    the untouched other half, so the realized rate is not the in-sample
    tautology that hits the budget by construction.

    The bootstrap puts an interval around that realized rate. Each replicate
    resamples the calibration half and the measurement half by account, with
    replacement, **stratified by half** -- the two halves are never mixed,
    because letting a calibration account into the measurement half would
    re-introduce exactly the in-sample optimism the split exists to remove.
    The threshold is re-fitted on every resampled calibration half, so
    threshold-selection variance is inside the interval rather than hidden
    under it (the same reason :func:`bootstrap_uplift_ci` re-derives its
    thresholds per replicate).

    The split reproduces :func:`holdout_fp_check` exactly (same shuffle, same
    cut), so on identical inputs the point estimate here equals that
    function's ``realized_fp_per_10k_baseline``.

    Two data-dependent cases are refused rather than reported, mirroring the
    CI validity gate in :func:`bootstrap_uplift_ci`: the dict comes back with
    ``computable: False`` and a stated reason instead of a number that would
    be an artifact of the input size:

    * fewer than 2 honest accounts -- there is nothing to split;
    * the budget floors to 0 permitted alerts on the calibration half -- the
      fitted threshold is then just the highest calibration score, and the
      measured rate describes the split, not the budget.

    Returns "budget_fp_per_10k", "calibration_fraction", "n_calibration",
    "n_measurement", "realized_fp_per_10k", "ci_low", "ci_high",
    "confidence", "n_resamples", "computable", "reason".
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    if budget_fp_per_10k < 0:
        raise ValueError("budget_fp_per_10k must be >= 0")

    base = {
        "budget_fp_per_10k": float(budget_fp_per_10k),
        "calibration_fraction": float(calibration_fraction),
        "confidence": float(confidence),
        "n_resamples": int(n_resamples),
    }

    honest_ids = sorted(a for a, is_shared in labels.items() if not is_shared)
    if len(honest_ids) < 2:
        return {**base, "n_calibration": None, "n_measurement": None,
                "realized_fp_per_10k": None, "ci_low": None, "ci_high": None,
                "computable": False,
                "reason": (f"only {len(honest_ids)} honest account(s); the "
                           f"check needs at least one honest account on each "
                           f"side of the calibration/measurement split")}

    # Same shuffle, same cut as holdout_fp_check, so the two functions agree
    # on the split (and therefore on the point estimate) for the same seed.
    rng = np.random.default_rng(seed)
    shuffled = list(honest_ids)
    rng.shuffle(shuffled)
    cut = max(1, min(len(shuffled) - 1,
                     int(round(len(shuffled) * calibration_fraction))))
    calibration, measurement = shuffled[:cut], shuffled[cut:]
    n_cal, n_meas = len(calibration), len(measurement)

    allowed_fp = math.floor(n_cal * budget_fp_per_10k / 10000.0)
    if allowed_fp == 0:
        need = (math.ceil(10000.0 / budget_fp_per_10k)
                if budget_fp_per_10k > 0 else None)
        reason = (f"a budget of {budget_fp_per_10k:g}/10k floors to 0 "
                  f"permitted alerts on the {n_cal}-account calibration half, "
                  f"so the fitted threshold would be the highest calibration "
                  f"score and the measured rate an artifact of the split, not "
                  f"of the budget")
        if need is not None:
            reason += f"; at least {need} honest accounts per half are needed"
        return {**base, "n_calibration": int(n_cal),
                "n_measurement": int(n_meas),
                "realized_fp_per_10k": None, "ci_low": None, "ci_high": None,
                "computable": False, "reason": reason}

    cal = np.array([scores[a] for a in calibration], dtype=float)
    meas = np.array([scores[a] for a in measurement], dtype=float)

    threshold = threshold_at_fp_budget(list(cal), budget_fp_per_10k)
    point = float(np.count_nonzero(meas > threshold)) / n_meas * 10000.0

    realized = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        rs_cal = cal[rng.integers(0, n_cal, n_cal)]
        rs_meas = meas[rng.integers(0, n_meas, n_meas)]
        if allowed_fp >= n_cal:
            t = -math.inf
        else:
            # (allowed_fp + 1)-th highest of the resampled calibration half:
            # the same arithmetic as threshold_at_fp_budget, via partition.
            k = n_cal - 1 - allowed_fp
            t = float(np.partition(rs_cal, k)[k])
        realized[i] = float(np.count_nonzero(rs_meas > t)) / n_meas * 10000.0

    tail = (1.0 - confidence) / 2.0 * 100.0
    return {**base, "n_calibration": int(n_cal), "n_measurement": int(n_meas),
            "realized_fp_per_10k": float(point),
            "ci_low": float(np.percentile(realized, tail)),
            "ci_high": float(np.percentile(realized, 100.0 - tail)),
            "computable": True, "reason": None}


def summarize_across_seeds(uplifts: list[float], target: float = 0.20) -> dict:
    """Distribution of U over independent synthetic worlds.

    One seed is one draw from the space of synthetic populations. Quoting a
    single seed's U as "the" result is cherry-picking whether or not it was
    chosen deliberately -- the reader cannot tell the difference, which is
    reason enough not to do it. The headline this function supports is the
    **median**; spread and pass-rate are reported alongside so the reader can
    see how much of any single number is luck.

    Args:
        uplifts: finite U values, one per seed.
        target: the bar to report a pass-rate against. Reported as a
            descriptive fraction only -- on synthetic data no pass/fail
            verdict applies (see report.py).

    Returns "median", "mean", "sd" (sample sd, n-1), "min", "max", "n_seeds"
    (finite values actually summarized), "n_seeds_supplied",
    "n_seeds_dropped_nonfinite" (seeds excluded because U was infinite),
    "target", "n_at_or_above_target", "fraction_at_or_above_target".
    """
    n_supplied = len(uplifts)
    finite = [float(u) for u in uplifts if math.isfinite(float(u))]
    if not finite:
        raise ValueError("no finite uplift values to summarize")
    # A seed whose baseline caught nothing yields U = +inf and cannot enter a
    # median. Dropping it is right; dropping it silently is not -- "n_seeds"
    # would then report the survivors and read as the number of seeds run.
    n_dropped = n_supplied - len(finite)

    ordered = sorted(finite)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    mean = sum(ordered) / n
    # Sample standard deviation (n-1); undefined for a single seed.
    sd = (math.sqrt(sum((u - mean) ** 2 for u in ordered) / (n - 1))
          if n > 1 else 0.0)
    n_pass = sum(1 for u in ordered if u >= target)

    return {
        "median": median,
        "mean": mean,
        "sd": sd,
        "min": ordered[0],
        "max": ordered[-1],
        "n_seeds": n,
        "n_seeds_supplied": int(n_supplied),
        "n_seeds_dropped_nonfinite": int(n_dropped),
        "target": float(target),
        "n_at_or_above_target": n_pass,
        "fraction_at_or_above_target": n_pass / n,
    }
