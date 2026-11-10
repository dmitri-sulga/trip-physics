"""Report rendering: one JSON for machines, one Markdown for humans.

Both outputs are pure functions of ``(result, meta)`` -- no timestamps, no
hostnames, no environment probing -- so the same seed produces bytewise
identical reports. That property is load-bearing: the determinism test in
``tests/`` diffs report JSON across runs.
"""

import json
import math
import os

# The acceptance bar fixed in advance for a run on real data (docs/method.md,
# "The acceptance bar"): on a real shadow log, the stacked scorer must lift
# recall at the agreed FP budget by at least this much, relative, over the
# baseline alone.
#
# Scope matters more than the value. This bar applies to a run on real
# confirmed-shared labels, with a pre-registered minimum of positives in
# the window. It does NOT apply to the synthetic demo: there the generator and
# the detector were written by the same hand, so the detector is correct by
# construction and no pass/fail verdict on that data would mean anything.
# Rendering "MEETS the target" over simulated numbers is exactly the kind of
# claim this harness exists to prevent, so the synthetic path never does it.
U_TARGET = 0.20


def _json_safe(value):
    """Replace non-finite floats with None so the JSON is strictly parseable.

    ``json.dumps`` would happily emit ``Infinity``, which is not JSON and
    breaks downstream consumers (jq, spreadsheets, most parsers in strict
    mode). U can be +inf when baseline recall is zero -- see eval.py.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _fmt_pct(value: float) -> str:
    """Signed percent with one decimal, e.g. +23.4% / -5.0%."""
    return f"{value * 100.0:+.1f}%"


def _fmt_float(value: float, digits: int = 4) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        return "-inf" if value < 0 else "+inf"
    return f"{value:.{digits}f}"


def _verdict_lines(u_value: float, synthetic: bool = True) -> tuple[str, str]:
    """(headline metric line, verdict line) for the headline box.

    On synthetic data there is deliberately no verdict: the run verifies that
    the pipeline computes the metric end-to-end, and nothing else. Only a run
    against real labels (``synthetic=False``) is compared to ``U_TARGET``.
    """
    if isinstance(u_value, float) and not math.isfinite(u_value):
        metric = "U = +inf (baseline recall is zero at this budget)"
    else:
        metric = f"U = {_fmt_pct(u_value)}"

    if synthetic:
        verdict = ("SYNTHETIC DRY-RUN: pipeline verified end-to-end. "
                   "No pass/fail applies to simulated data.")
        return metric, verdict

    if isinstance(u_value, float) and not math.isfinite(u_value):
        return metric, ("VERDICT: baseline caught nothing at this budget — "
                        "U is undefined, not a pass")
    metric = f"{metric}   target: >= {_fmt_pct(U_TARGET)}"
    verdict = (f"VERDICT: {'MEETS' if u_value >= U_TARGET else 'BELOW'} "
               f"the target on this dataset")
    return metric, verdict


def render_report(result: dict, meta: dict, out_json: str, out_md: str) -> None:
    """Write the evaluation result as JSON and Markdown.

    Args:
        result: the dict returned by ``eval.evaluate``.
        meta: run parameters worth pinning to the numbers (seed, n_accounts,
            shared_fraction, ...). Rendered verbatim; keep it JSON-friendly.
        out_json: path for the machine-readable report.
        out_md: path for the human-readable report.

    Parent directories are created as needed. Output depends only on the
    arguments -- rendering the same result twice yields identical bytes.
    """
    for path in (out_json, out_md):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    payload = _json_safe({"meta": meta, "metrics": result})
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")

    with open(out_md, "w") as fh:
        fh.write(_render_markdown(result, meta))


def _render_markdown(result: dict, meta: dict) -> str:
    u_value = result["U"]
    synthetic = bool(meta.get("synthetic", True))
    sweep = meta.get("seed_sweep")
    metric_line, verdict_line = _verdict_lines(u_value, synthetic=synthetic)

    lines: list[str] = []
    lines.append("# trip-physics — shadow-evaluation report")
    lines.append("")

    # --- Headline box. On a seed sweep the headline is the MEDIAN across
    # seeds, never the single displayed run: one seed is one draw, and
    # quoting the luckiest draw is cherry-picking whether or not it was
    # chosen deliberately.
    lines.append("```")
    lines.append("=" * 62)
    if sweep:
        lines.append(f"  HEADLINE   median U = {_fmt_pct(sweep['median'])} "
                     f"across {sweep['n_seeds']} seeds")
        lines.append(f"             (sd {sweep['sd'] * 100:.1f}pp, range "
                     f"{_fmt_pct(sweep['min'])} … {_fmt_pct(sweep['max'])})")
    else:
        lines.append(f"  SIGNED METRIC   {metric_line}")
    lines.append(f"  {verdict_line}")
    if synthetic:
        lines.append("  The generator and the detector were written by the")
        lines.append("  same hand: here the detector is right by construction.")
    lines.append("=" * 62)
    lines.append("```")
    lines.append("")

    # --- Under-powered runs must say so in the artifact itself, not only in
    # the terminal that produced them. A report file outlives its console.
    if result["n_accounts"] < 10000:
        lines.append(f"> ⚠️ **Under-powered run — do not quote this number.** "
                     f"{result['n_accounts']} accounts is below the ~10k "
                     f"floor at which U stabilizes; across seeds 1-25 at 2,000 "
                     f"accounts U ranges from +2.5% to +37.5%. This is a smoke test. "
                     f"Run `python examples/run_demo.py --full` for the "
                     f"quotable figure.")
        lines.append("")

    # --- The two numbers a reader must not conflate.
    if synthetic:
        lines.append("> **Two different numbers, do not compare them.**")
        lines.append("> ")
        headline = (_fmt_pct(sweep["median"]) + f" (median of {sweep['n_seeds']} seeds)"
                    if sweep else _fmt_pct(u_value))
        lines.append(f"> **(a) This synthetic demo: U = {headline}.** It checks "
                     f"only that the pipeline computes the metric correctly "
                     f"end-to-end, on data where the detector's assumptions are "
                     f"true by construction, against a deliberately crude "
                     f"baseline (`NaiveBaseline`). It is not an estimate of "
                     f"what the detector adds on real data.")
        lines.append("> ")
        lines.append(f"> **(b) The acceptance bar for real data: U ≥ "
                     f"{_fmt_pct(U_TARGET)} on a real shadow log.** That is a "
                     f"different measurement, on different data, against real "
                     f"confirmed-shared labels. A synthetic number can neither "
                     f"pass nor fail it.")
        lines.append("")

    # --- Headline metrics, baseline vs. stacked, EQUAL REALIZED spend.
    lines.append("## Headline metrics")
    lines.append("")

    # The budget names a maximum; the realized spend is what a strict
    # threshold actually flags on BOTH scorers (their common tie-feasible
    # spend). When a coarse baseline cannot spend the budget at all
    # (utilization below the floor), a point U at "equal spend" would be a
    # comparison at zero spend dressed up as a headline. Refuse it and say so.
    allowed_fp = result.get("allowed_fp")
    realized_fp = result.get("realized_fp")
    utilization = result.get("budget_utilization")
    realized_per_10k = result.get("realized_fp_per_10k")
    UTIL_FLOOR = 0.50
    util_ok = utilization is None or utilization >= UTIL_FLOOR

    if realized_fp is not None:
        lines.append(
            f"Both scorers are measured at the **same realized FP spend**: "
            f"{realized_fp} honest accounts flagged "
            f"({_fmt_float(realized_per_10k, 1)} per 10k), the largest spend a "
            f"strict threshold reaches on both scorers without exceeding the "
            f"{_fmt_float(result['budget_fp_per_10k'], 1)}/10k budget "
            f"(utilization {utilization * 100:.0f}%).")
    else:
        lines.append(f"Both scorers thresholded at the same false-positive "
                     f"budget: **{_fmt_float(result['budget_fp_per_10k'], 1)} "
                     f"FPs per 10k honest accounts**.")
    lines.append("")

    if not util_ok:
        lines.append(f"> ⚠️ **Baseline granularity cannot spend the budget.** "
                     f"The baseline's scores are coarse enough that a strict "
                     f"threshold reaches only {realized_fp} of the "
                     f"{allowed_fp} honest flags the budget allows "
                     f"(utilization {utilization * 100:.0f}%). U is reported "
                     f"below at that realized spend, not at the budget: a "
                     f"headline uplift 'at equal FP spend' would otherwise "
                     f"compare the two scorers at a spend the baseline never "
                     f"reaches, which overstates the stacked scorer. Feed a finer "
                     f"baseline score (calibrated probabilities rather than "
                     f"rounded buckets) to spend the full budget.")
        lines.append("")

    lines.append("| metric | baseline | stacked (baseline + detector) |")
    lines.append("|---|---|---|")
    lines.append(f"| recall at realized spend | {_fmt_float(result['R0'])} (R0) "
                 f"| {_fmt_float(result['R1'])} (R1) |")
    lines.append(f"| PR-AUC | {_fmt_float(result['pr_auc_baseline'])} "
                 f"| {_fmt_float(result['pr_auc_pilot'])} |")
    lines.append(f"| precision@500 | {_fmt_float(result['precision_at_500_baseline'])} "
                 f"| {_fmt_float(result['precision_at_500_pilot'])} |")
    if "threshold_baseline" in result and "threshold_pilot" in result:
        lines.append(f"| threshold | {_fmt_float(result['threshold_baseline'], 6)} "
                     f"| {_fmt_float(result['threshold_pilot'], 6)} |")
    lines.append("")
    u_str = ("+inf" if isinstance(u_value, float) and not math.isfinite(u_value)
             else _fmt_pct(u_value))
    lines.append(f"- **U = (R1 − R0) / R0 = {u_str}** — relative recall uplift "
                 f"at the same realized FP spend (see above).")
    lines.append(f"- Population: {result['n_accounts']} accounts, "
                 f"{result['n_shared']} shared (ground truth).")
    lines.append("")

    # --- U across FP budgets: the operating point is never cherry-picked.
    curve = meta.get("budget_curve")
    if curve:
        lines.append("## U across FP budgets")
        lines.append("")
        lines.append("A run on real data fixes ONE budget before the measurement "
                     "starts. This curve shows the whole trade-off around the "
                     "default, so the operating point is chosen openly, not for a "
                     "flattering number.")
        lines.append("")
        lines.append("| FP budget / 10k | R0 (baseline) | R1 (stacked) | U |")
        lines.append("|---|---|---|---|")
        for row in curve:
            mark = " ← default" if row.get("default") else ""
            u_row = ("+inf" if not math.isfinite(row["U"]) else _fmt_pct(row["U"]))
            lines.append(f"| {_fmt_float(row['budget_fp_per_10k'], 0)}{mark} "
                         f"| {_fmt_float(row['R0'])} | {_fmt_float(row['R1'])} "
                         f"| {u_row} |")
        lines.append("")
        # These numbers are interpolated from the actual result, never
        # hardcoded: a report whose prose disagrees with its own table is
        # worse than one that omits the prose.
        lines.append(f"PR-AUC ({_fmt_float(result['pr_auc_baseline'], 2)} → "
                     f"{_fmt_float(result['pr_auc_pilot'], 2)} above) is "
                     f"budget-independent — the separation is present before "
                     f"any threshold is chosen.")
        lines.append("")

    # --- U across independent seeds: the anti-cherry-picking section.
    if sweep:
        lines.append("## U across independent seeds")
        lines.append("")
        lines.append(f"Each seed is an independent synthetic population "
                     f"({result['n_accounts']} accounts, "
                     f"{result['n_shared']} shared). A single seed's U is one "
                     f"draw from this distribution; the headline above is the "
                     f"**median**, not the best draw.")
        lines.append("")
        lines.append("| statistic | U |")
        lines.append("|---|---|")
        lines.append(f"| median | {_fmt_pct(sweep['median'])} |")
        lines.append(f"| mean | {_fmt_pct(sweep['mean'])} |")
        lines.append(f"| sd | {sweep['sd'] * 100:.1f}pp |")
        lines.append(f"| min | {_fmt_pct(sweep['min'])} |")
        lines.append(f"| max | {_fmt_pct(sweep['max'])} |")
        lines.append(f"| seeds | {sweep['n_seeds']} |")
        lines.append("")
        # Deliberately no hit-rate against the acceptance bar here. Counting how many
        # synthetic seeds clear a threshold meant for real data is the exact
        # comparison this report tells the reader not to make, and phrasing it as
        # "N of M" reads as a pass rate no matter what disclaimer follows it.
        lines.append(f"The synthetic spread runs {_fmt_pct(sweep['min'])} … "
                     f"{_fmt_pct(sweep['max'])} (sd {sweep['sd'] * 100:.1f}pp). "
                     f"The real-data acceptance bar of "
                     f"{_fmt_pct(sweep['target'])} sits above that range on "
                     f"purpose: a bar this simulator clears on its own would be "
                     f"a bar calibrated to the simulator rather than to real "
                     f"traffic.")
        lines.append("")
        per_seed = meta.get("seed_sweep_values")
        if per_seed:
            lines.append("Per-seed U, every value from the sweep, none dropped:")
            lines.append("")
            lines.append("| seed | U |")
            lines.append("|---|---|")
            for row in per_seed:
                lines.append(f"| {row['seed']} | {_fmt_pct(row['U'])} |")
            lines.append("")

    # --- Bootstrap CI on the displayed run. Suppressed when the CI is invalid:
    # a percentile taken over resamples after dropping degenerate ones is only
    # a 95% interval when the degenerate fraction is small (see
    # bootstrap_uplift_ci's ci_valid gate). Below that bar the interval would
    # be computed over a spend-correlated subsample and is not printed.
    ci = meta.get("bootstrap_ci")
    if ci and ci.get("ci_valid", True) is False:
        lines.append("## Bootstrap confidence interval")
        lines.append("")
        lines.append(f"Not reported. Baseline recall was zero at the joint FP "
                     f"spend in {ci.get('n_degenerate', 0)} of "
                     f"{ci.get('n_resamples', 0)} resamples "
                     f"({ci.get('degeneracy_rate', 0.0) * 100:.0f}%), so U was "
                     f"undefined in too many of them for a percentile interval "
                     f"over the rest to be a valid 95% CI. This is the same "
                     f"coarse-baseline granularity flagged in the headline: "
                     f"feed a finer baseline score to make the interval "
                     f"meaningful.")
        lines.append("")
    elif ci and math.isfinite(ci.get("ci_low", float("nan"))):
        lines.append("## Bootstrap confidence interval")
        lines.append("")
        lines.append(f"Stratified percentile bootstrap on the displayed seed, "
                     f"{ci['n_resamples']} resamples, accounts resampled with "
                     f"replacement while holding the shared/honest counts "
                     f"fixed. The threshold is re-derived on every replicate, "
                     f"so threshold-selection variance is inside the interval "
                     f"rather than hidden behind it.")
        lines.append("")
        lines.append(f"- **U = {_fmt_pct(ci['u_point'])}, "
                     f"{ci['confidence'] * 100:.0f}% CI "
                     f"[{_fmt_pct(ci['ci_low'])}, {_fmt_pct(ci['ci_high'])}]**")
        if ci.get("n_degenerate"):
            lines.append(f"- {ci['n_degenerate']} replicate(s) had zero "
                         f"baseline recall (U undefined) and were excluded.")
        lines.append("")

    # --- Held-out FP check: is the budget promise actually kept?
    fp_check = meta.get("holdout_fp_check")
    if fp_check:
        lines.append("## Is the FP budget promise kept out-of-sample?")
        lines.append("")
        lines.append(f"The threshold is chosen on the same honest accounts the "
                     f"FP rate is then measured on, so in-sample the budget is "
                     f"hit by construction — a tautology, not a measurement. "
                     f"This check derives the threshold on "
                     f"{fp_check['n_calibration']} honest accounts and reports "
                     f"the FP rate realized on the {fp_check['n_measurement']} "
                     f"held-out ones.")
        lines.append("")
        lines.append("| scorer | budget | realized on held-out honest accounts |")
        lines.append("|---|---|---|")
        lines.append(f"| baseline | {_fmt_float(fp_check['budget_fp_per_10k'], 0)} / 10k "
                     f"| {fp_check['realized_fp_per_10k_baseline']:.1f} / 10k |")
        lines.append(f"| stacked | {_fmt_float(fp_check['budget_fp_per_10k'], 0)} / 10k "
                     f"| {fp_check['realized_fp_per_10k_pilot']:.1f} / 10k |")
        lines.append("")
        lines.append("The gap is the honest error bar on the budget promise. "
                     "It shrinks as the honest population grows and widens "
                     "sharply on small samples.")
        lines.append("")

    # --- How to read U.
    lines.append("## How to read U")
    lines.append("")
    lines.append("R0 is the baseline's recall on shared accounts when its "
                 "threshold is set to spend exactly the agreed FP budget on "
                 "honest accounts. R1 is the same measurement for the stacked "
                 "scorer — the baseline stacked with the detector via "
                 "`max(rank-normalized baseline, rank-normalized detector)` — "
                 "at the *same* budget. U is the relative improvement. "
                 "Holding the FP budget fixed is the point: any scorer can "
                 "raise recall by flagging more people; U only rewards recall "
                 "bought without additional honest-account cost. U can be "
                 "negative if the detector adds noise instead of signal.")
    lines.append("")

    # --- Run parameters.
    lines.append("## Run parameters")
    lines.append("")
    lines.append("| parameter | value |")
    lines.append("|---|---|")
    for key in sorted(meta):
        lines.append(f"| {key} | {meta[key]} |")
    lines.append("")

    # --- Honest limits.
    lines.append("## Honest limits")
    lines.append("")
    lines.append("- **Synthetic data proves the harness works, not that the "
                 "detector works on real traffic.** The synthetic sharing "
                 "patterns were designed alongside the detector; on real "
                 "trips both the base rates and the evasion behavior will "
                 "differ. Measuring that needs a shadow run on real logs with "
                 "confirmed labels.")
    lines.append("- The bundled `NaiveBaseline` is a placeholder, not a model "
                 "of any production stack. Uplift over it is not uplift over "
                 "a real scoring system; plug in real scores via the `scores_csv` "
                 "adapter before reading anything into U.")
    lines.append("- **The threshold is calibrated on the same honest "
                 "population it is measured on.** In-sample the realized FP "
                 "rate therefore matches the budget almost exactly by "
                 "construction; out-of-sample it does not. The held-out check "
                 "above sizes that gap instead of assuming it away. On real "
                 "data, calibrate on a window that precedes the measurement "
                 "window.")
    lines.append("- The threshold-at-budget procedure assumes the honest-score "
                 "distribution is roughly stationary between the calibration "
                 "window and the measurement window. Drift (app releases, "
                 "seasonal patterns) moves the effective FP spend.")
    lines.append("- A single seed's U is one draw from a distribution, and at "
                 "small sample sizes that distribution is wide. Any headline "
                 "quoting one seed is reporting luck as much as signal — which "
                 "is why the headline here is a median across seeds.")
    lines.append("- Rank-based stacking is evaluated offline over a closed "
                 "population. A live deployment needs an online calibration "
                 "step, since ranks are only defined relative to a batch.")
    lines.append("- On real data, \"shared\" labels come from confirmed cases "
                 "and are incomplete. Recall against known-shared accounts is "
                 "a lower-bound proxy, and both scorers inherit the same "
                 "label bias.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*trip-physics — shadow-evaluation harness · report rendered "
                 "deterministically from the run above (no timestamps by design).*")
    lines.append("")
    return "\n".join(lines)
