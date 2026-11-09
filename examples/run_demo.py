"""End-to-end synthetic demo.

Pipeline: synthetic dataset -> physics features -> detector scores ->
budgeted evaluation (which stacks detector with baseline internally) ->
report files.

What this run does and does not show
------------------------------------
It shows that the harness computes U correctly from raw trip fixes to a
rendered report. It does NOT show that the detector works on real traffic:
the generator and the detector were written by the same hand, so here the
detector's assumptions are true by construction.

Two design decisions exist to keep that honest:

1. **The headline is the median across seeds, never a single seed.** One
   seed is one draw from a distribution of synthetic worlds. At the old demo
   size (2000 accounts, 60 positives) that distribution has an sd of
   ~7.5pp and a range from +2.5% to +37.5% over seeds 1..25
   (`examples/sample_size_table.py`) — so a single-seed headline reported
   mostly luck. The seed shown in the detailed tables is just the first one; it is
   not selected for its result.

2. **The default population is 20,000 accounts.** The README warns that U
   goes noisy below roughly 10k scored accounts, and the pilot pre-registers
   a minimum of 150 confirmed positives. The old 2000-account default sat
   below both of its own thresholds: 600 positives instead of 60 cuts the
   across-seed sd from ~7.5pp to ~2pp.

Everything is seeded; running this twice produces byte-identical reports.

Usage
-----
    python examples/run_demo.py            # fast: 3 seeds x 20k, 123 s
    python examples/run_demo.py --full     # full sweep: 20 seeds x 20k, 542 s
    python examples/run_demo.py --quick    # smoke test: 2000 accounts, 17 s

    python examples/run_demo.py --accounts 5000 --seeds 10   # custom sweep

Runtimes measured 2026-08-04 on an idle 14-core Apple-silicon laptop, Python
3.14.5 with numpy 2.5.1; the sweep runs across processes, so wall time depends
heavily on core count, and peak memory for the default run is 5.0 GiB. The
README quotes the same figures and carries the measurement conditions in full
-- keep the two in step when either changes. These numbers were four to eight
times higher when last measured in July, so treat any inherited timing as
stale until re-run.

The full sweep is the number to quote. The default is trimmed only so the
first run finishes sooner; both use identical code paths and the same
20,000-account population, so the fast run is a smaller sample of the same
distribution, not a different measurement.

Because of that, the median this run prints will usually NOT equal the figure
quoted outside this repo, and the run says so itself: it reads the committed
sweep back out of ./out/report.json before overwriting it and prints both
numbers side by side, labelled, under "WHICH NUMBER IS WHICH". Two correct
numbers presented without that label read as one number that does not
reproduce, which costs a careful reader's trust for no reason.

Note that every run OVERWRITES ./out/report.json and ./out/report.md, which
are committed artifacts holding the quotable 20-seed sweep. After running
--quick or a custom sweep, `git checkout -- out/` restores them.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

from tripphysics.baseline import NaiveBaseline
from tripphysics.detector import score_account
from tripphysics.eval import (bootstrap_uplift_ci, evaluate, holdout_fp_check,
                              summarize_across_seeds)
from tripphysics.features import extract_features
from tripphysics.report import render_report
from tripphysics.synth import generate_dataset

OUT_DIR = "./out"

# Defaults: 20k accounts at a 3% base rate = 600 positives, comfortably above
# both the "10k accounts" noise floor and the pilot's 150-positive minimum.
N_ACCOUNTS = 20000
SHARED_FRACTION = 0.03
BUDGET_CURVE = (15.0, 30.0, 50.0, 100.0, 200.0)


def shipped_reference(path: str) -> dict | None:
    """The committed run's headline numbers, read back from out/report.json.

    Read rather than hardcoded on purpose. The figure quoted outside this
    repo is the median of the 20-seed sweep, and it lives in the committed
    report; a literal copied into this file would be a second source of
    truth that drifts the first time the sweep is re-run. If the two ever
    disagree, the disagreement should be visible in a diff of one artifact,
    not silently absorbed by a constant nobody re-checks.

    Returns None when the file is absent, carries no seed sweep, or carries
    a sweep that is not the 20-seed `--full` run at the default population --
    which is the normal state after a `--quick` or custom run has already
    overwritten it. Presence of a sweep alone is not enough: a 3-seed quick
    sweep would otherwise be echoed back labelled as the committed run. In
    that case the caller prints the run's own number and says the reference
    is unavailable, rather than inventing one.
    """
    try:
        with open(path) as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        return None
    sweep = meta.get("seed_sweep")
    if not isinstance(sweep, dict) or "median" not in sweep:
        return None
    if sweep.get("n_seeds") != 20 or meta.get("n_accounts") != N_ACCOUNTS:
        return None
    return {"median": sweep["median"], "n_seeds": sweep["n_seeds"],
            "sd": sweep.get("sd"),
            "display_seed": meta.get("display_seed"),
            "u_display_seed": payload.get("metrics", {}).get("U")}


def score_population(n_accounts: int, shared_fraction: float, seed: int):
    """Build one synthetic world and score it with both scorers."""
    dataset = generate_dataset(n_accounts=n_accounts,
                               shared_fraction=shared_fraction, seed=seed)
    accounts = dataset["accounts"]
    labels = {a["account_id"]: a["is_shared"] for a in accounts}

    # NaiveBaseline is a crude stand-in for "the current stack"; with real
    # data, replace it via baseline.load_scores_csv().
    naive = NaiveBaseline()
    baseline_scores = {a["account_id"]: naive.score(a) for a in accounts}
    detector_scores = {a["account_id"]: score_account(extract_features(a))
                       for a in accounts}
    return dataset["meta"], labels, baseline_scores, detector_scores


def _uplift_for_seed(args) -> dict:
    """One seed's U. Top-level so ProcessPoolExecutor can pickle it."""
    n_accounts, shared_fraction, seed = args
    _, labels, baseline_scores, detector_scores = score_population(
        n_accounts, shared_fraction, seed)
    result = evaluate(labels, baseline_scores, detector_scores)
    return {"seed": seed, "U": result["U"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="20-seed sweep at 20k accounts (slow, quotable)")
    parser.add_argument("--quick", action="store_true",
                        help="2k accounts, 3 seeds: smoke test only, U is noisy")
    parser.add_argument("--accounts", type=int, default=None, metavar="N",
                        help="override the account count (reproduces the "
                             "sample-size table in the README)")
    parser.add_argument("--seeds", type=int, default=None, metavar="K",
                        help="override the number of seeds in the sweep")
    args = parser.parse_args()

    # `is None`, not `or`: `--accounts 0` is falsy, so `args.accounts or
    # default` would silently swap an invalid 0 for the 20000-account default
    # and run a full sweep the caller never asked for. Explicit None checks
    # let the validation below actually see the 0 and reject it.
    n_accounts = args.accounts if args.accounts is not None else (
        2000 if args.quick else N_ACCOUNTS)
    n_seeds = args.seeds if args.seeds is not None else (20 if args.full else 3)
    if n_accounts < 1 or n_seeds < 1:
        parser.error("--accounts and --seeds must both be >= 1")
    seeds = list(range(1, n_seeds + 1))

    # Read the shipped reference BEFORE the run overwrites it. Step 5 below
    # rewrites out/report.json, so reading it afterwards would just echo this
    # run's own numbers back as though they were the quoted ones -- which is
    # precisely the confusion this reconciliation exists to remove.
    reference = shipped_reference(os.path.join(OUT_DIR, "report.json"))

    # 1. Seed sweep. Feature extraction dominates the runtime and each seed is
    #    independent, so the sweep runs across processes.
    print(f"sweeping {n_seeds} seeds x {n_accounts} accounts "
          f"(feature extraction dominates; this is the slow part)...")
    with ProcessPoolExecutor() as pool:
        sweep_rows = list(pool.map(
            _uplift_for_seed,
            [(n_accounts, SHARED_FRACTION, s) for s in seeds]))
    sweep = summarize_across_seeds([r["U"] for r in sweep_rows])

    # 2. Detailed tables come from the FIRST seed, not the best one. It is a
    #    worked example of the pipeline; the headline stays the median.
    display_seed = seeds[0]
    dataset_meta, labels, baseline_scores, detector_scores = score_population(
        n_accounts, SHARED_FRACTION, display_seed)
    result = evaluate(labels, baseline_scores, detector_scores)

    # 3. Bootstrap CI and the held-out FP-budget check on that same seed.
    ci = bootstrap_uplift_ci(labels, baseline_scores, detector_scores,
                             n_resamples=1000, seed=0)
    fp_check = holdout_fp_check(labels, baseline_scores, detector_scores, seed=0)

    # 4. U across FP budgets, so the operating point is never cherry-picked.
    curve = []
    for budget in BUDGET_CURVE:
        row = evaluate(labels, baseline_scores, detector_scores,
                       budget_fp_per_10k=budget)
        curve.append({"budget_fp_per_10k": budget, "R0": row["R0"],
                      "R1": row["R1"], "U": row["U"],
                      "default": budget == result["budget_fp_per_10k"]})

    # 5. Reports.
    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, "report.json")
    out_md = os.path.join(OUT_DIR, "report.md")
    meta = dict(dataset_meta)
    meta.update({
        "synthetic": True,
        "display_seed": display_seed,
        "seed_sweep": sweep,
        "seed_sweep_values": sweep_rows,
        "bootstrap_ci": ci,
        "holdout_fp_check": fp_check,
        "budget_curve": curve,
    })
    render_report(result, meta, out_json, out_md)

    # 6. Console output. The median leads; the single seed is labelled as an
    #    example so no one can read it as the headline.
    print(f"\nHEADLINE  median U = {sweep['median'] * 100:+.1f}% "
          f"across {sweep['n_seeds']} seeds  (THIS run)")
    print(f"          mean {sweep['mean'] * 100:+.1f}%   "
          f"sd {sweep['sd'] * 100:.1f}pp   "
          f"range {sweep['min'] * 100:+.1f}% … {sweep['max'] * 100:+.1f}%")
    print(f"          {sweep['n_at_or_above_target']}/{sweep['n_seeds']} seeds "
          f">= +20% (descriptive only — no pass/fail on synthetic data)")

    # Three correct numbers describe this simulator, and they differ. A reader
    # who ran the default command and then saw a different figure quoted
    # elsewhere has no way to tell a moving target from a lie, so the run
    # reconciles them itself, here, on the first screen -- rather than leaving
    # the arithmetic to a reader who has every reason to stop reading.
    print("\nWHICH NUMBER IS WHICH  (all three are correct; they measure "
          "different things)")
    print(f"  this run      median U = {sweep['median'] * 100:+.1f}%   "
          f"median of {sweep['n_seeds']} seeds at {n_accounts} accounts")
    if reference is not None:
        print(f"  quoted figure median U = {reference['median'] * 100:+.1f}%   "
              f"median of {reference['n_seeds']} seeds, the committed "
              f"`--full` run in out/report.json")
        if reference.get("u_display_seed") is not None:
            print(f"  worked example       U = "
                  f"{reference['u_display_seed'] * 100:+.1f}%   single seed "
                  f"{reference['display_seed']} of that committed run — one "
                  f"draw, not a headline")
    else:
        print("  quoted figure  unavailable: out/report.json does not carry "
              "the committed 20-seed `--full` sweep, so the reference cannot "
              "be shown. `git checkout -- out/` restores it.")
    if reference is not None and sweep["n_seeds"] != reference["n_seeds"]:
        print(f"  -> Your {sweep['n_seeds']}-seed median is a smaller sample of "
              f"the same distribution as the {reference['n_seeds']}-seed one. "
              f"Across-seed sd is ~{sweep['sd'] * 100:.1f}pp here, so a gap of "
              f"a few points between them is sampling noise, not a "
              f"discrepancy.")
        print(f"     Compare like for like: run `--full` and compare "
              f"{reference['n_seeds']}-seed medians. A single seed moves more "
              f"again than either median does.")

    print(f"\nworked example, seed {display_seed} "
          f"(one draw, NOT the headline):")
    print(f"accounts: {result['n_accounts']}   shared: {result['n_shared']}   "
          f"budget: {result['budget_fp_per_10k']} FP / 10k honest")
    print(f"{'metric':<18}{'baseline':>10}{'pilot':>10}")
    for name, r0, r1 in (
            ("recall @ budget", result["R0"], result["R1"]),
            ("PR-AUC", result["pr_auc_baseline"], result["pr_auc_pilot"]),
            ("precision @ 500", result["precision_at_500_baseline"],
             result["precision_at_500_pilot"])):
        print(f"{name:<18}{r0:>10.3f}{r1:>10.3f}")
    print(f"U = {ci['u_point'] * 100:+.1f}%  "
          f"95% CI [{ci['ci_low'] * 100:+.1f}%, {ci['ci_high'] * 100:+.1f}%]  "
          f"({ci['n_resamples']} bootstrap resamples)")

    print(f"\nFP budget honesty (threshold fitted on half, spent on the other):")
    print(f"  budget {fp_check['budget_fp_per_10k']:.0f}/10k  ->  realized "
          f"baseline {fp_check['realized_fp_per_10k_baseline']:.1f}/10k, "
          f"pilot {fp_check['realized_fp_per_10k_pilot']:.1f}/10k")

    print(f"\nU vs FP budget (uplift is not a single lucky point):")
    print(f"{'budget/10k':>12}{'R0':>8}{'R1':>8}{'U':>8}")
    for row in curve:
        star = "  <- default" if row["default"] else ""
        print(f"{row['budget_fp_per_10k']:>12.0f}{row['R0']:>8.3f}"
              f"{row['R1']:>8.3f}{row['U'] * 100:>7.0f}%{star}")

    print(f"\nreport: {out_json}  {out_md}")
    print("This proves the pipeline computes U end-to-end. It does NOT show "
          "the detector works on real traffic:")
    print("the generator and the detector were written by the same hand. "
          "The pilot bar (U >= +20%) is a")
    print("different measurement on your shadow log — see README.")


if __name__ == "__main__":
    main()
