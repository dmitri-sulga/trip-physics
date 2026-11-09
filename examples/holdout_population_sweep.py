"""Held-out FP check across 8 population seeds, plus its bootstrap intervals.

Two more checks on the held-out false-positive rate, both quoted in
docs/method.md:

1. the same single-split check (split seed 0) on population seeds 1..8, to
   see whether the committed population (seed 1) is typical;
2. `eval.holdout_fp_budget_ci` on population seed 1 for each scorer.

Usage: python examples/holdout_population_sweep.py   (about 10 minutes,
one 20,000-account population in memory at a time)
"""
import statistics

from tripphysics.eval import holdout_fp_budget_ci, holdout_fp_check, rank_normalize

try:
    from holdout_split_sweep import scored_population
except ImportError:  # pragma: no cover
    from examples.holdout_split_sweep import scored_population


def compute(n_seeds: int = 8) -> dict:
    rows, intervals = [], {}
    for seed in range(1, n_seeds + 1):
        labels, baseline, detector = scored_population(20_000, seed)
        r = holdout_fp_check(labels, baseline, detector, seed=0)
        rows.append({"seed": seed,
                     "baseline": r["realized_fp_per_10k_baseline"],
                     "stacked": r["realized_fp_per_10k_pilot"]})
        if seed == 1:
            br, dr = rank_normalize(baseline), rank_normalize(detector)
            stacked = {a: max(br[a], dr[a]) for a in labels}
            for name, scores in (("baseline", baseline), ("stacked", stacked)):
                intervals[name] = holdout_fp_budget_ci(labels, scores, seed=0)
    stacked_values = [r["stacked"] for r in rows]
    return {"rows": rows, "intervals": intervals,
            "stacked_median": statistics.median(stacked_values),
            "stacked_max_seed": rows[stacked_values.index(max(stacked_values))]["seed"],
            "stacked_not_above_baseline": sum(r["stacked"] <= r["baseline"] for r in rows)}


def main() -> None:
    out = compute()
    for r in out["rows"]:
        print(f"population seed {r['seed']}: baseline {r['baseline']:.1f}/10k  "
              f"stacked {r['stacked']:.1f}/10k")
    print(f"stacked over {len(out['rows'])} population seeds: median "
          f"{out['stacked_median']:.1f}/10k, highest at seed "
          f"{out['stacked_max_seed']}, at or below the baseline on "
          f"{out['stacked_not_above_baseline']} of {len(out['rows'])}")
    for name, ci in out["intervals"].items():
        print(f"bootstrap interval, population seed 1, {name}: point "
              f"{ci['realized_fp_per_10k']:.1f}/10k  [{ci['ci_low']:.1f}, "
              f"{ci['ci_high']:.1f}]  ({ci['n_resamples']} resamples)")


if __name__ == "__main__":
    main()
