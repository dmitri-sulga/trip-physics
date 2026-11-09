"""Held-out false-positive check repeated over 20 split seeds.

`run_demo.py` reports one split (seed 0). One split of 9,700 honest accounts
moves by about +/-20%, so this script repeats the same check over split seeds
0..19 on the same population (the committed report's population: 20,000
accounts, seed 1) and prints the median and the range for each scorer. It is
the source of the held-out table in docs/method.md and the README.

In code and JSON keys the stacked scorer is called `pilot`.

Usage: python examples/holdout_split_sweep.py [n_accounts] [population_seed] [n_splits]
Defaults: 20000 1 20. About 80 s, one population in memory.
"""
import statistics
import sys

from tripphysics.baseline import NaiveBaseline
from tripphysics.detector import score_account
from tripphysics.eval import holdout_fp_check
from tripphysics.features import extract_features
from tripphysics.synth import generate_dataset


def scored_population(n_accounts: int, seed: int):
    accounts = generate_dataset(n_accounts=n_accounts, shared_fraction=0.03,
                                seed=seed)["accounts"]
    labels = {a["account_id"]: a["is_shared"] for a in accounts}
    naive = NaiveBaseline()
    baseline = {a["account_id"]: naive.score(a) for a in accounts}
    detector = {a["account_id"]: score_account(extract_features(a))
                for a in accounts}
    return labels, baseline, detector


def compute(n_accounts: int = 20_000, population_seed: int = 1,
            n_splits: int = 20) -> dict:
    labels, baseline, detector = scored_population(n_accounts, population_seed)
    rows = [holdout_fp_check(labels, baseline, detector, seed=s)
            for s in range(n_splits)]
    out = {"budget": rows[0]["budget_fp_per_10k"],
           "n_measurement": rows[0]["n_measurement"]}
    for name, key in (("baseline", "baseline"), ("stacked", "pilot")):
        values = [r[f"realized_fp_per_10k_{key}"] for r in rows]
        ordered = sorted(values)
        out[name] = {"values": values,
                     "median": statistics.median(values),
                     "min": ordered[0], "max": ordered[-1],
                     "second_highest": ordered[-2],
                     "split_seed_0": values[0]}
    return out


def main() -> None:
    n_accounts = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
    population_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n_splits = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    out = compute(n_accounts, population_seed, n_splits)
    print(f"{n_accounts} accounts, population seed {population_seed}, "
          f"{n_splits} splits, {out['n_measurement']} honest accounts measured "
          f"per split, budget {out['budget']:g}/10k")
    for name in ("baseline", "stacked"):
        r = out[name]
        print(f"{name:8s} median {r['median']:.1f}/10k   "
              f"range {r['min']:.1f} ... {r['max']:.1f}   "
              f"second highest {r['second_highest']:.1f}   "
              f"split seed 0: {r['split_seed_0']:.1f}")


if __name__ == "__main__":
    main()
