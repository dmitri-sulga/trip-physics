"""How much does U move between seeds at each population size?

Source of the sample-size table in docs/method.md. For each population size
it builds seeds 1..K exactly as `run_demo.py` does (3% shared, NaiveBaseline
against the stacked scorer, default 50/10k budget), computes U per seed and
summarizes with `eval.summarize_across_seeds` (median, sample sd, range).

The 20,000-account row is the committed `--full` sweep: by default it is read
from out/report.json rather than recomputed, because recomputing it means 20
populations of 20,000 accounts. Pass --recompute-20000 to rebuild it here.

Usage:
    python examples/sample_size_table.py [--workers N] [--recompute-20000]

Runs one population per worker process; the default is one worker, which
keeps peak memory near one population (well under 1 GiB at 8,000 accounts).
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

from tripphysics.baseline import NaiveBaseline
from tripphysics.detector import score_account
from tripphysics.eval import evaluate, summarize_across_seeds
from tripphysics.features import extract_features
from tripphysics.synth import generate_dataset

SHARED_FRACTION = 0.03
ROWS = ((2_000, 25), (5_000, 10), (8_000, 10))
REPORT_JSON = os.path.join(os.path.dirname(__file__), "..", "out", "report.json")


def uplift(args) -> float:
    """U for one synthetic population, computed the way run_demo.py does."""
    n_accounts, seed = args
    accounts = generate_dataset(n_accounts=n_accounts,
                                shared_fraction=SHARED_FRACTION,
                                seed=seed)["accounts"]
    labels = {a["account_id"]: a["is_shared"] for a in accounts}
    naive = NaiveBaseline()
    baseline = {a["account_id"]: naive.score(a) for a in accounts}
    detector = {a["account_id"]: score_account(extract_features(a))
                for a in accounts}
    return evaluate(labels, baseline, detector)["U"]


def sweep(n_accounts: int, n_seeds: int, workers: int = 1) -> dict:
    jobs = [(n_accounts, s) for s in range(1, n_seeds + 1)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            values = list(pool.map(uplift, jobs))
    else:
        values = [uplift(j) for j in jobs]
    out = summarize_across_seeds(values)
    out["values"] = values
    return out


def committed_20000() -> dict:
    with open(REPORT_JSON, encoding="utf-8") as fh:
        meta = json.load(fh)["meta"]
    if meta.get("n_accounts") != 20_000 or meta["seed_sweep"]["n_seeds"] != 20:
        raise SystemExit("out/report.json does not hold the committed --full "
                         "sweep; run `git checkout -- out/` or --recompute-20000")
    return meta["seed_sweep"]


def compute(workers: int = 1, recompute_20000: bool = False) -> list[dict]:
    rows = []
    for n_accounts, n_seeds in ROWS:
        rows.append({"accounts": n_accounts, "seeds": n_seeds,
                     **sweep(n_accounts, n_seeds, workers)})
    big = sweep(20_000, 20, workers) if recompute_20000 else committed_20000()
    rows.append({"accounts": 20_000, "seeds": 20, **big})
    return rows


def format_row(r: dict) -> str:
    positives = round(r["accounts"] * SHARED_FRACTION)
    return (f"| {r['accounts']:,} | {positives} | {r['seeds']} | "
            f"{r['median'] * 100:+.1f}% | {r['sd'] * 100:.1f}pp | "
            f"{r['min'] * 100:+.1f}% … {r['max'] * 100:+.1f}% |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--recompute-20000", action="store_true")
    args = parser.parse_args()
    rows = compute(args.workers, args.recompute_20000)
    print("| accounts | positives | seeds | median U | sd of U | range |")
    print("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        print(format_row(r))
    if not args.recompute_20000:
        print("(20,000 row read from out/report.json, the committed --full sweep)")


if __name__ == "__main__":
    main()
