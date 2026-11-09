"""Does the held-out false-positive spread shrink as the honest population grows?

Builds one 20,000-account population (seed 1), then repeats the held-out FP
check over 200 split seeds on nested subsets of its honest accounts (every
shared account is kept) and compares the observed relative spread with the
counting-noise prediction sqrt(2 / expected_flags): Poisson noise in the
measured half plus the same amount again from fitting the threshold on the
other half. Source of the spread table in docs/method.md.

Usage: python examples/holdout_spread_vs_n.py      (about 90 s)
"""
import math
import statistics

from tripphysics.eval import holdout_fp_check

try:  # run as a script from the repository root or from examples/
    from holdout_split_sweep import scored_population
except ImportError:  # pragma: no cover
    from examples.holdout_split_sweep import scored_population

HONEST_SIZES = (2_425, 4_850, 9_700, 19_400)
N_SPLITS = 200
BUDGET = 50.0


def compute() -> list[dict]:
    labels, baseline, detector = scored_population(20_000, 1)
    honest = sorted(a for a, s in labels.items() if not s)
    shared = sorted(a for a, s in labels.items() if s)
    rows = []
    for n_honest in HONEST_SIZES:
        keep = set(honest[:n_honest]) | set(shared)
        lab = {a: labels[a] for a in keep}
        bas = {a: baseline[a] for a in keep}
        det = {a: detector[a] for a in keep}
        checks = [holdout_fp_check(lab, bas, det, seed=s) for s in range(N_SPLITS)]
        n_meas = checks[0]["n_measurement"]
        flags = n_meas * BUDGET / 10_000
        row = {"honest": n_honest, "n_measurement": n_meas,
               "expected_flags": flags,
               "predicted_rel_sd": math.sqrt(2 / flags)}
        for name, key in (("stacked", "pilot"), ("baseline", "baseline")):
            v = [c[f"realized_fp_per_10k_{key}"] for c in checks]
            mean, sd = statistics.mean(v), statistics.pstdev(v)
            row[name] = {"mean": mean, "sd": sd, "rel_sd": sd / mean}
        rows.append(row)
    return rows


def main() -> None:
    print("| honest accounts | in the measured half | expected flags | "
          "relative spread, stacked | relative spread, baseline | "
          "sqrt(2 / expected flags) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for r in compute():
        print(f"| {r['honest']:,} | {r['n_measurement']:,} | "
              f"{r['expected_flags']:.1f} | {r['stacked']['rel_sd'] * 100:.1f}% | "
              f"{r['baseline']['rel_sd'] * 100:.1f}% | "
              f"{r['predicted_rel_sd'] * 100:.1f}% |")
    print(f"({N_SPLITS} split seeds per row; population seed 1, 20,000 accounts)")
    # Under sqrt(2 / expected flags), with half the honest accounts measured
    # at 50/10k, the relative spread is sqrt(800 / honest accounts).
    for target in (0.10, 0.05):
        print(f"honest accounts for a {target:.0%} relative spread under "
              f"sqrt(2 / expected flags): {honest_needed(target):,.0f}")


def honest_needed(rel_spread: float) -> float:
    """Honest accounts at which sqrt(2 / expected flags) equals rel_spread."""
    flags = 2 / rel_spread ** 2
    return flags / (BUDGET / 10_000) * 2


if __name__ == "__main__":
    main()
