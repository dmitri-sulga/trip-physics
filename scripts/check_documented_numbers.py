"""Check that every number the documentation states is what the code produces.

Called by scripts/check-claimed-numbers.sh. Each check computes a figure (from
a committed artifact, from the registry, or by running the script that the
documentation names) and asserts that the documentation states exactly that
figure, formatted as the documentation formats it. A number that drifts from
its source fails here instead of waiting for a reader to re-run it.

Usage:
    python scripts/check_documented_numbers.py          # seconds
    python scripts/check_documented_numbers.py --slow   # adds the sweeps (~1 hour)

Fast mode reads out/report.json (the committed --full sweep) and the attack
registry, runs the bring-your-own-scores example and one 2,000-account seed.
Slow mode also runs examples/holdout_split_sweep.py,
examples/holdout_spread_vs_n.py, examples/holdout_population_sweep.py,
examples/sample_size_table.py and tripphysics.attack_measure.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples"))
os.chdir(ROOT)

failures: list[str] = []
checked = 0


def _norm(text: str) -> str:
    """Drop Markdown emphasis and collapse whitespace, so table padding and
    line wrapping do not matter but every character of a number does."""
    return re.sub(r"\s+", " ", text.replace("**", "").replace("_", " "))


_cache: dict[str, str] = {}


def doc(path: str) -> str:
    if path not in _cache:
        with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
            _cache[path] = _norm(fh.read())
    return _cache[path]


def expect(path: str, text: str) -> None:
    """Assert that `path` states `text` (after normalisation)."""
    global checked
    checked += 1
    if _norm(text) not in doc(path):
        failures.append(f"{path}: expected to find {text!r}")


def pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


# ---------------------------------------------------------------- fast checks

def check_committed_report() -> None:
    with open("out/report.json", encoding="utf-8") as fh:
        report = json.load(fh)
    meta, metrics = report["meta"], report["metrics"]
    sweep, ci = meta["seed_sweep"], meta["bootstrap_ci"]
    headline = (f"{pct(sweep['median'])} (range {pct(sweep['min'])} … "
                f"{pct(sweep['max'])})")
    expect("README.md", headline)
    expect("README.md", f"U {pct(metrics['U'])}, 95% CI [{pct(ci['ci_low'])}, "
                        f"{pct(ci['ci_high'])}]; PR-AUC "
                        f"{metrics['pr_auc_baseline']:.2f} → {metrics['pr_auc_pilot']:.2f}")
    expect("docs/method.md", f"| 20,000 | 600 | 20 | {pct(sweep['median'])} | "
                             f"{sweep['sd'] * 100:.1f}pp | {pct(sweep['min'])} … "
                             f"{pct(sweep['max'])} |")
    expect("docs/method.md", f"The synthetic median ({pct(sweep['median'])})")
    expect("docs/running.md", f"median U = {pct(sweep['median'])}")
    expect("docs/running.md", f"{pct(sweep['min'])} … {pct(sweep['max'])} range")
    expect("docs/running.md", f"seed 1 at {pct(metrics['U'])}")
    expect("CHANGELOG.md", f"{pct(sweep['median'])}, range {pct(sweep['min'])} … "
                           f"{pct(sweep['max'])}")
    fp = meta["holdout_fp_check"]
    expect("docs/method.md", f"{fp['realized_fp_per_10k_baseline']:.1f}/10k for the "
                             f"baseline and {fp['realized_fp_per_10k_pilot']:.1f}/10k "
                             f"for the stacked scorer")
    expect("docs/method.md", f"leaves {fp['n_measurement']:,} honest accounts in "
                             f"the measurement half")


def check_registry() -> None:
    from tripphysics.attacks import kind_counts, load_registry, verdict_counts

    reg = load_registry()
    v, k = verdict_counts(reg), kind_counts(reg)
    total = len(reg["attacks"])
    rerun = sum(1 for a in reg["attacks"] if "reproduce" in a)
    measured = total - v["unmeasured"]
    expect("README.md", f"{total} entries: {v['caught']} caught, {v['partial']} "
                        f"partial, {v['passes']} pass, {v['unmeasured']} unmeasured")
    expect("README.md", f"{k['telemetry']} telemetry manipulations, "
                        f"{k['simulator']} simulator settings, {k['unmeasured']} unmeasured")
    expect("README.md", f"{rerun} of {total}, each reproducing its recorded count")
    expect("README.md", f"a written record of {total} attacks")
    expect("README.md", f"{rerun} of them can be re-run")
    expect("README.md", f"{measured - rerun} of the {measured} measured registry entries")
    expect("README.md", f"The {k['simulator']} simulator-setting entries")
    expect("CHANGELOG.md", f"re-runs the {rerun} entries")


def check_byos_example() -> None:
    from tripphysics import byos

    buf = io.StringIO()
    out_dir = os.path.join(ROOT, "out", "byos-check")
    with redirect_stdout(buf):
        rc = byos.main(["--scores", "examples/byos_demo_scores.csv",
                        "--baseline-scores", "examples/byos_demo_baseline.csv",
                        "--out", out_dir])
    if rc != 0:
        failures.append(f"byos example exited {rc}")
        return
    printed = buf.getvalue().replace(out_dir, "out/byos")
    expect("docs/byos-guide.md", printed.strip())
    for ext in (".json", ".md"):
        if os.path.exists(out_dir + ext):
            os.remove(out_dir + ext)


def check_lucky_seed() -> None:
    from sample_size_table import uplift

    u = uplift((2_000, 42))
    expect("CHANGELOG.md", f"U = {pct(u)}")
    expect("tests/test_golden_metrics.py", f'"U": {u!r}')


# ---------------------------------------------------------------- slow checks

def check_holdout_split_sweep() -> None:
    import holdout_split_sweep

    r = holdout_split_sweep.compute()
    b, s = r["baseline"], r["stacked"]
    expect("README.md", f"median {s['median']:.1f} per 10,000 against a budget of "
                        f"{r['budget']:g}, range {s['min']:.1f} … {s['max']:.1f}")
    expect("docs/method.md", f"| baseline | 50 / 10k | {b['median']:.1f} / 10k | "
                             f"{b['min']:.1f} … {b['max']:.1f} / 10k |")
    expect("docs/method.md", f"| stacked | 50 / 10k | {s['median']:.1f} / 10k | "
                             f"{s['min']:.1f} … {s['max']:.1f} / 10k |")
    expect("docs/method.md", f"`seed=0` value of {s['split_seed_0']:.1f} is the "
                             f"highest of the 20 splits — the next one down is "
                             f"{s['second_highest']:.1f}")
    if s["split_seed_0"] != s["max"]:
        failures.append("split seed 0 is no longer the highest of the 20 splits")


def check_holdout_spread() -> None:
    import holdout_spread_vs_n

    rows = holdout_spread_vs_n.compute()
    first, last = rows[0], rows[-1]
    phrase = (f"{first['stacked']['rel_sd'] * 100:.1f}% at {first['honest']:,} "
              f"honest accounts")
    expect("docs/method.md", f"falls from {phrase} to "
                             f"{last['stacked']['rel_sd'] * 100:.1f}% at {last['honest']:,}")
    expect("CHANGELOG.md", f"{phrase} down to "
                           f"{last['stacked']['rel_sd'] * 100:.1f}% at {last['honest']:,}")
    n10 = holdout_spread_vs_n.honest_needed(0.10)
    n05 = holdout_spread_vs_n.honest_needed(0.05)
    expect("docs/method.md", f"about {n10:,.0f} honest accounts for ±10% and "
                             f"{n05:,.0f} for ±5%")


def check_holdout_population_sweep() -> None:
    import holdout_population_sweep

    r = holdout_population_sweep.compute()
    words = {5: "five", 8: "eight"}
    n = len(r["rows"])
    expect("docs/method.md", f"over {n} population seeds")
    expect("docs/method.md", f"the median is {r['stacked_median']:.1f}/10k")
    expect("docs/method.md", f"on {words.get(r['stacked_not_above_baseline'], r['stacked_not_above_baseline'])} "
                             f"of the {words.get(n, n)}")
    if r["stacked_max_seed"] != 1:
        failures.append("population seed 1 is no longer the highest of the 8")
    b, s = r["intervals"]["baseline"], r["intervals"]["stacked"]
    expect("docs/method.md", f"overlap on {s['ci_low']:.1f} … {b['ci_high']:.1f} — "
                             f"baseline [{b['ci_low']:.1f}, {b['ci_high']:.1f}], "
                             f"stacked [{s['ci_low']:.1f}, {s['ci_high']:.1f}]")


def check_sample_size_table() -> None:
    import sample_size_table

    rows = sample_size_table.compute()
    for r in rows:
        expect("docs/method.md", sample_size_table.format_row(r))
    small = rows[0]
    rng = f"{pct(small['min'])} to {pct(small['max'])}"
    expect("docs/method.md", f"anything from {rng}")
    expect("CHANGELOG.md", f"ranges from {rng}")
    expect("src/tripphysics/report.py", f"from {rng}")
    expect("examples/run_demo.py", f"range from {rng}")
    medians = [r["median"] for r in rows]
    expect("docs/method.md", f"it stays between {pct(min(medians))} and {pct(max(medians))}")


def check_attack_reruns() -> None:
    global checked
    from tripphysics.attack_measure import (DECOMPOSITION, MANIPULATIONS,
                                            _chain, measure, reference)
    from tripphysics.attacks import load_registry

    entries = {a["id"]: a for a in load_registry()["attacks"]}
    ref = reference()
    expect("attacks/registry.json", f"{ref['threshold']:.6f}")
    expect("attacks/registry.json", f"{ref['caught']}/{len(ref['accounts'])} "
                                    f"({ref['caught'] / len(ref['accounts']) * 100:.1f}%)")
    for attack_id in MANIPULATIONS:
        r = measure(attack_id, ref)
        recorded = entries[attack_id]["evidence"].split(" ", 1)[0]
        if recorded != f"{r['caught']}/{r['of']}":
            failures.append(f"attack {attack_id}: registry records {recorded}, "
                            f"re-run gives {r['caught']}/{r['of']}")
        checked += 1
    evidence = entries["full-client-evasion"]["evidence"]
    for label, steps in DECOMPOSITION:
        d = measure("full-client-evasion", ref, _chain(steps))
        if f"{d['caught']}/{d['of']}" not in evidence:
            failures.append(f"full-client-evasion decomposition '{label}': "
                            f"{d['caught']}/{d['of']} not in the evidence text")
        checked += 1


def main() -> int:
    slow = "--slow" in sys.argv[1:]
    steps = [check_committed_report, check_registry, check_byos_example,
             check_lucky_seed]
    if slow:
        steps += [check_attack_reruns, check_holdout_split_sweep,
                  check_holdout_spread, check_sample_size_table,
                  check_holdout_population_sweep]
    for step in steps:
        print(f"-> {step.__name__}", flush=True)
        step()
    if failures:
        print(f"x {len(failures)} of {checked} documented numbers do not match:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"ok: {checked} documented numbers match their source"
          + ("" if slow else " (fast mode; --slow adds the sweeps)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
