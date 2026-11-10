# trip-physics

A detector for shared driver accounts (one account, several people driving on
it), built from trip physics, and a harness that measures what such a detector
adds on top of an existing risk score at a fixed false-positive budget.

- **All data here is synthetic.** It comes from the generator in
  `src/tripphysics/synth.py`, from fixed seeds. The generator and the detector
  were written by the same person, so on this data the detector is right by
  construction. The numbers below check the measurement, not the detector.
- **The detector** turns a trip log into eleven features (impossible jumps,
  overlapping trips, an account "teleporting" between trips, mock-location
  flags, GPS against the serving cell, two driving styles on one account, and
  others) and combines them in a weighted sum with named constants. No fitted
  model.
- **The metric, U**, is the relative recall gain of the existing score stacked
  with the detector over the existing score alone, with both held to the same
  realized false-positive spend (default 50 per 10,000 honest accounts).
- **The attack registry** is a written record of 41 attacks and harder sharing
  scenarios with measured verdicts; 11 of them can be re-run from this
  repository.
- Python 3.11+, numpy as the only runtime dependency. PR-AUC, thresholding and
  U are written out by hand in `src/tripphysics/eval.py`.

## Results, on synthetic data

| what                                                     | value                                                            | command                                          |
| -------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------ |
| U, median of 20 seeds, 20,000 accounts each              | +15.5% (range +12.9% … +19.9%)                                   | `python examples/run_demo.py --full`             |
| worked example, seed 1                                   | U +14.3%, 95% CI [+9.6%, +18.5%]; PR-AUC 0.61 → 0.73             | same run, [`out/report.md`](out/report.md)       |
| held-out false-positive rate of the stacked scorer, 20 splits | median 49.5 per 10,000 against a budget of 50, range 24.7 … 70.1 | `python examples/holdout_split_sweep.py`         |
| attack registry                                          | 41 entries: 18 caught, 12 partial, 6 pass, 5 unmeasured           | `python -m tripphysics.attacks`                  |
| of which                                                 | 23 telemetry manipulations, 13 simulator settings, 5 unmeasured   | same                                             |
| re-runnable registry entries                             | 11 of 41, each reproducing its recorded count                     | `python -m tripphysics.attack_measure`           |
| test suite                                               | 555 passed, 1 xfailed                                             | `python -m pytest -q`                            |

**What +15.5% is.** A check that the measurement works, on a simulator where
the detector is right by construction, against a deliberately crude baseline
(`NaiveBaseline`). It is not an estimate of what the detector would add on real
data. The one expected test failure pins an open defect in the hand-written
PR-AUC (see [docs/method.md](docs/method.md#limits)).

## What CI verifies

On every push ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

- the test suite on Python 3.11 and 3.14; the golden metrics are asserted to
  1e-9, so a numpy build that moves one fails the job;
- the `--quick` demo, run twice, with byte-identical reports;
- `pip-audit` on every dependency installed from PyPI, and `bandit` on `src/`.

## What is not verified

- Anything about real traffic. No real trip, driver or company data was used.
- The 20-seed `--full` sweep and the held-out sweeps are too slow for CI.
  They are re-run by hand.
- 25 of the 36 measured registry entries. Their verdicts were measured with
  one-off scripts that are not in this repository; for those entries the
  registry is a record of what was run, with the population and seed stated.
  The 13 simulator-setting entries need generator parameters that
  `generate_dataset` does not expose.
- The Docker image is not built in CI.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python examples/run_demo.py --quick      # smoke test
python -m pytest -q
```

`python examples/run_demo.py --full` reproduces the committed report; it holds
one 20,000-account population per worker process, so on a small machine add
`--workers 2`. Timings, memory and Docker: [docs/running.md](docs/running.md).

To measure an existing score column against outcomes, without trip data:

```bash
python -m tripphysics.byos --scores examples/byos_demo_scores.csv \
  --baseline-scores examples/byos_demo_baseline.csv --out out/byos
```

Walkthrough: [docs/byos-guide.md](docs/byos-guide.md).

## Where the rest is

| file                                             | what it holds                                                                 |
| ------------------------------------------------ | ----------------------------------------------------------------------------- |
| [docs/method.md](docs/method.md)                 | the metric, stacking, reporting rules, sample size, held-out check, limits     |
| [docs/running.md](docs/running.md)               | install, which run to quote, timings, memory, Docker, the numbers check       |
| [docs/byos-guide.md](docs/byos-guide.md)         | bring your own scores                                                         |
| [attacks/ATTACKS.md](attacks/ATTACKS.md)         | the attack registry, generated from `attacks/registry.json`                   |
| [schema/input_schema.md](schema/input_schema.md) | the ten-field input contract for real trip logs                               |
| [CHANGELOG.md](CHANGELOG.md)                     | defects this harness had, and what each did to the numbers                    |
| `src/tripphysics/`                               | synth, features, detector, baseline, eval, report, byos, attacks, attack_measure |
| `examples/`                                      | the demo and the scripts behind every table in the docs                       |
| `tests/`                                         | hand-computed metric cases, golden values, regression tests for past defects  |

## How it was built and published

Developed July–November 2026 in a private working repository and published
here as a snapshot in November 2026: the first commit is that snapshot, and
the commits after it are corrections made in public. The CHANGELOG dates come
from the private history. Single author.

---

Dmitri Sulga · Tallinn · <europeworktallinn@gmail.com> · MIT
