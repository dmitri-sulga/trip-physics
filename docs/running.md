# Running it

## Install and the demo

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python examples/run_demo.py
```

Writes `./out/report.json` and `./out/report.md` and prints the metric table.
Everything is seeded; two runs produce byte-identical reports.

The virtualenv is needed on macOS with Homebrew Python and on current
Debian/Ubuntu, where a plain `pip install` into the system interpreter is
refused (PEP 668, `externally-managed-environment`). Python 3.11 or newer is
required; CI runs 3.11 (the version the container pins) and 3.14.

## Which run to quote

| command                               | population | seeds | what it is for                      |
| ------------------------------------- | ---------- | ----- | ----------------------------------- |
| `python examples/run_demo.py --full`  | 20,000     | 20    | the committed report, `out/report.md` |
| `python examples/run_demo.py`         | 20,000     | 3     | a faster sample of the same thing   |
| `python examples/run_demo.py --quick` | 2,000      | 3     | smoke test only; U is noisy here    |

`--accounts N` and `--seeds K` override the population and sweep size, and
`--workers W` caps the number of worker processes (each holds one population,
so it is the memory knob).

The figure quoted in the README, median U = +15.5%, is the 20-seed `--full`
sweep. The default command runs 3 seeds, so its median lands elsewhere in the
+12.9% … +19.9% range, and a single seed moves more again: the worked example
in the committed report is seed 1 at +14.3%. The demo prints all three side by
side under "WHICH NUMBER IS WHICH".

Every run overwrites `out/report.json` and `out/report.md`, which are committed
holding the 20-seed sweep. `git checkout -- out/` puts them back.

## Time and memory

Measured with `/usr/bin/time` on 2026-08-04, on an idle 14-core Apple-silicon
laptop with 36 GiB of RAM, Python 3.14.5 and numpy 2.5.1, from a clean
`git archive HEAD` checkout into a fresh virtualenv:

| command                               | runtime | peak RSS  |
| ------------------------------------- | ------- | --------- |
| `python examples/run_demo.py`         | 123 s   | 5.0 GiB   |
| `python examples/run_demo.py --full`  | 542 s   | not taken |
| `python examples/run_demo.py --quick` | 17 s    | 0.55 GiB  |
| `pytest`                              | 132 s   | not taken |

Re-measured on 2026-11-10 on a 14-core, 36 GiB Apple-silicon laptop that was
busy with other work, Python 3.15.2 and numpy 2.5.4:
`run_demo.py --full --workers 2` took 1,211 s and 1,844 s on two runs, with a
peak RSS of 5.0 GiB, and
`python -m pytest -q` took 185 s. Treat all of these as an order of
magnitude and measure on your own machine.

Memory is the constraint, not time: the sweep holds one 20,000-account
population per worker process. On an 8 GiB machine use `--workers 2`,
`--quick`, or a lower `--accounts`.

The table was first measured in July and read four to eight times higher
(989 s for the default run, 48 s for `--quick`, about 11 minutes for the
suite). Python 3.12 with the same numpy ran `--quick` in 11.9 s, so the
interpreter is not the explanation; the most likely cause is that the July
reading was taken while the machine was busy. Both readings are kept rather
than the newer one quietly replacing the older.

## Docker

```bash
docker build -t trip-physics .
mkdir -p out
docker run --rm --network=none -v "$PWD/out:/app/out" trip-physics
```

The harness makes no network calls at runtime, and running it with egress
blocked enforces that rather than asserting it. The image pins
`python:3.11.14-slim` and runs the default demo. It does not include
`attacks/`, so `python -m tripphysics.attacks` and
`python -m tripphysics.attack_measure` are for a source checkout.

## Tests

```bash
python -m pytest -q
```

`python -m pytest` rather than a bare `pytest`, so the tests run against the
interpreter the package was installed into. `tests/test_eval_math.py` doubles
as documentation: an 8-account fixture with hand-computed R0, R1 and U
asserted exactly.

## Checking the numbers in the documentation

```bash
bash scripts/check-claimed-numbers.sh          # the suite plus seconds
bash scripts/check-claimed-numbers.sh --slow   # adds the sweeps; under an hour
```

The fast mode runs the test suite and checks every number the README states
that can be read from committed artifacts (the report, the attack registry)
or computed in seconds. `--slow` also re-runs the scripts behind the held-out
tables, the sample-size table and the re-runnable attack entries, and fails
if any figure in the README or in `docs/method.md` differs from what they
print.
