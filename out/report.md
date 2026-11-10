# trip-physics — shadow-evaluation report

```
==============================================================
  HEADLINE   median U = +15.5% across 20 seeds
             (sd 2.0pp, range +12.9% … +19.9%)
  SYNTHETIC DRY-RUN: pipeline verified end-to-end. No pass/fail applies to simulated data.
  The generator and the detector were written by the
  same hand: here the detector is right by construction.
==============================================================
```

> **Two different numbers, do not compare them.**
> 
> **(a) This synthetic demo: U = +15.5% (median of 20 seeds).** It checks only that the pipeline computes the metric correctly end-to-end, on data where the detector's assumptions are true by construction, against a deliberately crude baseline (`NaiveBaseline`). It is not an estimate of what the detector adds on real data.
> 
> **(b) The acceptance bar for real data: U ≥ +20.0% on a real shadow log.** That is a different measurement, on different data, against real confirmed-shared labels. A synthetic number can neither pass nor fail it.

## Headline metrics

Both scorers are measured at the **same realized FP spend**: 97 honest accounts flagged (50.0 per 10k), the largest spend a strict threshold reaches on both scorers without exceeding the 50.0/10k budget (utilization 100%).

| metric | baseline | stacked (baseline + detector) |
|---|---|---|
| recall at realized spend | 0.5600 (R0) | 0.6400 (R1) |
| PR-AUC | 0.6075 | 0.7307 |
| precision@500 | 0.6760 | 0.7820 |
| threshold | 0.349296 | 0.980350 |

- **U = (R1 − R0) / R0 = +14.3%** — relative recall uplift at the same realized FP spend (see above).
- Population: 20000 accounts, 600 shared (ground truth).

## U across FP budgets

A run on real data fixes ONE budget before the measurement starts. This curve shows the whole trade-off around the default, so the operating point is chosen openly, not for a flattering number.

| FP budget / 10k | R0 (baseline) | R1 (stacked) | U |
|---|---|---|---|
| 15 | 0.5550 | 0.5850 | +5.4% |
| 30 | 0.5567 | 0.6083 | +9.3% |
| 50 ← default | 0.5600 | 0.6400 | +14.3% |
| 100 | 0.5667 | 0.6900 | +21.8% |
| 200 | 0.5767 | 0.7383 | +28.0% |

PR-AUC (0.61 → 0.73 above) is budget-independent — the separation is present before any threshold is chosen.

## U across independent seeds

Each seed is an independent synthetic population (20000 accounts, 600 shared). A single seed's U is one draw from this distribution; the headline above is the **median**, not the best draw.

| statistic | U |
|---|---|
| median | +15.5% |
| mean | +16.0% |
| sd | 2.0pp |
| min | +12.9% |
| max | +19.9% |
| seeds | 20 |

The synthetic spread runs +12.9% … +19.9% (sd 2.0pp). The real-data acceptance bar of +20.0% sits above that range on purpose: a bar this simulator clears on its own would be a bar calibrated to the simulator rather than to real traffic.

Per-seed U, every value from the sweep, none dropped:

| seed | U |
|---|---|
| 1 | +14.3% |
| 2 | +16.6% |
| 3 | +18.5% |
| 4 | +15.4% |
| 5 | +15.7% |
| 6 | +14.7% |
| 7 | +14.6% |
| 8 | +13.2% |
| 9 | +18.7% |
| 10 | +19.9% |
| 11 | +17.4% |
| 12 | +13.2% |
| 13 | +15.4% |
| 14 | +12.9% |
| 15 | +14.9% |
| 16 | +14.8% |
| 17 | +17.4% |
| 18 | +15.9% |
| 19 | +18.2% |
| 20 | +17.9% |

## Bootstrap confidence interval

Stratified percentile bootstrap on the displayed seed, 1000 resamples, accounts resampled with replacement while holding the shared/honest counts fixed. The threshold is re-derived on every replicate, so threshold-selection variance is inside the interval rather than hidden behind it.

- **U = +14.3%, 95% CI [+9.6%, +18.5%]**

## Is the FP budget promise kept out-of-sample?

The threshold is chosen on the same honest accounts the FP rate is then measured on, so in-sample the budget is hit by construction — a tautology, not a measurement. This check derives the threshold on 9700 honest accounts and reports the FP rate realized on the 9700 held-out ones.

| scorer | budget | realized on held-out honest accounts |
|---|---|---|
| baseline | 50 / 10k | 46.4 / 10k |
| stacked | 50 / 10k | 70.1 / 10k |

The gap is the honest error bar on the budget promise. It shrinks as the honest population grows and widens sharply on small samples.

## How to read U

R0 is the baseline's recall on shared accounts when its threshold is set to spend exactly the agreed FP budget on honest accounts. R1 is the same measurement for the stacked scorer — the baseline stacked with the detector via `max(rank-normalized baseline, rank-normalized detector)` — at the *same* budget. U is the relative improvement. Holding the FP budget fixed is the point: any scorer can raise recall by flagging more people; U only rewards recall bought without additional honest-account cost. U can be negative if the detector adds noise instead of signal.

## Run parameters

| parameter | value |
|---|---|
| bootstrap_ci | {'u_point': 0.14285714285714277, 'ci_low': 0.0961819653365291, 'ci_high': 0.1849587832346454, 'n_resamples': 1000, 'confidence': 0.95, 'n_degenerate': 0, 'degeneracy_rate': 0.0, 'ci_valid': True} |
| budget_curve | [{'budget_fp_per_10k': 15.0, 'R0': 0.555, 'R1': 0.585, 'U': 0.0540540540540539, 'default': False}, {'budget_fp_per_10k': 30.0, 'R0': 0.5566666666666666, 'R1': 0.6083333333333333, 'U': 0.09281437125748498, 'default': False}, {'budget_fp_per_10k': 50.0, 'R0': 0.56, 'R1': 0.64, 'U': 0.14285714285714277, 'default': True}, {'budget_fp_per_10k': 100.0, 'R0': 0.5666666666666667, 'R1': 0.69, 'U': 0.21764705882352936, 'default': False}, {'budget_fp_per_10k': 200.0, 'R0': 0.5766666666666667, 'R1': 0.7383333333333333, 'U': 0.2803468208092485, 'default': False}] |
| display_seed | 1 |
| holdout_fp_check | {'budget_fp_per_10k': 50.0, 'calibration_fraction': 0.5, 'n_calibration': 9700, 'n_measurement': 9700, 'realized_fp_per_10k_baseline': 46.391752577319586, 'realized_fp_per_10k_pilot': 70.10309278350516} |
| n_accounts | 20000 |
| seed | 1 |
| seed_sweep | {'median': 0.1554645883236209, 'mean': 0.15979529640871906, 'sd': 0.02000958738691707, 'min': 0.12923076923076934, 'max': 0.1986754966887419, 'n_seeds': 20, 'n_seeds_supplied': 20, 'n_seeds_dropped_nonfinite': 0, 'target': 0.2, 'n_at_or_above_target': 0, 'fraction_at_or_above_target': 0.0} |
| seed_sweep_values | [{'seed': 1, 'U': 0.14285714285714277}, {'seed': 2, 'U': 0.16566265060240964}, {'seed': 3, 'U': 0.1847133757961784}, {'seed': 4, 'U': 0.1536050156739812}, {'seed': 5, 'U': 0.1566265060240964}, {'seed': 6, 'U': 0.14696485623003216}, {'seed': 7, 'U': 0.1464968152866242}, {'seed': 8, 'U': 0.1323529411764707}, {'seed': 9, 'U': 0.1873015873015872}, {'seed': 10, 'U': 0.1986754966887419}, {'seed': 11, 'U': 0.17350157728706628}, {'seed': 12, 'U': 0.13173652694610785}, {'seed': 13, 'U': 0.1543026706231454}, {'seed': 14, 'U': 0.12923076923076934}, {'seed': 15, 'U': 0.14925373134328365}, {'seed': 16, 'U': 0.14826498422712941}, {'seed': 17, 'U': 0.17405063291139244}, {'seed': 18, 'U': 0.159159159159159}, {'seed': 19, 'U': 0.18181818181818185}, {'seed': 20, 'U': 0.17933130699088135}] |
| shared_fraction | 0.03 |
| synthetic | True |

## Honest limits

- **Synthetic data proves the harness works, not that the detector works on real traffic.** The synthetic sharing patterns were designed alongside the detector; on real trips both the base rates and the evasion behavior will differ. Measuring that needs a shadow run on real logs with confirmed labels.
- The bundled `NaiveBaseline` is a placeholder, not a model of any production stack. Uplift over it is not uplift over a real scoring system; plug in real scores via the `scores_csv` adapter before reading anything into U.
- **The threshold is calibrated on the same honest population it is measured on.** In-sample the realized FP rate therefore matches the budget almost exactly by construction; out-of-sample it does not. The held-out check above sizes that gap instead of assuming it away. On real data, calibrate on a window that precedes the measurement window.
- The threshold-at-budget procedure assumes the honest-score distribution is roughly stationary between the calibration window and the measurement window. Drift (app releases, seasonal patterns) moves the effective FP spend.
- A single seed's U is one draw from a distribution, and at small sample sizes that distribution is wide. Any headline quoting one seed is reporting luck as much as signal — which is why the headline here is a median across seeds.
- Rank-based stacking is evaluated offline over a closed population. A live deployment needs an online calibration step, since ranks are only defined relative to a batch.
- On real data, "shared" labels come from confirmed cases and are incomplete. Recall against known-shared accounts is a lower-bound proxy, and both scorers inherit the same label bias.

---

*trip-physics — shadow-evaluation harness · report rendered deterministically from the run above (no timestamps by design).*
