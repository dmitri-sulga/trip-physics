# Attack registry

GENERATED FILE -- do not edit. Rendered from `attacks/registry.json` by
`tripphysics.attacks.render_markdown`; `tests/test_attacks.py` asserts this
file is byte-identical to a fresh render.

This is the adversarial half of the harness, written as a registry: each entry records a manipulation or scenario, what it targets, and the measured verdict about whether the scoring in `src/tripphysics/detector.py` survives it. Entries are of two kinds: manipulations the owner of an account can apply to their own telemetry, and ways of sharing an account that need no manipulation at all (the `tier-*`, `mode-*` and similar rows, produced by changing the generator's parameters); a few were reasoned about but not measured. The counts by kind are in the summary below.

Several of these attacks defeat the detector outright, including one that reduces it to catching nothing at all. Those are the useful rows. A detector's evasion surface exists whether or not it is written down; the only variable is whether the team operating it finds out before an adversary does.

Measured verdicts name the population and seed that produced them. Unmeasured verdicts say so in those words and are not softened into prose that reads like a result.

The verdicts were first measured with one-off scripts that are not in this repository. `python -m tripphysics.attack_measure` re-implements the entries marked **Re-run** below from the public API and reproduces each of their counts exactly. Every other measured entry is a record of what was run, not something this repository can re-run: it states the manipulation, the population and the seed so that it can be re-implemented.

## Verdict summary

41 entries recorded. 11 of them can be re-run from this repository with `python -m tripphysics.attack_measure`, which reproduces their counts exactly; the others are a record.

| kind | count | meaning |
| --- | --- | --- |
| telemetry | 23 | a manipulation applied to the account's own trip log |
| simulator | 13 | a sharing scenario produced by changing the generator's parameters (`params` starts with `synth.py`); nothing is done to the log |
| unmeasured | 5 | reasoned about, not measured |

| verdict | count | meaning |
| --- | --- | --- |
| `caught` | 18 | the attack does not meaningfully help: the attacked population is still flagged at close to the un-attacked rate of 184/300 |
| `partial` | 12 | detection degrades materially but survives well above zero |
| `passes` | 6 | the attack defeats the current scoring |
| `unmeasured` | 5 | no measurement was run; the entry states what would have to be measured |

## caught (18)

### `split-trips` -- Trip splitting

**Mechanism.** Volume dilution without fabricating a single kilometre of driving. Cutting each uploaded trip in half at an existing fix doubles the gap count with a zero-distance, zero-time gap inserted at every cut, halving the mean handover score.

**Realization.** Split every trip with at least 12 fixes into two trips at the midpoint fix; the second half keeps the original timestamps and positions.

**Targets.** `handover_gap_score`

**Evidence.** 181/300 caught versus the unattacked 184/300 (attack population seed 777, threshold from the seed 4242 honest population) -- a 1pp difference, well inside the ~2.8pp binomial noise at this size.

**Re-run.** `python -m tripphysics.attack_measure split-trips` reproduces this count.

**Notes.** Surprising given the arithmetic: the zero-length gaps do halve the mean, but the surviving accounts are carried by overlap_rate and mock_rate, which splitting does not touch. Dilution only pays when it moves the feature that was actually driving the score.

### `client-speed-forge` -- Forged reported speed

**Mechanism.** max_speed_kmh and accel_p99_ms2 are read from the client-reported speed_mps field and are never cross-checked against position over time. A self-consistent falsified speed channel defeats both without touching the track.

**Realization.** Overwrite speed_mps on every fix with draws from a plausible urban cruise distribution (mean 10.5 m/s, sd 1.0, clipped to [0, 16]).

**Targets.** `max_speed_kmh`, `accel_p99_ms2`

**Evidence.** 173/300 caught versus the unattacked 184/300 (attack population seed 777, threshold from the seed 4242 honest population). An 11-account difference at ~2.8pp noise: the two speed features carry weight 0.8 and 0.6 against a 2.5/2.2 physics core, so removing them alone does not move the verdict.

**Re-run.** Not in this repository (telemetry entry).

**Notes.** The evasion is real -- both features are defeated -- but on this corpus they were not what was catching these accounts.

### `client-speed-smooth` -- Smoothed reported speed

**Mechanism.** A gentler form of client-speed-forge that keeps the reported speeds near their true values but removes the harsh-event tail a two-driver mixture produces, which is what accel_p99_ms2 measures.

**Realization.** Replace speed_mps with a 5-fix moving average of itself, per trip. Positions untouched.

**Targets.** `accel_p99_ms2`, `max_speed_kmh`

**Evidence.** 175/300 caught versus the unattacked 184/300 (attack population seed 777, threshold from the seed 4242 honest population).

**Re-run.** Not in this repository (telemetry entry).

**Notes.** Harder to spot in review than a wholesale rewrite, since the speeds still track the route.

### `client-mock-clear` -- Cleared mock-location flag

**Mechanism.** mock_rate is the OS mock-location flag taken verbatim from the client. Any tooling that can spoof position can also clear the flag reporting that it did.

**Realization.** Set is_mock=false on every fix.

**Targets.** `mock_rate`

**Evidence.** 167/300 caught versus the unattacked 184/300 (attack population seed 777, threshold from the seed 4242 honest population).

**Re-run.** `python -m tripphysics.attack_measure client-mock-clear` reproduces this count.

**Notes.** Costs the score a 1.2-weight signal, which alone is not decisive; it becomes decisive in combination -- see combined-client-scrub.

### `client-cell-forge` -- Forged serving-cell id

**Mechanism.** cell_mismatch_rate compares the reported cell_id against the cell implied by the reported position. In this synthetic world the expected cell is a pure function of x/y, so a client that computes the same function on its spoofed coordinates always agrees with itself.

**Realization.** Recompute cell_id = cell_id_from_xy(x_m, y_m) on every fix.

**Targets.** `cell_mismatch_rate`

**Evidence.** 153/300 caught versus the unattacked 184/300 (attack population seed 777, threshold from the seed 4242 honest population).

**Re-run.** `python -m tripphysics.attack_measure client-cell-forge` reproduces this count.

**Notes.** The most important caveat in this registry: this attack is trivially available ONLY because the synthetic cell grid is a function of position. On live data cell_id must come from an operator-side tower join, which the account cannot forge, and this feature becomes one of the strongest signals rather than one of the weakest. The synthetic corpus understates it.

### `bridging` -- Bridging a spatial jump

**Mechanism.** Handover excess is scored only between CONSECUTIVE trips. Inserting one plausible intermediate trip splits a large spatial jump into two individually explainable legs, each covered by the generous 54 km/h door-to-door allowance.

**Realization.** For every inter-trip gap over 1.5 km with more than 3 minutes available, insert a fabricated leg from the previous trip's end to the midpoint, timed inside the existing gap.

**Targets.** `handover_gap_score`

**Evidence.** 166/300 caught versus the unattacked 184/300 (attack population seed 777, threshold from the seed 4242 honest population).

**Re-run.** Not in this repository (telemetry entry).

**Notes.** Weaker than the README's framing implies, at least in this form: bridging to the midpoint halves each leg's distance but also adds gaps, and the mean is taken over more terms. A bridge placed to make each leg individually plausible under the 15 m/s allowance would do better; that stronger variant is not measured here.

### `split-and-bridge` -- Trip splitting plus bridging

**Mechanism.** Compose the two handover attacks: halve every trip to double the gap count, then bridge whatever large jumps remain.

**Realization.** Apply split-trips, then bridging, in that order.

**Targets.** `handover_gap_score`

**Evidence.** 168/300 caught versus the unattacked 184/300 (attack population seed 777, threshold from the seed 4242 honest population).

**Re-run.** Not in this repository (telemetry entry).

**Notes.** Composing two weak handover attacks stays weak, in contrast to full-client-evasion where composing four weak attacks was decisive. The difference is that these two target the same feature, so the second has little left to remove, while the client-scrub components each removed a different contribution from the sum.

### `drop-endpoint-fixes` -- Trimmed trip endpoints

**Mechanism.** handover_gap_score is computed from the last fix of one trip and the first fix of the next. Deleting a few fixes from each end of every trip moves those anchors inward, shortening the apparent inter-trip distance, and costs nothing in nonfinite_rate because the fixes are simply absent rather than malformed.

**Realization.** Drop the first 3 and last 3 fixes of every trip with at least 10 fixes.

**Targets.** `handover_gap_score`, `max_jump_m`

**Evidence.** 175/300 caught versus the unattacked 184/300 (attack population seed 777, threshold from the seed 4242 honest population).

**Re-run.** `python -m tripphysics.attack_measure drop-endpoint-fixes` reproduces this count.

**Notes.** Trimming 3 fixes moves an endpoint by ~300 m at urban speed, which is small against handover excesses measured in kilometres. A deeper trim would trade against the trip looking implausibly short.

### `kill-in-trip-teleports` -- Re-anchored in-trip teleports

**Mechanism.** max_jump_m measures consecutive-fix displacement in excess of a 250 km/h cap. A spoofer that re-anchors its fake route mid-trip produces km-scale jumps; rescaling each offending hop back to the cap and shifting the trip remainder with it removes the excess while preserving the route shape.

**Realization.** For each consecutive fix pair exceeding 69.4 m/s, rescale the displacement to exactly the cap and translate all downstream fixes in the trip by the same vector.

**Targets.** `max_jump_m`

**Evidence.** 172/300 caught versus the unattacked 184/300 (attack population seed 777, threshold from the seed 4242 honest population).

**Re-run.** `python -m tripphysics.attack_measure kill-in-trip-teleports` reproduces this count.

**Notes.** max_jump_m carries weight 1.5 but is largely redundant with handover_gap_score on these accounts, so removing it alone changes little.

### `inject-jitter` -- Injected GPS scatter

**Mechanism.** Spoofed tracks are glassy; gps_jitter_m measures positional scatter. The detector's transform is one-sided -- it only raises suspicion for scatter ABOVE 25 m -- so a spoofer adding device-typical noise removes the smoothness tell from any human review without paying anything to the score.

**Realization.** Add independent N(0, 6 m) noise to x and y on every fix.

**Targets.** `gps_jitter_m`

**Evidence.** 188/300 caught versus the unattacked 184/300 (attack population seed 777, threshold from the seed 4242 honest population) -- if anything marginally more, within noise.

**Re-run.** Not in this repository (telemetry entry).

**Notes.** The attack does not help because the feature was never scoring these accounts: a glassy track sits at low jitter, and low jitter contributes near-zero evidence under a threshold that only fires high. The signal a human reviewer would use is not the signal the detector uses.

### `shift-night-trips` -- Restamped night trips

**Mechanism.** night_share counts trips starting between 00:00 and 06:00. A shift-shared account covers the clock; restamping the night driver's trips into daytime hours collapses the feature.

**Realization.** Shift every trip starting in the night window forward by 10 hours.

**Targets.** `night_share`

**Evidence.** 183/300 caught versus the unattacked 184/300 (attack population seed 777, threshold from the seed 4242 honest population).

**Re-run.** `python -m tripphysics.attack_measure shift-night-trips` reproduces this count.

**Notes.** Expected: night_share carries the smallest weight in the score (0.3) precisely because honest night drivers are common. Removing the weakest signal changes nothing, which is the correct behaviour.

### `speed-cap-under-center` -- Speed clamped below the suspicion midpoint

**Mechanism.** A boundary attack on SPEED_CENTER_KMH = 150: reported speeds clamped just under the midpoint keep the soft threshold's evidence below 0.5 while remaining high enough to look like ordinary motorway driving.

**Realization.** Clamp speed_mps so no fix exceeds 140 km/h.

**Targets.** `max_speed_kmh`

**Evidence.** 183/300 caught versus the unattacked 184/300 (attack population seed 777, threshold from the seed 4242 honest population).

**Re-run.** `python -m tripphysics.attack_measure speed-cap-under-center` reproduces this count.

**Notes.** Boundary-hugging works as designed against this one feature and is worth nothing on its own at weight 0.8. Listed because it composes -- it is a component of full-client-evasion.

### `nonfinite-flood` -- Non-finite coordinate flood

**Mechanism.** Tests the anti-evasion guard directly: emitting unparseable coordinates removes fixes from every aggregate. The guard's claim is that garbage must raise suspicion rather than lower it, via nonfinite_rate.

**Realization.** Set x_m to NaN on a random 30% of fixes.

**Targets.** `nonfinite_rate`, `all`

**Evidence.** 204/300 caught, ABOVE the unattacked 184/300 (attack population seed 777, threshold from the seed 4242 honest population). Median score rises 0.79 -> 0.87.

**Re-run.** Not in this repository (telemetry entry).

**Notes.** The guard works and is worth stating as a positive result: the attack is counterproductive, costing the attacker 20 additional flagged accounts. nonfinite_rate saturates at 2% so a 30% flood is far into saturation.

### `nonfinite-trickle` -- Non-finite trickle at the saturation knee

**Mechanism.** The subtler version of nonfinite-flood: a rate at the NONFINITE_RATE_SATURATION = 0.02 knee, where the attacker pays full evidence for the smallest possible amount of destroyed data.

**Realization.** Set x_m to NaN on a random 2% of fixes.

**Targets.** `nonfinite_rate`

**Evidence.** 205/300 caught, ABOVE the unattacked 184/300 (attack population seed 777, threshold from the seed 4242 honest population).

**Re-run.** Not in this repository (telemetry entry).

**Notes.** Same conclusion as the flood, reached more cheaply: at the knee the account already pays the full 0.5-weight penalty, so there is no rate at which this trade is favourable.

### `nonfinite-targeted-endpoints` -- Targeted non-finite poisoning of trip endpoints

**Mechanism.** The sharpest form of the garbage-in evasion, because the attacker chooses WHICH evidence to destroy rather than poisoning at random: blanking the first and last fixes of every trip removes exactly the anchors handover_gap_score is computed from, at a nonfinite_rate cost of only a few percent.

**Realization.** Set x_m to NaN on fixes 0, 1 and the last two of every trip with at least 6 fixes.

**Targets.** `nonfinite_rate`, `handover_gap_score`, `max_jump_m`

**Evidence.** 203/300 caught, ABOVE the unattacked 184/300 (attack population seed 777, threshold from the seed 4242 honest population).

**Re-run.** Not in this repository (telemetry entry).

**Notes.** The most interesting of the three non-finite results. Targeting is strictly smarter than flooding and still loses, because the endpoints removed are replaced by the next fixes inward -- a few hundred metres away -- while the nonfinite_rate penalty is paid in full. The guard in features.py holds against a deliberate attacker, not just a careless one. Compare drop-endpoint-fixes, which achieves the same anchor shift with no penalty at all and is the better attack.

### `tier-blatant` -- Hardness tier: blatant

**Mechanism.** Not an evasion but the calibration floor: the caricature end of the corpus, with frequent switching, 8-18 km zone separation, 2-8 minute remote handovers and 30% spoofed trips.

**Realization.** synth.py tier 'blatant': p_switch 0.62, switch_modes (0.30, 0.55, 0.08), zone_sep_km (8, 18), handover_gap_min (2, 8), style_sep_mps 3.0, p_mock_trip 0.30, p_night_split 0.70.

**Targets.** `all`

**Evidence.** 292/300 caught (population forced to this tier, seed 777, threshold from the seed 4242 honest population). Median score 0.99.

**Re-run.** Not in this repository (simulator entry).

**Notes.** Establishes that the detector is not broken: given unambiguous physics it flags 97% at a 0.5% false-positive budget. Every harder row below should be read against this ceiling.

### `tier-moderate` -- Hardness tier: moderate

**Mechanism.** The middle of the corpus and the largest tier by weight (45%): 5-12 km zones, 4-14 minute handovers, 10% spoofed trips.

**Realization.** synth.py tier 'moderate': p_switch 0.42, switch_modes (0.16, 0.52, 0.12), zone_sep_km (5, 12), handover_gap_min (4, 14), style_sep_mps 2.0, p_mock_trip 0.10, p_night_split 0.55.

**Targets.** `all`

**Evidence.** 201/300 caught (population forced to this tier, seed 777, threshold from the seed 4242 honest population).

**Re-run.** Not in this repository (simulator entry).

### `mode-overlap` -- Switch mode: overlapping trips

**Mechanism.** The second driver starts before the first finishes. Physically impossible for one person and directly visible in the uploaded timestamps.

**Realization.** synth.py switch_modes forced to (1.0, 0.0, 0.0) on the moderate tier.

**Targets.** `overlap_rate`

**Evidence.** 293/300 caught (population seed 777, threshold from the seed 4242 honest population). Median score 0.97.

**Re-run.** Not in this repository (simulator entry).

**Notes.** The strongest signal available, and the reason overlap carries weight 2.5. Also the reason deoverlap-restamp is the most valuable single attack.

## partial (12)

### `vol-dilution-25` -- Volume dilution, 25 filler trips

**Mechanism.** handover_gap_score is the MEAN over consecutive inter-trip gaps, so appending geometrically clean trips lowers the average without removing the original handover. The damning gap is still in the log; it is simply outvoted.

**Realization.** Append 25 fabricated trips that start where the previous trip ended, at a plausible cruise, with correct cell ids and no mock flags.

**Targets.** `handover_gap_score`, `mock_rate`, `cell_mismatch_rate`, `night_share`, `overlap_rate`

**Evidence.** 125/300 caught, down from the unattacked 184/300 (attack population seed 777, threshold from the seed 4242 honest population). Reproduces the README's independently measured 33/50 -> 19/50 at 25 filler trips.

**Re-run.** Not in this repository (telemetry entry).

**Notes.** A max or a high quantile over gaps would resist this; the mean is what ships.

### `vol-dilution-100` -- Volume dilution, 100 filler trips

**Mechanism.** Same averaging weakness as vol-dilution-25, pushed until the original handover is a rounding error in the mean. Every rate feature (mock, cell mismatch, night share) dilutes on the same denominator.

**Realization.** Append 100 fabricated clean trips, otherwise identical to vol-dilution-25.

**Targets.** `handover_gap_score`, `mock_rate`, `cell_mismatch_rate`, `night_share`, `overlap_rate`

**Evidence.** 82/300 caught, down from the unattacked 184/300 (attack population seed 777, threshold from the seed 4242 honest population). Median score falls 0.79 -> 0.41.

**Re-run.** Not in this repository (telemetry entry).

**Notes.** Cost to the attacker is fabricating 100 plausible trips, which is cheap for anyone already running a spoofer.

### `combined-client-scrub` -- Combined client-side scrub

**Mechanism.** The four manipulations above applied together. Individually each removes one mid-weight signal and detection barely moves; together they remove enough that the remaining physics core no longer clears the threshold.

**Realization.** Forge speed, clear mock flags, forge cell ids, in one pass.

**Targets.** `max_speed_kmh`, `accel_p99_ms2`, `mock_rate`, `cell_mismatch_rate`

**Evidence.** 134/300 caught, down from the unattacked 184/300 (attack population seed 777, threshold from the seed 4242 honest population). Median score 0.79 -> 0.67.

**Re-run.** Not in this repository (telemetry entry).

**Notes.** The gap between this (50 accounts lost) and the sum of its parts (11 + 17 + 31 lost individually) is the interaction: the score is a sum, and accounts sitting just above threshold fall below it once several contributions go at once.

### `deoverlap-restamp` -- De-overlapped timestamps

**Mechanism.** overlap_rate is the heaviest signal in the score (weight 2.5) and is computed purely from uploaded trip timestamps. A client that buffers and restamps concurrent trips to run sequentially removes the signal entirely while two people keep driving at the same time.

**Realization.** Sort trips by start; whenever a trip starts before the running maximum end time, shift its whole fix series forward so it starts 120 s after that maximum.

**Targets.** `overlap_rate`

**Evidence.** 130/300 caught, down from the unattacked 184/300 (attack population seed 777, threshold from the seed 4242 honest population). Median score 0.79 -> 0.42, the largest single-feature drop measured here.

**Re-run.** `python -m tripphysics.attack_measure deoverlap-restamp` reproduces this count.

**Notes.** Confirms the weighting is doing what it claims -- overlap is the strongest evidence in the score -- and therefore that it is the most valuable single thing for an attacker to remove. Server-side receipt timestamps, which the client does not control, are what make this attack expensive.

### `tier-subtle` -- Hardness tier: subtle

**Mechanism.** The hard end reached without any deliberate evasion at all -- just two drivers who happen to operate close together, hand over unhurriedly and rarely spoof. 3-8 km zones, 8-25 minute handovers, 3% spoofed trips.

**Realization.** synth.py tier 'subtle': p_switch 0.28, switch_modes (0.06, 0.40, 0.16), zone_sep_km (3, 8), handover_gap_min (8, 25), style_sep_mps 1.2, p_mock_trip 0.03, p_night_split 0.35.

**Targets.** `all`

**Evidence.** 77/300 caught, versus 292/300 for the blatant tier (population forced to this tier, seed 777, threshold from the seed 4242 honest population). Median score 0.18, below the honest population's own p99.

**Re-run.** Not in this repository (simulator entry).

**Notes.** The most operationally important row in this registry after full-client-evasion, and it involves no attacker at all. Three quarters of merely subtle sharing is missed at this budget. Any recall figure quoted from a mixed corpus is a weighted average over these tiers and moves with the tier mix, not only with the detector.

### `mode-remote` -- Switch mode: remote handover

**Mechanism.** The next trip starts across town after a gap far too short to have driven the distance. The classic teleport signature, caught by handover_gap_score rather than by overlap.

**Realization.** synth.py switch_modes forced to (0.0, 1.0, 0.0) on the moderate tier.

**Targets.** `handover_gap_score`, `max_jump_m`

**Evidence.** 136/300 caught (population seed 777, threshold from the seed 4242 honest population).

**Re-run.** Not in this repository (simulator entry).

**Notes.** Markedly weaker than overlap despite being the signature the harness is named for: a 5-12 km separation against a 15 m/s door-to-door allowance leaves excess only when the gap is genuinely short.

### `mode-meet` -- Switch mode: co-located handover

**Mechanism.** The two drivers meet in person and pass the phone. This is a DESIGNED blind spot, stated as such in synth.py: the geometry is continuous by construction, so no positional feature can see it. Only style and schedule signals remain, and both are weak.

**Realization.** synth.py switch_modes forced to (0.0, 0.0, 1.0) on the moderate tier.

**Targets.** `handover_gap_score`, `max_jump_m`, `overlap_rate`, `style_bimodality`, `night_share`

**Evidence.** 122/300 caught (population seed 777, threshold from the seed 4242 honest population).

**Re-run.** Not in this repository (simulator entry).

**Notes.** The accounts still caught here are caught by the spoofing and night-shift behaviour that happens to accompany the handover in this tier, not by the handover itself. Strip those and nothing is left -- see meet-clean.

### `mode-shift-change` -- Switch mode: shift change

**Mechanism.** The incoming driver waits for their own shift, so there are hours of gap and the geometry is plausible. The tell is supposed to be the day/night and style split in aggregate.

**Realization.** synth.py switch_modes forced to (0.0, 0.0, 0.0) on the moderate tier, leaving the shift-change remainder branch.

**Targets.** `handover_gap_score`, `night_share`, `style_bimodality`

**Evidence.** 118/300 caught (population seed 777, threshold from the seed 4242 honest population). Median score 0.42.

**Re-run.** Not in this repository (simulator entry).

**Notes.** The features this mode is meant to be caught by are the two weakest in the score (night_share 0.3, style_bimodality 0.3). The catches here come mostly from the spoofing that accompanies the second driver in this tier.

### `zone-sep-under-jump-center` -- Zone separation held under the jump midpoint

**Mechanism.** Boundary attack on JUMP_CENTER_M = 3000: two drivers operating 2.5-2.9 km apart hand over remotely, keeping the displacement under the soft threshold's midpoint so max_jump evidence stays below 0.5.

**Realization.** synth.py moderate tier with zone_sep_km (2.5, 2.9) and switch_modes (0.0, 1.0, 0.0).

**Targets.** `max_jump_m`, `handover_gap_score`

**Evidence.** 126/300 caught, versus 136/300 for the same mode at the tier's normal 5-12 km separation (population seed 777, threshold from the seed 4242 honest population).

**Re-run.** Not in this repository (simulator entry).

**Notes.** A 10-account difference at ~2.8pp noise, so the boundary-hugging buys the attacker very little here. handover_gap_score is scaled against a time-dependent allowance rather than a fixed distance midpoint, so shrinking the distance also shrinks what the gap has to explain.

### `handover-gap-stretched` -- Handover gap stretched past the plausibility allowance

**Mechanism.** handover_gap_score subtracts what a 15 m/s door-to-door average covers in the observed gap. Waiting 25-45 minutes instead of 4-14 gives that allowance 22-40 km of credit, which covers any separation the tier produces, driving the excess to zero.

**Realization.** synth.py moderate tier with handover_gap_min (25.0, 45.0) and switch_modes (0.0, 1.0, 0.0).

**Targets.** `handover_gap_score`

**Evidence.** 122/300 caught, versus 136/300 for the same mode at the tier's normal 4-14 minute gaps (population seed 777, threshold from the seed 4242 honest population).

**Re-run.** Not in this repository (simulator entry).

**Notes.** The operational cost to the attacker is real -- half an hour of idle phone between shifts -- but it is the cheapest thing on this list to arrange, requiring no software at all. The residual catches come from mock and cell signals, not from the handover.

### `mock-rate-under-saturation` -- Spoofing held under the mock saturation knee

**Mechanism.** Boundary attack on MOCK_RATE_SATURATION = 0.05: the mock evidence ramps linearly and caps at 5% of fixes, so a spoofer that uses location faking on only a small share of trips pays a proportionally smaller penalty.

**Realization.** synth.py moderate tier with p_mock_trip reduced from 0.10 to 0.02.

**Targets.** `mock_rate`, `cell_mismatch_rate`, `gps_jitter_m`

**Evidence.** 162/300 caught, versus 201/300 for the unmodified moderate tier (population seed 777, threshold from the seed 4242 honest population).

**Re-run.** Not in this repository (simulator entry).

**Notes.** Note this attack also reduces cell_mismatch_rate and the glassy-track signature as a side effect, since in synth.py all three are produced by the same spoofed trips. The 39-account drop is therefore not attributable to the mock feature alone.

### `no-switch` -- Two drivers, one uploading

**Mechanism.** The account is shared in the ground-truth sense but the phone almost never changes hands during the observation window -- one driver does nearly all the uploading. There is no handover to detect because no handover happened.

**Realization.** synth.py moderate tier with p_switch reduced from 0.42 to 0.02.

**Targets.** `handover_gap_score`, `overlap_rate`, `max_jump_m`, `style_bimodality`

**Evidence.** 44/300 caught, versus 201/300 for the unmodified moderate tier (population seed 777, threshold from the seed 4242 honest population). Median score 0.15, at the honest median.

**Re-run.** Not in this repository (simulator entry).

**Notes.** A label-noise result rather than an evasion: these accounts are labelled shared but behave as single-driver accounts within the window. Any real deployment will contain them, and they place a hard ceiling on achievable recall that has nothing to do with the detector. Worth knowing before quoting any recall number as a product claim.

## passes (6)

### `full-client-evasion` -- Full client-side evasion

**Mechanism.** The complete set of manipulations available to a client that controls its own telemetry and nothing else: de-overlap the timestamps, scrub the reported fields, and re-anchor in-trip teleports so no consecutive pair breaks the 250 km/h cap. No change to how anyone actually drives.

**Realization.** De-overlap by restamping concurrent trips to start after the previous trip ends; clear is_mock; forge cell_id from position; clamp speed to 140 km/h; rescale any consecutive-fix displacement exceeding 250 km/h back to the cap, shifting the remainder of the trip with it.

**Targets.** `overlap_rate`, `mock_rate`, `cell_mismatch_rate`, `max_speed_kmh`, `accel_p99_ms2`, `max_jump_m`

**Evidence.** 0/300 caught, down from the unattacked 184/300 (attack population seed 777, threshold from the seed 4242 honest population). Median score 0.79 -> 0.19. Decomposed on the same population to confirm the result is an interaction and not a defect in the rig: de-overlap alone 130/300, mock-clear plus cell-forge alone 142/300, the three together 6/300, adding teleport re-anchoring 2/300, adding the speed clamp 0/300. A spot check of one attacked account confirms it is still scored on real evidence rather than routed to the evidence-free floor -- 8 trips, 215 fixes, handover_gap_score still 0.71 contributing 0.92 to a raw score of 1.74, which simply sits below the 2.5 bias.

**Re-run.** `python -m tripphysics.attack_measure full-client-evasion` reproduces this count.

**Notes.** This is the attack that defeats this detector. It is worth being precise about why, because the number is more alarming than the finding: every feature it removes is either client-reported (speed, mock flag, cell id) or a within-log consistency check the client can satisfy by construction (overlap timestamps, in-trip jump cap). What it leaves untouched is handover_gap_score -- the account still teleports between trips -- but that single 2.2-weight signal against a 2.5 bias does not clear the threshold alone. The lesson is not that the detector is broken; it is that a scoring function built mostly from client-controlled fields inherits their trustworthiness, and the fix is server-side corroboration (operator tower join, server-timestamped trip boundaries), not reweighting.

### `full-client-evasion-padded` -- Full client-side evasion plus volume dilution

**Mechanism.** full-client-evasion leaves handover_gap_score intact as the only surviving signal; padding the history with clean trips dilutes that mean as well. Belt and braces from the attacker's side.

**Realization.** Apply full-client-evasion, additionally restamp night trips into daytime, then append 25 fabricated clean trips.

**Targets.** `overlap_rate`, `mock_rate`, `cell_mismatch_rate`, `max_speed_kmh`, `accel_p99_ms2`, `max_jump_m`, `handover_gap_score`, `night_share`

**Evidence.** 2/300 caught, down from the unattacked 184/300 (attack population seed 777, threshold from the seed 4242 honest population). Median score 0.13, below the honest population's median of 0.14.

**Re-run.** Not in this repository (telemetry entry).

**Notes.** Recorded separately from full-client-evasion because it demonstrates the endpoint: the attacked accounts do not merely fall below the threshold, they rank below the median honest account. At that point the detector is not just failing to catch them, it is actively vouching for them.

### `silence-no-trips` -- Silence: submit no trips

**Mechanism.** An account with no trips produces the all-zero feature vector, which detector._is_evidence_free maps to a hard 0.0 -- the minimum possible score. Absence of evidence is deliberately not scored as evidence of guilt, but in pure ranking terms it sorts below every honest account.

**Realization.** Upload no trips at all.

**Targets.** `evidence_free_path`, `score_ordering`, `all`

**Evidence.** 0/300 caught (attack population seed 777, threshold from the seed 4242 honest population); every attacked account scores exactly 0.0.

**Re-run.** `python -m tripphysics.attack_measure silence-no-trips` reproduces this count.

**Notes.** Not a defect in the scoring so much as a routing question the scoring cannot answer. Empty accounts belong in their own queue -- 'why is a live account emitting nothing?' -- not at the bottom of this ranking. Any deployment that ranks on this score alone inherits the blind spot.

### `silence-empty-fixes` -- Silence: trips with no fixes

**Mechanism.** The same evidence-free floor reached by a different route: trips are present in the upload but every one carries an empty fix list, so extraction has nothing to measure and returns the all-zero vector.

**Realization.** Keep every trip record but set fixes to an empty list.

**Targets.** `evidence_free_path`, `score_ordering`, `all`

**Evidence.** 0/300 caught (attack population seed 777, threshold from the seed 4242 honest population); every attacked account scores exactly 0.0.

**Re-run.** `python -m tripphysics.attack_measure silence-empty-fixes` reproduces this count.

**Notes.** Worth listing separately from silence-no-trips because it is the version that survives a naive 'does this account have trips?' guard: the trip count looks normal and only the fix count is zero.

### `meet-clean` -- Co-located handover with no corroborating signal

**Mechanism.** The designed blind spot isolated. Drivers meet to pass the phone, neither spoofs location, and neither works nights, so the account is geometrically continuous, has no mock fixes, no cell mismatch, no overlap and an ordinary schedule. The only remaining evidence is style_bimodality, which is a coin flip.

**Realization.** synth.py moderate tier with switch_modes (0.0, 0.0, 1.0), p_mock_trip 0.0, p_night_split 0.0.

**Targets.** `handover_gap_score`, `max_jump_m`, `overlap_rate`, `mock_rate`, `cell_mismatch_rate`, `night_share`, `style_bimodality`

**Evidence.** 0/300 caught (population seed 777, threshold from the seed 4242 honest population). Median score 0.15, statistically indistinguishable from the honest median of 0.14.

**Re-run.** Not in this repository (simulator entry).

**Notes.** Requires no technical sophistication whatsoever -- two people meeting to hand over a phone defeat every physics feature in this detector, because there is no physics violation to detect. This is the honest ceiling of the whole approach, and it is why trip physics belongs in a stack with device and identity signals rather than alone.

### `style-snr-ceiling` -- Style bimodality at and beyond its SNR ceiling

**Mechanism.** The flagship self-failure. style_bimodality is documented in detector.py as empirically weak (ROC-AUC 0.51 over 7 seeds at n=2000) because the two drivers' cruise separation is smaller than the spread of one account's own per-trip median speeds. This entry tests whether the feature rescues the co-located blind spot when the separation is widened far beyond anything the tiers use.

**Realization.** synth.py moderate tier with style_sep_mps raised from 2.0 to 6.0 (three times the widest tier), switch_modes (0.0, 0.0, 1.0), p_mock_trip 0.0, p_night_split 0.0 -- i.e. meet-clean with an implausibly large style difference.

**Targets.** `style_bimodality`

**Evidence.** 1/300 caught (population seed 777, threshold from the seed 4242 honest population). Median score 0.16, against the honest median of 0.14.

**Re-run.** Not in this repository (simulator entry).

**Notes.** Independently confirms the demotion from weight 2.0 to 0.3 documented in detector.py, from the opposite direction: even handed a driving-style separation three times wider than the corpus ever produces, the feature contributes essentially nothing at its current weight. Carrying it at 2.0 would have made a coin flip the third heaviest voice in the score. The construct is sound; the measurement ceiling is real; the honest response was to demote rather than to delete or to quietly retune.

## unmeasured (5)

### `handover-max-not-mean` -- Single handover in a long clean history

**Mechanism.** The limiting case of volume dilution stated as its own attack: one damning handover inside a long, otherwise impeccable history. Because handover_gap_score averages, the account's score converges to that of an honest account as the history grows, no matter how implausible the single event was.

**Realization.** Not measured. Would require generating accounts with exactly one remote handover at a controlled position in a history of controlled length, sweeping the length.

**Targets.** `handover_gap_score`

**Evidence.** unmeasured -- reasoning only

**Re-run.** Not in this repository (unmeasured entry).

**Notes.** The two dilution rows measure this effect indirectly by padding an existing account; what is not measured is the sweep that would say how many clean trips buy how much cover, which is the number an operator would need to choose between a mean, a max and a high quantile.

### `bridging-optimal-placement` -- Optimally placed bridge legs

**Mechanism.** The bridging attack measured above places its intermediate leg at the geometric midpoint, which is convenient rather than optimal. A bridge placed and timed so that EACH resulting leg is individually covered by the 15 m/s allowance would zero the handover excess exactly, rather than merely reducing it.

**Realization.** Not measured. Would require solving, per gap, for the intermediate position and timestamp that keep both legs under the allowance, then verifying the fabricated leg is itself kinematically plausible.

**Targets.** `handover_gap_score`

**Evidence.** unmeasured -- reasoning only

**Re-run.** Not in this repository (unmeasured entry).

**Notes.** Expected to be strictly stronger than the measured bridging row, which came out at 166/300. Left unmeasured rather than asserted: the arithmetic is straightforward but whether the fabricated legs survive the other features is exactly the kind of thing that is wrong when guessed.

### `cell-join-server-side` -- Serving-cell forgery against a real tower join

**Mechanism.** client-cell-forge succeeds only because the synthetic world's expected cell is a pure function of the reported position. Against an operator-side tower join the client cannot compute the expected value, so forging cell_id requires knowing which towers actually served the handset -- which is the whole point of the feature.

**Realization.** Not measurable in this harness. Would require a corpus with an independent cell-location ground truth rather than synth.cell_id_from_xy.

**Targets.** `cell_mismatch_rate`

**Evidence.** unmeasured -- reasoning only

**Re-run.** Not in this repository (unmeasured entry).

**Notes.** Recorded because the synthetic result on client-cell-forge is misleading in the pessimistic direction and a reader should not carry it into a production discussion. This is the single largest gap between what this corpus can show and what a shadow pilot would.

### `replay-another-driver` -- Replayed genuine trip logs

**Mechanism.** Rather than fabricating physics, the account uploads real trip logs captured from a genuine single-driver account. Every physical feature is by construction that of an honest driver, because it was one.

**Realization.** Not measured. Would require a donor-and-recipient corpus and a decision about how the replayed timestamps are re-based, which is where the attack would either succeed or expose itself.

**Targets.** `all`

**Evidence.** unmeasured -- reasoning only

**Re-run.** Not in this repository (unmeasured entry).

**Notes.** Listed because it is the attack most likely to be seen in practice at scale and the one this harness is least equipped to reason about: nothing in a physics-first feature set can distinguish a genuine track from the same genuine track uploaded by someone else. Detecting it requires cross-account correlation, which is a different tool.

### `threshold-crowding` -- Crowding the honest tail

**Mechanism.** The threshold is a quantile of the honest population, so it is set by the honest tail rather than by any absolute standard. A coordinated fleet that pushes many accounts to just below the operating point raises nothing, but a fleet that pushes honest-looking accounts INTO the tail raises the threshold and lets genuinely shared accounts underneath it.

**Realization.** Not measured. Would require simulating a population where an attacker controls a non-trivial share of the honest-labelled accounts and sweeping that share.

**Targets.** `score_ordering`

**Evidence.** unmeasured -- reasoning only

**Re-run.** Not in this repository (unmeasured entry).

**Notes.** A property of budgeted-threshold evaluation generally, not of these features. Relevant to any deployment where one operator controls many accounts.
