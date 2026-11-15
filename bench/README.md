# bench: the frozen corpus and the one scoreboard

Every detector change in sferic-watch is scored here, against the same captures and the same injected pulses, beside
OnAir's own detector. "Ours vs OnAir" is then a diff on one table, not an argument.

## Files

- `corpus.json`: the frozen regression corpus. It holds 14 golden 600 s discone captures plus tonight's live capture,
  which is used for spot checks only (`golden: false`). Each entry carries its roles and its census labels.
  It also carries the h2h whole-file numbers and rough hum on/off seconds (`hum_on_s`, `hum_frac`).
  The file also stores the injection pieces, the 386 frozen real-sferic cut starts, and the h2h reference numbers.
  Paths start with `~`, and there is no station position anywhere.
- `inject.py`: the waveform banks (real, real_snr, synth), the injection schedule, and the peak-SNR yardstick.
- `harness.py`: the scoring, the baseline detectors and the CLI.

## Roles in the golden set

| role | captures |
|---|---|
| clean_day | 10-19 15Z (also the source of the real bank), 11-05 15Z |
| clean_evening | 11-03 21Z |
| sparse_buzz | 11-05 20Z |
| hum_switch | 11-05 04:30Z (on at 450 s), 11-06 02:30Z (on at 90 s) |
| steady_hum | 11-04 01Z (also second_train), 11-06 05Z, 11-06 22Z (also dusk_onset), 10-20 08Z, 10-20 11:30Z (dawn) |
| intermittent / raised floor | 11-06 10Z, 11-06 08:30Z |
| daytime_hum | 11-06 20Z |

The rough hum labels come from the July rule, per 10 s block:

- `hum_on_s` (and `hum_frac`): dense buzz, more than 20 edges/s with a comb share above 0.5.
- `buzz_suspect_s`: sparse or intermittent buzz, at most 20 edges/s but at least 20 edges with a comb share above 0.5.
  It is kept only when it totals 30 s or more, because busy clean skies give one or two stray blocks.
  It flags 11-05 20Z (530 s), 11-05 04:30Z (250 s before the dense switch-on at 450 s) and 11-06 10Z (250 s).
- `hum_label_complete: false` on those three captures. There `hum_on_s` is a lower bound on the buzz, not a
  clean label.

Plan step 2 replaces these labels.

## Run it

Use at most 2 workers on the shared box, and cap memory:

```
export SFERIC_ONAIR_BIN=/path/to/onair_sferics_h2h     # else $PATH, else ~/.local/share/sferic-watch/
export SFERIC_TMP=/some/scratch/dir                     # temporary .cs16 files (default /var/tmp; /tmp is tmpfs)
export SFERIC_BANK_NPY=/some/scratch/dir/bank_real.npy  # optional cache for the 386-cut bank (6 MB, gitignored)
systemd-run --user --scope -q -p MemoryMax=4G nice -n 10 python bench/harness.py all --detector july onair_bp --out results.json
python bench/harness.py list
```

### The OnAir binary (optional, built outside this repo)

The `onair_*` baselines run an external program, `onair_sferics_h2h`: a small C++ wrapper that links OnAir's
`libdect2core.a` and feeds a .cs16 file through OnAir's `SfericsReceiver`. Because it links OnAir's GPL-3 library,
its source and the built binary are GPL-3 and are **kept out of this MIT repo**: nothing of OnAir's is copied here,
and the harness only runs the binary as a separate program.

To build it, keep the wrapper source next to your OnAir checkout (not in sferic-watch), build OnAir first so that
`build/libdect2core.a` exists, then compile and link the wrapper against it, for example:

```
cd ~/src/onair-h2h          # your own folder holding onair_sferics_h2h.cpp (+ sos.h for the band-pass)
g++ -O2 -std=c++20 -I ~/src/OnAir-gps/core/include onair_sferics_h2h.cpp \
    ~/src/OnAir-gps/build/libdect2core.a -o ~/.local/share/sferic-watch/onair_sferics_h2h
```

The interface the harness expects (`run_onair` in harness.py), if you write your own wrapper:

```
onair_sferics_h2h IN.cs16 OUT_PREFIX MAINS(0|1) 0 FRONT(bp|raw)
```

- input: int16 IQ at 62.5 kS/s;
- `FRONT=bp` puts a causal 4th-order 2.5-9 kHz band-pass in front of the receiver, `raw` feeds the IQ as shipped;
- output: `OUT_PREFIX.times`, one line per event, first column the time in s and second the peak in dB;
- the last stdout line is JSON with `cpu_s`, `mains`, `tooLong`, `floorDb_min` and `floorDb_max`.

The harness finds the binary through `$SFERIC_ONAIR_BIN`, then `onair_sferics_h2h` on `$PATH`, then the corpus.json
`onair_bin` path (`~/.local/share/sferic-watch/onair_sferics_h2h`). **If none exists, the onair baselines are
skipped**: the harness prints one line (`OnAir binary not found ... skipping ['onair_bp']`) and scores the other
detectors. Calling `run_onair` directly without the binary raises FileNotFoundError with the same hint.

The modes are `inject`, `poisson`, `rule`, `corpus`, `all`, `label` and `list`. The docstring at the top of
`harness.py` says what each number means.

## Plugging in a detector

```python
import sferic_watch
@sferic_watch.register("median_floor", ungated="median_floor_nogate")
def median_floor(x, fs):          # x: complex64, full scale 1.0, 62.5 kS/s centred 15 kHz
    return times_s, peaks_db     # optional third item: dict of extra numbers
```

Any module inside the `sferic_watch` package is imported automatically. If a module fails to import, the failure is
printed and the baselines still run.

## Numbers as of 2026-11-13 (engine-v2, step 1)

These are chance-corrected recall within ±2 ms, using six 120 s pieces (200-320 s): 1 clean, 3 steady hum and
2 intermittent.

| set | july | onair_bp |
|---|---|---|
| native (h2h reproduction, bit-exact injected pieces) | 0.45 (hum 0.08-0.30, clean 0.71) | 0.705 (hum 0.60-0.65, clean 0.75) |
| real_snr 12 / 18 / 24 / 30 dB | 0.24 / 0.68 / 0.68 / 0.66 | 0.61 / 0.91 / 0.91 / 0.91 |
| synth 12 / 18 / 24 / 30 dB | 0.24 / 0.68 / 0.68 / 0.69 | 0.60 / 0.91 / 0.91 / 0.90 |

### False mains rejection with no mains present

The `rule` mode runs a behavioural model of OnAir's train-vote mains rule on bare Poisson times. That model
follows OnAir's GPL-3 rule step for step, so it is kept outside this MIT repo and loaded from
`$SFERIC_ONAIR_RULE` (default `~/.local/share/sferic-watch/onair_rule_model.py`) when present; otherwise the
mode is skipped and its two tests skip. It rejects 4.2 / 13.3 / 17.7 % at
40 / 60 / 80 per s, within 0.25 points of the earlier mains_sim.py study (4.3 / 13.5 / 17.9 %, and 4.26 / 13.47 /
17.95 % from the model it replaced).

The real OnAir binary on a synthetic signal, compared with its own mains-off run: 1.7 / 16.1 / 15.4 % at
40 / 60 / 80 per s.

### Whole-file rates

The whole-file rates are identical to the h2h table. For example, 10-19 15Z gives july 44.4/min and onair_bp
56.3/min. The comb share is measured against its chance level of 0.144.

### Caveats

- The real bank was picked by A and OnAir together, so it flatters both of them. Read the synth rows too.
- The real_snr bank only takes cuts whose 2.5-9 kHz peak lies -0.5 to +1 ms from the labelled onset. That is 349 of
  the 386 cuts. Without this check, 7 of the 24 cuts had a stronger second sferic 2.8-7.8 ms later. The gain was then
  set by that later sferic, which capped every detector near 0.8 (the old table read 0.15 / 0.45 / 0.45 / 0.54 and
  0.37 / 0.61 / 0.65 / 0.78).
- Of the eligible cuts, the 24 strongest are used. Only 8 of them reach 30 dB native SNR, and the 24th is 24.3 dB.
  A cut with native SNR S, injected at T dB, carries its copy of the source floor at T - S dB relative to the host
  floor, for the 16 ms the cut lasts. At the 30 dB level that copy sits up to 5.7 dB ABOVE the host floor around the
  pulse; at 24 dB and below it is at or under it. This may be why july's clean recall dips from 0.99 at 24 dB to
  0.94 at 30 dB. The synth set has no such floor copy (1.00 at 30 dB).

## Checked against satellite lightning (GOES GLM)

Two scripts score the detectors against GOES GLM L2 lightning (GOES-East G19 and GOES-West G18) on all 151
banked captures. They are separate from `harness.py` because the truth here is outside data, not injected pulses.

- `glm_fetch.py`: downloads the 20 s GLM files for each capture's time window (padded 60 s each side) from the
  public NOAA buckets, by time only, and decodes them. The times are scaled unsigned int16 relative to the file's
  `time_coverage_start`; `validate` checks that against the metadata. It merges the two satellites (G19 wins
  within 50 ms and 50 km) and writes one private npz per capture, with a distance from the station for each
  flash and group.
- `glm_truth.py`: runs july, onair_bp, v2_detect and v2 on every capture (`events`), then fits the clock and
  scores precision, recall and chance (`match`). Every output is an aggregate.

```
export SFERIC_ONAIR_BIN=/path/to/onair_sferics_h2h
export SFERIC_GLM_EVENTS=/some/scratch/dir/events        # cached detector events (one npz per capture)
R="systemd-run --user --scope -q -p MemoryMax=4G nice -n 10 python"
$R bench/glm_fetch.py fetch                               # download + decode (at most 8 connections)
$R bench/glm_fetch.py validate                            # decode check against the file metadata
$R bench/glm_fetch.py lighttime                           # G18 vs G19: are GLM times at the flash?
$R bench/glm_truth.py events --workers 2
$R bench/glm_truth.py match --workers 2 --boot 1000 --out results.json   # --lag-mode pooled for the sensitivity run
```

**Station position.** Distances need it, so `glm_fetch.py` reads `$SFERIC_STATION` (default
`~/.local/share/sferic-watch/station.json`, chmod 600, rounded to about 1 km) inside the process only. The per-flash
npz files (`$SFERIC_GLM_DIR`, default `~/.local/share/sferic-watch/glm`) hold distances and stay there, outside
the repo. `glm_truth.py` never reads the station file. Nothing positional is printed or written to results: only
counts (rounded to 10), rates, lags and distance bins of 250 km or wider, with no bearings.

### Method

- **Clock.** Detector time minus GLM time minus the waveguide delay d / v, histogrammed over ±2 s in 1 ms bins.
  The clean peak comes from v2 events confirmed at 16-30 kHz against GLM within 2000 km. The fitted speed is
  3.75 ± 0.13 ms per 1000 km (v = 0.89 c, 0.86-0.92) for our 2.5-9 kHz threshold crossing.
  The pooled lag is +25.9 ms (z ≈ 1200): the sidecar `utc` is stamped after the settle reads, so sample 0 is about
  26 ms older than its stamp. Per capture it is 24.8-26.9 ms (p25-p75).
- **Light time.** GLM L2 times are at the flash. The same group seen by G18 and G19 carries the same time
  (slope -0.02 against their light-time difference; +1 would mean satellite time). The VLF residual does not move
  with the light time (slope -0.03, block CI -0.19 to 0.05; satellite time would give -1).
- **Lag per capture.** Cross-fitted: events in the even 20 s blocks are scored with the lag fitted on the odd
  blocks, and the other way round. One pooled lag instead lowers v2 precision by 0.002 (0.494 → 0.492).
- **Match.** A detector event is matched when a GLM group lies within ±8.5 ms after alignment. A flash is caught
  when an event lies within ±8.5 ms of one of its groups (the "group" rule). The "span" and "first event"
  rules are also written.
- **Chance.** The same counts with GLM shifted by 40 offsets of ±5-60 s. The excess over chance is what counts:
  precision q = (matched - chance) / (1 - chance).
- **Uncertainty.** Captures are not independent, so the headline 95 % interval is a moving-block bootstrap of 8
  consecutive captures (1000 resamples). A whole-day bootstrap (7 days, crude) and the naive per-capture one are
  written too, plus a per-day table.

### Numbers as of 2026-11-14 (151 captures, 7 days)

Precision within 4000 km (lower bound, 95 % block CI) and recall by distance (group rule, chance-corrected):

| detector | precision | real sferics/min | 0-500 km | 500-1000 | 1000-2000 | 2000-3000 | 3000-4000 | 50 % range |
|---|---|---|---|---|---|---|---|---|
| july | 0.003 (0.001-0.006) | 9.7 | 0.82* | 0.49* | 0.10 | 0.08 | 0.015 | — |
| onair_bp | 0.33 (0.24-0.39) | 30.0 | 0.83 | 0.48 | 0.09 | 0.07 | 0.021 | 681 km |
| v2_detect | 0.24 (0.17-0.29) | 39.2 | 0.85 | 0.51 | 0.10 | 0.08 | 0.022 | 733 km |
| v2 | 0.49 (0.38-0.57) | 33.6 | 0.87 | 0.52 | 0.10 | 0.08 | 0.021 | 738 km |
| v2_conf | 0.63 (0.55-0.69) | 29.4 | 0.87 | 0.51 | 0.09 | 0.07 | 0.019 | 717 km |

\* july fires about 3300 times a minute on buzz; its recall is mostly the chance correction.

- v2 recall intervals: 0-500 km 0.74-0.92, 500-1000 km 0.34-0.71, 1000-2000 km 0.06-0.14. Recall falls to 50 % at
  740 km (560-1030) and to 10 % at 1570 km (1430-2440).
- v2 minus onair_bp: precision +0.16 (0.07-0.26), recall at 0-1000 km +0.045 (0.016-0.091), no difference
  beyond 1000 km.
- v2 precision with no distance cap 0.53 (0.43-0.61), within 2000 km 0.36 (0.22-0.47). By day it ranges from 0.11
  (11-05) to 0.59 (11-06). It is 0.62 at night and 0.42 by day, 0.53 on hum captures and 0.43 on clean ones.
- Precision rises with peak level: 0.34 below 12 dB, 0.58-0.68 from 12 to 30 dB.
- Within 500 km a flash gives about 11 v2 events (18 per caught flash). From 1000 to 4000 km it gives 2.0-2.5 per
  caught flash.

### Caveats

- 99 % of the GLM flashes within 500 km fall on 2026-11-06. The 0-500 km recall is that storm's number, and its
  whole-day interval goes below 0.
- Precision is a lower bound. GLM misses flashes, and its field of view and detection efficiency are not corrected.
- The chance level depends on the shift range. A 20 s block shuffle lowers v2 precision by about 0.008. Shifts
  of 0.1-1 s lower it to 0.44, because real lightning clusters within a second. The 5-60 s range is a choice.
- The ±8.5 ms window and the waveguide speed are fitted on all the data (not cross-fitted). The window is wide
  compared with the lag spread, so the effect is small.
- The G18/G19 merge leaves some twins: about 5-9 % of flashes at 1000-4000 km. Precision is not affected,
  recall ratios hardly at all.
- `tests/test_glm_truth.py` covers the lag recovery on synthetic skies (flash- and satellite-referenced), the
  light-time geometry, the resampling and an end-to-end match. The GLM decoding is checked by `glm_fetch.py
  validate` against real files, not by a unit test.
