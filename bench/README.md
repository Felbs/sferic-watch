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
