# sferic-watch ⚡📡

**Single-station storm sensing by radio — lightning is the transmitter you
don't own.**

**Project site:** [felbs.software](https://felbs.software) · **Contact:** [E@felbs.software](mailto:E@felbs.software)

Born 2026-08-02, mid-thunderstorm, from the
[Software-TV-Tuner](https://github.com/Felbs/Software-TV-Tuner) lineage
(TV Tuna → [Radio Tuna](https://github.com/Felbs/gr-radiotuna) →
[wxTuna](https://github.com/Felbs/wxTuna) →
[aeroTuna](https://github.com/Felbs/aeroTuna) →
[dlayer-diary](https://github.com/Felbs/dlayer-diary) → sferic-watch, formerly stormTuna).
Every lightning stroke fires a broadband VLF impulse into the
Earth-ionosphere waveguide. One SDR and one antenna hear it. The hard part
is not hearing lightning — it is not mistaking the house wiring for it.

## What engine v2 does today

Engine v2 (the `sferic_watch` package) works on **banked captures**: 600 s
int16 IQ files at 62.5 kS/s centred on 15 kHz. Per capture it gives:

1. **A sferic count you can trust or not** — impulse detector over a
   floor-tracking front end, then a mains gate that finds 100/120 Hz buzz
   trains and removes them. Rate per minute, plus how much of the time was
   gated and a residual 120 Hz comb share against its chance level (0.144).
2. **A hum flag** — per 10 s block, is the band buzzing? Minutes with too
   much buzz or too much gated time are marked *untrusted*, not counted.
3. **Receiver health** — NAA-line clock/LO check and dropout tests.
4. **A session report** — `<id>.session.json` + `<id>.events.csv` per
   capture, and a one-row-per-capture census.

```bash
python -m sferic_watch analyze ~/path/vlf_HHMMZ.cs16 --out OUTDIR
python -m sferic_watch census ~/path/to/vlf_folders --out census.csv --workers 2
# installed: pip install .   then   sferic-watch --help
```

### The scoreboard (bench/, 2026-11-13)

Chance-corrected recall within ±2 ms of pulses injected into six 120 s
pieces of real air (1 clean, 3 steady hum, 2 intermittent). `july` is the
old July rule, `onair_bp` is OnAir's detector behind our band-pass.

| detector | native recall | clean | steady hum | hum at 12/18/24/30 dB (real pulses) |
|---|---|---|---|---|
| july | 0.45 | 0.71 | 0.23 | 0.09 / 0.39 / 0.39 / 0.38 |
| onair_bp | 0.71 | 0.75 | 0.62 | 0.54 / 0.83 / 0.82 / 0.82 |
| **v2** | **0.77** | **0.79** | **0.75** | **0.72 / 0.96 / 1.00 / 1.00** |

Residual 120 Hz comb share on the 9 hum captures (median): july 0.998,
onair_bp 0.242, v2 0.165 (one capture, 11-06 02:30Z, still over 0.25).

False mains rejection on synthetic skies with **no mains at all**: v2 0 %
at every rate from 2 to 200 per s (6 seeds each). OnAir's rule throws away
about 15 % at 60 per s.

CPU: v2 costs 1.2 s per minute of signal (detector alone 0.48; july 0.33,
OnAir 0.22). Target was 0.3: fine for banked files, still heavy for a live loop.

### The buzz problem, honestly

Most nights here the band is full of 120 Hz buzz from local wiring (the
2026-11-14 night capture is buzz from end to end: the July rule reads
4589 "strikes"/min, comb share 0.998). v2 gates it down to ~76/min with a
comb share of 0.17, and 52/min of that are confirmed in two bands at once.
But:

- a weak 60 Hz source with a broad (~2 ms) phase spread is now flagged but
  not removed (comb share 0.35–0.52 on 11-03 18–19:30Z and 11-05 13–14:30Z);
  removing it would cost about 12 % of real strokes;
- the "clean" evening of 11-03 21Z still loses 16 % between detector and
  gate, most likely to that weak source (removed events are 13 % confirmed
  against 52 % for kept ones), but that is not proven;
- the indoor antenna (port B) hears almost no sferics; use the outdoor ones.

Every "better" above is injection recall or detectors agreeing with each
other. The next section is the first check against an outside truth.

### Checked against satellite lightning (GOES GLM)

**How.** The GOES-East and GOES-West Geostationary Lightning Mappers (GLM)
see lightning flashes from space. Their L2 files are public on NOAA's
buckets, 20 s each. `bench/glm_fetch.py` downloads the files for the time
window of each of the 151 banked captures (by time only, nothing else is
sent) and decodes them. The decoding was checked against each file's own
time metadata. `bench/glm_truth.py` then runs the detectors on the same
captures and asks two questions:

- **precision**: what share of a detector's events have a GLM flash at the
  same moment?
- **recall**: what share of GLM flashes did the detector catch, by distance?

Chance is measured, not assumed. The same counts are redone with the GLM
times shifted by 5-60 s, and only the excess over that counts. The
uncertainty comes from resampling blocks of 8 consecutive captures, because
one storm fills many captures in a row. The scored numbers below are
"chance-corrected" in that sense.

**The clock.** VLF and GLM line up sharply (z ≈ 1200) once two things are
removed:

- the travel time through the waveguide: 3.75 ± 0.13 ms per 1000 km, which
  is an effective 0.89 c for our 2.5-9 kHz threshold crossing;
- a fixed +26 ms offset in our recorder. The sidecar time is stamped after
  the settle reads, so sample 0 is about 26 ms older than its stamp. It
  varies by ±1.5 ms from capture to capture (p5-p95 17-33 ms).

GLM times are at the flash, not at the satellite. That was tested two ways:
the same flash seen by both satellites has the same time, and the residual
does not move with the flash-to-satellite light time. The match window is
±8.5 ms.

**Precision** (share of events that are real lightning within 4000 km, 95 %
block interval). This is a lower bound: GLM misses some flashes and does not
see the whole sky.

| detector | precision | real lightning sferics per minute | recall 0-500 km | recall 500-1000 km | 50 % range |
|---|---|---|---|---|---|
| july | 0.003 (0.001-0.006) | 9.7 (of 3281 events/min) | 0.82* | 0.49* | — |
| onair_bp | 0.33 (0.24-0.39) | 30.0 | 0.83 | 0.48 | 681 km |
| v2_detect (no mains gate) | 0.24 (0.17-0.29) | 39.2 | 0.85 | 0.51 | 733 km |
| **v2** | **0.49 (0.38-0.57)** | **33.6** | **0.87** | **0.52** | **738 km** |
| **v2 + 16-30 kHz confirmation** | **0.63 (0.55-0.69)** | **29.4** | **0.87** | **0.51** | **717 km** |

\* july fires 3281 times a minute on buzz, so its recall is mostly the
chance correction. Its precision is the honest number.

- v2 minus onair_bp precision: +0.16 (0.07-0.26). v2 catches more real
  sferics per minute (+3.6/min, 0.1-8.0) and fewer false ones.
- v2 minus v2_detect precision: +0.25 (0.15-0.34). The mains gate removes
  buzz, not lightning.
- Recall within 1000 km: v2 beats onair_bp by +0.045 (0.016-0.091). Beyond
  1000 km the detectors cannot be told apart.

**Range curve** (v2, share of GLM flashes caught, group rule):

| distance | recall | 95 % block interval |
|---|---|---|
| 0-500 km | 0.87 | 0.74-0.92 (one storm, see below) |
| 500-1000 km | 0.52 | 0.34-0.71 |
| 1000-2000 km | 0.10 | 0.06-0.14 |
| 2000-3000 km | 0.08 | 0.05-0.12 |
| 3000-4000 km | 0.02 | 0.01-0.04 |
| over 4000 km | 0.001 | — |

Recall falls to 50 % at about 740 km (560-1030) and to 10 % at about 1570 km
(1430-2440). Bigger flashes are heard farther: at 1000-2000 km the most
energetic third is caught 15 % of the time and the weakest third 4 %.
Peak level falls about 11-13 dB per tenfold distance.

**What this proves**

- Most of what v2 counts is real lightning. At least half of its events
  (63 % with confirmation) line up with a GLM flash within ±8.5 ms. That
  share is a floor.
- v2 is better than OnAir's rule on real lightning, not only on injected
  pulses. The difference is clear for precision and small but real for
  near-range recall.
- The station hears nearby storms well (most flashes within 500 km), about
  half the flashes at 500-1000 km, and only the big ones beyond that.

**What it does not prove**

- The 0-500 km number is one storm. 99 % of the near flashes are from
  2026-11-06. Resampled by whole days, its interval goes below 0.
- The other ~half of v2's events are not shown to be false. GLM misses
  flashes (especially weak or low ones) and only sees part of the far sky.
  Some of the "unmatched" events are surely lightning.
- No range or bearing per stroke. The distances come from GLM, not from the
  radio.
- Day to day it moves a lot. v2 precision by day runs from 0.11 (11-05) to
  0.59 (11-06). It is higher at night (0.62) than by day (0.42).

**Caveats**

- 151 captures over 7 days, mostly autumn evenings and nights.
- The chance level depends on how far the shifts go. A 20 s block shuffle
  lowers v2 precision by about 0.008. Shifts under 1 s lower it to 0.44,
  because real flashes cluster within a second.
- The ±8.5 ms window and the waveguide speed were fitted on the same data.
  The per-capture clock lags are cross-fitted (half the capture fits the
  lag for the other half).
- GOES-East and GOES-West overlap. Some flashes are counted twice at
  1000-4000 km (about 5-9 %), which does not move the headline numbers.

**Privacy.** Distances need the station position. It is read only from a
private file outside this repo
(`~/.local/share/sferic-watch/station.json`, rounded to about 1 km), only
inside the code, and is never printed, sent anywhere or committed. The
per-flash GLM files with distances stay in the same private folder. The
repo and the results hold only aggregates: counts, rates, lags and
distance bins of 250 km or wider, no bearings.

See [bench/README.md](bench/README.md#checked-against-satellite-lightning-goes-glm)
for how to run it.

### Not built yet

- **Live daemon** — no streaming chain, no systemd unit, no radio lock.
- **Alerts** — no trigger on the gated sky rate, no replay test.
- **Network truth** — GLM is done (above); no comparison with a ground
  lightning network yet, which would give stroke-level timing.
- **Range / tweeks** — there is an audit (`tweek_audit.py`) of the old
  tweek-height method, and its verdict is that the random-window null does
  not hold up (daytime "textbook tweeks" pass 5.8x the null). **No per-stroke
  range is delivered.**
- **Lightning-jump warnings** and **Early-VLF event hunting** — not in the
  engine. Ideas, not features.
- **Bearing** — one antenna gives no direction; that needs a crossed loop.

## The panel

```bash
python tools/storm_panel.py --session <session-dir>
# -> http://127.0.0.1:8647
```

Reads a storm-session directory from the older capture chain (per-leg JSON
summaries + carrier `.npy` traces). File-reader only — it never touches
the radio, so it coexists with any capture chain. Some of its panes (range
history, jump banner) show the older chain's numbers, which engine v2 does
not stand behind (see *Not built yet*).

## The bench

`bench/` holds the frozen corpus, injection banks and one scoring harness —
every detector change is scored there against the same captures. See
[bench/README.md](bench/README.md). The OnAir comparison is optional and
its wrapper is built outside this repo (it links GPL code).

## Lineage & inspiration

The tweek dispersion physics and quality gates follow our
[dlayer-diary](https://github.com/Felbs/dlayer-diary) measurements
(77 night tweeks → ~82 km ceiling). The mains-train idea is credited to
OnAir's sferics receiver (GPL-3; ideas only, no code copied — our models of
it were written from prose descriptions of its behaviour). The wider world:
[Blitzortung](https://www.blitzortung.org) (network TOA geolocation),
Paul Nicholson's `vlfrx-tools` (the serious amateur VLF toolkit),
DL4YHF's Spectrum Lab (the classic VLF spectrogram viewer), and
Boltek-class commercial single-station detectors (crossed-loop bearing —
the natural hardware upgrade for this project).

## Roadmap
- fix the false-train bug in the mains gate, then a live daemon
- outside truth before any alert
- capture chain moves in-repo (today it rides the fleet's storm_watch)
- crossed-loop antenna → bearing → a real single-station storm plot
- Mode S BDS 4,4 wind/temp harvesting (aircraft as weather probes) via aeroTuna
- storm-night combined dataset: strokes + ionosphere + ADS-B deviations

## License
MIT — see [LICENSE](LICENSE).
