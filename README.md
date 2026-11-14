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
other. **There is no outside truth yet** (no GLM / network comparison), so
nobody can say how many of v2's strokes are real.

### Not built yet

- **Live daemon** — no streaming chain, no systemd unit, no radio lock.
- **Alerts** — no trigger on the gated sky rate, no replay test.
- **Outside truth** — no comparison with a lightning network or GOES GLM.
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
