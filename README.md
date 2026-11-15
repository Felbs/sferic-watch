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

- **Live daemon on the real radio** — built and stress-tested on the mock
  radio (next section); not yet run against the SDR itself.
- **Network truth** — GLM is done (above); no comparison with a ground
  lightning network yet, which would give stroke-level timing.
- **Range / tweeks** — there is an audit (`tweek_audit.py`) of the old
  tweek-height method, and its verdict is that the random-window null does
  not hold up (daytime "textbook tweeks" pass 5.8x the null). **No per-stroke
  range is delivered.**
- **Lightning-jump warnings** and **Early-VLF event hunting** — not in the
  engine. Ideas, not features.
- **Bearing** — one antenna gives no direction; that needs a crossed loop.

## The live watcher

`python -m sferic_watch live` is a daemon that keeps running: radio in, strokes,
minute rates and a storm level out. Code: `sferic_watch/radio.py` (radio
sources) and `sferic_watch/live.py` (chain, files, alert, lock).

**In plain words.** It listens to the VLF band all the time, runs the same
engine v2 as the batch tools, and every minute writes down how many
lightning clicks it heard and whether that minute can be trusted (mains
buzz, a dead or overloaded front end and radio dropouts make a minute
untrusted). From the trusted minutes it keeps a storm level: quiet, active,
storm or severe. Each stroke is written about 1-2 minutes after it
happened. After a start or a dropout the first minute takes about 11 minutes
to appear. Where it lost data, it writes a gap row instead of leaving a
silent hole. It uses about 6 % of one CPU core and up to 1.7 GB of memory
for about a second each minute. So far it has only run against recorded
captures on a fake ("mock") radio. It has not yet run on the real SDR.

| question | answer |
|---|---|
| how do I run it | `python -m sferic_watch live --radio soapy --antenna 'Antenna C'` (below) |
| where does it write | `~/.local/state/sferic-watch/`: `state.json`, `minutes.jsonl`, `events.jsonl`, `alerts.jsonl` |
| what does an alert mean | "much more trusted lightning than usual for this hour" (a relative level, not a distance) |
| how often is it wrong | 0 false rises in 25 h of replayed nights checked against GOES GLM satellite lightning; with that little data the honest upper bound is about 3 false rises per day |
| how do I stop it | Ctrl-C, or `kill <pid>` of the daemon (the `pid` in `state.json`) |

```bash
# the real radio (owns the SDR; refuses to start while dlayer_run.py runs or the lock is held,
# and closes the radio and exits 3 if dlayer_run.py is started later)
python -m sferic_watch live --radio soapy --antenna 'Antenna C'
# replay captures as if they were the radio (no hardware): real time, N x, or --speed 0 unpaced
python -m sferic_watch live --radio mock --mock-files ~/path/vlf_0200Z.cs16 --speed 10 --duration 600 \
    --state-dir /tmp/sw-test [--mock-faults '[{"kind": "overflow", "at_s": 300, "n": 3}]']
python tools/storm_panel.py --live          # the panel on the watcher's files
```

A unit file is in `systemd/sferic-watch.service` (not installed; the
comments in it say how). Read the open items under *Known, not fixed*
before installing it: a systemd stop currently loses the last minutes.

**Radio.** The recipe every corpus capture was made with: SDRplay RSPdx via
SoapySDR, 62.5 kS/s, centre 15 kHz, Antenna C (`--antenna`), AGC off,
IFGR 40, RFGR 0, bias-T and notches off, CS16, a 1 s settle read thrown
away. A reader thread fills a 120 s ring buffer; overflows, timeouts, short
reads, read errors and ring drops are counted, and the device is reopened
with backoff (1 s ... 60 s) when it fails. Each block is stamped from the
host clock with the recorder's measured +26 ms stamp lag removed (GOES GLM
pooled lag +25.9 ms, see below), and the host-vs-sample clock is checked on
every read: a step or silent slip over 20 ms starts a new segment. The
anchor after a gap comes from one read and is late by that read's jitter;
it is refined on the next 16-read lag minimum (simulated at 10 ms mean
jitter: median error 8.3 -> 0.1 ms, worst 41 -> 3 ms). The watcher looks
for `dlayer_run.py` every 10 s (also `python3 -X dev -u dlayer_run.py`)
and yields the radio to it. `SFERIC_WATCH_NO_SDR=1` makes the soapy radio
refuse to open (tests and the stress bench set it).

**Chain.** Per minute, the expensive local parts are computed once (front
end powers with 2 s of context each side, frames on absolute positions, the
hum flag's 10 s blocks with their edges, health); the gate and detector
are then rerun on a rolling window of cached minutes (8 back, 1 ahead: as
long as a banked capture, so the "whole" fold sees as much as offline),
and only the oldest finished minute is written. **Delay: a stroke is
written 61-121 s after it happened** (1 minute look-ahead + 2 s guard +
about 1-2 s of CPU). **Warm-up:** after a start or a gap the window keeps
its full 10 minutes, so the first minutes wait until 10 minutes are in
(minute 0 is written about 11 min after it began) and a segment of up to
10 minutes is gated exactly as the batch engine gates a banked capture.
Without it (`--no-warmup`) the first minutes after every gap were gated on
2-9 minutes, missed the slow "whole" buzz trains, kept 13-80 % more events
than batch on hum captures and turned 94 batch-untrusted minutes trusted in
the corpus soak. `--lookahead-min 0 --no-warmup` gives 2-62 s.

**Outputs** (default `~/.local/state/sferic-watch`): `events.jsonl` (one line
per stroke that passed the gate: time, peak dB, 16-30 kHz confirmation,
hum block), `minutes.jsonl` (gated rate, confirmed rate, two-band rate,
hum_frac, gated share, comb share, trusted, health: floor, NAA level, 0 Hz
tone, clipping, radio faults in that minute, clock drift; plus explicit
gap rows `{"gap": true, t, t_end, missing_min, reason}` where no minute
was written: radio loss, a chain error or the watcher not running), `alerts.jsonl`,
and `state.json` (heartbeat every 5 s, written atomically). Files rotate
at UTC midnight into `archive/` (kept 30 days); optional raw recording
(`--raw`) stops when the disk has under 5 GB free.

**Storm level.** quiet / active / storm / severe from the *trusted*,
gated, 16-30 kHz *confirmed* rate: the 10-minute median against a baseline
(larger of the last 6 h and the same UTC hour band on earlier days, never
under 10/min), with hysteresis (up after 3 minutes over 3x / 6x / 12x the
baseline and over 40 / 80 / 200 per minute, down after 15 minutes under
70 % of that). Untrusted minutes (comb share over 0.25 or over 20 % gated)
never count, so buzz cannot raise it; nor do minutes whose front end is
dead (floor under -120 dBFS), swamped (floor over -40 dBFS) or clipping
(over 0.01 % of samples at full scale): those used to read as trusted quiet
sky. Only a new trusted minute moves the level up, and when fewer than 5
of the last 10 minutes are trusted for 15 minutes the level steps down one
step per 15 minutes (`reason: no_trusted_minutes`) instead of holding for
ever. On start the level and its baselines are rebuilt from the minute
files of the last 8 days. Alert minutes stay out of the
baseline for up to 6 h. A notifier hook exists (`--notify-cmd`); nothing
goes to the network by default.

### Measured (mock radio on real captures, 2026-11-15, after the repair round)

| | result |
|---|---|
| stream vs batch, all 150 corpus captures in time order (each its own segment, warm-up on) | 126 captures identical; 149 of 150 within 2 % in total; 99.6 % of minutes within 2 %; 99.2 % of batch events found within 1 ms; 1 trust flip in 1496 minutes. The one outlier (11-06 11:30Z, -28 %) lost 88 s of samples to ring drops when the box was overloaded during the run |
| the same, before warm-up (first-minutes window of 2-9 min) | hum captures **not** within 2 %: +13 % to +80 % per capture (pooled +3.3 %, which hid it), dense hum too (10-20 09:30Z, about 117 per min, +23 %), and 94 minutes batch calls untrusted came out trusted. An earlier version of this README said only sparse-hum captures (5-18 per min) differed; that was wrong |
| CPU per minute of signal | 3.0-3.4 s (worker + main): every minute is now gated on a full 10-min window. That is about 6 % of one core at real time |
| memory | worker about 0.2 GB between minutes and 1.6 GB peak for about 1 s per minute (the 10-min gate window); the daemon process holds 0.1-0.24 GB, most of it the bounded ring and queue buffers |

Steady state cannot be checked against batch with the corpus: no capture is
longer than 10 minutes. Live, minute c is gated on minutes c-8 ... c+1. A
buzz train that starts in the minutes after c+1 is invisible to it, while
batch saw the whole file. Between gaps the live window is as long as the
batch one, but it is placed differently.

Alert replay over the corpus (150 captures, 25 h of signal, 4 nights, the
live daemon itself): the level left quiet at 11-06 04:07Z (active) and
04:37Z (storm). GOES GLM matched lightning rose from 2-4 to 65-260 per
minute exactly then, so both were real. No rise without a GLM surge in
25 h, i.e. an observed false-alert rate of 0 per day (with so few hours, an
upper bound of about 3 per day). The level stepped down to active at 06:35Z
because the hum-gated 06:30Z capture left too few trusted minutes to read;
it now says so instead of holding a level it cannot see. Not caught: 10-20
08-11Z had 50-90 GLM-matched per minute all night long, so its own baseline
was already high. The level is relative: it says "much more than usual",
not "lightning nearby".

### Stress test (`bench/stress_live.py`, mock radio only, 2026-11-15, repair round)

| | target | measured | |
|---|---|---|---|
| soak A: all 150 captures, time order, 15x | no crash | 25 h of signal in 100 min, rc 0, 1497 minutes, 101 458 events, 0 chain errors, 0 bad lines, heartbeat never older than 5.5 s. 84 ring drops (88 s of one capture) while the test suite ran on the same box: at 15x the chain now needs 0.74 core, so there is little headroom left | pass (drops caused by the test setup) |
| soak A, alone: 40 captures (6.7 h), 10x | RSS growth < 20 MB | daemon 108 -> 111 MB (+3). Chain worker: no ratchet any more; its per-10-min floor moves between 104 and 209 MB with no trend (25 h run: first hour 173, last 177 MB). Before: steps from 254 to 463 MB. 0 ring drops; lag behind the input 17 -> 22 s of wall time (max 27 s). 400 of 400 minutes and every event equal to batch | pass |
| CPU | report | 2.7-3.0 s per signal minute; 0.05 s of that in the daemon process | |
| faults C: overflow bursts, device gone 5 and 60 s, timeouts, short reads, dropped and duplicated samples, host clock +-2 s and +-1 h, disk blocked 30 s at 10x and at 1x, disk low, ENOSPC, SIGTERM, SIGKILL + restart (with a torn line and a NUL block), second instance, dlayer_run running at start or started later, a dead and a clipped front end, worker SIGKILLed 4 times, daemon SIGKILLed 6 times, everything mixed | no crash, gaps honest, no fake events | all rc 0, 3 or 1 as intended. Every hole in `minutes.jsonl` has its gap row (mixed: 9 of 9, reasons overflows, timeouts, read errors, clock steps, ring drops, not written). Near segment edges there are fewer events than without the fault, never more. The worker's death ends the daemon in 1 s (rc 1); a SIGKILLed daemon leaves no worker. A later dlayer_run makes the watcher yield in 4 s (rc 3). Dead or clipped minutes are untrusted | pass |
| alerts D vs GOES GLM (25 h soak) | list, check | rises at 11-06 04:07Z (active) and 04:37Z (storm), both with 5-7 GLM flashes/min within 1000 km; 0 false rises (95 % bound 2.9 per day). New: storm -> active at 06:35Z, after 15 minutes with too few trusted minutes (the 06:30Z capture is gated), where the old run held "storm" until 16:36Z | pass |

Fixed in this round:
- a SIGKILLed daemon could leave its chain worker blocked for ever in a
  half-read pipe message;
- a dead worker could hang the daemon (in `_send`, or at exit joining the
  queue thread), and the exit then aborted with a core dump;
- dead, swamped or clipped input counted as trusted quiet sky;
- the storm level rose on stale re-checks and never decayed without
  trusted minutes; it is now restored on restart;
- a NUL block inside a `.jsonl` file survived a restart;
- the anchor after a gap was late by one read's jitter;
- `clock_drift_ppm` decayed after a re-anchor;
- `python3 -X dev -u dlayer_run.py` went unseen, and so did a dlayer_run
  started after the watcher;
- the live raw sidecars were replayed 26 ms late;
- a looping mock on an empty file spun for ever;
- the worker's memory floor ratcheted up;
- gaps had no rows;
- cold gate windows after every gap.

Known, not fixed:
- **The event set depends on the phase of the 1 s frames.** Batch on the same capture
  started at a whole second keeps 99.5 % of its events. Started at any other
  sample offset, only about a third of the events fall within 1 ms, at the
  same count. The live default `--align utc` keeps every segment on whole UTC
  seconds; the mock's `align=start` does not. Minute rates move by a median
  5-13 % after a phase change.
- Warm-up costs delay after every start or gap (minute 0 is written about
  11 minutes after it began). If the real radio overflowed every few
  minutes, every minute would wait for its segment's end.
- After a host clock step the minute times follow the host clock; +-1 h
  leaves a hole of an hour and one step back in time.
- A `--radio mock` run takes no lock, so it can write into a running watcher's state dir.
- Segment numbers restart at 0 after a restart.

Open, found by the independent verifier after the repair round (not fixed
yet; fix these before running it unattended under systemd):
- **systemd stop loses the backlog.** systemd's default stop sends SIGTERM to
  the whole process group, and the chain worker dies at once. The minutes
  still waiting in the warm-up and look-ahead window are lost, and the run
  still reports rc 0 with no error. Stopping the daemon process alone
  (Ctrl-C, or `kill` on its pid) is fine. Fix: ignore SIGTERM in the worker,
  or `KillMode=mixed` in the unit.
- **The reopen wait never resets.** After a stream that ran well, the wait
  before a reopen does not go back to 1 s. Rare, widely spaced driver errors
  climb 1, 2, 4 ... 60 s. From the 7th one on, every hiccup costs at least
  60 s of data.
- **A read that never returns is never recovered.** If the USB driver
  wedges inside a read, the watchdog asks for a reopen that never happens.
  The daemon stays `stalled` for ever and does not exit, so systemd's
  `Restart=on-failure` never fires.
- **A part-written line during a run.** If the disk fills in the middle of
  a write, a torn line stays, and the next write is glued onto it (one
  corrupt line). This is repaired only at the next start.
- A forward host-clock jump of more than 30 days would prune the whole
  archive. A kill during rotation onto an existing archive file can
  duplicate that day's rows.
- A mock file that does not exist makes the mock retry for ever instead of
  failing.

## The panel

```bash
python tools/storm_panel.py --session <session-dir>
python tools/storm_panel.py --live [state-dir]     # the live watcher's minutes + storm level
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
- run the live watcher on the radio itself (built, mock-tested)
- outside truth before any alert
- capture chain moves in-repo (today it rides the fleet's storm_watch)
- crossed-loop antenna → bearing → a real single-station storm plot
- Mode S BDS 4,4 wind/temp harvesting (aircraft as weather probes) via aeroTuna
- storm-night combined dataset: strokes + ionosphere + ADS-B deviations

## License
MIT — see [LICENSE](LICENSE).
