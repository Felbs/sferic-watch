# stormTuna ⚡📡

**Single-station storm sensing by radio — lightning is the transmitter you
don't own.**

Born 2026-08-02, mid-thunderstorm, from the
[Software-TV-Tuner](https://github.com/Felbs/Software-TV-Tuner) lineage
(TV Tuna → [Radio Tuna](https://github.com/Felbs/gr-radiotuna) →
[wxTuna](https://github.com/Felbs/wxTuna) →
[aeroTuna](https://github.com/Felbs/aeroTuna) →
[dlayer-diary](https://github.com/Felbs/dlayer-diary) → stormTuna).
Every lightning stroke fires a broadband VLF impulse into the
Earth-ionosphere waveguide. One SDR and one loop antenna turn that into
four instruments:

1. **Intensity** — sferic rate per minute is a direct convective-intensity
   proxy, and a sudden rate spike (the published "lightning jump") precedes
   severe weather by 10–30 minutes. The panel fires a banner on it.
2. **Range** — distant strokes arrive as dispersed *tweeks*; fitting the
   waveguide group delay hands you propagation distance per stroke and the
   ionosphere's reflection height. One antenna gives **range, not bearing**
   — the panel draws honest range history, never a fake map.
3. **The ionosphere's flinch** — the same capture carries the Navy's VLF
   transmitters (NAA 24.0, NLK 24.8, NML 25.2 kHz). A strong stroke can
   heat/ionize the D-layer over the path and the carrier visibly steps —
   an *Early VLF event* (Inan et al. 2010), hunted here with the causative
   stroke and the probe in the same IQ file.
4. **Honesty dials** — a 120 Hz comb test separates the sky from the house
   wiring before any number is believed.

## The panel

```bash
python tools/storm_panel.py --session <session-dir>
# -> http://127.0.0.1:8647
```

Reads a storm-session directory (per-leg JSON summaries + carrier `.npy`
traces) and renders: intensity curve with jump alerting, tweek-range
history, the VLF-carrier ionosphere watch, and the sky-vs-mains dial.
File-reader only — it never touches the radio, so it coexists with any
capture chain.

## Lineage & inspiration

The tweek dispersion physics and quality gates follow our
[dlayer-diary](https://github.com/Felbs/dlayer-diary) measurements
(77 night tweeks → ~82 km ceiling). The wider world:
[Blitzortung](https://www.blitzortung.org) (network TOA geolocation),
Paul Nicholson's `vlfrx-tools` (the serious amateur VLF toolkit),
DL4YHF's Spectrum Lab (the classic VLF spectrogram viewer), and
Boltek-class commercial single-station detectors (crossed-loop bearing —
the natural hardware upgrade for this project).

## Roadmap
- capture chain moves in-repo (today it rides the fleet's storm_watch)
- crossed-loop antenna → bearing → a real single-station storm plot
- Mode S BDS 4,4 wind/temp harvesting (aircraft as weather probes) via aeroTuna
- storm-night combined dataset: strokes + ionosphere + ADS-B deviations

## License
MIT
