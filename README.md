# Lumen Atlas

**Where and when can you land at the Moon's south pole and still have sunlight and a line to Earth?**

Lumen Atlas is our entry to the NASA Space Apps Challenge 2026, *Lunar Landing Site Navigator*.
It computes, for every pixel of NASA's Artemis III candidate regions, how the real terrain blocks
the Sun and Earth, and turns that into fast, side-by-side answers for mission planners, educators
and the public: time to sunset and Earthset, power and comms windows, and Plan-B sites and dates
when a schedule slips.

> Work in progress during the hackathon (Nov 14-15, 2026).

## How it works (short)
1. **Terrain**: NASA LRO LOLA elevation maps (20 m south-polar, ~474 m global).
2. **Sky**: Sun and Earth positions in the Moon's Mean-Earth frame (DE421, via Skyfield),
   checked against NASA JPL Horizons.
3. **Horizons on the GPU**: a custom Metal kernel (Apple M5 Max, via MLX) traces 360 rays from
   every observer pixel out to 200 km, including the Moon's curvature. A float64 CPU reference
   with a different math path must agree before any result is used.

## Data (NASA)
- LRO LOLA gridded DEMs `LDEM_80S_20M`, `LDEM_80S_40M`, `LDEM_64` (PDS Geosciences Node / MIT).
- NAIF SPICE kernels: `de421.bsp`, `moon_pa_de421_1900-2050.bpc`, `moon_080317.tf`, `pck00008.tpc`.
- JPL Horizons (validation).
- USGS Astrogeology: Down Selected Artemis III Candidate Landing Site Navigational Grids
  (McClernan 2024, doi:10.5066/P1MEQ6UK, CC0).

## Use of AI
Code in this repository was written with AI assistance (Claude, by Anthropic) and reviewed, run
and tested by the author. Every source file carries this note in its header.

## License
Apache-2.0
