# Lumen Atlas

**Where and when can you land at the Moon's south pole and keep both the Sun and Earth above the real skyline?**

Lumen Atlas is a project for NASA Space Apps Challenge 2026, challenge *Lunar Landing Site Navigator*
(Boston local event, solo team: Bignesh Das).

- **Live app:** https://bignesh14.github.io/lumen-atlas/
- **Demo video (2 min 51 s, narrated):** [docs/LumenAtlas-demo.mp4](https://youtu.be/UzDrP0IF2t0)

## What it does
- **Pick a spot and an hour** (2026–2030): any pixel of NASA's nine Artemis III candidate regions, or a famous
  landing site (Apollo 11–17, Chandrayaan-3, Chang'e-4, Chang'e-6, IM-2).
- **See the skyline**: a 360° terrain horizon with the Sun and Earth tracks for the next 14 days.
- **Countdowns**: time to sunset or sunrise, and to Earthset or Earthrise, to the minute.
- **Every-pixel maps** of each region for 2027–2028: average sunlight, Earth in view, Sun and Earth together,
  longest night.
- **Mission check and Plan B**: set the stay (default 6.5 days), sunlight and Earth limits and a launch slip.
  It lists other dates at the same site and other spots in the same region that still work.
- **Compare** up to four site and date scenarios side by side.

## How it works
1. **Terrain**: NASA LRO LOLA elevation, `LDEM_80S_20M` (20 m, south pole) and `LDEM_64` (~474 m, global).
2. **Sky**: Sun and Earth positions in the Moon's Mean-Earth frame (NAIF SPICE DE421 + lunar orientation kernels,
   via Skyfield), the same frame as the terrain. Apparent positions, libration included.
3. **Horizons on the GPU**: a custom Metal kernel (MLX `mx.fast.metal_kernel`, Apple M5 Max) traces 360 rays from
   every 20 m pixel out to 200 km along great circles, with the Moon's curvature.
4. **Visibility statistics on the GPU**: a second Metal kernel walks every pixel through every hour of 2027–2028:
   fraction of the solar disk above the skyline, Earth above the skyline, and the longest runs of each.
5. **Web app**: a static page (no server) that reads the precomputed horizons and an hourly Sun/Earth table and
   does all date-specific calculations in the browser.

Scale: 22.9 million pixels × 360 directions (8.2 billion rays) and about 400 billion hourly visibility checks,
about 66 s of GPU time.

## Validation
| Check | Result |
|---|---|
| Sun and Earth az/el vs JPL Horizons (86° S, 0° E, Dec 2026, 62 values) | < 0.0001° |
| Metal horizon kernel vs independent float64 reference (4 sites × 360 az) | ≤ 0.0024° |
| Metal statistics kernel vs float64 reference (6 pixels × 17,544 h) | identical |
| Terrain orientation: 5 south-pole craters must be craters in only one of 8 orientations | pass |
| Ground heights at Apollo 11, 14, 17, Chang'e-4, Chang'e-6 vs LROC published values | within 7 m |

Performance note: assigning neighbouring GPU threads to neighbouring pixels on the same bearing
(parallel rays, coherent memory reads) is 7× faster than neighbouring bearings from one pixel, with identical output.

## Limits
- A planning and teaching aid, not flight software. Observer height is fixed at 2 m above the ground.
- In the browser, horizons come from sample points 200–900 m apart; the region maps use every 20 m pixel.
- Terrain beyond the edge of the 80° S map is ignored (it subtends small angles at these distances).
- Sites north of 80° S use the ~474 m global map.

## Run it yourself
```
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt   # Apple Silicon for the Metal kernels
python -m tests.test_geometry
python scripts/00_selftest_dem.py      # needs the LOLA files in ../data.nosync/dem
python scripts/01_check_ephem.py       # needs the SPICE kernels and Horizons tables
python scripts/02_horizon_referee.py
python scripts/03_fetch_regions.py     # USGS outlines (download the lon/lat zip first, see the script)
python scripts/04_sky_tables.py
python scripts/05_region_maps.py
python scripts/06_sites.py
python scripts/07_overview.py
python -m http.server 8000             # then open http://localhost:8000/web/
```

## Data and credits
- NASA LRO LOLA gridded data records (PDS Geosciences Node, via MIT): `LDEM_80S_20M`, `LDEM_80S_40M`, `LDEM_64`.
- NAIF SPICE: `de421.bsp`, `moon_pa_de421_1900-2050.bpc`, `moon_080317.tf`, `pck00008.tpc`.
- NASA JPL Horizons (validation).
- USGS Astrogeology, Down Selected Artemis III Candidate Landing Site Navigational Grids
  (McClernan 2024, doi:10.5066/P1MEQ6UK, CC0): region outlines.
- Site coordinates: LROC / Wagner et al. 2017 (Icarus 283) via NASA NSSDC; LROC featured images (Chang'e-4, Chang'e-6,
  IM-2); IAU / ISRO (Chandrayaan-3, Statio Shiv Shakti).
- Libraries: NumPy, MLX, Skyfield, Pillow, pyshp, pyproj. Fonts: Barlow (Google Fonts, OFL).

## Use of AI
Code in this repository was written with AI assistance (Claude, by Anthropic) and reviewed, run and tested
by the author on his own hardware. Every source file carries this note in its header. Not an official NASA product.

## License
Apache-2.0
