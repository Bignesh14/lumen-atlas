# NASA Space Apps 2026 submission text (Lumen Atlas)

## Project Name
Lumen Atlas

## High-Level Project Summary
Lumen Atlas shows, for any spot near the Moon's south pole and any hour from 2026 to 2030, whether the Sun and
Earth are above the real terrain skyline, and for how long. It answers the challenge's core questions in one screen:
time to sunset and Earthset, power and direct-to-Earth windows, and fast comparison of sites and dates. When a
launch slips, it finds Plan B: other dates at the same site, or other spots in the same region that still meet the
mission's sunlight and Earth-visibility limits. Behind it, custom GPU kernels traced the horizon of every 20 m pixel
in all nine Artemis III candidate regions, and every number is checked against NASA JPL Horizons and LROC.

## Link to Project Demo
https://github.com/Bignesh14/lumen-atlas/blob/main/docs/LumenAtlas-slides.pdf

## Link to Final Project
https://bignesh14.github.io/lumen-atlas/   (code: https://github.com/Bignesh14/lumen-atlas)

## Detailed Project Description
**What it does.** Near the lunar south pole the Sun never rises more than a few degrees, so a ridge tens of
kilometres away can switch a solar-powered lander off for days, and Earth can drop behind the same ridge and cut
direct radio contact. Lumen Atlas lets a mission planner, teacher or member of the public:
- choose any pixel of NASA's nine Artemis III candidate regions, or a famous site (Apollo 11–17, Chandrayaan-3,
  Chang'e-4, Chang'e-6, IM-2), and any hour from November 2026 to December 2030;
- see a 360° panorama of the real skyline with the Sun and Earth paths for the next 14 days;
- read the time to sunset or sunrise and to Earthset or Earthrise, to the minute;
- view every-pixel maps of each region for 2027–2028 (average sunlight, Earth in view, Sun and Earth together,
  longest night), which show why moving a few hundred metres or a few days changes everything;
- run a mission check (stay length, minimum sunlight, minimum Earth visibility, Earth at touchdown, launch slip)
  and get Plan B: launch windows at the same site in the next 60 days, and the nearest other spots in the same
  region that pass for the new date;
- compare up to four site and date scenarios in one table.
Explore mode explains the geometry in plain language; Plan mode adds the mission tools.

**How it works.** Terrain comes from NASA LRO LOLA elevation maps (20 m at the south pole, ~474 m globally). Sun and
Earth positions come from NAIF SPICE DE421 and the lunar orientation kernels, in the Moon's Mean-Earth frame, the same
frame the LOLA maps use, with light time, aberration and libration. A custom Metal GPU kernel (Apple M5 Max, via MLX)
traces 360 great-circle rays from every 20 m pixel out to 200 km, including the Moon's curvature. A second kernel
walks every pixel through every hour of 2027–2028. In total: 22.9 million pixels, 8.2 billion rays and about
400 billion visibility checks in about 66 seconds of GPU time. The web app is a static page: it loads the
precomputed horizons and an hourly Sun/Earth table and does all date-specific calculations in the browser, so it
needs no server.

**Validation.** Sun and Earth positions match JPL Horizons to better than 0.0001°. GPU horizons match an independent
float64 reference to within 0.0024°, and the statistics kernel matches its reference exactly. Ground heights read at
the Apollo 11, 14 and 17 and Chang'e-4 and -6 landers match LROC's published values within 7 m. A crater test confirms
the terrain map orientation.

**Benefits and goals.** Fast "what if" answers when hardware or launch dates slip; a shared picture for science and
engineering teams; and a way for students to see why the lunar south pole is so hard to plan for.

**Tools.** Python, NumPy, MLX (custom Metal kernels), Skyfield, Pillow, pyshp, pyproj; HTML, CSS and JavaScript
(canvas); Apple Mac Studio (M5 Max). Limits are stated in the app and README: a planning aid, not flight software;
in-browser horizons use sample points 200–900 m apart.

## NASA Data
- LRO LOLA gridded elevation maps LDEM_80S_20M, LDEM_80S_40M and LDEM_64 (PDS Geosciences Node): the terrain for
  every horizon and the region maps.
- NAIF SPICE kernels (de421.bsp, moon_pa_de421_1900-2050.bpc, moon_080317.tf, pck00008.tpc): Sun and Earth positions
  in the Moon's Mean-Earth frame.
- JPL Horizons: reference values used to validate our Sun and Earth positions.
- NASA's nine Artemis III candidate regions, as published by USGS (below); LROC-derived landing site coordinates
  (Wagner et al. 2017, NASA NSSDC; LROC featured images for Chang'e-4, Chang'e-6 and IM-2).
- Quickmap, Moon Trek and Horizons from the challenge's resources informed what the tool must do faster.

## Space Agency Partner & Other Data
- USGS Astrogeology: Down Selected Artemis III Candidate Landing Site Navigational Grids (McClernan 2024,
  doi:10.5066/P1MEQ6UK, CC0), region outlines.
- ISRO / IAU: Chandrayaan-3 landing site (Statio Shiv Shakti, 69.373° S, 32.319° E).
- Open-source software: NumPy, MLX, Skyfield, Pillow, pyshp, pyproj (their own licenses); Barlow font (Google Fonts, OFL).
- Our code: Apache-2.0, https://github.com/Bignesh14/lumen-atlas

## Use of Artificial Intelligence
Yes. The code and some text were written with AI assistance (Claude, by Anthropic). I designed the approach,
checked the sources, ran every step on my own hardware and verified the results against NASA data (JPL Horizons,
LROC heights) and independent reference implementations. Every source file and the README state that the code is
AI-assisted. No AI-generated images, video or audio are used; all images are screenshots of the app and maps
computed from NASA data. No NASA branding is used or modified.
