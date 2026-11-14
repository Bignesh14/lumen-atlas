# Lumen Atlas - AI-assisted code (Claude, Anthropic); reviewed and run by Bignesh Das. Apache-2.0
"""Step 2: the referee for the Metal horizon kernel, then a speed test.

  a) GPU kernel (float32, Metal) vs NumPy float64 reference (different math path) on
     test observers at real south-pole sites  ->  must agree within 0.02 deg
  b) a 250 x 250 grid (62,500 observers x 360 azimuths) with both thread layouts:
     identical results required, then timing  (memory-coherence experiment)
Run:  python scripts/02_horizon_referee.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lumen.config import DEM_DIR, OUT_DIR                          # noqa: E402
from lumen.dem import PolarDEM                                      # noqa: E402
from lumen.horizon import horizon_gpu, horizon_ref, observer_table  # noqa: E402

TEST_SITES = [("Malapert Massif area", -86.0, 0.0), ("Mons Mouton area", -84.6, 30.0),
              ("Shackleton rim area", -89.45, -137.0), ("Haworth rim area", -87.0, -10.0)]


def load_dem():
    ori = json.load(open(OUT_DIR / "orientation.json"))["polar"]
    return PolarDEM(DEM_DIR / "LDEM_80S_20M.IMG", orientation=(ori["sx"], ori["sy"], ori["swap"]))


def main():
    dem = load_dem()
    print(f"DEM {dem.W}x{dem.H} @ {dem.scale_m} m, orientation sx={dem.sx} sy={dem.sy} swap={dem.swap}")

    # a) referee ----------------------------------------------------------------------
    lat = np.array([s[1] for s in TEST_SITES]); lon = np.array([s[2] for s in TEST_SITES])
    obs = observer_table(dem, lat, lon)
    t0 = time.time(); g = horizon_gpu(dem, obs); t_gpu = time.time() - t0
    worst = 0.0
    for k, (name, la, lo) in enumerate(TEST_SITES):
        t0 = time.time(); ref = horizon_ref(dem, la, lo, obs[k, 3]); t_ref = time.time() - t0
        d = np.nanmax(np.abs(g[k] - ref)); worst = max(worst, d)
        print(f"  {name:22s} ground {obs[k,3]-2:7.0f} m  horizon min/max {np.nanmin(ref):6.2f}/{np.nanmax(ref):6.2f} deg"
              f"  |GPU-ref| max {d:.4f} deg   (ref {t_ref:.1f} s)")
    print(f"REFEREE: worst |GPU - float64 reference| = {worst:.4f} deg -> {'PASS' if worst < 0.02 else 'FAIL'}"
          f"   (first GPU call incl. 1.85 GB upload + compile: {t_gpu:.1f} s)")

    # b) layouts + speed ---------------------------------------------------------------
    r0, c0 = dem.ll_to_rc(-86.0, 0.0)
    rr, cc = np.mgrid[0:250, 0:250]
    rows = float(r0) - 125 * 3 + rr * 3; cols = float(c0) - 125 * 3 + cc * 3   # 60 m spacing, 15 x 15 km
    glat, glon = dem.rc_to_ll(rows.ravel(), cols.ravel())
    big = observer_table(dem, glat, glon)
    res = {}
    for layout in ("az_major", "pixel_major"):
        horizon_gpu(dem, big[:1024], layout=layout)            # warm-up
        t0 = time.time(); res[layout] = horizon_gpu(dem, big, layout=layout); dt = time.time() - t0
        rays = big.shape[0] * 360
        print(f"  layout {layout:11s}: {big.shape[0]} observers x 360 az in {dt:6.2f} s  "
              f"({rays / dt / 1e6:6.1f} M rays/s)")
    same = np.array_equal(res["az_major"], res["pixel_major"])
    print(f"layouts identical: {same}")
    np.save(OUT_DIR / "referee_grid_malapert_60m.npy", res["az_major"])


if __name__ == "__main__":
    main()
