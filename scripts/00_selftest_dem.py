# Lumen Atlas - AI-assisted code (Claude, Anthropic); reviewed and run by Bignesh Das. Apache-2.0
"""Step 0: prove we read the NASA terrain correctly.

Polar map: try all 8 possible axis orientations and keep the one in which known south-pole
craters are actually craters (centre much lower than rim).  Global map: same depth test on
famous nearside craters.  Writes out/orientation.json and a quick-look hillshade PNG.
Run:  python scripts/00_selftest_dem.py
"""
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lumen.config import DEM_DIR, OUT_DIR          # noqa: E402
from lumen.dem import PolarDEM, CylDEM             # noqa: E402

# (name, lat, lon, diameter_km)  - IAU/LROC published crater centres
POLAR_CRATERS = [("Shackleton", -89.67, 129.78, 21), ("Shoemaker", -88.1, 44.9, 51),
                 ("Haworth", -87.45, -4.98, 51), ("Faustini", -87.3, 77.0, 39),
                 ("Cabeus", -85.33, -42.13, 98)]
GLOBAL_CRATERS = [("Copernicus", 9.62, -20.08, 93), ("Tycho", -43.31, -11.36, 85),
                  ("Aristarchus", 23.7, -47.4, 40)]


def ring(lat, lon, dist_m, R, n=36):
    a = np.radians(np.arange(n) * 360.0 / n); d = dist_m / R
    p1, l1 = np.radians(lat), np.radians(lon)
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(a))
    l2 = l1 + np.arctan2(np.sin(a) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2))
    return np.degrees(p2), np.degrees(l2)


def depth(dem, lat, lon, diam_km):
    r = diam_km * 500.0
    c = np.nanmean([dem.height_ll(*ring(lat, lon, f * r, dem.R, 12)) for f in (0.0, 0.1, 0.2)])
    rim = np.nanmedian(dem.height_ll(*ring(lat, lon, r, dem.R, 72)))
    return rim - c


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dem = PolarDEM(DEM_DIR / "LDEM_80S_40M.IMG")
    print(f"polar map {dem.W}x{dem.H} @ {dem.scale_m} m, factor {dem.factor}, R {dem.R}")
    results = []
    for sx, sy, swap in product((1, -1), (1, -1), (False, True)):
        dem.set_orientation(sx, sy, swap)
        d = [depth(dem, la, lo, dk) for _, la, lo, dk in POLAR_CRATERS]
        results.append(((sx, sy, swap), d))
    results.sort(key=lambda r: -np.nansum(r[1]))
    print("\norientation (sx, sy, swap) -> crater depths (m): " + ", ".join(c[0] for c in POLAR_CRATERS))
    for o, d in results:
        print(f"  {str(o):22s} " + "  ".join(f"{x:7.0f}" for x in d) + f"   sum {np.nansum(d):8.0f}")
    best, d_best = results[0]
    ok = all(x > 1000 for x in d_best) and np.nansum(d_best) > 2 * max(np.nansum(results[1][1]), 1)
    print(f"\nchosen orientation {best}  ->  {'PASS' if ok else 'CHECK MANUALLY'}")

    g = CylDEM(DEM_DIR / "LDEM_64.IMG")
    gd = [depth(g, la, lo, dk) for _, la, lo, dk in GLOBAL_CRATERS]
    gok = all(x > 1500 for x in gd)
    print("global map: " + ", ".join(f"{n} {x:.0f} m" for (n, *_), x in zip(GLOBAL_CRATERS, gd))
          + f"  ->  {'PASS' if gok else 'CHECK MANUALLY'}")

    json.dump({"polar": {"sx": best[0], "sy": best[1], "swap": best[2], "depths_m": d_best, "pass": bool(ok)},
               "global": {"depths_m": gd, "pass": bool(gok)}},
              open(OUT_DIR / "orientation.json", "w"), indent=2)

    # quick-look hillshade (every 8th pixel of the 40 m map = 320 m/px), craters marked
    from PIL import Image, ImageDraw
    dem.set_orientation(*best)
    z = np.asarray(dem.data[::8, ::8], dtype=np.float32) * dem.factor
    gy, gx = np.gradient(z, 320.0)
    shade = np.clip(0.5 + 0.5 * (-gx - gy) / np.sqrt(1 + gx ** 2 + gy ** 2) * 2.0, 0, 1)
    img = Image.fromarray((shade * 255).astype(np.uint8)).convert("RGB")
    dr = ImageDraw.Draw(img)
    for name, la, lo, dk in POLAR_CRATERS:
        r, c = dem.ll_to_rc(la, lo); r, c = float(r) / 8, float(c) / 8
        rad = dk * 500.0 / 320.0
        dr.ellipse([c - rad, r - rad, c + rad, r + rad], outline=(255, 80, 40), width=2)
        dr.text((c + rad + 3, r - 6), name, fill=(255, 220, 0))
    for lo, lab in ((0, "0°"), (90, "90°E"), (180, "180°"), (-90, "90°W")):
        r, c = dem.ll_to_rc(-80.5, lo); dr.text((float(c) - 12, float(r) - 6), lab, fill=(80, 200, 255))
    img.save(OUT_DIR / "quicklook_south_pole.png")
    print(f"wrote {OUT_DIR / 'orientation.json'} and {OUT_DIR / 'quicklook_south_pole.png'}")


if __name__ == "__main__":
    main()
