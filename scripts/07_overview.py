# Lumen Atlas - AI-assisted code (Claude, Anthropic); reviewed and run by Bignesh Das. Apache-2.0
"""Step 7: south-pole overview map for the web app (hillshade, 400 m/pixel, +-240 km around the pole).
Run:  python scripts/07_overview.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lumen.config import DEM_DIR, OUT_DIR, WEB_ASSETS  # noqa: E402
from lumen.dem import PolarDEM                          # noqa: E402
from lumen.imaging import hillshade                     # noqa: E402

HALF_KM, STEP = 240, 20     # 20 x 20 m = 400 m per overview pixel


def main():
    ori = json.load(open(OUT_DIR / "orientation.json"))["polar"]
    dem = PolarDEM(DEM_DIR / "LDEM_80S_20M.IMG", orientation=(ori["sx"], ori["sy"], ori["swap"]))
    half = int(HALF_KM * 1000 / dem.scale_m)
    rc = int(round(dem.line_off)); cc = int(round(dem.sample_off))
    r0, c0 = rc - half, cc - half
    z = dem.data[r0:rc + half:STEP, c0:cc + half:STEP].astype(np.float32) * dem.factor
    sh = hillshade(z, dem.scale_m * STEP, az_deg=315, alt_deg=30)
    # gentle elevation tint so plateaus and crater floors read at a glance
    t = np.clip((z - z.min()) / (np.percentile(z, 99.5) - z.min()), 0, 1)
    rgb = np.stack([0.80 * sh + 0.20 * t, 0.80 * sh + 0.17 * t, 0.84 * sh + 0.10 * t], -1)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(WEB_ASSETS / "overview.png", optimize=True)
    json.dump({"r0": r0, "c0": c0, "step": STEP, "H": z.shape[0], "W": z.shape[1]},
              open(WEB_ASSETS / "overview.json", "w"))
    print(f"overview {z.shape[1]}x{z.shape[0]} px -> {WEB_ASSETS / 'overview.png'}")


if __name__ == "__main__":
    main()
