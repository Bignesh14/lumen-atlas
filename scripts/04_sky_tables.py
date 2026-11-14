# Lumen Atlas - AI-assisted code (Claude, Anthropic); reviewed and run by Bignesh Das. Apache-2.0
"""Step 4: hourly Sun and Earth positions (Moon-fixed MOON_ME_DE421 frame, apparent).

  data.nosync/out/sky_stats.npz   2027-01-01 .. 2029-01-01 hourly  -> statistics maps
  web/assets/ephem.bin + .json    2026-11-01 .. 2031-01-01 hourly  -> the web app
ephem.bin layout: float32 little-endian, 7 per hour:
  sun unit vector (3), sun angular radius deg (1), Earth vector from Moon centre km (3)
Run:  python scripts/04_sky_tables.py
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lumen.config import OUT_DIR, WEB_ASSETS            # noqa: E402
from lumen.ephem import Ephem, sun_angular_radius_deg  # noqa: E402


def hourly(eph, start, end):
    n = int((end - start).total_seconds() // 3600)
    t = eph.ts.utc(start.year, start.month, start.day, start.hour + np.arange(n))
    s, e = eph.vectors_km(t)
    return n, s, e, sun_angular_radius_deg(np.linalg.norm(s, axis=1))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True); WEB_ASSETS.mkdir(parents=True, exist_ok=True)
    eph = Ephem()
    a, b = datetime(2027, 1, 1, tzinfo=timezone.utc), datetime(2029, 1, 1, tzinfo=timezone.utc)
    n, s, e, r = hourly(eph, a, b)
    np.savez(OUT_DIR / "sky_stats.npz", sun_km=s, earth_km=e, sun_r_deg=r, start=a.isoformat(), hours=n)
    print(f"stats window {a:%Y-%m-%d} .. {b:%Y-%m-%d}: {n} hours")

    a, b = datetime(2026, 11, 1, tzinfo=timezone.utc), datetime(2031, 1, 1, tzinfo=timezone.utc)
    n, s, e, r = hourly(eph, a, b)
    su = s / np.linalg.norm(s, axis=1, keepdims=True)
    tab = np.concatenate([su, r[:, None], e], axis=1).astype("<f4")
    tab.tofile(WEB_ASSETS / "ephem.bin")
    json.dump({"start_utc": a.strftime("%Y-%m-%dT%H:%M:%SZ"), "step_hours": 1, "hours": n, "fields": 7,
               "frame": "MOON_ME_DE421 (apparent, from Moon centre)",
               "layout": ["sun_ux", "sun_uy", "sun_uz", "sun_radius_deg", "earth_x_km", "earth_y_km", "earth_z_km"]},
              open(WEB_ASSETS / "ephem.json", "w"), indent=1)
    print(f"web table {a:%Y-%m-%d} .. {b:%Y-%m-%d}: {n} hours, {tab.nbytes/1e6:.2f} MB -> {WEB_ASSETS/'ephem.bin'}")


if __name__ == "__main__":
    main()
