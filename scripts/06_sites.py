# Lumen Atlas - AI-assisted code (Claude, Anthropic); reviewed and run by Bignesh Das. Apache-2.0
"""Step 6: famous landing sites anywhere on the Moon (curated list, coordinates from sources below).

South of 80 S the 20 m polar map is used (GPU); elsewhere the global ~474 m LOLA map (CPU float64).
Writes web/assets/sites.json (horizon profile + 2027-2028 statistics per site).
Run:  python scripts/06_sites.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lumen.config import DEM_DIR, MAST_M, OUT_DIR, WEB_ASSETS  # noqa: E402
from lumen.dem import CylDEM, PolarDEM                         # noqa: E402
from lumen.horizon import horizon_gpu, horizon_ref, observer_table  # noqa: E402
from lumen.visibility import stats_ref                          # noqa: E402

LROC = "LROC / Wagner et al. 2017, Icarus 283 (via NASA NSSDC)"
SITES = [
    ("Apollo 11", "Apollo", 0.67416, 23.47314, LROC), ("Apollo 12", "Apollo", -3.0128, -23.4219, LROC),
    ("Apollo 14", "Apollo", -3.64589, -17.47194, LROC), ("Apollo 15", "Apollo", 26.13239, 3.63330, LROC),
    ("Apollo 16", "Apollo", -8.9734, 15.5011, LROC), ("Apollo 17", "Apollo", 20.1911, 30.7723, LROC),
    ("Chandrayaan-3 (Statio Shiv Shakti)", "ISRO", -69.373, 32.319, "IAU-approved site name; ISRO coordinates"),
    ("Chang'e-4", "CNSA", -45.4561, 177.5885, "LROC NAC DTM (Robinson 2019)"),
    ("Chang'e-6", "CNSA", -41.6385, -153.9852, "LROC (Wagner 2024)"),
    ("IM-2 Athena (intended site)", "CLPS", -84.78, 29.13, "LROC featured image 1401 (2025)"),
]


def main():
    ori = json.load(open(OUT_DIR / "orientation.json"))["polar"]
    pdem = PolarDEM(DEM_DIR / "LDEM_80S_20M.IMG", orientation=(ori["sx"], ori["sy"], ori["swap"]))
    gdem = CylDEM(DEM_DIR / "LDEM_64.IMG")
    sky = np.load(OUT_DIR / "sky_stats.npz")
    out = []
    for name, kind, lat, lon, src in SITES:
        if lat < -80.5:
            obs = observer_table(pdem, [lat], [lon]); hz = horizon_gpu(pdem, obs)[0].astype(np.float64)
            dem_name = "LOLA LDEM_80S_20M (20 m)"
        else:
            obs = observer_table(gdem, [lat], [lon])
            hz = horizon_ref(gdem, lat, lon, obs[0, 3]); dem_name = "LOLA LDEM_64 (~474 m)"
        st = stats_ref(hz, obs[0], sky["sun_km"], sky["earth_km"], sky["sun_r_deg"])
        out.append({"name": name, "kind": kind, "lat": lat, "lon": lon, "source": src, "dem": dem_name,
                    "height_m": round(float(obs[0, 3] - MAST_M)), "horizon": np.round(hz, 2).tolist(),
                    "stats": {"avg_illum": round(st[0] * 100, 1), "earth_pct": round(st[2] * 100, 1),
                              "both_pct": round(st[3] * 100, 1), "longest_night_h": int(st[4]),
                              "longest_no_earth_h": int(st[5])}})
        print(f"  {name:36s} {lat:9.4f} {lon:10.4f}  h {obs[0,3]-MAST_M:7.0f} m  horizon max {hz.max():5.2f} deg  "
              f"illum {st[0]*100:5.1f}%  Earth {st[2]*100:5.1f}%  longest night {st[4]/24:4.1f} d")
    json.dump(out, open(WEB_ASSETS / "sites.json", "w"))
    print(f"wrote {WEB_ASSETS / 'sites.json'}")


if __name__ == "__main__":
    main()
