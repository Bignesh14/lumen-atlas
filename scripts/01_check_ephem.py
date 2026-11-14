# Lumen Atlas - AI-assisted code (Claude, Anthropic); reviewed and run by Bignesh Das. Apache-2.0
"""Step 1: our Sun/Earth positions vs NASA JPL Horizons (the answer key downloaded on Friday).
Observer: 86 S, 0 E, on the 1737.4 km sphere, December 2026, daily.
Run:  python scripts/01_check_ephem.py
"""
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lumen.config import REF_DIR       # noqa: E402
from lumen.ephem import Ephem, topo_altaz  # noqa: E402

LAT, LON, H = -86.0, 0.0, 0.0
TOL_DEG = 0.02


def read_horizons(path):
    txt = Path(path).read_text()
    body = txt.split("$$SOE")[1].split("$$EOE")[0]
    rows = []
    for line in body.strip().splitlines():
        tok = line.split()
        if len(tok) >= 4 and re.match(r"\d{4}-\w{3}-\d{2}$", tok[0]):
            rows.append((datetime.strptime(tok[0] + " " + tok[1][:5], "%Y-%b-%d %H:%M"), float(tok[-2]), float(tok[-1])))
    return rows


def main():
    eph = Ephem()
    worst = 0.0
    for code, name, idx in (("10", "Sun", 0), ("399", "Earth", 1)):
        rows = read_horizons(REF_DIR / f"horizons_{code}.txt")
        t = eph.ts.utc([r[0].year for r in rows], [r[0].month for r in rows], [r[0].day for r in rows],
                       [r[0].hour for r in rows], [r[0].minute for r in rows])
        vec = eph.vectors_km(t)[idx]
        az, el, _ = topo_altaz(LAT, LON, H, vec)
        h_az = np.array([r[1] for r in rows]); h_el = np.array([r[2] for r in rows])
        d_el = el - h_el; d_az = (az - h_az + 180) % 360 - 180
        worst = max(worst, np.abs(d_el).max(), np.abs(d_az).max())
        print(f"\n{name}: {len(rows)} epochs   max |d_el| = {np.abs(d_el).max():.4f} deg   "
              f"max |d_az| = {np.abs(d_az).max():.4f} deg")
        for k in range(0, len(rows), 6):
            print(f"  {rows[k][0]:%Y-%m-%d %H:%M}  Horizons az {h_az[k]:8.3f} el {h_el[k]:7.3f}   "
                  f"ours az {az[k]:8.3f} el {el[k]:7.3f}")
    print(f"\nRESULT: worst difference {worst:.4f} deg -> {'PASS' if worst < TOL_DEG else 'FAIL'} (tolerance {TOL_DEG} deg)")


if __name__ == "__main__":
    main()
