# Lumen Atlas - AI-assisted code (Claude, Anthropic); reviewed and run by Bignesh Das. Apache-2.0
"""Step 3: download the official outlines of NASA's 9 Artemis III candidate regions.

Source: USGS Astrogeology, "Down Selected Artemis III Candidate Landing Site Navigational
Grids" (McClernan 2024), doi:10.5066/P1MEQ6UK, CC0.  We use the 1 km grid shapefiles:
the union of a region's 1 km cells is the region.  Coordinates are converted from the
USGS Lunar Polar Stereographic (LPS) projection to lat/lon with the .prj that ships with them.
Writes data.nosync/regions/regions.json (outlines from the release's lon/lat file)
Run:  python scripts/03_fetch_regions.py
"""
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lumen.config import DATA  # noqa: E402

ITEM = "671a6fa8d34efed5620f89f8"
API = f"https://www.sciencebase.gov/catalog/item/{ITEM}?format=json"
REGIONS = {  # key as used in USGS file names -> display name
    "Peak_Near_Cabeus_B": "Peak near Cabeus B", "Haworth": "Haworth", "Malapert_Massif": "Malapert Massif",
    "Mons_Mouton_Plateau": "Mons Mouton Plateau", "Mons_Mouton": "Mons Mouton", "Nobile_Rim_1": "Nobile Rim 1",
    "Nobile_Rim_2": "Nobile Rim 2", "de_Gerlache_Rim_2": "de Gerlache Rim 2", "Slater_Plain": "Slater Plain",
}
DEST = DATA / "regions"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "lumen-atlas/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def region_of(name):
    n = name.lower()
    for key in sorted(REGIONS, key=len, reverse=True):     # longest first: Mons_Mouton_Plateau before Mons_Mouton
        if key.lower() in n:
            return key
    return None


def all_files(item):
    files = list(item.get("files", []))
    for facet in item.get("facets", []) or []:
        files += facet.get("files", []) or []
    return files


LONLAT_URL = "https://sciencebase.usgs.gov/manager/download/cm4aea2di00gx0uo7c1j7dlg3"
LONLAT_ZIP = "A3_9regions-lonlat-simplified.zip"


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main():
    """Uses the region outlines the USGS release ships in lon/lat ("A3_9regions-lonlat-simplified.zip").
    ScienceBase sits behind a bot check, so the zip is downloaded with curl (or a browser) first:
      curl -L -A "Mozilla/5.0" -o data.nosync/regions/A3_9regions-lonlat-simplified.zip <LONLAT_URL>
    """
    import shapefile
    from pyproj import CRS, Transformer
    DEST.mkdir(parents=True, exist_ok=True)
    z = DEST / LONLAT_ZIP
    if not z.exists() or z.stat().st_size < 1000:
        sys.exit(f"missing {z}\ndownload it first:\n  curl -L -A 'Mozilla/5.0' -o '{z}' '{LONLAT_URL}'")
    zipfile.ZipFile(z).extractall(DEST / "lonlat")
    shps = sorted((DEST / "lonlat").rglob("*.shp"))
    print("shapefiles:", [p.name for p in shps])
    found = {}
    keys = {norm(k): k for k in REGIONS}
    for shp in shps:
        prj = shp.with_suffix(".prj")
        tr = None
        if prj.exists():
            crs = CRS.from_wkt(prj.read_text())
            if not crs.is_geographic:
                tr = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
            print(f"  {shp.name}: {crs.name} (geographic={crs.is_geographic})")
        rd = shapefile.Reader(str(shp))
        fields = [f[0] for f in rd.fields[1:]]
        for sr in rd.iterShapeRecords():
            rec = dict(zip(fields, list(sr.record)))
            text = " ".join(str(v) for v in rec.values()) + " " + shp.stem
            hit = None
            for nk in sorted(keys, key=len, reverse=True):     # longest first (Mons Mouton Plateau)
                if nk in norm(text):
                    hit = keys[nk]; break
            if hit is None:
                print("    unmatched record:", rec); continue
            pts = np.asarray(sr.shape.points, dtype=np.float64)
            parts = list(sr.shape.parts) + [len(pts)]
            for a, b in zip(parts[:-1], parts[1:]):
                ring = pts[a:b]
                lon, lat = (tr.transform(ring[:, 0], ring[:, 1]) if tr else (ring[:, 0], ring[:, 1]))
                found.setdefault(hit, []).append(np.round(np.stack([lat, lon], 1), 6).tolist())
    out = []
    for key, display in REGIONS.items():
        cells = found.get(key)
        if not cells:
            print(f"  WARNING: no outline for {key}"); continue
        allpts = np.concatenate([np.asarray(c) for c in cells])
        u = np.stack([np.cos(np.radians(allpts[:, 0])) * np.cos(np.radians(allpts[:, 1])),
                      np.cos(np.radians(allpts[:, 0])) * np.sin(np.radians(allpts[:, 1])),
                      np.sin(np.radians(allpts[:, 0]))], 1).mean(0)
        clat = float(np.degrees(np.arcsin(u[2] / np.linalg.norm(u)))); clon = float(np.degrees(np.arctan2(u[1], u[0])))
        out.append({"key": key, "name": display, "center": [round(clat, 5), round(clon, 5)],
                    "n_cells": len(cells), "cells": cells})
        print(f"  {display:20s} {len(cells):3d} ring(s), {len(allpts):5d} pts, centre {clat:8.4f}, {clon:9.4f}")
    json.dump(out, open(DEST / "regions.json", "w"))
    print(f"wrote {DEST / 'regions.json'} ({len(out)} regions)")


if __name__ == "__main__":
    main()
