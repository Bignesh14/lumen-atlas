# Lumen Atlas - AI-assisted code (Claude, Anthropic); reviewed and run by Bignesh Das. Apache-2.0
"""Step 3: download the official outlines of NASA's 9 Artemis III candidate regions.

Source: USGS Astrogeology, "Down Selected Artemis III Candidate Landing Site Navigational
Grids" (McClernan 2024), doi:10.5066/P1MEQ6UK, CC0.  We use the 1 km grid shapefiles:
the union of a region's 1 km cells is the region.  Coordinates are converted from the
USGS Lunar Polar Stereographic (LPS) projection to lat/lon with the .prj that ships with them.
Writes data.nosync/regions/regions.json
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


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    item = json.loads(get(API))
    files = all_files(item)
    print(f"ScienceBase item has {len(files)} files")
    wanted = []
    for f in files:
        name = f.get("name", ""); url = f.get("downloadUri") or f.get("url")
        if not url or not re.search(r"1\s*km", name, re.I):
            continue
        key = region_of(name)
        if key:
            wanted.append((key, name, url))
    if not wanted:
        print("No 1 km files matched. File names on the item are:")
        for f in files:
            print("   ", f.get("name"))
        sys.exit(1)
    for key, name, url in wanted:
        d = DEST / key; d.mkdir(exist_ok=True)
        blob = get(url)
        if name.lower().endswith(".zip") or blob[:2] == b"PK":
            zipfile.ZipFile(io.BytesIO(blob)).extractall(d)
        else:
            (d / name).write_bytes(blob)
        print(f"  {key:22s} <- {name} ({len(blob)/1e3:.0f} kB)")

    import shapefile
    from pyproj import CRS, Transformer
    out = []
    for key, display in REGIONS.items():
        shps = sorted((DEST / key).rglob("*.shp"))
        cells = []
        for shp in shps:
            prj = shp.with_suffix(".prj")
            crs = CRS.from_wkt(prj.read_text())
            tr = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
            rd = shapefile.Reader(str(shp))
            shapes = rd.shapes()
            if rd.shapeType in (5, 15, 25):            # polygons: each one is a 1 km cell
                rings = [np.asarray(s.points, dtype=np.float64) for s in shapes]
            else:                                      # grid lines: the region is their bounding box
                pts = np.concatenate([np.asarray(s.points, dtype=np.float64) for s in shapes])
                x0, y0 = pts.min(0); x1, y1 = pts.max(0)
                rings = [np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])]
            print(f"    {shp.name}: shapeType {rd.shapeType}, {len(shapes)} shapes")
            for ring in rings:
                lon, lat = tr.transform(ring[:, 0], ring[:, 1])
                cells.append(np.round(np.stack([lat, lon], 1), 6).tolist())
        if not cells:
            print(f"  WARNING: no shapes for {key}")
            continue
        allpts = np.concatenate([np.asarray(c) for c in cells])
        u = np.stack([np.cos(np.radians(allpts[:, 0])) * np.cos(np.radians(allpts[:, 1])),
                      np.cos(np.radians(allpts[:, 0])) * np.sin(np.radians(allpts[:, 1])),
                      np.sin(np.radians(allpts[:, 0]))], 1).mean(0)
        clat = float(np.degrees(np.arcsin(u[2] / np.linalg.norm(u)))); clon = float(np.degrees(np.arctan2(u[1], u[0])))
        out.append({"key": key, "name": display, "center": [round(clat, 5), round(clon, 5)],
                    "n_cells": len(cells), "cells": cells, "crs": crs.name})
        print(f"  {display:20s} {len(cells):5d} cells, centre {clat:8.4f}, {clon:9.4f}   crs: {crs.name}")
    json.dump(out, open(DEST / "regions.json", "w"))
    print(f"wrote {DEST / 'regions.json'} ({len(out)} regions)")


if __name__ == "__main__":
    main()
