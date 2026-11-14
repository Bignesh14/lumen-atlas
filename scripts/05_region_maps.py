# Lumen Atlas - AI-assisted code (Claude, Anthropic); reviewed and run by Bignesh Das. Apache-2.0
"""Step 5: every pixel of every Artemis III candidate region.

For each region (USGS outline) and each 20 m pixel in its bounding box:
  Metal kernel 1 -> 360-direction terrain horizon (rays to 200 km, curvature included)
  Metal kernel 2 -> 2027-2028 hourly Sun/Earth visibility statistics
Outputs for the web app (web/assets/regions/<key>/):
  hillshade.png, sun.png, earth.png, both.png, night.png   colour maps (region outline kept bright)
  stats.bin   uint8 x4 per image pixel: avg illumination %, Earth-visible %, Sun+Earth %, longest night (h/4)
  hz.bin      uint8 x360 per grid point: horizon elevation, (deg + 12.8) * 10
  hzh.bin     int16 per grid point: ground height (m)
and web/assets/regions/index.json (projection, layout, per-region summaries).
Referee: on the first region, the GPU statistics are checked against a float64 NumPy reference.
Run:  python scripts/05_region_maps.py --stride 4 --only Malapert_Massif     (quick test, 80 m)
      python scripts/05_region_maps.py                                       (all regions, 20 m)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lumen.config import DATA, DEM_DIR, MAST_M, OUT_DIR, WEB_ASSETS   # noqa: E402
from lumen.dem import PolarDEM, ll_to_unit                             # noqa: E402
from lumen.horizon import horizon_mx                                   # noqa: E402
from lumen.imaging import block_mean, colorize, hillshade              # noqa: E402
from lumen.visibility import sky_arrays, stats_gpu, stats_ref          # noqa: E402

N_AZ = 360
MAX_IMG = 1000          # web image max side (pixels)
MAX_GRID = 90           # web horizon grid max side (points)
CHUNK = 65536           # observers per GPU launch


def load_dem():
    ori = json.load(open(OUT_DIR / "orientation.json"))["polar"]
    return PolarDEM(DEM_DIR / "LDEM_80S_20M.IMG", orientation=(ori["sx"], ori["sy"], ori["swap"]))


def region_box(dem, cells, margin_px=25):
    pts = np.concatenate([np.asarray(c) for c in cells])
    r, c = dem.ll_to_rc(pts[:, 0], pts[:, 1])
    r0 = int(np.floor(r.min())) - margin_px; r1 = int(np.ceil(r.max())) + margin_px
    c0 = int(np.floor(c.min())) - margin_px; c1 = int(np.ceil(c.max())) + margin_px
    return max(r0, 0), min(r1, dem.H - 2), max(c0, 0), min(c1, dem.W - 2)


def region_mask(dem, cells, r0, c0, H, W, stride):
    img = Image.new("L", (W, H), 0); dr = ImageDraw.Draw(img)
    for cell in cells:
        cell = np.asarray(cell)
        r, c = dem.ll_to_rc(cell[:, 0], cell[:, 1])
        dr.polygon(list(zip((c - c0) / stride, (r - r0) / stride)), fill=255)
    return np.asarray(img) > 0


def obs_for(dem, rows, cols):
    lat, lon = dem.rc_to_ll(rows, cols)
    u = ll_to_unit(lat, lon)
    h = dem.data[rows.astype(np.int64), cols.astype(np.int64)].astype(np.float64) * dem.factor + MAST_M
    return np.concatenate([u, h[:, None]], axis=1), lat, lon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1, help="observer spacing in 20 m pixels (1 = every pixel)")
    ap.add_argument("--only", default="", help="comma-separated region keys")
    args = ap.parse_args()

    dem = load_dem()
    regions = json.load(open(DATA / "regions" / "regions.json"))
    if args.only:
        keep = set(args.only.split(",")); regions = [r for r in regions if r["key"] in keep]
    sky = np.load(OUT_DIR / "sky_stats.npz")
    sun, earth = sky_arrays(sky["sun_km"], sky["earth_km"], sky["sun_r_deg"])
    print(f"DEM 20 m loaded; sky window from {sky['start']} ({sun.shape[0]} hours); stride {args.stride}")

    idx_path = WEB_ASSETS / "regions" / "index.json"
    index = json.load(open(idx_path)) if idx_path.exists() else {"regions": {}}
    index["projection"] = {"R": dem.R, "scale_m": dem.scale_m, "line_off": dem.line_off, "sample_off": dem.sample_off,
                           "sx": dem.sx, "sy": dem.sy, "swap": dem.swap}
    index["window"] = {"start": str(sky["start"]), "hours": int(sun.shape[0])}
    refereed = False
    for reg in regions:
        key = reg["key"]; t_reg = time.time()
        r0, r1, c0, c1 = region_box(dem, reg["cells"])
        s = args.stride
        rows = np.arange(r0, r1, s); cols = np.arange(c0, c1, s)
        H, W = len(rows), len(cols)
        mask = region_mask(dem, reg["cells"], r0, c0, H, W, s)
        RR, CC = np.meshgrid(rows, cols, indexing="ij")
        obs, lat, lon = obs_for(dem, RR.ravel().astype(np.float64), CC.ravel().astype(np.float64))
        N = obs.shape[0]
        g = int(np.ceil(max(H, W) / MAX_GRID))                       # web horizon grid stride
        gsel = (np.arange(H)[::g, None] * W + np.arange(W)[None, ::g]).ravel()
        gpos = {int(v): k for k, v in enumerate(gsel)}
        hz_grid = np.zeros((len(gsel), N_AZ), dtype=np.float32)
        stats = np.zeros((N, 8), dtype=np.float32)
        t0 = time.time()
        import mlx.core as mx
        for a in range(0, N, CHUNK):
            sub = obs[a:a + CHUNK]; n = sub.shape[0]
            hz = horizon_mx(dem, sub, n_az=N_AZ)
            stats[a:a + n] = stats_gpu(hz, sub, sun, earth, n_az=N_AZ)
            want = [i for i in range(a, a + n) if i in gpos]
            if want:
                cols_ = mx.array(np.array(want, dtype=np.int32) - a)
                hz_grid[[gpos[i] for i in want]] = np.array(mx.take(hz.reshape(N_AZ, n), cols_, axis=1)).T
            if not refereed:                                          # float64 referee on 6 observers
                pick = np.linspace(0, n - 1, 6).astype(int)
                hz_np = np.array(mx.take(hz.reshape(N_AZ, n), mx.array(pick.astype(np.int32)), axis=1)).T
                worst = np.zeros(7)
                for k, p in enumerate(pick):
                    ref = stats_ref(hz_np[k].astype(np.float64), sub[p].astype(np.float64),
                                    sky["sun_km"], sky["earth_km"], sky["sun_r_deg"])
                    worst = np.maximum(worst, np.abs(ref[:7] - stats[a + p, :7]))
                print(f"  STATS REFEREE (GPU vs float64, 6 observers x {sun.shape[0]} h): "
                      f"max |d| fractions {worst[:4].max()*100:.3f} %-points, runs {worst[4:7].max():.0f} h  -> "
                      f"{'PASS' if worst[:4].max() < 0.005 and worst[4:7].max() <= 2 else 'CHECK'}")
                refereed = True
        t_gpu = time.time() - t0
        S = stats.reshape(H, W, 8)
        out = WEB_ASSETS / "regions" / key; out.mkdir(parents=True, exist_ok=True)
        np.save(OUT_DIR / f"stats_{key}_s{s}.npy", S.astype(np.float16))

        # ---- images ------------------------------------------------------------------
        f = max(1, int(np.ceil(max(H, W) / MAX_IMG)))
        z = dem.data[r0:r1:s, c0:c1:s].astype(np.float32) * dem.factor
        shade = block_mean(hillshade(z, dem.scale_m * s), f)
        m = block_mean(mask.astype(np.float32), f) > 0.5
        Sd = block_mean(S, f)
        dim = np.where(m, 1.0, 0.45)
        Image.fromarray((shade * 255).astype(np.uint8)).save(out / "hillshade.png")
        for name, v, ramp in (("sun", Sd[..., 0], "sun"), ("earth", Sd[..., 2], "earth"),
                              ("both", Sd[..., 3], "both"), ("night", Sd[..., 4] / (24 * 15.0), "night")):
            im = np.asarray(colorize(v, ramp)).astype(np.float32)
            im = im * (0.55 + 0.45 * shade[..., None]) * dim[..., None]
            Image.fromarray(np.clip(im, 0, 255).astype(np.uint8)).save(out / f"{name}.png")
        Image.fromarray((m * 255).astype(np.uint8)).save(out / "mask.png")
        q = np.stack([np.round(Sd[..., 0] * 100), np.round(Sd[..., 2] * 100), np.round(Sd[..., 3] * 100),
                      np.clip(np.round(Sd[..., 4] / 4), 0, 255)], axis=-1).astype(np.uint8)
        q.tofile(out / "stats.bin")
        hq = np.clip(np.round((hz_grid + 12.8) * 10), 0, 255).astype(np.uint8)
        hq.tofile(out / "hz.bin")
        gh = np.round(obs[gsel, 3] - MAST_M).astype("<i2"); gh.tofile(out / "hzh.bin")
        gm = mask[::g, ::g].astype(np.uint8)

        # ---- summary -----------------------------------------------------------------
        inside = mask.ravel()
        Sm = stats[inside]
        best_both = int(np.flatnonzero(inside)[np.argmax(Sm[:, 3])])
        best_sun = int(np.flatnonzero(inside)[np.argmax(Sm[:, 0])])

        def pt(i):
            return {"lat": round(float(lat[i]), 5), "lon": round(float(lon[i]), 5), "row": int(i // W), "col": int(i % W),
                    "avg_illum": round(float(stats[i, 0]) * 100, 1), "earth_pct": round(float(stats[i, 2]) * 100, 1),
                    "both_pct": round(float(stats[i, 3]) * 100, 1), "longest_night_h": int(stats[i, 4]),
                    "height_m": round(float(obs[i, 3] - MAST_M))}
        index["regions"][key] = {
            "key": key, "name": reg["name"], "center": reg["center"], "cells": reg["cells"],
            "box": {"r0": int(r0), "c0": int(c0), "stride": s, "H": H, "W": W},
            "image": {"f": f, "H": Sd.shape[0], "W": Sd.shape[1]},
            "grid": {"g": g, "H": int(len(np.arange(H)[::g])), "W": int(len(np.arange(W)[::g])),
                     "mask": gm.ravel().tolist()},
            "summary": {"pixels": int(inside.sum()), "mean_avg_illum": round(float(Sm[:, 0].mean()) * 100, 1),
                        "mean_earth_pct": round(float(Sm[:, 2].mean()) * 100, 1),
                        "mean_both_pct": round(float(Sm[:, 3].mean()) * 100, 1),
                        "best_both": pt(best_both), "best_sun": pt(best_sun)},
        }
        print(f"{reg['name']:20s} {H}x{W} = {N:,} pixels ({inside.sum():,} inside)  GPU {t_gpu:6.1f} s  "
              f"total {time.time()-t_reg:5.1f} s | mean illum {Sm[:,0].mean()*100:5.1f}%  Earth {Sm[:,2].mean()*100:5.1f}%  "
              f"best Sun+Earth {Sm[:,3].max()*100:5.1f}%")
        json.dump(index, open(idx_path, "w"))
    print(f"wrote {idx_path}")


if __name__ == "__main__":
    main()
